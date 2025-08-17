import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import win32gui
import win32con


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01  

# ===== INIT =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

last_right_click_time = 0
last_arrow_time = 0
last_key_time = 0
last_single_click_time = 0
last_double_click_time = 0
# Затримки
SINGLE_CLICK_DEBOUNCE = 8.0  
DOUBLE_CLICK_DEBOUNCE = 8.0  
RIGHT_CLICK_DEBOUNCE = 8.0   
ARROW_DEBOUNCE = 0.5
KEY_DEBOUNCE = 0.7

cursor_x, cursor_y = pyautogui.position()
arrow_mode = False
keyboard_mode = False  
button_pressed = False
keyboard_button_pressed = False  
caps_lock = False  

# ===== Віртуальна клавіатура =====
keyboard_layout = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
    ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/'],
    ['space'],
    ['backspace'],
    ['enter'],
    ['caps'] 
]

keyboard_pos = (10, 100)
key_size = (40, 40)
key_spacing = 8

def draw_keyboard(img):
    if not keyboard_mode:
        return
    x, y = keyboard_pos
    for row_idx, row in enumerate(keyboard_layout):
        for col_idx, key in enumerate(row):
            key_x = x + col_idx * (key_size[0] + key_spacing)
            if row_idx == 4 and key == 'backspace':
                key_x += key_spacing * 2
            elif row_idx == 4 and key == 'enter':
                key_x += key_spacing * 5
            elif row_idx == 4 and key == 'caps':
                key_x += key_spacing * 8  
            key_y = y + row_idx * (key_size[1] + key_spacing)
            if key == 'space':
                cv2.rectangle(img, (key_x, key_y), (key_x + key_size[0]*4, key_y + key_size[1]), (100, 100, 100), -1)
                cv2.putText(img, 'space', (key_x + 5, key_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            elif key == 'backspace':
                cv2.rectangle(img, (key_x, key_y), (key_x + key_size[0]*3, key_y + key_size[1]), (100, 100, 100), -1)
                cv2.putText(img, 'backspace', (key_x + 5, key_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 250), 2)
            elif key == 'enter':
                cv2.rectangle(img, (key_x, key_y), (key_x + key_size[0]*3, key_y + key_size[1]), (100, 100, 100), -1)
                cv2.putText(img, 'enter', (key_x + 5, key_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            elif key == 'caps':
                color = (0, 200, 0) if caps_lock else (100, 100, 100)  # Зелений, якщо Caps Lock увімкнено
                cv2.rectangle(img, (key_x, key_y), (key_x + key_size[0]*2, key_y + key_size[1]), color, -1)
                cv2.putText(img, 'caps', (key_x + 5, key_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                cv2.rectangle(img, (key_x, key_y), (key_x + key_size[0], key_y + key_size[1]), (100, 100, 100), -1)
                cv2.putText(img, key.upper() if caps_lock and key.isalpha() else key, 
                           (key_x + 10, key_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def check_key_press(x, y, threshold):
    global last_key_time, caps_lock
    if not keyboard_mode:
        return
    current_time = time.time()
    if current_time - last_key_time < KEY_DEBOUNCE:
        return

    img_x = int(x * frame.shape[1])
    img_y = int(y * frame.shape[0])
    kx, ky = keyboard_pos
    for row_idx, row in enumerate(keyboard_layout):
        for col_idx, key in enumerate(row):
            key_x = kx + col_idx * (key_size[0] + key_spacing)
            if row_idx == 4 and key == 'backspace':
                key_x += key_spacing * 2
            elif row_idx == 4 and key == 'enter':
                key_x += key_spacing * 5
            elif row_idx == 4 and key == 'caps':
                key_x += key_spacing * 8
            key_y = ky + row_idx * (key_size[1] + key_spacing)
            key_w = key_size[0] * (4 if key == 'space' else 3 if key in ['backspace', 'enter'] else 2 if key == 'caps' else 1)
            if key_x <= img_x <= key_x + key_w and key_y <= img_y <= key_y + key_size[1]:
                try:
                    if key == 'caps':
                        caps_lock = not caps_lock  
                        print(f"Caps Lock {'ON' if caps_lock else 'OFF'}")
                    elif key == 'space':
                        pyautogui.write(' ')
                    elif key == 'backspace':
                        pyautogui.press('backspace')
                    elif key == 'enter':
                        pyautogui.press('enter')
                    else:
                        pyautogui.write(key.upper() if caps_lock and key.isalpha() else key)
                    print(f"pressed: {key}")
                    last_key_time = current_time
                except Exception as e:
                    print(f"error during key press: {e}")
                break


def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 +
                   (p1.y - p2.y) ** 2 +
                   (p1.z - p2.z) ** 2)

def detect_left_click(thumb_tip, index_tip, threshold):
    global last_single_click_time
    current_time = time.time()
    if not keyboard_mode:  # Затримка застосовується тільки поза режимом клавіатури
        if current_time - last_single_click_time < SINGLE_CLICK_DEBOUNCE:
            return False
    dist = calculate_distance(thumb_tip, index_tip)
    if dist < threshold:
        if not keyboard_mode:
            last_single_click_time = current_time
        return True
    return False

def detect_double_click(thumb_tip, ring_tip, threshold):
    global last_double_click_time
    current_time = time.time()
    if current_time - last_double_click_time < DOUBLE_CLICK_DEBOUNCE:
        return
    dist = calculate_distance(thumb_tip, ring_tip)
    if dist < threshold:
        pyautogui.doubleClick()
        print("double click")
        last_double_click_time = current_time
        time.sleep(0.2)

def detect_right_click(thumb_tip, middle_tip, threshold):
    global last_right_click_time
    current_time = time.time()
    if current_time - last_right_click_time < RIGHT_CLICK_DEBOUNCE:
        return
    dist = calculate_distance(thumb_tip, middle_tip)
    if dist < threshold:
        pyautogui.rightClick()
        print("right click")
        last_right_click_time = current_time

def detect_simple_arrow(index_tip, palm_center, sensitivity=0.1):
    global last_arrow_time
    global arrow_mode
    current_time = time.time()
    if not arrow_mode:
        return
    if current_time - last_arrow_time < ARROW_DEBOUNCE:
        return

    dx = index_tip.x - palm_center[0]
    dy = index_tip.y - palm_center[1]

    if abs(dx) > abs(dy):
        if dx > sensitivity:
            pyautogui.press('right')
            print("arrow →")
            last_arrow_time = current_time
        elif dx < -sensitivity:
            pyautogui.press('left')
            print("arrow ←")
            last_arrow_time = current_time
    else:
        if dy < -sensitivity:
            pyautogui.press('up')
            print("arrow ↑")
            last_arrow_time = current_time
        elif dy > sensitivity:
            pyautogui.press('down')
            print("arrow ↓")
            last_arrow_time = current_time

# ===== INTERFACE: кнопки =====
button_text = "Toggle Arrow Mode"
keyboard_button_text = "Toggle Keyboard"
button_pos = (10, 60)
keyboard_button_pos = (240, 60)
button_size = (220, 40)

def draw_button(img, pressed, text, pos):
    x, y = pos
    w, h = button_size
    color = (0, 200, 0) if pressed else (0, 100, 0)
    cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
    cv2.putText(img, text, (x+10, y+28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

def check_button_click(x, y, pos):
    bx, by = pos
    bw, bh = button_size
    return bx <= x <= bx+bw and by <= y <= by+bh


def nothing(x):
    pass

# ===== Вікно із налаштуваннями =====
cv2.namedWindow("Settings", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Settings", 300, 600) 
cv2.moveWindow("Settings", 650, screen_h - 480)  
cv2.createTrackbar("Smoothing", "Settings", 5, 20, nothing)
cv2.createTrackbar("Move Scale", "Settings", 25, 50, nothing)
cv2.createTrackbar("Click Threshold", "Settings", 12, 30, nothing)
cv2.createTrackbar("Arrow Sensitivity", "Settings", 10, 30, nothing)
cv2.createTrackbar("Single Click Delay", "Settings", 80, 200, nothing)
cv2.createTrackbar("Double Click Delay", "Settings", 80, 200, nothing)
cv2.createTrackbar("Right Click Delay", "Settings", 80, 200, nothing)

# ===== Миша для інтерфейсу  =====
def mouse_callback(event, x, y, flags, param):
    global arrow_mode, button_pressed, keyboard_mode, keyboard_button_pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        if check_button_click(x, y, button_pos):
            arrow_mode = not arrow_mode
            button_pressed = arrow_mode
            print(f"Arrow mode toggled {'on' if arrow_mode else 'off'}")
        elif check_button_click(x, y, keyboard_button_pos):
            keyboard_mode = not keyboard_mode
            keyboard_button_pressed = keyboard_mode
            print(f"Keyboard mode toggled {'on' if keyboard_mode else 'off'}")
       

# ===== Налаштування вікна =====
cv2.namedWindow("Gesture Mouse Controller", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Mouse Controller", 640, 480)
cv2.moveWindow("Gesture Mouse Controller", 0, screen_h - 480)
cv2.setMouseCallback("Gesture Mouse Controller", mouse_callback)

# ===== Головний цикл =====
try:
    cursor_x, cursor_y = pyautogui.position()
    last_right_click_time = 0

    while cap.isOpened():
        SMOOTHING = cv2.getTrackbarPos("Smoothing", "Settings")
        MOVE_SCALE = cv2.getTrackbarPos("Move Scale", "Settings") / 10
        CLICK_THRESHOLD = cv2.getTrackbarPos("Click Threshold", "Settings") / 100
        ARROW_SENSITIVITY = cv2.getTrackbarPos("Arrow Sensitivity", "Settings") / 100
        SINGLE_CLICK_DEBOUNCE = cv2.getTrackbarPos("Single Click Delay", "Settings") / 10
        DOUBLE_CLICK_DEBOUNCE = cv2.getTrackbarPos("Double Click Delay", "Settings") / 10
        RIGHT_CLICK_DEBOUNCE = cv2.getTrackbarPos("Right Click Delay", "Settings") / 10

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            palm_center = ((wrist.x + middle_mcp.x) / 2, (wrist.y + middle_mcp.y) / 2)

            # Оновлення позиції курсора лише поза режимом клавіатури
            if not keyboard_mode:
                target_x = np.interp(palm_center[0], [0.2, 0.8], [0, screen_w])
                target_y = np.interp(palm_center[1], [0.2, 0.8], [0, screen_h])

                cursor_x += (target_x - cursor_x) / SMOOTHING * MOVE_SCALE
                cursor_y += (target_y - cursor_y) / SMOOTHING * MOVE_SCALE

                cursor_x = max(0, min(screen_w - 1, cursor_x))
                cursor_y = max(0, min(screen_h - 1, cursor_y))

                pyautogui.moveTo(cursor_x, cursor_y)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]

            if not keyboard_mode:
                if detect_left_click(thumb_tip, index_tip, CLICK_THRESHOLD):
                    pyautogui.click()
                    print("single click")
                    time.sleep(0.15)
                detect_double_click(thumb_tip, ring_tip, CLICK_THRESHOLD)
                detect_right_click(thumb_tip, middle_tip, CLICK_THRESHOLD)
                detect_simple_arrow(index_tip, palm_center, sensitivity=ARROW_SENSITIVITY)
            else:
                if detect_left_click(thumb_tip, index_tip, CLICK_THRESHOLD):
                    img_x = int(index_tip.x * frame.shape[1])
                    img_y = int(index_tip.y * frame.shape[0])
                    if check_button_click(img_x, img_y, button_pos):
                        arrow_mode = not arrow_mode
                        button_pressed = arrow_mode
                        print(f"Arrow mode toggled {'on' if arrow_mode else 'off'}")
                        time.sleep(0.15)
                    elif check_button_click(img_x, img_y, keyboard_button_pos):
                        keyboard_mode = not keyboard_mode
                        keyboard_button_pressed = keyboard_mode
                        print(f"Keyboard mode toggled {'on' if keyboard_mode else 'off'}")
                        if keyboard_mode:
                            hwnd = win32gui.FindWindow(None, "Gesture Mouse Controller")
                            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                        else:
                            hwnd = win32gui.FindWindow(None, "Gesture Mouse Controller")
                            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                        time.sleep(0.15)
                    else:
                        check_key_press(index_tip.x, index_tip.y, CLICK_THRESHOLD)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        draw_button(frame, button_pressed, button_text, button_pos)
        draw_button(frame, keyboard_button_pressed, keyboard_button_text, keyboard_button_pos)
        draw_keyboard(frame)

        status_text = f"Arrow: {'ON' if arrow_mode else 'OFF'} | Keyboard: {'ON' if keyboard_mode else 'OFF'} | Caps: {'ON' if caps_lock else 'OFF'}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Gesture Mouse Controller", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()