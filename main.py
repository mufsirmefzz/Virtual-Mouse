import cv2
import mediapipe as mp
import pyautogui
import random
import util
import time
from pynput.mouse import Button, Controller
mouse = Controller()


screen_width, screen_height = pyautogui.size()

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)


def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]  # Assuming only one hand is detected
        index_finger_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
        return index_finger_tip
    return None, None


def is_in_roi(x, y):
    return 0.25 < x < 0.75 and 0.25 < y < 0.75

def move_mouse(index_finger_tip):
    if index_finger_tip is not None and is_in_roi(index_finger_tip.x, index_finger_tip.y):
        x = int(index_finger_tip.x * screen_width)
        y = int(index_finger_tip.y / 2 * screen_height)
        pyautogui.moveTo(x, y)

def is_left_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) > 90 and
            thumb_index_dist > 50
    )


def is_right_click(landmark_list, thumb_index_dist):
    # Check if the middle finger is bent (angle between middle finger bones < 50 degrees)
    # and thumb and index fingers are apart (> 50 distance)
    return (
        util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and  # Middle finger bent
        util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and    # Index finger not bent
        thumb_index_dist > 50  # Thumb and index are apart
    )



def is_double_click(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist > 50
    )


def is_screenshot(landmark_list, thumb_index_dist):
    return (
            util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
            util.get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
            thumb_index_dist < 50
    )


# Add a timestamp for debounce mechanism
last_left_click_time = 0
click_delay = 0.5  # 500 ms delay between clicks

def detect_gesture(frame, landmark_list, processed):
    global last_left_click_time
    if len(landmark_list) >= 21:

        index_finger_tip = find_finger_tip(processed)
        thumb_index_dist = util.get_distance([landmark_list[4], landmark_list[5]])

        # Move mouse if thumb and index are apart
        if util.get_distance([landmark_list[4], landmark_list[5]]) < 50 and util.get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90:
            move_mouse(index_finger_tip)

        # Detect left click gesture
        elif is_left_click(landmark_list, thumb_index_dist) and time.time() - last_left_click_time > click_delay:
            last_left_click_time = time.time()
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Detect right click gesture
        elif is_right_click(landmark_list, thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Other gestures like double-click and screenshot
        elif is_double_click(landmark_list, thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        elif is_screenshot(landmark_list, thumb_index_dist):
            im1 = pyautogui.screenshot()
            label = random.randint(1, 1000)
            im1.save(f'my_screenshot_{label}.png')
            cv2.putText(frame, "Screenshot Taken", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
calibrated_finger_tip = None

def calibrate_gesture(processed):
    global calibrated_finger_tip
    if processed.multi_hand_landmarks:
        calibrated_finger_tip = find_finger_tip(processed)
        print("Gesture calibrated!")

def main():
    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    is_calibrated = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = hands.process(frameRGB)

        # Call calibration once
        if not is_calibrated:
            calibrate_gesture(processed)
            is_calibrated = True

        if is_calibrated:
            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            detect_gesture(frame, landmark_list, processed)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()





