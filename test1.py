import cv2
import mediapipe as mp
import pyautogui
import numpy as np


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Get screen size
screen_width, screen_height = pyautogui.size()

# Capture video from the webcam
cap = cv2.VideoCapture(0)

def get_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        return result.multi_hand_landmarks, result.multi_handedness  # Return both landmarks and handedness
    return None, None

def calculate_finger_positions(landmarks, image_width, image_height):
    positions = {}
    for id, lm in enumerate(landmarks.landmark):
        cx, cy = int(lm.x * image_width), int(lm.y * image_height)
        positions[id] = (cx, cy)
    return positions

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image_height, image_width, _ = image.shape
    hand_landmarks, handedness = get_hand_landmarks(image)

    if hand_landmarks:
        for i, landmarks in enumerate(hand_landmarks):
            positions = calculate_finger_positions(landmarks, image_width, image_height)
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the label (left or right hand)
            label = handedness[i].classification[0].label

            # Define finger tip landmarks
            index_tip = positions[8]
            middle_tip = positions[12]
            thumb_tip = positions[4]
            wrist = positions[0]

            # Actions for the right hand
            if label == "Right":
                # Move mouse cursor
                index_x, index_y = index_tip
                pyautogui.moveTo(index_x * screen_width / image_width, index_y * screen_height / image_height)

                # Click if index and middle fingers are touching
                distance_between_index_and_middle = np.linalg.norm(np.array(index_tip) - np.array(middle_tip))
                if distance_between_index_and_middle < 50:
                    pyautogui.click()

                # Select text if thumb and index finger are touching
                distance_between_thumb_and_index = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))

                if distance_between_thumb_and_index < 20:
                    # Simulate pressing the left mouse button
                    pyautogui.mouseDown()


            # Actions for the left hand
            if label == "Left":
                # Scroll
                distance_between_thumb_and_wrist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))

                if distance_between_thumb_and_wrist < 160:
                    pyautogui.scroll(35)  # Scroll up
                elif distance_between_thumb_and_wrist > 200:
                    pyautogui.scroll(-35)  # Scroll down

    cv2.imshow('Hand Mouse Control', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
