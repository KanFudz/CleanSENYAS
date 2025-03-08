import cv2
import mediapipe as mp
import os

dataset_path = "dataset/"  # Path to store images
labels = [str(i) for i in range(10)]  # Labels from 0-9

# Ensure dataset directory and subdirectories exist
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
for label in labels:
    label_path = os.path.join(dataset_path, label)
    if not os.path.exists(label_path):
        os.makedirs(label_path)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
current_label_index = 0
count = 0
max_samples = 200  # Number of samples per number
capturing = False  # Flag to start capturing

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # Flip the camera horizontally
    h, w, c = frame.shape
    roi_size = int(min(w, h) * 0.4)  # Adjusted ROI size to prevent exceeding resolution
    roi_x = min((2 * w) // 3, w - roi_size)
    roi_y = min(h // 4, h - roi_size)
    roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size].copy()  # Copy ROI before processing
    
    img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    
    if result.multi_hand_landmarks and capturing and count < max_samples:
        filename = os.path.join(dataset_path, labels[current_label_index], f"{count}.jpg")
        cv2.imwrite(filename, roi)  # Save image without overlay
        count += 1
        if count >= max_samples:
            capturing = False  # Stop capturing when limit is reached
    
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_size, roi_y+roi_size), (0, 255, 0), 2)
    cv2.putText(frame, f"Capturing: {labels[current_label_index]} ({count}/{max_samples})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "Press Space to Start/Stop", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press N/B to Change Number", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Dataset Collection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        capturing = not capturing if count < max_samples else False  # Toggle capturing
        count = 0 if capturing else count  # Reset count when starting new capture
    elif key == ord('n'):
        current_label_index = (current_label_index + 1) % 10  # Next number
        count = 0  # Reset count
    elif key == ord('b'):
        current_label_index = (current_label_index - 1) % 10  # Previous number
        count = 0  # Reset count

cap.release()
cv2.destroyAllWindows()
