import cv2
import numpy as np
import tensorflow.lite as tflite
import mediapipe as mp

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="asl_number_classifier.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Label map
labels = [str(i) for i in range(10)]

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip and process frame
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    roi = frame[:, w//2:]  # Right side ROI
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    result = hands.process(roi_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.extend([lm.x, lm.y, lm.z])
                
                # Update bounding box
                # x_min = min(x_min, x)
                # y_min = min(y_min, y)
                # x_max = max(x_max, x)
                # y_max = max(y_max, y)
            
            # Convert landmarks to NumPy array and reshape
            input_data = np.array(landmarks, dtype=np.float32).reshape(1, -1)
            
            # Perform inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            prediction = np.argmax(output_data)
            
            # Display prediction
            cv2.putText(frame, f"Prediction: {labels[prediction]}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Draw bounding box
            # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Draw landmarks
            mp_drawing.draw_landmarks(roi, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Show frame
    cv2.imshow("ASL Number Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
