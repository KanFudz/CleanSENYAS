import cv2
import mediapipe as mp
import numpy as np
import tensorflow.lite as tflite

# Load TFLite model
TFLITE_MODEL_PATH = "gesture_model.tflite"
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

# Define gestures (make sure they match your training labels)
GESTURES = ["hello", "i love you", "thanks"]  # Modify this based on your training data

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract (x, y, z) coordinates
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            # Reshape for TFLite model input
            landmarks = np.expand_dims(landmarks, axis=0).astype(np.float32)

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], landmarks)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Get predicted gesture
            gesture_index = np.argmax(output_data)
            predicted_gesture = GESTURES[gesture_index]
            confidence = output_data[0][gesture_index]

            # Display the predicted gesture
            cv2.putText(frame, f"Gesture: {predicted_gesture} ({confidence:.2f})", 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display video feed
    cv2.imshow("Gesture Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
