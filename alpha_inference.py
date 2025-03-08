import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Load trained model
model = load_model("asl_mlp_model.h5")

# Load label encoder from pickle
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip and convert color
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame
    result = hands.process(rgb_frame)
    
    # Extract hand landmarks
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract (x, y) coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])
            
            # Convert to NumPy array
            landmarks = np.array(landmarks).flatten().reshape(1, -1)  # Shape (1, 42)
            
            # Normalize landmarks (same as during training)
            landmarks = landmarks.astype('float32') / np.max(landmarks)

            # Predict ASL letter
            prediction = model.predict(landmarks)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            
            # Display prediction
            cv2.putText(frame, f"Prediction: {predicted_label}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow("ASL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
