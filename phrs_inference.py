import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
sequence = []
sentence = []
predictions = []
threshold = 0.85  # Increased from 0.7 to 0.85
sequence_length = 30  # Match the sequence length used in training
min_consecutive_predictions = 5  # Require more consecutive predictions
prediction_history_size = 15  # Increased prediction history size
confidence_threshold = 0.90  # Additional threshold for high confidence predictions

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='MODEL/gesture_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# After initializing the interpreter, print the expected input shape
print("Expected input shape:", input_details[0]['shape'])

# Actions/Classes
actions = np.array(['hello', 'thanks', 'iloveyou'])

# Visualization colors
colors = [(245,117,16), (117,245,16), (16,117,245)]

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    )
    # Draw left hand connections
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    )
    # Draw right hand connections
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )

def extract_keypoints(results):
    # Pose landmarks (33 points * 4 values = 132 features)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Left hand landmarks (21 points * 3 values = 63 features)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # Right hand landmarks (21 points * 3 values = 63 features)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, lh, rh])  # Total: 132 + 63 + 63 = 258 features

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, f'{actions[num]} ({prob:.2f})', (0, 85+num*40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

def has_hands(results):
    """Check if at least one hand is detected"""
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

def detect_motion(keypoints, threshold=0.01):
    """Check if there's meaningful motion in the keypoints"""
    # Only check hand keypoints (last 126 values of the 258 features)
    hand_points = keypoints[-126:]  # 21 points * 3 coords * 2 hands
    return np.mean(np.abs(hand_points)) > threshold

# Start capture
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally for natural movement
        frame = cv2.flip(frame, 1)
            
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # Process frames
        keypoints = extract_keypoints(results)

        # Only append keypoints if there's meaningful motion or hands are detected
        if has_hands(results) and detect_motion(keypoints):
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]  # Keep only last 30 frames
        else:
            # Clear sequence when no meaningful motion is detected
            sequence = []
            predictions = []  # Also clear predictions to avoid false positives
        
        if len(sequence) == sequence_length:
            try:
                # Prepare input data
                input_data = np.array(sequence, dtype=np.float32)
                
                # Normalize
                input_data = (input_data - np.mean(input_data)) / (np.std(input_data) + 1e-6)
                
                # Reshape according to model's expected input shape (1, 30, 258)
                input_data = np.expand_dims(input_data, axis=0)
                
                # Set the tensor and run inference
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                
                # Get prediction results
                res = interpreter.get_tensor(output_details[0]['index'])[0]
                max_prob = res[np.argmax(res)]
                
                # Only predict if we have very high confidence
                if max_prob > threshold:
                    prediction = actions[np.argmax(res)]
                    predictions.append(prediction)
                    
                    # Keep prediction history manageable
                    if len(predictions) > prediction_history_size:
                        predictions = predictions[-prediction_history_size:]
                    
                    # Update sentence only if we have consistent predictions
                    if (len(predictions) >= min_consecutive_predictions and 
                        len(set(predictions[-min_consecutive_predictions:])) == 1 and 
                        max_prob > confidence_threshold):
                        
                        current_pred = predictions[-1]
                        if len(sentence) == 0 or current_pred != sentence[-1]:
                            sentence.append(current_pred)
                            
                    # Keep last 5 words in sentence
                    if len(sentence) > 5:
                        sentence = sentence[-5:]
                
                # Visualize probabilities
                image = prob_viz(res, actions, image, colors)
                
            except Exception as e:
                print(f"Inference error: {e}")
                print(f"Input shape: {input_data.shape}")
                print(f"Expected shape: {input_details[0]['shape']}")
        
        # Show predictions and info
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Calculate actual FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if 'prev_time' in locals() else 0.0
        prev_time = current_time
        
        cv2.putText(image, f'FPS: {int(fps)}', (550, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show frame
        cv2.imshow('Sign Language Recognition', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()