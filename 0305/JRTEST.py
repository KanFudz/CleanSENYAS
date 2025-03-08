import pygame
import sys
import cv2
import os
import json
import numpy as np
import tensorflow.lite as tflite
import mediapipe as mp

def get_collision_rect(img):
    """ Returns a rect representing the non-transparent area of an image """
    mask = pygame.mask.from_surface(img)
    if mask.count() == 0:
        return img.get_rect()
    
    collision_rect = mask.get_bounding_rects()[0]
    full_rect = img.get_rect()
    collision_rect.x += full_rect.x
    collision_rect.y += full_rect.y
    return collision_rect

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

class NumberRecognitionState:
    def __init__(self, game):
        self.game = game
        self.cap = cv2.VideoCapture(0)
        self.prediction = ""
    
    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        roi = frame[:, w//2:]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        result = hands.process(roi_rgb)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                input_data = np.array(landmarks, dtype=np.float32).reshape(1, -1)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                self.prediction = labels[np.argmax(output_data)]
    
    def render(self):
        font = pygame.font.Font(None, 36)
        text_surface = font.render(f"Detected Number: {self.prediction}", True, (255, 255, 255))
        self.game.screen.blit(text_surface, (50, 50))
    
    def exit(self):
        self.cap.release()
        cv2.destroyAllWindows()

# Example integration into the game class
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1024, 600))
        pygame.display.set_caption("SENYAS - Number Recognition")
        self.number_recognition = NumberRecognitionState(self)
        self.clock = pygame.time.Clock()
    
    def run(self):
        while True:
            self.screen.fill((0, 0, 0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.number_recognition.update()
            self.number_recognition.render()
            pygame.display.flip()
            self.clock.tick(30)

if __name__ == "__main__":
    Game().run()
