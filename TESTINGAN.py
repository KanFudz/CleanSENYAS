import tensorflow.lite as tflite
import os

model_path = "asl_number_classifier.tflite"

if os.path.exists(model_path):
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("Model loaded successfully!")
        print("Input details:", interpreter.get_input_details())
        print("Output details:", interpreter.get_output_details())
    except Exception as e:
        print("Error loading model:", e)
else:
    print("Model file not found.")
