import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('MODEL/asl_mlp_model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model to a file
with open('MODEL/asl_mlp_model.tflite', 'wb') as f:
    f.write(tflite_model)