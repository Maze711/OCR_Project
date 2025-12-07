import tensorflow as tf
import os
from Trainer import AttentionLayer  # Import the custom AttentionLayer class

def convert_keras_to_tflite(keras_model_path, tflite_model_path):
    """
    Converts a Keras model to TensorFlow Lite format.

    Args:
        keras_model_path (str): Path to the Keras model file (.keras or .h5).
        tflite_model_path (str): Path to save the converted TensorFlow Lite model (.tflite).
    """
    # Load the Keras model with the custom AttentionLayer
    model = tf.keras.models.load_model(
        keras_model_path, custom_objects={"AttentionLayer": AttentionLayer}
    )
    print(f"Keras model loaded from: {keras_model_path}")

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable optimizations
    tflite_model = converter.convert()
    print("Model converted to TensorFlow Lite format.")

    # Save the TensorFlow Lite model
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"TensorFlow Lite model saved to: {tflite_model_path}")

if __name__ == "__main__":
    # Define paths
    keras_model_path = "ocr_models/CNN_BiLSTM.keras"  # Path to the Keras model
    tflite_model_path = "ocr_models/ocr_model_production.tflite"  # Path to save the TFLite model

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(tflite_model_path), exist_ok=True)

    # Convert the model
    convert_keras_to_tflite(keras_model_path, tflite_model_path)