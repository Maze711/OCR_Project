import tensorflow as tf

keras_model_path = "ocr_models/CNN_BiLSTM.keras"
new_model_path = "ocr_models/CNN_BiLSTM_v2.keras"

try:
    print(f"Loading model from: {keras_model_path}")
    model = tf.keras.models.load_model(keras_model_path)
    print("Model loaded successfully!")

    print(f"Re-saving model to: {new_model_path}")
    model.save(new_model_path)
    print(f"Model re-saved to: {new_model_path}")
except Exception as e:
    print(f"Error occurred: {e}")