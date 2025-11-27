import os
import tensorflow as tf

# Ensure output directory exists
models_dir = "ocr_modelsv2"
os.makedirs(models_dir, exist_ok=True)

# Load the saved Keras prediction model
model_path = os.path.join(models_dir, "ocr_model_keras.keras")
ocr_model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

# TensorFlow Lite Conversion
converter = tf.lite.TFLiteConverter.from_keras_model(ocr_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_fp16 = converter.convert()
out_path = os.path.join(models_dir, "ocr_model_production_fp16.tflite")

with open(out_path, "wb") as f:
    f.write(tflite_fp16)

print("Saved converted lite model successfully", out_path)