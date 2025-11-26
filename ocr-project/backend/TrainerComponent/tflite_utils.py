import os
import tensorflow as tf

def convert_to_tflite_with_flex(pred_model, models_dir):
    keras_model_path = os.path.join(models_dir, "ocr_model_keras.keras")
    pred_model.save(keras_model_path)
    print(f"Saved Keras model to: {keras_model_path}")
    converter = tf.lite.TFLiteConverter.from_keras_model(pred_model)
    converter.experimental_new_converter = True
    converter.experimental_enable_resource_variables = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.allow_custom_ops = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    try:
        tflite_model = converter.convert()
        tflite_path = os.path.join(models_dir, "ocr_model_production_fp16.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ Successfully converted to TFLite: {tflite_path}")
        print(f"Model size: {len(tflite_model) / (1024*1024):.2f} MB")
        return tflite_path
    except Exception as e:
        print(f"❌ TFLite conversion failed: {e}")
        return None

def test_tflite_model(tflite_path, test_images):
    print("\n🧪 Testing TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"📊 Input details: {input_details[0]}")
    print(f"📊 Output details: {output_details[0]}")
    print(f"📊 Output shape (should be [1, time_steps, num_classes]): {output_details[0]['shape']}")
    if len(test_images) > 0:
        test_image = test_images[0:1]
        print(f"🧪 Test image shape: {test_image.shape}")
        interpreter.set_tensor(input_details[0]['index'], test_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(f"✅ TFLite model works! Output shape: {output_data.shape}")
        return True
    else:
        print("❌ No test images available")
        return False