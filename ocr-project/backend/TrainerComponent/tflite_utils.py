import os
import tensorflow as tf
import numpy as np

def convert_to_tflite_optimized(pred_model, models_dir, X_train=None):
    """Convert to TFLite with FP16 quantization"""
    keras_model_path = os.path.join(models_dir, "ocr_model_keras.keras")
    pred_model.save(keras_model_path)
    print(f"Saved Keras model to: {keras_model_path}")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(pred_model)
    
    # Apply DEFAULT optimizations (without quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Apply FP16 quantization for GPU acceleration and size reduction
    converter.target_spec.supported_types = [tf.float16]
    
    # Essential settings for compatibility
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS,    # Fallback for unsupported ops
    ]
    
    converter.experimental_new_converter = True
    converter.experimental_enable_resource_variables = True
    converter.allow_custom_ops = True
    
    try:
        tflite_model = converter.convert()
        tflite_path = os.path.join(models_dir, "ocr_model_production_fp16.tflite")
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ Successfully converted to FP16 TFLite: {tflite_path}")
        print(f"Model size: {len(tflite_model) / (1024*1024):.2f} MB")
        
        # Model analysis
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"📊 Input details: {input_details[0]['dtype']}, shape: {input_details[0]['shape']}")
        print(f"📊 Output details: {output_details[0]['dtype']}, shape: {output_details[0]['shape']}")
        
        return tflite_path
    except Exception as e:
        print(f"❌ FP16 TFLite conversion failed: {e}")

def test_tflite_model(tflite_path, test_images):
    print("\n🧪 Testing TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"📊 Input details: {input_details[0]}")
    print(f"📊 Output details: {output_details[0]}")
    
    if len(test_images) > 0:
        test_image = test_images[0:1]
        
        # For FP16 models, input is still expected as float32
        # TFLite handles internal FP16 conversion
        interpreter.set_tensor(input_details[0]['index'], test_image.astype(np.float32))
        
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"✅ TFLite model works! Output shape: {output_data.shape}")
        print(f"Output range: [{output_data.min():.4f}, {output_data.max():.4f}]")
        return True
    else:
        print("❌ No test images available")
        return False