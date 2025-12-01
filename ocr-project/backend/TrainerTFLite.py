import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
)
from TrainerComponent.model_builder import build_ocr_model_tflite_compatible
from TrainerComponent.data_utils import prepare_data, load_samples, NUM_CHARS, IMAGE_WIDTH, IMAGE_HEIGHT
from TrainerComponent.callbacks import TerminalLogger, ValidationCallback
from TrainerComponent.tflite_utils import convert_to_tflite_with_flex, test_tflite_model
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(BASE_DIR, 'ocr_dataset')
logs_dir = os.path.join(BASE_DIR, 'logs')
models_dir = os.path.join(BASE_DIR, 'ocr_modelsv2')
sample_logs_dir = os.path.join(BASE_DIR, 'sample_logs')

os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(sample_logs_dir, exist_ok=True)

if os.path.exists(logs_dir):
    shutil.rmtree(logs_dir)
os.makedirs(logs_dir, exist_ok=True)

class SaveKerasOnBest(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_path, keras_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.keras_path = keras_path
        self.last_best_loss = None

    def on_epoch_end(self, epoch, logs=None):
        # Only copy if the checkpoint file exists and val_loss improved
        if os.path.exists(self.checkpoint_path):
            current_loss = logs.get('val_loss')
            if self.last_best_loss is None or (current_loss is not None and current_loss < self.last_best_loss):
                # Copy best_model.keras to ocr_model_keras.keras
                import shutil
                shutil.copy2(self.checkpoint_path, self.keras_path)
                print(f"📦 Saved best Keras model to: {self.keras_path}")
                self.last_best_loss = current_loss

if __name__ == "__main__":
    # Load real samples only
    real_samples = load_samples(
        os.path.join(dataset_dir, "Training", "training_labels.csv"),
        os.path.join(dataset_dir, "Training", "training_words")
    )
    all_samples = real_samples  # No synthetic samples

    print(f"Total samples loaded: {len(all_samples)}")
    if len(all_samples) == 0:
        print("No samples found!")
        exit(1)
    random.shuffle(all_samples)
    num_total = len(all_samples)
    num_train = int(0.8 * num_total)
    num_val = min(500, int(0.1 * num_total))
    num_test = min(500, num_total - num_train - num_val)
    train_samples = all_samples[:num_train]
    val_samples = all_samples[num_train:num_train+num_val]
    test_samples = all_samples[num_train+num_val:num_train+num_val+num_test]

    # Prepare data
    X_train, y_train, il_train, ll_train = prepare_data(train_samples)
    X_val, y_val, il_val, ll_val = prepare_data(val_samples)
    X_test, y_test, il_test, ll_test = prepare_data(test_samples)

    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    # print(X_train.shape, y_train.shape, il_train.shape, ll_train.shape)
    # print("Max train label length:", np.max(ll_train))
    # print("Max val label length:", np.max(ll_val))
    # print("Train samples:", X_train.shape[0])
    # print("Val samples:", X_val.shape[0])

    # Build TFLite compatible model
    train_model, pred_model, time_steps = build_ocr_model_tflite_compatible()
    print(f"📐 Model time steps: {time_steps}")
    print(f"📐 Number of characters: {NUM_CHARS}")
    print(f"📐 Expected output shape: (batch_size, {time_steps}, {NUM_CHARS + 1})")
    print(f"📐 Actual pred_model output shape: {pred_model.output_shape}")

    # Before defining callbacks
    validation_callback = ValidationCallback(pred_model, X_val, y_val, logs_dir)

    # Define callbacks
    checkpoint_path = os.path.join(models_dir, "best_model.keras")
    keras_model_path = os.path.join(models_dir, "ocr_model_keras.keras")
    if os.path.exists(checkpoint_path):
        print("📥 Restoring best model weights for training...")
        train_model.load_weights(checkpoint_path)

    callbacks = [
        TensorBoard(log_dir=logs_dir, histogram_freq=1),
        TerminalLogger(validation_callback=validation_callback),  # <-- Pass the callback here
        ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            save_weights_only=False,
            monitor='val_loss',
            verbose=1
        ),
        SaveKerasOnBest(checkpoint_path, keras_model_path),  # <-- Add this callback
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        # EarlyStopping(
        #     monitor='val_loss',
        #     patience=10,
        #     restore_best_weights=True
        # ),
        validation_callback  # <-- Add this to callbacks if you want TensorBoard metrics
    ]

    print("🚀 Starting training...")
    print(f"📊 Logs will be saved to: {logs_dir}")

    history = train_model.fit(
        [X_train, y_train, il_train, ll_train],
        np.zeros(len(X_train)),
        validation_data=([X_val, y_val, il_val, ll_val], np.zeros(len(X_val))),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    if os.path.exists(checkpoint_path):
        print("📥 Loading best model...")
        pred_model.load_weights(checkpoint_path)

    # Convert to TFLite
    print("🔄 Converting to TFLite...")
    tflite_path = convert_to_tflite_with_flex(pred_model, models_dir)

    # Check TFLite output shape
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    output_details = interpreter.get_output_details()
    print(f"📊 TFLite output shape: {output_details[0]['shape']}")  # <-- Add this line

    # Test the TFLite model
    test_tflite_model(tflite_path, X_test)

    print(f"✅ Training completed! TFLite model saved to: {tflite_path}")
    print(f"📊 Training logs saved to: {logs_dir}")