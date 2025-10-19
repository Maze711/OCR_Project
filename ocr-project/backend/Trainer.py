import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM, Dense, BatchNormalization, SpatialDropout2D, Lambda, Attention
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
import random
import string
from PIL import Image, ImageDraw, ImageFont
import datetime
import gc
import shutil

CHARACTERS = string.ascii_letters + string.digits + " -'.,"
NUM_CHARS = len(CHARACTERS)
BLANK_TOKEN = NUM_CHARS
IMAGE_WIDTH, IMAGE_HEIGHT = 160, 64

dataset_dir = 'ocr_dataset'
logs_dir = 'ocr_logs'
models_dir = 'ocr_models'
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention = Attention()

    def call(self, inputs):
        return self.attention([inputs, inputs])

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def generate_text_sample():
    cursive_patterns = [
        "flourish", "script", "elegant", "brush", "swing",
        "grace", "flow", "curve", "loop", "connect"
    ]
    
    if random.random() < 0.7:
        parts = [
            ("fl", "ow", "ing"),
            ("br", "ush", ""),
            ("sc", "rip", "t"),
            ("co", "nn", "ect"),
            ("cur", "ve", "")
        ]
        base = random.choice(parts)
        return base[0] + base[1] + base[2]
    else:
        return ''.join(random.choices(string.ascii_uppercase, k=random.randint(3, 6)))

def generate_image(text, image_path):
    img = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), 255)
    draw = ImageDraw.Draw(img)
    
    try:
        if text.islower():
            cursive_fonts = [
                # "fonts/AlexBrush-Regular.ttf",
                # "fonts/DancingScript-Regular.ttf", 
                "fonts/GreatVibes-Regular.ttf"
            ]
            selected_font = random.choice(cursive_fonts)
            font = ImageFont.truetype(selected_font, size=32)
            slant = random.randint(-30, 30)
            img = img.transform(
                img.size, 
                Image.AFFINE, 
                (1, slant/35, random.randint(-10, 10),
                 0, 1, random.randint(-5, 5))
            )
            draw = ImageDraw.Draw(img)
            x, y = 15, 20
        else:
            font = ImageFont.truetype("fonts/ARIAL.ttf", size=28)
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            x = (IMAGE_WIDTH - text_width) // 2
            y = (IMAGE_HEIGHT - 40) // 2
            
        draw.text((x, y), text, font=font, fill=0)
        img.save(image_path, "PNG")
        return True

    except Exception as e:
        print(f"Error in image generation: {e}")
        return False

def generate_dataset(num_samples, folder):
    samples = []
    os.makedirs(os.path.join(dataset_dir, folder), exist_ok=True)
    
    for i in range(num_samples):
        text = generate_text_sample()
        img_path = os.path.join(dataset_dir, folder, f"{folder}_{i}.png")
        generate_image(text, img_path)
        samples.append((img_path, text))
    
    return samples

def augment_image(img):
    # Add Gaussian noise
    if random.random() < 0.3:  # Lower probability
        noise = np.random.normal(0, 0.07, img.shape)  # Lower stddev
        img = img + noise
        img = np.clip(img, 0, 1)
    # Add blur
    if random.random() < 0.3:  # Lower probability
        ksize = random.choice([3])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    # Add grain (salt & pepper)
    if random.random() < 0.15:  # Lower probability
        amount = 0.01  # Lower amount
        num_salt = np.ceil(amount * img.size * 0.5)
        num_pepper = np.ceil(amount * img.size * 0.5)
        coords = [np.random.randint(0, i, int(num_salt)) for i in img.shape]
        img[tuple(coords)] = 1
        coords = [np.random.randint(0, i, int(num_pepper)) for i in img.shape]
        img[tuple(coords)] = 0
    return img

def prepare_data(samples):
    images = []
    texts = []
    
    for img_path, text in samples:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img = (img / 255.0).astype(np.float32)
        # img = augment_image(img)  # <-- Add this line
        img = np.expand_dims(img, axis=-1)
        images.append(img)

        text_labels = [CHARACTERS.index(c) for c in text if c in CHARACTERS]
        if not text_labels:
            continue
        texts.append(text_labels)
    
    max_text_len = max(len(t) for t in texts)
    padded_texts = np.ones((len(texts), max_text_len), dtype='int32') * BLANK_TOKEN
    for i, text in enumerate(texts):
        padded_texts[i, :len(text)] = text
    
    feature_width = IMAGE_WIDTH // 4
    input_length = np.ones((len(images), 1), dtype='int64') * feature_width
    label_length = np.array([[len(t)] for t in texts], dtype='int64')
    
    return np.array(images), padded_texts, input_length, label_length

def build_ocr_model():
    input_img = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    
    x = Conv2D(64, (5,5), activation='swish', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    
    x = Conv2D(128, (3,3), activation='swish', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,1))(x)
    
    _, h, w, c = x.shape
    x = Reshape((w, h * c))(x)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(x)
    x = AttentionLayer()(x)
    x = Bidirectional(LSTM(96, return_sequences=True, dropout=0.3))(x)

    output = Dense(NUM_CHARS + 1, activation='softmax')(x)
    
    labels = Input(name='labels', shape=(None,), dtype='int32')
    input_length = Input(name='input_length', shape=(1,), dtype='int64')
    label_length = Input(name='label_length', shape=(1,), dtype='int64')
    
    loss_out = Lambda(ctc_lambda_func, name='ctc')([output, labels, input_length, label_length])    
    train_model = Model(
        inputs=[input_img, labels, input_length, label_length],
        outputs=loss_out
    )
    train_model.compile(
        optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5, beta_1=0.9, beta_2=0.999),
        loss={'ctc': lambda y_true, y_pred: y_pred}
    )
    
    pred_model = Model(inputs=input_img, outputs=output)
    
    return train_model, pred_model

class TerminalLogger(tf.keras.callbacks.Callback):
    def __init__(self, validation_callback=None):
        super().__init__()
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_filename = f"{now}-result.txt"
        self.log_filepath = os.path.join(logs_dir, self.log_filename)
        self.validation_callback = validation_callback

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_line = (
            f"Epoch {epoch+1:2d}/{self.params['epochs']} "
            f"- accuracy: {logs.get('accuracy', 0):.4f} "
            f"- loss: {logs.get('loss', 0):.4f} "
            f"- val_accuracy: {logs.get('val_accuracy', 0):.4f} "
            f"- val_loss: {logs.get('val_loss', 0):.4f}"
        )

        # Add custom metrics if available
        if self.validation_callback is not None:
            log_line += (
                f" - val_word_accuracy: {getattr(self.validation_callback, 'last_word_acc', 0):.4f}"
                f" - val_precision: {getattr(self.validation_callback, 'last_precision', 0):.4f}"
                f" - val_recall: {getattr(self.validation_callback, 'last_recall', 0):.4f}"
                f" - val_f1_score: {getattr(self.validation_callback, 'last_f1_score', 0):.4f}"
            )
        with open(self.log_filepath, "a") as f:
            f.write(log_line + "\n")

class ImageLogger(tf.keras.callbacks.Callback):
    def __init__(self, images, log_dir):
        super().__init__()
        self.images = images
        self.log_dir = log_dir

    def on_train_begin(self, logs=None):
        # Select up to 10 images to log
        imgs = self.images[:10]
        imgs = np.array(imgs)
        # Add batch dimension if needed
        if imgs.ndim == 3:
            imgs = np.expand_dims(imgs, -1)
        file_writer = tf.summary.create_file_writer(self.log_dir)
        with file_writer.as_default():
            tf.summary.image("Training samples", imgs, step=0)

class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, pred_model, X_val, y_val, log_dir):
        super().__init__()
        self.pred_model = pred_model
        self.X_val = X_val
        self.y_val = y_val
        self.writer = tf.summary.create_file_writer(log_dir)
        self.best_accuracy = 0.0

    def decode_prediction(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        decoded, _ = K.ctc_decode(pred, input_length=input_len, greedy=True, beam_width=5)
        decoded_text = ''.join([CHARACTERS[idx] for idx in K.get_value(decoded[0][0]) if idx != -1 and idx < NUM_CHARS])
        return decoded_text

    def on_epoch_end(self, epoch, logs=None):
        sample_images = []
        correct = 0
        TP, FP, FN = 0, 0, 0
        total = min(10, len(self.X_val))

        for i in range(total):
            img = (self.X_val[i] * 255).astype(np.uint8).squeeze()
            pred = self.pred_model.predict(np.expand_dims(self.X_val[i], axis=0))
            pred_text =  self.decode_prediction(pred)
            true_indices = self.y_val[i]
            true_text = ''.join([CHARACTERS[idx] for idx in true_indices if idx < NUM_CHARS])

            if pred_text == true_text:
                correct += 1
                TP += 1
            else:
                FP += 1
                FN += 1

            img_rgb = np.stack([img]*3, axis=-1)
            pil_img = Image.fromarray(img_rgb)
            canvas = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT + 30), "white")
            canvas.paste(pil_img, (0, 0))
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()
            draw.text((5, IMAGE_HEIGHT + 5), f"True: {true_text}", fill=(0, 0, 0), font=font)
            color = (0, 200, 0) if pred_text == true_text else (200, 0, 0)
            draw.text((5, IMAGE_HEIGHT + 15), f"Pred: {pred_text}", fill=color, font=font)
            img_arr = np.array(canvas).astype(np.float32) / 255.0
            sample_images.append(img_arr)

        word_acc = correct / total if total > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        with self.writer.as_default():
            tf.summary.image("Image + Text OCR Results", np.stack(sample_images), step=epoch)
            tf.summary.scalar('val_word_accuracy', word_acc, step=epoch)
            tf.summary.scalar('val_precision', precision, step=epoch)
            tf.summary.scalar('val_recall', recall, step=epoch)
            tf.summary.scalar('val_f1_score', f1_score, step=epoch)
            self.writer.flush()

        # Save best accuracy model
        if word_acc > self.best_accuracy:
            self.best_accuracy = word_acc
            self.pred_model.save(os.path.join(models_dir, 'ocr_model_best_accuracy.h5'))
            self.model.save(os.path.join(models_dir, "ocr_model_final.h5"))
            self.pred_model.save(os.path.join(models_dir, 'ocr_pred_model.h5'))
            self.model.save_weights(os.path.join(models_dir, "ocr_model_best_weights.h5"))


# Delete and recreate ocr_logs for a fresh TensorBoard run
if os.path.exists(logs_dir):
    shutil.rmtree(logs_dir)
os.makedirs(logs_dir, exist_ok=True)

if __name__ == "__main__":
    train_samples = generate_dataset(5000, 'train')  # Increased to 1500 samples
    test_samples = generate_dataset(1000, 'test')
    
    X_train, y_train, il_train, ll_train = prepare_data(train_samples)
    X_test, y_test, il_test, ll_test = prepare_data(test_samples)

    final_weights_path = os.path.join(models_dir, "ocr_model_final_weights.h5")
    checkpoint_weights_path = os.path.join(models_dir, "ocr_model_best_weights.h5")

    # Always rebuild the model and load weights if available
    if os.path.exists(final_weights_path):
        print("Loading final model weights...")
        train_model, pred_model = build_ocr_model()
        train_model.load_weights(final_weights_path)
    elif os.path.exists(checkpoint_weights_path):
        print("Loading checkpoint model weights...")
        train_model, pred_model = build_ocr_model()
        train_model.load_weights(checkpoint_weights_path)
    else:
        print("Creating new model...")
        train_model, pred_model = build_ocr_model()

    tensorboard_callback = TensorBoard(log_dir=logs_dir)
    validation_callback = ValidationCallback(pred_model, X_test, y_test, logs_dir)
    terminal_logger = TerminalLogger(validation_callback=validation_callback)

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_weights_path,
        save_best_only=False,
        save_weights_only=True,
        verbose=1
    )

    reduce_lr_callback = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = train_model.fit(
        [X_train, y_train, il_train, ll_train],
        np.zeros(len(X_train)),
        validation_data=([X_test, y_test, il_test, ll_test], np.zeros(len(X_test))),
        epochs=50,
        batch_size=32,
        callbacks=[
            tensorboard_callback,
            terminal_logger,
            checkpoint_callback,
            validation_callback,
            reduce_lr_callback,
            early_stop
        ]
    )

    # Save final weights after training
    train_model.save_weights(final_weights_path)