import os
import cv2
import gc
import random
import string
import shutil
import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Reshape, Bidirectional, LSTM,
    Dense, BatchNormalization, SpatialDropout2D, LayerNormalization,
    Dropout, Lambda, Attention
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import (
    TensorBoard, ModelCheckpoint, ReduceLROnPlateau,
    EarlyStopping
)
from tensorflow.keras import backend as K

# ---------------------- CONFIG ----------------------
CHARACTERS = string.ascii_letters + string.digits + " -'.,"
NUM_CHARS = len(CHARACTERS)
BLANK_TOKEN = NUM_CHARS
IMAGE_WIDTH, IMAGE_HEIGHT = 160, 64

dataset_dir = 'data'
logs_dir = 'logs'
models_dir = 'models'
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# ---------------------- ATTENTION ----------------------
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention = Attention()

    def call(self, inputs):
        return self.attention([inputs, inputs])

# ---------------------- LOSS ----------------------
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# ---------------------- DATASET ----------------------
def generate_text_sample():
    words = [
        "curve", "flow", "loop", "connect", "flourish", "brush", "script",
        "wave", "shadow", "motion", "twist", "blend", "curveflow", "grace"
    ]
    if random.random() < 0.6:
        return random.choice(words)
    else:
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(3, 6)))

def generate_image(text, path):
    img = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), 255)
    draw = ImageDraw.Draw(img)
    try:
        if text.islower():
            cursive_fonts = [
                "fonts/AlexBrush-Regular.ttf",
                "fonts/DancingScript-Regular.ttf",
                "fonts/GreatVibes-Regular.ttf"
            ]
            font = ImageFont.truetype(random.choice(cursive_fonts), 32)
        else:
            font = ImageFont.truetype("fonts/ARIAL.ttf", 28)

        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        x = (IMAGE_WIDTH - text_width) // 2
        y = (IMAGE_HEIGHT - 40) // 2
        draw.text((x, y), text, font=font, fill=0)
        img.save(path, "PNG")
        return True
    except:
        return False

def generate_dataset(n, subset):
    folder = os.path.join(dataset_dir, subset)
    os.makedirs(folder, exist_ok=True)
    samples = []
    for i in range(n):
        txt = generate_text_sample()
        path = os.path.join(folder, f"{subset}_{i}.png")
        if generate_image(txt, path):
            samples.append((path, txt))
    return samples

def augment_image(img):
    if random.random() < 0.25:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if random.random() < 0.25:
        noise = np.random.normal(0, 0.04, img.shape)
        img = np.clip(img + noise, 0, 1)
    return img

def prepare_data(samples):
    imgs, texts = [], []
    for p, t in samples:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        im = cv2.resize(im, (IMAGE_WIDTH, IMAGE_HEIGHT))
        im = (im / 255.0).astype(np.float32)
        im = augment_image(im)
        imgs.append(np.expand_dims(im, -1))
        texts.append([CHARACTERS.index(c) for c in t if c in CHARACTERS])

    maxlen = max(len(t) for t in texts)
    padded = np.ones((len(texts), maxlen), dtype='int32') * BLANK_TOKEN
    for i, t in enumerate(texts):
        padded[i, :len(t)] = t
    feat_w = IMAGE_WIDTH // 4
    in_len = np.ones((len(imgs), 1)) * feat_w
    lbl_len = np.array([[len(t)] for t in texts])
    return np.array(imgs), padded, in_len, lbl_len

# ---------------------- MODEL ----------------------
def build_model():
    inp = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1))

    x = Conv2D(64, (5, 5), activation='swish', padding='same')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='swish', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1))(x)

    x = SpatialDropout2D(0.2)(x)
    _, h, w, c = x.shape
    x = Reshape((w, h * c))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    x = AttentionLayer()(x)
    x = Bidirectional(LSTM(96, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)

    out = Dense(NUM_CHARS + 1, activation='softmax')(x / 1.1)

    labels = Input(name='labels', shape=(None,), dtype='int32')
    input_len = Input(name='input_length', shape=(1,), dtype='int64')
    label_len = Input(name='label_length', shape=(1,), dtype='int64')

    loss_out = Lambda(ctc_lambda_func, name='ctc')([out, labels, input_len, label_len])

    train_model = Model(inputs=[inp, labels, input_len, label_len], outputs=loss_out)
    train_model.compile(
        optimizer=AdamW(learning_rate=2e-4, weight_decay=1e-5),
        loss={'ctc': lambda y_true, y_pred: y_pred}
    )
    pred_model = Model(inputs=inp, outputs=out)
    return train_model, pred_model

# ---------------------- CALLBACKS ----------------------
class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, Xv, Yv, logdir):
        super().__init__()
        self.model = model
        self.Xv = Xv
        self.Yv = Yv
        self.writer = tf.summary.create_file_writer(logdir)
        self.best = 0

    @staticmethod
    def collapse_repeated(text):
        res, prev = [], None
        for c in text:
            if c != prev:
                res.append(c)
            prev = c
        return ''.join(res)

    def decode(self, p):
        il = np.ones(p.shape[0]) * p.shape[1]
        dec, _ = K.ctc_decode(p, input_length=il, greedy=True)
        seq = [CHARACTERS[i] for i in K.get_value(dec[0][0]) if i < NUM_CHARS and i != -1]
        return self.collapse_repeated(''.join(seq))

    def on_epoch_end(self, epoch, logs=None):
        correct, total = 0, min(10, len(self.Xv))
        imgs = []
        for i in range(total):
            pr = self.model.predict(np.expand_dims(self.Xv[i], 0))
            pred = self.decode(pr)
            gt = ''.join([CHARACTERS[i] for i in self.Yv[i] if i < NUM_CHARS])
            if pred == gt: correct += 1
            img = np.stack([self.Xv[i].squeeze()] * 3, -1)
            canvas = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT + 20), "white")
            canvas.paste(Image.fromarray((img * 255).astype(np.uint8)), (0, 0))
            d = ImageDraw.Draw(canvas)
            d.text((2, IMAGE_HEIGHT + 3), f"T:{gt} | P:{pred}", fill=(0, 0, 0))
            imgs.append(np.array(canvas).astype(np.float32) / 255.)
        acc = correct / total
        with self.writer.as_default():
            tf.summary.image("Validation Samples", np.stack(imgs), step=epoch)
            tf.summary.scalar("val_word_accuracy", acc, step=epoch)
            self.writer.flush()
        if acc > self.best:
            self.best = acc
            self.model.save(os.path.join(models_dir, "ocr_best_pred_model.h5"))

# ---------------------- MAIN ----------------------
if os.path.exists(logs_dir):
    shutil.rmtree(logs_dir)
os.makedirs(logs_dir, exist_ok=True)

if __name__ == "__main__":
    train_samples = generate_dataset(1000, 'train')
    test_samples = generate_dataset(200, 'test')

    X_train, y_train, il_train, ll_train = prepare_data(train_samples)
    X_test, y_test, il_test, ll_test = prepare_data(test_samples)

    final_path = os.path.join(models_dir, "ocr_final_weights.h5")
    train_model, pred_model = build_model()
    if os.path.exists(final_path):
        train_model.load_weights(final_path)

    callbacks = [
        TensorBoard(log_dir=logs_dir),
        ModelCheckpoint(
            filepath=os.path.join(models_dir, "ocr_checkpoint.h5"),
            save_weights_only=True, save_best_only=True
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ValidationCallback(pred_model, X_test, y_test, logs_dir)
    ]

    history = train_model.fit(
        [X_train, y_train, il_train, ll_train],
        np.zeros(len(X_train)),
        validation_data=([X_test, y_test, il_test, ll_test], np.zeros(len(X_test))),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    train_model.save_weights(final_path)
    gc.collect()
