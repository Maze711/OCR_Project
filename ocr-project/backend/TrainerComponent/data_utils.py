import string
import cv2
import numpy as np
import pandas as pd
import os

CHARACTERS = string.ascii_letters + string.digits + " -'.,:"
NUM_CHARS = len(CHARACTERS)
BLANK_TOKEN = NUM_CHARS
IMAGE_WIDTH, IMAGE_HEIGHT = 160, 64

def ctc_lambda_func(args):
    from tensorflow.keras import backend as K
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def prepare_data(samples):
    images = []
    texts = []
    
    # FIXED: time_steps is 20, not 40
    time_steps = 20  # Must match model_builder.py
    
    for img_path, text in samples:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        img = (img / 255.0).astype(np.float32)
        img = np.expand_dims(img, axis=-1)
        
        text_labels = [CHARACTERS.index(c) for c in text if c in CHARACTERS]
        
        # Check against correct max_label_length
        if len(text_labels) > time_steps:
            print(f"⚠️ Truncating label '{text}' from {len(text_labels)} to {time_steps} chars")
            text_labels = text_labels[:time_steps]
        
        images.append(img)
        texts.append(text_labels)
    
    max_text_len = max(len(t) for t in texts)
    padded_texts = np.ones((len(texts), max_text_len), dtype='int32') * BLANK_TOKEN
    for i, text in enumerate(texts):
        padded_texts[i, :len(text)] = text
    
    # FIXED: input_length should be time_steps (20)
    input_length = np.ones((len(images), 1), dtype='int32') * time_steps
    label_length = np.array([[len(t)] for t in texts], dtype='int32')
    
    return np.array(images), padded_texts, input_length, label_length

def load_samples(label_csv_path, images_folder):
    df = pd.read_csv(label_csv_path)
    samples = []
    # Support both 'MEDICINE_NAME' and 'NOTE_TEXT' columns
    label_col = None
    for col in ['MEDICINE_NAME', 'NOTE_TEXT', 'label', 'LABEL', 'text']:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        raise ValueError(f"No valid label column found in {label_csv_path}")
    for _, row in df.iterrows():
        img_filename = row['IMAGE']
        label = str(row[label_col])
        img_path = os.path.join(images_folder, img_filename)
        if os.path.exists(img_path):
            samples.append((img_path, label))
        else:
            print("Missing image:", img_path)
    return samples