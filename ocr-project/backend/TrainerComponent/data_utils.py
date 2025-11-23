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

def resize_and_preserve_aspect_ratio(img, target_height, target_width):
    """Preserve aspect ratio with padding instead of distortion"""
    h, w = img.shape
    
    # Calculate scaling factor to fit height
    scale_h = target_height / h
    new_w = int(w * scale_h)
    
    if new_w > target_width:
        # If still too wide, scale to fit width
        scale_w = target_width / w
        new_h = int(h * scale_w)
        img = cv2.resize(img, (target_width, new_h))
        # Pad height to maintain target dimensions
        pad_top = (target_height - new_h) // 2
        pad_bottom = target_height - new_h - pad_top
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, 
                               cv2.BORDER_CONSTANT, value=0)
    else:
        # Scale to target height, pad width
        img = cv2.resize(img, (new_w, target_height))
        pad_left = (target_width - new_w) // 2
        pad_right = target_width - new_w - pad_left
        img = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right, 
                               cv2.BORDER_CONSTANT, value=0)
    
    # Ensure exact target dimensions (fix for any off-by-one errors)
    if img.shape[0] != target_height or img.shape[1] != target_width:
        img = cv2.resize(img, (target_width, target_height))
    
    return img

def preprocess_text(text):
    """Add spaces before and after text as recommended in the paper"""
    return f" {text.strip()} "

def prepare_data(samples):
    images = []
    texts = []
    feature_width = IMAGE_WIDTH // 4
    
    print(f"Processing {len(samples)} samples...")
    
    for i, (img_path, text) in enumerate(samples):
        if i % 500 == 0:
            print(f"Processed {i}/{len(samples)} samples...")
            
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        try:
            # Apply aspect ratio preserving resize
            img = resize_and_preserve_aspect_ratio(img, IMAGE_HEIGHT, IMAGE_WIDTH)
            
            # Normalize
            img = (img / 255.0).astype(np.float32)         
            img = np.expand_dims(img, axis=-1)
            
            # Verify the shape is correct
            if img.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 1):
                print(f"Warning: Image shape {img.shape} doesn't match expected {(IMAGE_HEIGHT, IMAGE_WIDTH, 1)}")
                # Force correct shape
                img = cv2.resize(img.squeeze(), (IMAGE_WIDTH, IMAGE_HEIGHT))
                img = np.expand_dims(img, axis=-1)
            
            # Preprocess text with spaces
            processed_text = preprocess_text(text)
            text_labels = [CHARACTERS.index(c) for c in processed_text if c in CHARACTERS]
            
            if len(text_labels) > feature_width:
                print(f"Skipping sample: label too long ({len(text_labels)} > {feature_width}) for {img_path}")
                continue
                
            images.append(img)
            texts.append(text_labels)
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue
    
    if len(images) == 0:
        print("No valid images found!")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    print(f"Successfully processed {len(images)} images")
    
    # Pad sequences
    max_text_len = max(len(t) for t in texts) if texts else 0
    if max_text_len > 0:
        padded_texts = np.ones((len(texts), max_text_len), dtype='int32') * BLANK_TOKEN
        for i, text in enumerate(texts):
            padded_texts[i, :len(text)] = text
    else:
        padded_texts = np.array([]).reshape(0, 0)
    
    # Convert images to numpy array with explicit shape check
    images_array = []
    for img in images:
        if img.shape == (IMAGE_HEIGHT, IMAGE_WIDTH, 1):
            images_array.append(img)
        else:
            print(f"Warning: Skipping image with wrong shape {img.shape}")
    
    if len(images_array) == 0:
        print("No valid images after shape check!")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    images_array = np.array(images_array)
    print(f"Final images array shape: {images_array.shape}")
    
    feature_width = IMAGE_WIDTH // 4
    input_length = np.ones((len(images_array), 1), dtype='int32') * feature_width
    label_length = np.array([[len(t)] for t in texts], dtype='int32') if texts else np.array([])
    
    return images_array, padded_texts, input_length, label_length

def load_samples(label_csv_path, images_folder):
    df = pd.read_csv(label_csv_path)
    samples = []
    for _, row in df.iterrows():
        img_filename = row['IMAGE']
        label = str(row['MEDICINE_NAME'])
        img_path = os.path.join(images_folder, img_filename)
        if os.path.exists(img_path):
            samples.append((img_path, label))
        else:
            print("Missing image:", img_path)
    return samples