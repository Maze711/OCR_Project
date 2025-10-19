import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image, ImageDraw, ImageFont
import string
import argparse
import matplotlib.pyplot as plt

CHARACTERS = string.ascii_uppercase + string.digits
IMAGE_WIDTH, IMAGE_HEIGHT = 128, 64
FONT_PATH = "fonts/ARIAL.ttf"
MAX_TEXT_LENGTH = 10

def create_argument_parser():
    parser = argparse.ArgumentParser(description='OCR Text Recognition')
    parser.add_argument('--model', type=str, default='ocr_models/ocr_pred_model.h5', help='Path to prediction model')
    parser.add_argument('--image', type=str, help='Path to image file for recognition')
    parser.add_argument('--generate', type=str, help='Generate an image with this text and recognize it')
    parser.add_argument('--test', action='store_true', help='Run tests on generated samples')
    parser.add_argument('--debug', action='store_true', help='Enable detailed debugging')
    return parser

def validate_input_text(text):
    validated = []
    for c in text.upper():
        if c in CHARACTERS:
            validated.append(c)
    return ''.join(validated)[:MAX_TEXT_LENGTH]

def load_ocr_model(model_path='ocr_models/ocr_pred_model.h5', debug=False):
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def show_image_with_prediction(image_path, original_text, predicted_text):
    img = cv2.imread(image_path)
    if img is not None:
        plt.figure(figsize=(10, 4))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"Original: '{original_text}'\nPredicted: '{predicted_text}'")
        plt.axis('off')
        plt.show()

def decode_predictions(pred):
    pred = pred[:, :, :len(CHARACTERS)]
    input_length = np.ones(pred.shape[0]) * pred.shape[1]
    decoded = K.ctc_decode(pred, input_length=input_length, greedy=True, beam_width=5, top_paths=1)
    decoded_indices = K.get_value(decoded[0][0])[0]
    result = []
    for idx in decoded_indices:
        if idx == -1:
            continue
        if 0 <= idx < len(CHARACTERS):
            result.append(CHARACTERS[idx])
    return ''.join(result)

def generate_text_image(text, output_path):
    img = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, size=24)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (IMAGE_WIDTH - text_width) // 2
    y = (IMAGE_HEIGHT - text_height) // 2
    draw.text((x, y), text, font=font, fill=0)
    img.save(output_path)

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=(0, -1))

def predict_with_retry(model, image_path, original_text, max_retries=3):
    for attempt in range(max_retries):
        processed_img = preprocess_image(image_path)
        pred = model.predict(processed_img)
        predicted_text = decode_predictions(pred)
        show_image_with_prediction(image_path, original_text, predicted_text)
        if predicted_text == original_text:
            return predicted_text
        if attempt < max_retries - 1:
            choice = input("Try again? (y/n): ").lower()
            if choice != 'y':
                break
    return predicted_text

def run_tests(model):
    test_texts = ["ABC", "123", "HELLO", "WORLD", "OCR", "TEST", "PYTHON", "TENSOR", "12345", "MODEL"]
    results = []
    for text in test_texts:
        img_path = f"test_{text}.jpg"
        generate_text_image(text, img_path)
        predicted = predict_with_retry(model, img_path, text, max_retries=1)
        correct = predicted == text
        results.append((text, predicted, correct))
    correct_predictions = sum(1 for _, _, correct in results if correct)
    total_tests = len(results)
    accuracy = correct_predictions / total_tests
    print(f"Accuracy: {correct_predictions}/{total_tests} ({accuracy:.2%})")

def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    print("\nLoading OCR model...")
    try:
        model = load_ocr_model(args.model)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    if args.generate:
        text = validate_input_text(args.generate)
        img_path = f"generated_{text}.jpg"
        generate_text_image(text, img_path)
        predicted = predict_with_retry(model, img_path, text)
        print(f"\nFinal Prediction: '{predicted}'")
    elif args.image:
        processed_img = preprocess_image(args.image)
        pred = model.predict(processed_img)
        predicted_text = decode_predictions(pred)
        print(f"\nPredicted Text: '{predicted_text}'")
    elif args.test:
        run_tests(model)
    else:
        while True:
            choice = input("Select option (1: Generate, 2: Recognize, 3: Test, 4: Exit): ").strip()
            if choice == '1':
                text = input("Enter text: ").strip()
                text = validate_input_text(text)
                img_path = f"generated_{text}.jpg"
                generate_text_image(text, img_path)
                predicted = predict_with_retry(model, img_path, text)
                print(f"Final Result: '{predicted}'")
            elif choice == '2':
                img_path = input("Enter image path: ").strip()
                predicted = predict_with_retry(model, img_path, "unknown")
                print(f"Predicted: '{predicted}'")
            elif choice == '3':
                run_tests(model)
            elif choice == '4':
                break

if __name__ == "__main__":
    main()