# OCR Project

This project implements an Optical Character Recognition (OCR) system using deep learning techniques. The OCR model is built with TensorFlow and Keras, and it is designed to recognize both printed and cursive text from images.

## Project Structure

```
ocr-project
├── backend
│   ├── Trainer.py          # Main training logic for the OCR model
│   ├── Main.py             # Entry point for the OCR application
│   └── config.yaml         # Configuration file for model parameters and settings
├── ocr_models
│   ├── ocr_checkpoint.h5   # Checkpoint of the trained OCR model
│   ├── ocr_full_model.h5   # Complete architecture and weights of the trained model
│   ├── ocr_pred_model.h5   # Model for making predictions
│   └── ocr_train_weights.h5 # Weights of the trained model
├── ocr_dataset
│   ├── train               # Directory for training images and labels
│   └── test                # Directory for test images and labels
├── ocr_logs                # Directory for TensorBoard logs
├── fonts
│   ├── ARIAL.ttf           # Font for generating printed text images
│   ├── AlexBrush-Regular.ttf # Font for generating cursive text images
│   ├── DancingScript-Regular.ttf # Font for generating cursive text images
│   └── GreatVibes-Regular.ttf # Font for generating cursive text images
├── requirements.txt         # Python dependencies for the project
└── README.md                # Documentation for the project
```

## Setup Instructions

1. **Clone the Repository**
   ```
   git clone <repository-url>
   cd ocr-project
   ```

2. **Install Dependencies**
   Ensure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**
   Place your training and testing images in the `ocr_dataset/train` and `ocr_dataset/test` directories, respectively.

4. **Train the Model**
   Run the training script:
   ```
   python backend/Trainer.py
   ```

5. **Run the OCR Application**
   Use the main script to recognize text from images:
   ```
   python backend/Main.py --image <path-to-image>
   ```

## Usage Guidelines

- The model can recognize both printed and cursive text.
- You can generate images with text using the command line options in `Main.py`.
- For debugging and testing, use the `--test` option to run predefined tests.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.