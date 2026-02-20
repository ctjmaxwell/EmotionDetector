# Emotion Detector

A real-time emotion detection application using Convolutional Neural Networks (CNN) with Keras and OpenCV. This project can classify human facial expressions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Features

- **Real-time Detection**: Uses a webcam to detect faces and predict emotions on the fly.
- **Deep Learning Model**: A custom CNN architecture trained on grayscale images.
- **Data Augmentation**: Includes rescaling, rotation, zoom, and horizontal flipping to improve model robustness.
- **Performance Evaluation**: Scripts to generate confusion matrices and detailed classification reports.
- **Imbalance Handling**: Uses class weights during training to account for uneven distribution of emotion samples.

## Project Structure

- `main.py`: Script to train the CNN model.
- `evaluate_model.py`: Script to evaluate the trained model on a test dataset.
- `test.py`: Real-time emotion detection using OpenCV and the trained model.
- `data/`: Directory containing `train` and `test` datasets (organized by emotion labels).
- `emotion_detector_model.keras`: The saved trained model (generated after running `main.py`).

## Getting Started

### Prerequisites

- Python 3.x
- Virtual environment (recommended)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd EmotionDectector
   ```

2. Install dependencies:
   ```bash
   pip install keras tensorflow opencv-python matplotlib scikit-learn seaborn numpy
   ```

### Usage

1. **Train the Model**:
   Ensure your dataset is in the `data/train` and `data/test` folders. Then run:
   ```bash
   python main.py
   ```
   This will train the model and save it as `emotion_detector_model.keras`.

2. **Evaluate the Model**:
   To see the performance metrics and confusion matrix:
   ```bash
   python evaluate_model.py
   ```

3. **Real-time Detection**:
   To start the webcam-based emotion detector:
   ```bash
   python test.py
   ```
   *Press 'q' to exit the camera view.*

## Model Architecture

The model is built using Keras `Sequential` API and includes:
- 4 Convolutional layers with Batch Normalization and Max Pooling.
- Dropout layers (ranging from 0.2 to 0.4) to prevent overfitting.
- A Dense layer with 512 units.
- Output layer with Softmax activation for 7 emotion classes.

## License

This project is open-source and available under the MIT License.
