# Emotion Recognition System

## Overview
This project implements a real-time facial emotion recognition system using a Convolutional Neural Network (CNN). The system is designed to detect human faces in video streams or images and classify the detected faces into one of seven emotional categories: Angry, Disgust, Scared, Happy, Sad, Surprised, and Neutral.

The project provides two primary interfaces: a traditional OpenCV-based real-time video processor and a modern web interface built with Streamlit.

## Features
- Real-time face detection using Haar Cascade classifiers.
- Emotion classification using a pre-trained mini-XCEPTION model.
- Dual interface support: Streamlit web application and OpenCV standalone script.
- Visual representation of emotion probabilities.
- Support for both live webcam feed and static image processing.

## Technical Architecture
The system follows a pipeline architecture:
1. **Face Detection**: Utilizes OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`) to locate faces within the input frame.
2. **Preprocessing**: The detected face region is extracted, converted to grayscale, resized to 64x64 pixels, and normalized for the neural network.
3. **Inference**: The processed image is fed into the mini-XCEPTION model, which outputs a probability distribution across the seven emotion classes.
4. **Visualization**: Results are displayed with bounding boxes around detected faces and a probability bar chart showing the confidence for each emotion.

## Requirements
The project requires Python 3.x and the following libraries:
- TensorFlow / Keras
- OpenCV (cv2)
- Pandas
- NumPy
- Imutils
- Scikit-learn
- Streamlit (for the web interface)

## Installation

### Prerequisites
- Python 3.10 or higher is recommended.
- A functional webcam for real-time features.

### Step-by-Step Setup
1. Clone the repository:
```bash
git clone https://github.com/your-username/Emotion-recognition.git
cd Emotion-recognition
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Web Interface (Recommended)
The Streamlit interface provides the most user-friendly experience, allowing you to start/stop the webcam feed directly from your browser.
```bash
streamlit run streamlit_app.py
```
After running, navigate to `http://localhost:8501` in your web browser.

### 2. Standalone Video Processor
For a lightweight, window-based real-time processor:
```bash
python real_time_video.py
```
- A window named "your_face" will appear with the detection.
- A second window "Probabilities" will show the confidence scores.
- Press **'q'** on your keyboard to exit the application.

### 3. Model Training
To train the model from scratch using the FER2013 dataset:
1. Download the `fer2013.csv` file from Kaggle.
2. Place it at `fer2013/fer2013/fer2013.csv`.
3. Execute the training script:
```bash
python train_emotion_classifier.py
```

## Troubleshooting

- **Camera Not Detected**: Ensure no other application (like Zoom or Teams) is using your webcam. If you have multiple cameras, you may need to change the index in `cv2.VideoCapture(0)` to `1` or `2` in the source code.
- **Model Loading Errors**: Verify that the `.hdf5` file exists in the `models/` directory.
- **Performance Issues**: If the video feed is laggy, try closing other heavy applications or lowering the resolution in the scripts.

## Project Structure
- `models/`: Contains the pre-trained mini-XCEPTION model and training logs.
- `haarcascade_files/`: XML files for face detection.
- `emotions/`: Sample images for testing.
- `streamlit_app.py`: Main entry point for the web interface.
- `real_time_video.py`: Main entry point for the OpenCV interface.
- `train_emotion_classifier.py`: Script for model training.
- `load_and_process.py`: Utility functions for data handling.

## Dataset
The model was trained on the FER-2013 dataset, which contains 35,887 grayscale images of faces at 48x48 pixel resolution. Each image is categorized into one of seven emotions.

## Credits
This project is inspired by the face_classification work by oarriaga and utilizes resources from various computer vision research papers and tutorials.
