# Driver Drowsiness Detection System

## About the Project

The Driver Drowsiness Detection System is a Python-based project designed to detect signs of driver drowsiness by monitoring eye movements through a webcam. It provides an alert when the driver's eyes remain closed for a specified period.

## Features

* Real-time webcam monitoring
* Face and eye landmark detection using MediaPipe
* Eye Aspect Ratio (EAR) calculation
* Drowsiness detection
* Audio alarm when drowsiness is detected
* Machine learning model for eye-state classification
* Model training using open-eye and closed-eye images

## Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy
* Scikit-learn
* Streamlit
* Streamlit WebRTC
* Joblib

## Project Structure

```text
Driver-Drowsiness-Detection/
├── app.py
├── trainmodel.py
├── requirements.txt
├── README.md
├── dataset/
│   ├── close/
│   └── open/
└── models/
    └── eye_state_model.pkl
```

## Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the Project Folder

```bash
cd YOUR_PROJECT_FOLDER
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python trainmodel.py
```

This command trains the eye-state classification model and saves it in the `models` folder.

### 5. Run the Application

```bash
python -m streamlit run app.py
```

## How It Works

1. The webcam captures video frames.
2. MediaPipe detects facial landmarks.
3. The system calculates the Eye Aspect Ratio (EAR).
4. If the eyes remain closed beyond the configured time, the system triggers an audio alert.
5. The trained machine learning model can classify eye images as open or closed when integrated into the live detection pipeline.

## Dataset

The dataset contains two categories:

* `open`: Images of open eyes.
* `close`: Images of closed eyes.

## Machine Learning Model

A Support Vector Machine (SVM) classifier is trained using resized grayscale eye images. The images are resized to 32 × 32 pixels and normalized before training.

## Important Note

This project is intended for educational and demonstration purposes. It should not be relied upon as the sole method for preventing driver drowsiness or ensuring road safety.

## Author

Aditi Singh
