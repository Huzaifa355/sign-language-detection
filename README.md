# 🤟 Sign Language Detection Using Mediapipe & LSTM

This project is a deep learning-based **sign language detection system** that uses **MediaPipe**, **OpenCV**, and **TensorFlow** to recognize hand and body gestures in real-time via webcam. It captures keypoints from the face, hands, and pose, and classifies them using a trained LSTM neural network.

---

## 📌 Features

- Real-time sign language recognition
- Uses Mediapipe Holistic for full-body landmark detection
- Keypoint extraction from face, hands, and pose
- Sequence-based classification using LSTM deep learning model
- Data augmentation for improved robustness
- Visual sentence overlay on webcam feed

---

## 🧰 Libraries Used

- `OpenCV`
- `MediaPipe`
- `TensorFlow / Keras`
- `NumPy`
- `Matplotlib`
- `Scikit-learn`

---

## 🗂️ Dataset Creation

- Actions defined: `Hello`, `Thanks`, `I like it`
- Each action is recorded for:
  - 20 sequences
  - 30 frames per sequence
- Keypoints are extracted and saved as `.npy` files per frame

---

## 🧠 Model Architecture

- Stacked LSTM layers (512 → 256 → 128)
- Fully connected dense layers
- Dropout for regularization
- Categorical cross-entropy loss
- Adam optimizer
- Trained for up to 2000 epochs with early stopping and learning rate scheduling

---

## 📈 Performance

- Training/Validation accuracy shown via TensorBoard
- Final model evaluated with:
  - Confusion Matrix
  - Accuracy Score

---

## 🎥 Live Prediction

- 30-frame sliding window used for prediction
- Confidence threshold of `0.7`
- Displays recognized words in real-time over webcam

---

## 💾 Model Saving

- Final trained model saved as:  
  `saved_models/action_recognition_model.h5`

---

## 🚀 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/Huzaifa355/sign-language-detection.git
   cd sign-language-detection
