# 🗣️ Speech Emotion Recognition using CNN+LSTM

![Python](https://img.shields.io/badge/Python-3.7+-blue)
![Keras](https://img.shields.io/badge/Keras-TensorFlow-orange)
![Librosa](https://img.shields.io/badge/Librosa-Audio_Processing-blueviolet)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## 🔍 Overview

This project demonstrates how to classify human emotions from speech using a deep learning pipeline that combines **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** layers. The hybrid architecture effectively captures both **spatial** and **temporal** characteristics from audio signals for accurate emotion recognition.

---

## 🎯 Objectives

- Extract rich audio features using `librosa`
- Apply audio augmentation to increase robustness
- Train a CNN+LSTM hybrid model to classify 8 emotions
- Fine-tune the model for optimal performance
- Evaluate using metrics like F1-score, confusion matrix, and accuracy

---

## 🗃️ Dataset

- **Dataset**: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Classes**:
  - `neutral`
  - `calm`
  - `happy`
  - `sad`
  - `angry`
  - `fearful`
  - `disgust`
  - `surprised`

---

## 📊 Features Extracted

| Feature               | Description                             |
|------------------------|-----------------------------------------|
| MFCC                  | Mel-Frequency Cepstral Coefficients     |
| Chroma STFT           | Harmonic/pitch features                 |
| Mel Spectrogram       | Time-frequency energy spectrum          |
| Zero Crossing Rate    | Audio waveform sign changes             |
| RMS Energy            | Root Mean Square amplitude              |
| Spectral Contrast     | Peak-to-valley contrast in spectrum     |
| Tonnetz               | Tonal centroid features                 |

---

## 🧠 Model Architecture: CNN + LSTM

```text
Input Shape: (Timesteps, Features)

→ Conv1D (128 filters, kernel=5) + ReLU
→ MaxPooling1D + Dropout

→ Conv1D (64 filters, kernel=3) + ReLU
→ MaxPooling1D + Dropout + BatchNormalization

→ LSTM (64 units) + Dropout + BatchNormalization

→ Dense (32 units, ReLU) + Dropout
→ Output Layer (Softmax for 8 emotions)
