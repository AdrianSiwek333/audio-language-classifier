# Audio Language Identifier

A Deep Learning-based spoken language identification system designed to classify languages from short audio clips. The project leverages advanced digital signal processing techniques for feature extraction and utilizes deep neural networks implemented in TensorFlow/Keras for robust classification.

---

## 1. Abstract

This project focuses on classifying spoken languages from audio recordings. The system utilizes the **Librosa** library to extract 128 Mel-Frequency Cepstral Coefficients (MFCCs), which are subsequently time-averaged into feature vectors. A Deep Neural Network (DNN) model was developed using **TensorFlow/Keras**. To mitigate the risk of overfitting, strict regularization techniques and Early Stopping were implemented. The model successfully learns the frequency characteristics of distinct languages, achieving stable and satisfying classification accuracy on the test dataset.

---

## 2. Introduction

Spoken Language Identification (LID) is a fundamental challenge in digital signal processing and machine learning. Automatically identifying the language spoken in an audio stream enables efficient routing to downstream speech-to-text (STT) and translation pipelines. 

The primary challenge in such systems is finding a suitable representation for non-stationary audio signals so they can be processed by a classifier. In this project, rather than feeding raw waveforms directly into the network, spectral analysis was employed. The system relies on a fully connected (Dense) Deep Neural Network architecture that analyzes aggregated acoustic features extracted from speech recordings.

---

## 3. Project Context & Objectives

The context of this project is to build a tool that automates the initial preprocessing and analysis of multilingual audio recordings. The main goal was to design, optimize, and train a classification model capable of distinguishing between different languages based on short voice samples.

### Specific Objectives:
* **Efficient Feature Extraction:** Mass-processing and time-optimized extraction of acoustic features from thousands of audio files.
* **Generalization & Regularization:** Building a model robust against overfitting, capable of high generalization on unseen data.
* **Hyperparameter Tuning:** Selecting optimal hyperparameters and performing an objective evaluation of the neural network architecture.

---

## 4. Dataset

The model was trained and evaluated using audio samples from the **Mozilla Common Voice** dataset. The pipeline is designed to balance the data across targeted languages to ensure unbiased classification boundaries.

---

## 5. Model Architecture & Training

The network architecture is a fully connected Deep Neural Network (DNN) optimized for tabularized spectral features:
* **Input Layer:** 128 features (Time-averaged MFCCs).
* **Hidden Layers:** Multiple `Dense` layers with `ReLU` activation.
* **Regularization:** Incorporated `Dropout` layers and $L_2$ regularization to prevent overfitting.
* **Output Layer:** `Dense` layer with `Softmax` activation for multi-class language classification.
* **Training Strategy:** Optimized using the `Adam` optimizer and `Categorical Crossentropy` loss, combined with an `EarlyStopping` callback tracking validation loss.

---

## Tech Stack & Libraries

* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Audio Processing:** Librosa
* **Data Manipulation & Visualization:** NumPy, Pandas, Matplotlib / Seaborn

---

## How to Run

1. Clone the repository:
```bash
   git clone [https://github.com/AdrianSiwek333/YOUR_REPO_NAME.git](https://github.com/AdrianSiwek333/YOUR_REPO_NAME.git)
