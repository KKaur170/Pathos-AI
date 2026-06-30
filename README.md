<div align="center">

# ✨ Pathos AI: Multimodal Emotion Recognition for Indic Languages

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

**Pathos AI** is a multimodal emotion recognition framework engineered to provide inclusive, emotion-aware AI for Indic languages. By fusing high-fidelity acoustic signal processing with semantic transformer-based embeddings, the platform enables precise emotional state classification in **Malayalam** and **Punjabi**.

**[GitHub Repository](https://github.com/KKaur170/Pathos-AI)**

<br>

*(Dashboard Overview)*
<img src="assets/pathos_ai_dash_mal.png" width="100%" alt="Pathos AI Dashboard">

</div>

---

##  Real-World Impact (Why This Matters)
Mainstream voice assistants remain largely English-centric and emotion-blind, creating a significant accessibility gap for millions of Indic language speakers.

Pathos AI transforms raw audio data into structured emotional intelligence. By replacing generic, one-size-fits-all models with specialized domain-specific architectures for Malayalam and Punjabi, this system enables more natural, empathetic, and inclusive human-computer interactions.

###  Approach Comparison
| Approach | Limitation | Pathos AI's Solution |
| :--- | :--- | :--- |
| **Standard SER Models** | Blind to linguistic context; ignore semantic cues. | Multimodal fusion (Text BERT + Acoustic Features). |
| **Generic Classifiers** | Struggle with long-tail minority emotion classes. | Custom synthetic augmentation & Margin-based logic. |
| **Black-box Argmax** | Forces predictions even on low-confidence samples. | Margin-based decision thresholding for high-reliability outputs. |

---

##  System Architecture

```
                  User Audio
                      │
                      ▼
             Audio Normalization
        (FFmpeg + Pydub → 16kHz Mono)
                      │
                      ▼
              Language Selection
          ┌────────────────────────┐
          │                        │
          ▼                        ▼
 Malayalam Pipeline         Punjabi Pipeline
          │                        │
          ▼                        ▼
 Feature Extraction        Feature Extraction
(Text + Audio Fusion)     (Prosodic Features)
          │                        │
          ▼                        ▼
 Ensemble Prediction      Stacking Prediction
          │                        │
          └──────────┬─────────────┘
                     ▼
             Emotion Prediction
```

---

##  Engineering Highlights

### 1️ Malayalam – Multimodal Fusion Pipeline

The Malayalam pipeline combines **semantic understanding** with **acoustic information** to improve emotion recognition.

#### Semantic Integration

- Uses **bert-base-multilingual-cased** to generate contextual text embeddings.
- Captures the semantic meaning of spoken content.

#### Acoustic Features

Extracts multiple speech descriptors including:

- MFCC
- Chroma
- Tonnetz
- Spectral Contrast

#### Fusion Strategy

Text embeddings are concatenated with acoustic features and passed into a **Soft Voting Ensemble** for final prediction.

#### Confidence-based Decision Layer

A custom **Margin-based Thresholding** strategy evaluates:

```
(probability − threshold)
```

Low-confidence predictions are rejected, improving prediction reliability.

---

### 2️ Punjabi – Prosodic & Voice-Quality Pipeline

Instead of relying on textual information, the Punjabi pipeline focuses on **speech production characteristics**.

#### Parselmouth/Praat Features

Specialized voice-quality metrics include:

- Fundamental Frequency (F0)
- F0 Slope
- Jitter (Local)
- Shimmer (Local)
- Harmonics-to-Noise Ratio (HNR)

These physiological speech properties help capture emotional variations.

#### Stacking Ensemble

Base Models:

- LightGBM
- CatBoost
- Random Forest

Meta Learner:

- Logistic Regression

The stacking strategy combines predictions from multiple classifiers to improve generalization.

#### Class Weight Calibration

Manual class weights are incorporated to address class imbalance and improve minority class recognition.

---

### 3️ Production-Ready Inference Engine

The application is designed for efficient real-time deployment.

#### Universal Prediction Pipeline

`app.py` automatically routes incoming audio to the correct prediction pipeline based on the selected language.

#### Audio Normalization

Incoming audio is standardized using:

- FFmpeg
- Pydub

Supported formats:

- WAV
- MP3
- WebM

All files are converted to:

- 16 kHz
- Mono channel

for consistent feature extraction.

#### Efficient Model Loading

To minimize latency:

- `st.cache_resource` caches trained models.
- `joblib` loads serialized ensemble checkpoints.

This enables **sub-second inference** during repeated predictions.

---

##  Performance Insights

| Pipeline | Model Architecture | Validation Accuracy |
| :--- | :--- | :---: |
| **Malayalam** | Voting Ensemble (Soft-Voting) | 85.1% |
| **Punjabi** | Stacking Ensemble (LGBM meta-learner) | 82.5% |

<br>

<div align="center">
  <img src="assets/confusion_matrix_pun.png" width="45%" alt="Punjabi Confusion Matrix">
  <img src="assets/correlation_heatmap_mal.png" width="45%" alt="Malayalam Heatmap">
</div>

---

##  Technology Stack

### Machine Learning

- PyTorch
- LightGBM
- CatBoost
- Scikit-learn

### NLP

- Hugging Face Transformers
- bert-base-multilingual-cased
- SentenceTransformers

### Audio Processing

- Librosa
- Praat-Parselmouth
- Pydub
- FFmpeg

### Deployment

- Streamlit
- Joblib

---

##  Project Structure

```
Pathos-AI/
│
├── src/
│   ├── app.py
│
├── notebooks/
|   ├── 01_EDA_and_Visualization.ipynb
|   ├── 02_EDA_Malayalam.ipynb
|   ├── 03_Punjabi_Pipeline_Training.ipynb
|   ├── 04_Malayalam_Pipeline_Training.ipynb
│
├── assets/
|   ├── classification_report_mal.png
|   ├── classification_report_pun.png
|   ├── confusion_matrix_pun.png
|   ├── correlation_heatmap_mal.png
|   ├── data_insights_pun.png
|   ├── pathos_ai_dash_mal.png
|   ├── pathos_ai_dash_pun.png   
|   ├── results_mal.png
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

---

##  Setup & Usage

### 1. Requirements
Ensure you have `ffmpeg` installed on your system for audio processing capabilities.

### 2. Installation
```bash
git clone [https://github.com/KKaur170/Pathos-AI.git](https://github.com/KKaur170/Pathos-AI.git)
cd Pathos-AI
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run src/app.py
```

---

##  Supported Audio Formats

- WAV
- MP3
- WebM

All uploaded files are automatically normalized before inference.

---

##  Predicted Emotion Classes

Depending on the selected language and trained model, the system predicts emotions such as:

-  Happy
-  Sad
-  Anger
-  Fear
-  Neutral

---

##  Future Improvements

- Support for additional Indian languages
- Transformer-based multimodal fusion
- Explainable AI (XAI) visualizations
- Mobile-friendly deployment
- ONNX optimization for edge deployment

---

##  Author & AI Architect

**Khushnoor Kaur** 

*B.E. Computer Engineering | Thapar Institute of Engineering and Technology*

**Let's Connect:** [GitHub](https://github.com/KKaur170) | [LinkedIn](https://www.linkedin.com/in/khushnoor-kaur-bb7684345/)

---

##  License & Acknowledgements
This project is intended for academic and research purposes.
