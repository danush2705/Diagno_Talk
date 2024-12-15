# RemoteCura: AI-Powered Telehealth Diagnostic Platform

## Overview
RemoteCura is an innovative telehealth application that combines two powerful diagnostic approaches:
1. Symptom-based Disease Prediction
2. Medical Image Classification

## Components

### Symptom Diagnosis

#### Features
- **Text Input**: Detailed symptom description via typing
- **Voice Input**: Symptom description through speech recognition
- **Translation**: Automatic translation of voice inputs to English
- **Intelligent Matching**: Advanced symptom-to-disease prediction

#### Diagnostic Process
- Preprocesses text inputs through:
  - Tokenization
  - Punctuation removal
  - Stop word elimination
  - Lemmatization
- Uses cosine similarity with TF-IDF vectors
- Provides top matched disease labels

### Medical Imaging Classification

#### Features
- Brain Tumor MRI Classification
  - Detects glioma, meningioma, pituitary tumors
  - Identifies tumor absence
- Breast Ultrasound Classification
  - Identifies benign, malignant, and normal conditions
- Lung Cancer Detection
  - Classifies various lung cancer types:
    * Squamous cell carcinoma
    * Large cell carcinoma
    * Adenocarcinoma
  - Distinguishes malignant, benign, and normal cases

## Requirements
```
streamlit
pandas
SpeechRecognition
googletrans
scikit-learn
nltk
torch
torchvision
```

## Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Execution Instructions

### Launch Symptom Diagnosis Module
```bash
streamlit run src/symptom_diagnosis/final.py
```

### Launch Medical Image Classification Module
```bash
streamlit run src/medical_imaging/prediction/final_comb_disease_mod.py
```

## Usage Guidelines
1. Select input method (text or voice) for symptom diagnosis
2. Enter or speak symptoms
3. View potential disease matches
4. For medical imaging, upload relevant scan
5. Receive AI-powered disease predictions

## Disclaimer
RemoteCura is an AI-assisted diagnostic tool. Always consult healthcare professionals for definitive medical advice.
