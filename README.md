# Diagno-Talk

## Overview

This application helps users find potential diseases based on symptoms. It provides two input methods: text input and voice input. Users can describe symptoms either by typing or speaking, and the application will match those symptoms with potential diseases.

## Features

- **Text Input**: Users can type in descriptions of symptoms.
- **Voice Input**: Users can speak descriptions of symptoms, which are then transcribed and matched with potential diseases.
- **Translation**: For voice input, the application can translate the transcribed text to English before matching with diseases.
- **Top Matches**: The application displays the top matched disease labels based on the input symptoms.

## Requirements

Make sure you have the following dependencies installed:

- Streamlit
- Pandas
- SpeechRecognition
- Googletrans
- Scikit-learn
- NLTK

You can install them using pip: 

    - pip install streamlit
    - pip install pandas
    - pip install SpeechRecognition
    - pip install googletrans
    - pip install scikit-learn
    - pip install nltk


## How to Run

1. Clone this repository to your local machine.
2. Navigate to the directory containing the code.
3. Run the following command:
   ```bash
   streamlit run final.py

This will start the Streamlit server, and you can access the application through your web browser.

## Usage

1. Upon running the application, you will see a sidebar with two options: "Text to Text" and "Voice to Text".
2. Choose your preferred input method.
3. If you select "Text to Text", simply type in the description of the symptoms in the text box provided.
4. If you select "Voice to Text", click on the button and speak into your microphone when prompted.
5. The application will transcribe your speech (if voice input is chosen) and match the symptoms with potential diseases.
6. The top matched disease labels will be displayed below the input box.

## Additional Information

- The dataset used for matching symptoms with diseases is loaded from a CSV file named "Symptom2Disease.csv".
- The application preprocesses the text input by tokenizing, removing punctuation, stop words, and lemmatizing the words.
- Cosine similarity is used to match the input symptoms with potential diseases based on TF-IDF vectors.
- For voice input, the application uses Google Speech Recognition to transcribe the speech.
- The transcribed text (for voice input) is translated to English (if needed) using Google Translate.
