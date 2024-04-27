import streamlit as st
import pandas as pd
import speech_recognition as sr
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
from collections import Counter

# Load the dataset
symptom_data = pd.read_csv("Symptom2Disease.csv")

# Function to preprocess text
def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove punctuation
    tokens = [word.lower() for word in tokens if word.isalpha()]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Function to transcribe speech
def transcribe_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("Please speak:")
        audio = recognizer.listen(source)

    try:
        with st.spinner('Transcribing...'):
            text = recognizer.recognize_google(audio, language="en-US")
        return text
    except sr.UnknownValueError:
        st.write("Sorry, could not understand the audio.")
    except sr.RequestError as e:
        st.write("Could not request results from Google Speech Recognition service; {0}".format(e))

# Function to translate text
def translate_text(text, src_lang, dest_lang):
    translator = Translator()
    translated_text = translator.translate(text, src=src_lang, dest=dest_lang).text
    return translated_text

# Function to get top diseases
def get_top_diseases(text):
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Compute TF-IDF vectors for the dataset text
    vectorizer = TfidfVectorizer()
    dataset_text = symptom_data['text'].apply(preprocess_text)
    dataset_tfidf = vectorizer.fit_transform(dataset_text)

    # Compute TF-IDF vector for the input text
    input_tfidf = vectorizer.transform([processed_text])

    # Calculate cosine similarity between input and dataset
    similarities = cosine_similarity(input_tfidf, dataset_tfidf)

    # Find the indices of the top most similar rows
    top_indices = similarities.argsort(axis=1)[0][-3:][::-1]

    # Get the corresponding disease labels
    top_diseases = [symptom_data.loc[index, 'label'] for index in top_indices]

    return top_diseases

def main():
    # Customizing the sidebar with a background color and title
    st.sidebar.title("Options")
    st.sidebar.markdown("---")
    st.sidebar.write("Choose the input method:")

    text_button = st.sidebar.button("Text to Text")
    voice_button = st.sidebar.button("Voice to Text")

    if text_button:
        st.session_state.input_method = "Text to Text"
    elif voice_button:
        st.session_state.input_method = "Voice to Text"

    st.sidebar.markdown("---")

    # Displaying a header with an icon and a title
    st.title("Disease Symptom Matcher")
    st.markdown("<h2 style='text-align: center; color: #ff6347;'>Symptom Matcher</h2>", unsafe_allow_html=True)

    # Displaying additional information and instructions
    st.write("This app helps you find potential diseases based on symptoms.")

    if "input_method" not in st.session_state:
        return

    # Based on user's choice, displaying either text input or voice input option
    if st.session_state.input_method == "Text to Text":
        # Text input
        user_input = st.text_input("Please describe the disease symptom:")

        if user_input:
            # Get top diseases
            top_diseases = get_top_diseases(user_input)
            
            # Display top diseases
            st.subheader("Top matched disease labels:")
            st.write(top_diseases)

    elif st.session_state.input_method == "Voice to Text":
        # Transcribe speech
        user_input_voice = transcribe_speech()

        if user_input_voice:
            # Translate transcribed text to English
            translated_text = translate_text(user_input_voice, 'ta', 'en')

            # Get top diseases
            top_diseases = get_top_diseases(translated_text)

            # Display top diseases
            st.subheader("Top matched disease labels:")
            st.write(top_diseases)

if __name__ == "__main__":
    main()
