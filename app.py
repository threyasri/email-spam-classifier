#import libraries
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

import streamlit as st
import pickle

# Load feature extractor from pickle files
with open("tfidf_vectorizer.pkl", "rb") as f:
    feature_extraction = pickle.load(f)

# Load model from pickle files
with open("linear_svm_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Email Spam Message Classifier", page_icon="✉️", layout="wide")

# For precise styling
st.markdown("""
    <style>
    .big-title {
        color: #2196F3;
        font-size: 3em;
        font-weight: 700;
        margin-bottom: 0.5em;
        font-family: Arial, sans-serif;
    }
    .custom-label {
        font-size: 1.25em;
        margin-bottom: 0.2em;
        font-family: Arial, sans-serif;
    }
    .custom-textarea textarea {
        font-size: 1.15em !important;
        background-color: #fcfefd !important;
        border: 2px solid #90caf9 !important;
        border-radius: 7px !important;
        min-height: 90px !important;
        height: 85px !important;
        color: #222 !important;
    }
    .stButton > button {
        color: white !important;
        background-color: #2196F3 !important;
        font-size: 1.18em !important;
        font-weight: 500 !important;
        padding: 5px 26px;
        border-radius: 6px;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="big-title">Email Spam Message Classifier</div>', unsafe_allow_html=True)

# Label
st.markdown('<div class="custom-label">Enter your mail context:</div>', unsafe_allow_html=True)

# Single large textarea
mail_text = st.text_area("", height=90, key="mail_text", help="", placeholder="", label_visibility="collapsed")

# Blue Analyze button
analyze = st.button("Analyze")

# backend prediction logic
if analyze:
    if mail_text.strip() == "":
        st.warning("Please enter your mail context.")
    else:
        # Transform input to vectorized form
        features = feature_extraction.transform([mail_text])
        # prediction
        pred = model.predict(features)[0]
        # Map to label
        result = "Spam" if pred == 0 else "Not Spam"
        st.success(f"Result: {result}")

