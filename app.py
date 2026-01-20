import os
# Force CPU and suppress TF logs BEFORE importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)

MAX_LEN = 26  # must match training

# -----------------------------
# Load Assets (Cached)
# -----------------------------
@st.cache_resource
def load_assets():
    model = load_model("sentiment_bigru_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("eii_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    sia = SentimentIntensityAnalyzer()

    return model, tokenizer, scaler, sia


model, tokenizer, scaler, sia = load_assets()
lexicon = sia.lexicon

# -----------------------------
# UI
# -----------------------------
st.title("💬 Sentiment Analysis App")
st.write("BiGRU + EII (VADER) hybrid sentiment classifier")

text = st.text_area(
    "Enter text to analyze:",
    placeholder="Type a sentence here...",
    height=120
)

analyze = st.button("Analyze Sentiment")

# -----------------------------
# Prediction Logic
# -----------------------------
if analyze:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        # Tokenize & pad
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

        # EII calculation
        tokens = text.lower().split()
        eii = sum([abs(lexicon[t])/4.0 for t in tokens if t in lexicon]) / len(tokens) if tokens else 0
        eii_scaled = scaler.transform([[eii]])

        # Predict
        with st.spinner("Analyzing..."):
            preds = model.predict([padded, eii_scaled], verbose=0)

        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        label = sentiment_map[int(np.argmax(preds))]
        # confidence = float(np.max(preds))

        # -----------------------------
        # Output
        # -----------------------------
        st.subheader("Result")
        st.markdown(f"**Sentiment:** `{label}`")
        # st.markdown(f"**Confidence:** `{confidence:.2%}`")

        st.progress(confidence)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("CPU-only inference • BiGRU + VADER EII")
