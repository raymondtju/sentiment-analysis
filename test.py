import os
# Force CPU and clean logs before importing TF
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load assets once
print("Loading model and assets...")
model = load_model("sentiment_bigru_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("eii_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Initialize VADER once
sia = SentimentIntensityAnalyzer()
lexicon = sia.lexicon
MAX_LEN = 26 # Adjust this to match your notebook training

print("Ready! Enter text below:")

while True:
    text = input("> ")
    if text.lower() == 'exit': break
    
    # Preprocess
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    
    # EII logic
    tokens = text.lower().split()
    eii = sum([abs(lexicon[t])/4.0 for t in tokens if t in lexicon]) / len(tokens) if tokens else 0
    eii_scaled = scaler.transform([[eii]])
    
    # Predict
    print("Analyzing...")
    preds = model.predict([padded, eii_scaled], verbose=0)
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    print(f"Result: {sentiment_map[np.argmax(preds)]}")