import streamlit as st
import pickle
import re
import numpy as np
from scipy.sparse import hstack
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r'[^a-z0-9 +\-*/<>=]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


with open("models/vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/classifier.pkl", "rb") as f:
    clf = pickle.load(f)

with open("models/regressor.pkl", "rb") as f:
    reg = pickle.load(f)
def extract_numeric_features(text):
    text_length = len(text)

    num_math_symbols = sum(text.count(s) for s in ["+", "-", "*", "/", "<", ">", "="])

    keywords = ["dp", "graph", "tree", "greedy", "math", "string"]
    kw_counts = [text.count(kw) for kw in keywords]

    return np.array([[text_length, num_math_symbols] + kw_counts])  
st.title("AutoJudge 🧠")
st.write("Predict programming problem difficulty using text only")

desc = st.text_area("Problem Description")
inp = st.text_area("Input Description")
out = st.text_area("Output Description")

if st.button("Predict"):
    if desc.strip() == "":
        st.error("Please enter problem description")
    else:
        full_text = desc + " " + inp + " " + out
        full_text = clean_text(full_text)

        X_tfidf = tfidf.transform([full_text])
        X_numeric =  extract_numeric_features(full_text)
        X_final = hstack([X_tfidf, X_numeric])
        pred_class = clf.predict(X_final)[0]
        pred_score = reg.predict(X_final)[0]

        st.success(f"Predicted Difficulty Class: **{pred_class}**")
        st.success(f"Predicted Difficulty Score: **{round(pred_score, 2)}**")
