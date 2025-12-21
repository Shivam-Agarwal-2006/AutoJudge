import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from preprocess import clean_text

def build_features(df,vectorizer=None,fit=True):
    df['full_text'] = (
    df['title'] + " " +
    df['description'] + " " +
    df['input_description'] + " " +
    df['output_description']
    )
    df['full_text'] = df['full_text'].apply(clean_text)
    tfidf = TfidfVectorizer(
    max_features=8000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3
    )
    X_tfidf = tfidf.fit_transform(df["full_text"])
    df["text_length"] = df["full_text"].apply(len)

    df["num_math_symbols"] = df["full_text"].apply(
    lambda x: sum(x.count(s) for s in ["+", "-", "*", "/", "<", ">", "="])
    )

    keywords = ["dp", "graph", "tree", "greedy", "math", "string"]

    for kw in keywords:
        df[f"kw_{kw}"] = df["full_text"].apply(lambda x: x.count(kw))
    X_numeric = df[["text_length", "num_math_symbols"] +[f"kw_{kw}" for kw in keywords]].values

    X_final = hstack([X_tfidf, X_numeric])
    return X_final,tfidf
