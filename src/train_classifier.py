import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MaxAbsScaler
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from features import build_features
import os
os.makedirs("models", exist_ok=True)
data = []

with open("data/problems_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
df.replace("",np.nan,inplace=True)
df = df.dropna()
df = df.reset_index(drop=True)
X, vectorizer = build_features(df, fit=True)
y = df["problem_class"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
clf = LogisticRegression(
    max_iter=15000,
    class_weight="balanced",
    solver="saga"
)
scaler = MaxAbsScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
cm = confusion_matrix(y_test, y_pred)
print(cm)
with open("models/classifier.pkl", "wb") as f:
    pickle.dump(clf, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)