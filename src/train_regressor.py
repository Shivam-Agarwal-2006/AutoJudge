import pandas as pd
import numpy as np  
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from features import build_features
data = []

with open("C:/Users/shiva/OneDrive/Desktop/Projects/AutoJudge/data/problems_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
df.replace("",np.nan,inplace=True)
df = df.dropna()
df = df.reset_index(drop=True)
X, vectorizer = build_features(df, fit=True)
y_score = df["problem_score"]
X_train, X_test, y_train,y_test = train_test_split(
    X,
    y_score,
    test_size=0.2,
    random_state=42
)

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)
with open("models/regressor.pkl", "wb") as f:
    pickle.dump(rf, f)