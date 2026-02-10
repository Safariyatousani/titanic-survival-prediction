import pandas as pd
def load_data(path):
    return pd.read_csv(path)

from data_loader import load_data
from preprocessing import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
df = load_data()
df = preprocess_data(df)

x = df[["Pclass","Age","Fare"]]
y = df["Survived"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = LogisticRegression(max_iter = 1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Accuracy :", accuracy_score(y_test,y_pred))

import joblib
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok = True)

joblib.dump(model, MODEL_DIR / "logistic_model.pkl")