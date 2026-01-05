import pandas as pd
import os

def download_data():
    os.makedirs("data", exist_ok=True)

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal", "target"
    ]

    df = pd.read_csv(url, names=columns)
    df.to_csv("data/heart.csv", index=False)

    print("Dataset downloaded successfully")

if __name__ == "__main__":
    download_data()

