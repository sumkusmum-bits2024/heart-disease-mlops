from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Heart Disease ML API is live!"}

model = joblib.load("api/model.pkl")
scaler = joblib.load("api/scaler.pkl")

columns = ["age", "sex", "cp", "trestbps", "chol", "fbs",
           "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

@app.post("/predict")
def predict(features: dict):
    # Convert dict to DataFrame with correct column names
    x = pd.DataFrame([features], columns=columns)
    
    # Now scaler sees proper feature names → no warning
    x_scaled = scaler.transform(x)

    prob = model.predict_proba(x_scaled)[0][1]

    return {
        "prediction": int(prob > 0.5),
        "confidence": float(prob)
    }

