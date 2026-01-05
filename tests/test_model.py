import pandas as pd
import joblib


def test_model_prediction():
    model = joblib.load("api/model.pkl")
    scaler = joblib.load("api/scaler.pkl")

    # Column names from your training data
    columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    # Use a DataFrame instead of np.array
    sample = pd.DataFrame(
        [[60, 1, 3, 140, 290, 0, 0, 150, 0, 1.2, 2, 0, 2]], columns=columns
    )
    sample_scaled = scaler.transform(sample)

    pred = model.predict(sample_scaled)
    assert pred[0] in [0, 1]
