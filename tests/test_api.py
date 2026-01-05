from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)


def test_prediction_endpoint():
    payload = {
        "age": 60,
        "sex": 1,
        "cp": 3,
        "trestbps": 140,
        "chol": 289,
        "fbs": 0,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 2,
        "ca": 0,
        "thal": 2
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
