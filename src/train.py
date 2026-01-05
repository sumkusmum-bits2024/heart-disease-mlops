import mlflow
import mlflow.sklearn
import joblib

from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import prepare_features

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

df = load_and_clean_data("data/heart.csv")

X_train, X_test, y_train, y_test, scaler = prepare_features(df)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
}

mlflow.set_experiment("HeartDisease_MLOps")

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)

        mlflow.log_param("model_name", name)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "model")

        # Save best model manually (for API)
        joblib.dump(model, "api/model.pkl")
        joblib.dump(scaler, "api/scaler.pkl")

        print(f"{name} ROC-AUC: {auc}")

