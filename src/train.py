import mlflow
import mlflow.sklearn
import joblib
import numpy as np

from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import prepare_features

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)


# ---------------------------------------------------
# Load data
# ---------------------------------------------------
df = load_and_clean_data("data/heart.csv")

X_train, X_test, y_train, y_test, scaler = prepare_features(df)


# ---------------------------------------------------
# Models to evaluate
# ---------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
}

mlflow.set_experiment("HeartDisease_MLOps")

best_auc = 0
best_model = None


# ---------------------------------------------------
# Training + Cross Validation
# ---------------------------------------------------
for name, model in models.items():
    with mlflow.start_run(run_name=name):

        # ---------- Cross Validation ----------
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="roc_auc"
        )

        mlflow.log_metric("cv_auc_mean", np.mean(cv_scores))
        mlflow.log_metric("cv_auc_std", np.std(cv_scores))

        # ---------- Train on full training set ----------
        model.fit(X_train, y_train)

        # ---------- Test set evaluation ----------
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(model, "model")

        print(f"{name}")
        print(f"CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"Test ROC-AUC: {auc:.3f}\n")

        # ---------- Select best model ----------
        if auc > best_auc:
            best_auc = auc
            best_model = model


# ---------------------------------------------------
# Save best model for API
# ---------------------------------------------------
joblib.dump(best_model, "api/model.pkl")
joblib.dump(scaler, "api/scaler.pkl")

print("Best model saved for API inference")
