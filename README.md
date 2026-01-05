## Heart Disease Prediction – End-to-End MLOps Pipeline

## 1. Problem Statement

Cardiovascular diseases are among the leading causes of mortality worldwide. Early prediction of heart disease using clinical attributes can significantly improve preventive care and medical decision-making.

The objective of this project is to **design, develop, and deploy a complete end-to-end machine learning system** that predicts the **risk of heart disease** using patient health data, while strictly following **modern MLOps best practices** including reproducibility, automation, CI/CD readiness, containerization, deployment, monitoring, and documentation.

---

## 2. Dataset Description

* **Dataset Name:** Heart Disease UCI Dataset
* **Source:** UCI Machine Learning Repository
* **Data Format:** CSV
* **Features:**
  * age, sex, chest pain type (cp)
  * resting blood pressure (trestbps)
  * cholesterol (chol)
  * fasting blood sugar (fbs)
  * resting ECG (restecg)
  * maximum heart rate (thalach)
  * exercise induced angina (exang)
  * ST depression (oldpeak)
  * slope, number of major vessels (ca), thal
* **Target Variable:**
  * Binary classification
  * `1` → Presence of heart disease
  * `0` → Absence of heart disease

The dataset is programmatically downloaded to ensure  **reproducibility** .

---

## 3. Project Structure

<pre class="overflow-visible! px-0!" data-start="1844" data-end="2300"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>heart-disease-mlops/
heart-disease-mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # CI pipeline (pytest + training)
│
├── api/
│   ├── __init__.py
│   ├── main.py                     # FastAPI inference service
│   ├── model.pkl                   # Best trained model
│   └── scaler.pkl                  # Feature scaler
│
├── artifacts/
│   └── EDA/
│       └── EDA_PDF.pdf             # EDA report/screenshots
│
├── data/
│   ├── download_data.py            # Dataset download script
│   └── heart.csv                   # Heart disease dataset
│
├── docker/
│   └── Dockerfile                  # Docker image definition
│
├── k8s/
│   ├── deployment.yaml             # Kubernetes deployment
│   └── service.yaml                # Kubernetes service
│
├── mlruns/
│   └── 1/
│       └── models/                 # MLflow tracked models
│           ├── m-2c276f734e944375bc7cb769706df139
│           ├── m-5701d2bf217a49c195b5cb534d67dee6
│           ├── m-e91463929cb44eaeb092979904146240
│           └── m-fa21e5ea68cf4e6685e974180aeced0f
│
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory Data Analysis
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py       # Data cleaning & preparation
│   ├── feature_engineering.py      # Scaling & feature handling
│   ├── monitoring.py               # Model monitoring logic
│   └── train.py                    # Training + MLflow logging
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py                 # API endpoint tests
│   ├── test_data.py                # Data validation tests
│   └── test_model.py               # Model inference tests
│
├── venv/                           # Virtual environment (local)
│
├── mlflow.db                       # MLflow backend store
├── pytest.ini                      # Pytest configuration
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── Report.docx                     # Final assignment report
├── Documentation.docx              # Supporting documentation
└── Documentation.pdf               # Supporting documentation (PDF)
</span><span>.md</span><span>
</span></span></code></div></div></pre>

---

## 4. Data Acquisition & Exploratory Data Analysis (EDA)

The dataset is **downloaded and cleaned inside the EDA notebook** (`01_eda.ipynb`) to ensure a reproducible experimental workflow.

### Key EDA Steps:

* Handling missing values (`?` replaced with NaN and dropped)
* Target variable binarization
* Feature distribution analysis
* Class balance visualization
* Correlation analysis using heatmaps

### Visualizations Included:

* Target class distribution
* Age distribution histogram
* Cholesterol vs target boxplot
* Feature correlation heatmap

Screenshots from the EDA notebook are included in the final report.

---

## 5. Data Preprocessing Pipeline

Implemented in `src/data_preprocessing.py`.

### Steps:

* Load CSV data
* Replace missing values
* Remove invalid records
* Convert multi-class target into binary classification

This preprocessing logic is  **shared across training, testing, and inference** , ensuring consistency.

---

## 6. Feature Engineering

Implemented in `src/feature_engineering.py`.

### Techniques Used:

* Feature scaling using `StandardScaler`
* Train–test split (80/20)
* Fixed random seed for reproducibility

The trained scaler is persisted and reused during inference to avoid training-serving skew.

---

## 7. Model Training & Evaluation

Implemented in `src/train.py`.

### Models Trained:

* Logistic Regression
* Random Forest Classifier

### Evaluation Metrics:

* ROC-AUC (primary metric)
* Accuracy, Precision, Recall (secondary)

### Model Selection:

* Both models are evaluated
* **Best model is selected automatically** based on ROC-AUC
* Only the best-performing model is saved for deployment

---

## 8. Experiment Tracking with MLflow

MLflow is used to track:

* Model parameters
* Evaluation metrics
* Model artifacts
* Experiment metadata

Each model training run is logged under the experiment:

<pre class="overflow-visible! px-0!" data-start="4163" data-end="4189"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>HeartDisease_MLOps</span><span>
</span></span></code></div></div></pre>

This ensures **traceability, comparability, and reproducibility** of experiments.

---

## 9. Model Packaging & Reproducibility

* Final model saved using `joblib`
* Preprocessing scaler saved separately
* `requirements.txt` ensures environment reproducibility
* Fixed random seeds used across all scripts

All scripts can be executed from a **clean environment** without manual intervention.

---

## 10. API Development (Model Serving)

Implemented using **FastAPI** (`api/main.py`).

### Endpoint:

<pre class="overflow-visible! px-0!" data-start="4692" data-end="4713"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>POST /predict
</span></span></code></div></div></pre>

### Input:

* JSON object with clinical feature values

### Output:

* Binary prediction (0 or 1)
* Confidence score (probability)

The API is designed to be  **stateless, lightweight, and production-ready** .

---

## 11. Containerization (Docker)

A Docker container is created using:

<pre class="overflow-visible! px-0!" data-start="4998" data-end="5023"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>docker/Dockerfile
</span></span></code></div></div></pre>

### Features:

* Python 3.10 base image
* Dependency installation via `requirements.txt`
* FastAPI server exposed on port `8000`

The container can be built and run locally for testing.

---

## 12. Kubernetes Deployment

Deployment manifests are provided in the `k8s/` directory.

### Components:

* **Deployment:** Multiple replicas for scalability
* **Service:** LoadBalancer for external access

This setup demonstrates cloud-native deployment readiness.

---

## 13. CI/CD & Automated Testing

Unit tests are written using **Pytest** and include:

* Data validation tests
* Model sanity tests
* API endpoint tests

These tests are designed to fail the pipeline if:

* Data preprocessing breaks
* Model inference fails
* API response is invalid

This ensures  **reliability and robustness** .

---

## 14. Monitoring & Alarm Strategy

A basic monitoring mechanism is implemented in `src/monitoring.py`.

### Monitoring Logic:

* Detect model performance degradation
* Trigger alerts if performance drops below threshold

In real-world systems, this can be extended to:

* Prometheus metrics
* Grafana dashboards
* Automated retraining triggers

---

## 15. Deployment Strategy (Production Readiness)

The system supports:

* **Blue-Green Deployment:** Zero-downtime releases
* **Canary Deployment:** Gradual rollout and validation

These strategies allow safe model updates and quick rollback.

---

## 16. Reproducibility Guarantee

* Programmatic data download
* Fixed random seeds
* Version-controlled dependencies
* Consistent preprocessing across pipeline
* Containerized execution

Any evaluator can reproduce the results from scratch.

---

## 17. Conclusion

This project demonstrates a  **complete, production-grade MLOps pipeline** , covering:

* Data ingestion
* Model development
* Experiment tracking
* Deployment
* Monitoring
* Testing
* Documentation

The solution mirrors real-world ML systems and adheres to modern MLOps best practices.

---

## 18. How to Run (Quick Start)

<pre class="overflow-visible! px-0!" data-start="7013" data-end="7106"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>pip install -r requirements.txt
python src/train.py
uvicorn api.main:app --reload
</span></span></code></div></div></pre>
