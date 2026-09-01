# Customer Churn Prediction API

An end-to-end machine learning project that predicts customer churn for a
telecom company, from raw data to a containerized, callable API.

Thefocus is not just training a model, but shipping one: clean data pipelines,
honest model evaluation, tested inference code, and a Dockerized service.

---

## Problem

Given a telecom customer's account details (contract type, tenure, billing
info, services subscribed to), predict whether they are likely to churn
(cancel their subscription).

**Business motivation:** acquiring a new customer costs significantly more
than retaining an existing one. A model that flags at-risk customers early
lets a business intervene (retention offers, outreach) before they leave.

**Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(Kaggle) — 7,043 customers, 21 raw features, binary target (`Churn`).

---

## Results

Two models were trained and compared honestly — not just the best one
reported in isolation.

| Model | Precision (Churn) | Recall (Churn) | F1 (Churn) | Accuracy |
|---|---|---|---|---|
| Logistic Regression (unweighted) | 0.66 | 0.57 | 0.61 | 0.81 |
| Logistic Regression (class-weighted) | 0.51 | 0.79 | 0.62 | 0.74 |
| **Random Forest (class-weighted)** | **0.54** | **0.78** | **0.64** | **0.77** |

**Final model: Random Forest, class-weighted.**

### Why accuracy isn't the headline metric here

The dataset is imbalanced (~73% no-churn / ~27% churn). A model that always
predicts "no churn" scores 73% accuracy while being useless. This project
optimizes for **recall on the churn class** instead — missing an actual
churner (a false negative) is the costlier business mistake, since it means
no retention action is ever taken. Class weighting was applied specifically
to address this, trading some precision for a large recall gain (0.57 →
0.78–0.79), a deliberate, explainable trade-off rather than a default.

---

## Architecture

```
Raw CSV
  │
  ▼
Preprocessing (src/preprocessing.py)
  - Fix TotalCharges (blank → 0, tied to tenure == 0)
  - Encode binary columns (label encoding)
  - Encode multi-category columns (one-hot encoding)
  │
  ▼
Stratified Train/Test Split (src/train_test_split.py)
  │
  ▼
Model Training (src/train_model.py)
  - Random Forest, class_weight='balanced'
  │
  ▼
Evaluation (src/evaluate.py)
  - Confusion matrix, precision/recall/F1
  │
  ▼
Model + Feature Columns saved (models/*.pkl)
  │
  ▼
FastAPI Service (api/main.py)
  - Validates input (Pydantic)
  - Preprocesses a single record (inference-safe path)
  - Returns prediction + probability
  │
  ▼
Docker Container (portable, runs anywhere)
```

**Why training and inference preprocessing diverge:** one-hot encoding a
full training set (`pd.get_dummies`) and one-hot encoding a single incoming
API request are *not* the same operation — a single row can't reveal all
possible categories the way thousands of training rows can.
`preprocess_single_record` handles this by aligning against the exact
column layout saved from training, rather than re-deriving categories from
one row. (This was a real bug caught by the test suite — see below.)

---

## Project Structure

```
churn-prediction-api/
├── api/
│   └── main.py              # FastAPI app: /health, /predict
├── data/
│   ├── raw/                 # Original CSV (gitignored)
│   └── processed/
├── models/
│   ├── churn_rf_model.pkl        # Trained model (gitignored)
│   └── feature_columns.pkl       # Training column layout (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing_check.ipynb
│   └── 03_baseline_model.ipynb
├── src/
│   ├── preprocessing.py     # Cleaning + encoding (train & inference paths)
│   ├── train_test_split.py  # Stratified split
│   ├── train_model.py       # Training, scaling, save/load
│   ├── evaluate.py          # Metrics reporting
│   └── train.py             # One-command, end-to-end training script
├── tests/
│   └── test_preprocessing.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites
- Python 3.12
- Docker (optional, for containerized run)

### Local setup

```bash
git clone <repo-url>
cd churn-prediction-api

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Download the dataset from
[Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
and place it at `data/raw/telco_churn.csv`.

### Train the model

```bash
python src/train.py
```

This runs the full pipeline — load, clean, encode, split, train, evaluate,
save — and prints the evaluation report to the console.

### Run the tests

```bash
pytest tests/ -v
```

### Run the API locally

```bash
uvicorn api.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

### Run with Docker

```bash
docker build -t churn-prediction-api .
docker run -p 8000:8000 churn-prediction-api
```

---

## API Reference

### `GET /health`
Health check.

**Response:**
```json
{"status": "ok"}
```

### `POST /predict`
Predicts churn for a single customer.

**Request body:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": "29.85"
}
```

**Response:**
```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.7196
}
```

---

## Engineering Notes

A few decisions and issues worth calling out — the kind of thing that comes
up in a technical interview about this project.

- **`TotalCharges` cleaning:** 11 rows had a blank `TotalCharges`, all with
  `tenure == 0` (brand-new customers, not yet billed). Filled with `0`
  rather than the column mean, since that's the value that's actually
  logically correct for this case — not just a generic missing-data default.

- **Data leakage avoidance:** `StandardScaler` is fit only on the training
  split and applied (not re-fit) to the test split, to avoid test-set
  statistics leaking into preprocessing.

- **Single-row one-hot encoding bug (caught by tests):** `pd.get_dummies`
  derives categories from the data it's given. On a full training set this
  is safe; on a single incoming API request, it can only ever "see" one
  category per column, so encoding silently drops the customer's real
  category instead of erroring. A dedicated test
  (`test_preprocess_single_record_matches_training_columns`) caught this —
  fixing it changed one test customer's predicted churn probability by
  roughly 5x (0.24 → 0.05), since their `Contract` type, one of the
  strongest churn signals in the dataset, was previously being silently
  zeroed out.

- **Environment isolation issue:** a system-wide `PYTHONPATH` variable (from
  an unrelated ROS installation) was leaking dozens of unrelated packages
  into `pip freeze` output and crashing both `pytest` and the Docker build.
  Diagnosed via `echo $PYTHONPATH` and fixed by clearing it at venv
  activation and regenerating a clean `requirements.txt`.

---

## Possible Extensions

- Threshold tuning on top of class weighting, to fine-tune the
  precision/recall trade-off for a specific business cost model
- Hyperparameter tuning (`GridSearchCV`) for the Random Forest
- CI/CD (GitHub Actions) to run tests automatically on push
- Deploy the container to a cloud platform (Render, Railway, AWS) for a
  live public endpoint
- Model monitoring / drift detection for a production deployment

---

## Tech Stack

| Purpose | Tool |
|---|---|
| Data manipulation | pandas, numpy |
| Modeling | scikit-learn |
| API | FastAPI, uvicorn |
| Testing | pytest |
| Containerization | Docker |
