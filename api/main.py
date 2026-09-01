import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI
from pydantic import BaseModel
from train_model import load_model, load_feature_columns
from preprocessing import preprocess_single_record

app = FastAPI(title="Churn Prediction API")

model = load_model("models/churn_rf_model.pkl")
feature_columns = load_feature_columns("models/feature_columns.pkl")


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: CustomerData):
    record = data.model_dump()
    X = preprocess_single_record(record, feature_columns)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4)
    }