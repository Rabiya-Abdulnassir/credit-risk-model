from fastapi import FastAPI
import pandas as pd
import mlflow.pyfunc
import joblib

from src.api.pydantic_models import CustomerData, PredictionResponse
from src.data_processing import build_feature_pipeline

app = FastAPI(title="Credit Risk API")

# =========================
# LOAD FEATURE PIPELINE
# =========================
# BEST PRACTICE: load pre-fitted pipeline (recommended)
# feature_pipeline = joblib.load("feature_pipeline.pkl")

# TEMP DEV OPTION: fit at startup (NOT production-safe)
feature_pipeline = build_feature_pipeline()

@app.on_event("startup")
def load_pipeline():
    global feature_pipeline
    df = pd.read_csv("data/raw/data.csv")
    feature_pipeline.fit(df)

# =========================
# LOAD MODEL FROM MLFLOW
# =========================
MODEL_NAME = "CreditRiskModel"
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"message": "Credit Risk Model API is running"}

# =========================
# PREDICTION ENDPOINT
# =========================
@app.post("/predict", response_model=PredictionResponse)
def predict(request: CustomerData):

    # Convert request -> DataFrame safely
    df = pd.DataFrame([request.model_dump()])

    # Apply feature pipeline (must match training pipeline)
    df = feature_pipeline.transform(df)

    # Predict using MLflow model
    preds = model.predict(df)

    return PredictionResponse(predictions=preds.tolist())