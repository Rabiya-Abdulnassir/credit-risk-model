from fastapi import FastAPI
import pandas as pd
import mlflow.pyfunc

from src.api.pydantic_models import CustomerData, PredictionResponse
from src.data_processing import build_feature_pipeline

app = FastAPI(title="Credit Risk API")


# =========================
# MODEL (GLOBAL - OK)
# =========================


MODEL_NAME = "CreditRiskModel"
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")


# =========================
# APP STATE INITIALIZATION
# =========================


@app.on_event("startup")
def load_pipeline():
    df = pd.read_csv("data/raw/data.csv")

    pipeline = build_feature_pipeline()
    pipeline.fit(df)

    app.state.feature_pipeline = pipeline


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

    df = pd.DataFrame([request.model_dump()])

    # Get pipeline from app state
    pipeline = app.state.feature_pipeline
    df = pipeline.transform(df)

    preds = model.predict(df)

    return PredictionResponse(predictions=preds.tolist())
