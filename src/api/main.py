from fastapi import FastAPI
import pandas as pd
import mlflow.pyfunc

from src.api.pydantic_models import CustomerData, PredictionResponse

app = FastAPI(title="Credit Risk API")

# =========================
# LOAD MODEL FROM MLFLOW
# =========================
MODEL_NAME = "best_model"

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

    df = pd.DataFrame(request.data)

    preds = model.predict(df)

    return PredictionResponse(predictions=preds.tolist())