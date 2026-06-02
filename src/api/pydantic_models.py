from pydantic import BaseModel
from typing import List


# =========================
# INPUT SCHEMA
# =========================
class CustomerData(BaseModel):
    data: List[dict]


# =========================
# OUTPUT SCHEMA
# =========================
class PredictionResponse(BaseModel):
    predictions: List[float]