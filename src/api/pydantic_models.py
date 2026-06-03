from pydantic import BaseModel

class CustomerData(BaseModel):
    age: int
    income: float
    loan_amount: float
    credit_score: int
    employment_length: int
    home_ownership: str
    purpose: str

class PredictionResponse(BaseModel):
    predictions: list