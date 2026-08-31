from pydantic import BaseModel,Field
from typing import Literal

class TransactionFeatures(BaseModel):
    Time: float = Field(...,description="Seconds elapsed since the first transaction")

    #Principal components V1 TO V28

    V1:float
    V2:float
    V3:float
    V4:float
    V5:float
    V6:float
    V7:float
    V8:float
    V9:float
    V10:float
    V11:float
    V12:float
    V13:float
    V14:float
    V15:float
    V16:float
    V17:float
    V18:float
    V19:float
    V20:float
    V21:float
    V22:float
    V23:float
    V24:float
    V25:float
    V26:float
    V27:float
    V28:float

    Amount:float = Field(...,description="Transaction amount")

class PredictionResponse(BaseModel):
    is_fraud:bool = Field(..., description="True if the transaction is flagged as fraud")
    fraud_probability:float= Field(..., ge=0.0, le=1.0, description="Model prediction probability score")
    risk_level:Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]= Field(..., description="Categorized risk based on probability")
    decision_threshold:float= Field(..., description="The cutoff probability used to classify fraud")
    latency_ms:float= Field(..., description="Inference execution time in milliseconds")

class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Current operational status of the service")
    model_version: str = Field(..., description="The version string of the active ML model, e.g., '1.4.2'")