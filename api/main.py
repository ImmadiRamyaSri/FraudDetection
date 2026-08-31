import time
import joblib
import pandas as pd
from fastapi import FastAPI,HTTPException
from contextlib import asynccontextmanager
from api.schemas import HealthResponse,PredictionResponse,TransactionFeatures

ml_models = {}

def get_risk_level(probability : float) -> str:
    if probability < 0.30:
        return "LOW"
    elif probability < 0.70:
        return "MEDIUM"
    elif probability < 0.90:
        return "HIGH"
    else:
        return "CRITICAL"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: Loading ML models and artifacts...")
    try:
        ml_models["model"] = joblib.load("models/fraud_xgboost.joblib")
        ml_models["scaler"] = joblib.load("models/scaler.joblib")
        
        ml_models["decision_threshold"] = 0.50
        ml_models["model_version"] = "1.0.0"
        
    except FileNotFoundError as e:
        print(f"Error loading artifacts: {e}. API will run without model context.")

    yield

    print("Shutting down: Clearing ML models from memory...")
    ml_models.clear()

app = FastAPI(title="Fraud Detection API",
              description="Real-time transaction risk scoring API using machine learning",
              version="1.0.0",
              lifespan=lifespan)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    if not ml_models or "model" not in ml_models:
        raise HTTPException(
            status_code=503, 
            detail="Service Unavailable: ML models failed to load."
        )
        
    return HealthResponse(
        status="healthy",
        model_version=ml_models.get("model_version", "unknown")
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_transaction(payload: TransactionFeatures):
    start_time = time.perf_counter()

    if "model" not in ml_models or "scaler" not in ml_models:
        raise HTTPException(
            status_code=503, 
            detail="Model server artifacts are not loaded."
        )
    
    input_dict = payload.model_dump()
    df = pd.DataFrame([input_dict])

    try:
        scaled_features = ml_models["scaler"].transform(df[["Amount", "Time"]])
        df["Amount"] = scaled_features[:, 0]
        df["Time"] = scaled_features[:, 1]
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Feature scaling transformation failed: {str(e)}"
        )

    # 4. Run model inference to determine probabilities
    # Align DataFrame columns to match exact features expected by the model
    feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X_input = df[feature_order]

    try:
        # Extract the probability of the transaction being fraudulent (class 1)
        probabilities = ml_models["model"].predict_proba(X_input)
        fraud_probability = float(probabilities[0][1])
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Model prediction inference failed: {str(e)}"
        )

    # 5. Evaluate fraud status against your decision configurations
    threshold = ml_models.get("decision_threshold", 0.50)
    is_fraud = fraud_probability >= threshold
    risk_level = get_risk_level(fraud_probability)

    # 6. Track performance overhead latency
    latency_ms = (time.perf_counter() - start_time) * 1000

    # 7. Deliver validated structural response payload
    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=fraud_probability,
        risk_level=risk_level,
        decision_threshold=threshold,
        latency_ms=latency_ms
    )