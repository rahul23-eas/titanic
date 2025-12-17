"""
FastAPI service for Titanic survival prediction.
Loads the trained classification pipeline and exposes a /predict endpoint.
"""

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Configuration (ABSOLUTE PATH — REQUIRED FOR DOCKER)
# -----------------------------------------------------------------------------
MODEL_PATH = Path("/app/models/global_best_model_optuna.pkl")

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="FastAPI service for predicting passenger survival on the Titanic",
    version="1.0.0",
)

# -----------------------------------------------------------------------------
# Load model at startup
# -----------------------------------------------------------------------------
def load_model(path):
    path = Path(path)  # Defensive: ensures Path even if string passed

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    print(f"Loading model from: {path}")
    model = joblib.load(path)
    print("✓ Model loaded successfully")

    if hasattr(model, "named_steps"):
        print(f"Pipeline steps: {list(model.named_steps.keys())}")

    return model


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print("✗ ERROR: Failed to load model")
    raise RuntimeError(f"Model loading failed: {e}")

# -----------------------------------------------------------------------------
# Request / Response Schemas
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[int]
    count: int

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "Titanic Survival Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.instances:
        raise HTTPException(
            status_code=400,
            detail="No instances provided. Please provide at least one instance.",
        )

    try:
        X = pd.DataFrame(request.instances)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input format: {e}",
        )

    required_columns = {
        "Pclass",
        "Age",
        "Fare",
        "SibSp",
        "Parch",
        "sex",
    }

    missing = required_columns - set(X.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {sorted(missing)}",
        )

    try:
        predictions = model.predict(X)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {e}",
        )

    preds_list = [int(p) for p in predictions]

    return PredictResponse(
        predictions=preds_list,
        count=len(preds_list),
    )


@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 80)
    print("Titanic Survival Prediction API - Starting Up")
    print("=" * 80)
    print(f"Model path: {MODEL_PATH}")
    print(f"Model loaded: {model is not None}")
    print("API is ready to accept requests")
    print("=" * 80 + "\n")
