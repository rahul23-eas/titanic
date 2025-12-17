from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Path resolution (WORKS LOCALLY + DOCKER + CLOUD)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent        # api/
MODEL_PATH = BASE_DIR / "models" / "global_best_model_optuna.pkl"

app = FastAPI(
    title="Titanic Survival Prediction API",
    version="1.0.0",
)

# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
def load_model(path: Path):
    print(f"Looking for model at: {path}")
    print(f"Exists: {path.exists()}")

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = joblib.load(path)
    print("✅ Model loaded successfully")
    return model


try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print("❌ Model loading failed")
    raise RuntimeError(e)

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[int]
    count: int

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.instances:
        raise HTTPException(status_code=400, detail="No instances provided")

    df = pd.DataFrame(request.instances)
    preds = model.predict(df)

    return PredictResponse(
        predictions=[int(p) for p in preds],
        count=len(preds),
    )