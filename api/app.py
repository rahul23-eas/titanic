from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ------------------------------------------------------
# Path resolution (WORKS LOCALLY + DOCKER + CLOUD)
# ------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # api/
MODEL_PATH = BASE_DIR / "models" / "global_best_model_optuna.pkl"

# ------------------------------------------------------
# App
# ------------------------------------------------------
app = FastAPI(
    title="Titanic Survival Prediction API",
    version="1.0.0",
)

# ------------------------------------------------------
# Load model at startup
# ------------------------------------------------------
def load_model(path: Path):
    print(f"Looking for model at: {path}")
    print(f"Exists: {path.exists()}")

    if not path.exists():
        raise RuntimeError(f"Model file not found: {path}")

    return joblib.load(path)

model = load_model(MODEL_PATH)

# ------------------------------------------------------
# Request schema
# ------------------------------------------------------
class TitanicInput(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float

# ------------------------------------------------------
# Routes
# ------------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: TitanicInput):
    try:
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
