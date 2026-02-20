"""
src/api/predictor.py
Inference wrapper for serving models outside FastAPI context.
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
MODELS_DIR    = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

class YieldPredictor:
    def __init__(self, threshold=0.15):
        self.model     = joblib.load(MODELS_DIR / "lgbm_model.pkl")
        with open(PROCESSED_DIR / "feature_names.json") as f:
            self.features = json.load(f)
        self.threshold = threshold
        logger.info(f"YieldPredictor loaded. Features: {len(self.features)}")

    def predict(self, sensor_readings: dict) -> dict:
        X = pd.DataFrame([{f: sensor_readings.get(f, 0.0) for f in self.features}])
        proba = float(self.model.predict_proba(X)[0, 1])
        pred  = int(proba >= self.threshold)
        return {
            "failure_probability": round(proba, 4),
            "prediction": pred,
            "prediction_label": "FAIL" if pred == 1 else "PASS",
            "confidence": round(proba if pred == 1 else 1 - proba, 4),
        }

    def predict_batch(self, readings_list: list) -> list:
        return [self.predict(r) for r in readings_list]
