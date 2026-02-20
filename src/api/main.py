"""
src/api/main.py
---------------
FastAPI REST API for semiconductor yield prediction.
Serves the LightGBM model with real-time inference.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from loguru import logger
from src.api.schemas import PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse, HealthResponse

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
MODELS_DIR    = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Global model store
model_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info("Loading models...")
    try:
        model_store["lgbm"] = joblib.load(MODELS_DIR / "lgbm_model.pkl")
        with open(PROCESSED_DIR / "feature_names.json") as f:
            model_store["features"] = json.load(f)
        logger.success(f"Model loaded. Features: {len(model_store['features'])}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield
    model_store.clear()
    logger.info("Models unloaded.")

app = FastAPI(
    title="Semiconductor Yield Prediction API",
    description="""
    REST API for wafer yield prediction using LightGBM + GCN ensemble.
    
    ## Endpoints
    - **POST /predict** — single wafer prediction
    - **POST /predict/batch** — batch wafer predictions  
    - **GET /health** — API health check
    - **GET /features** — list of required features
    - **GET /results/summary** — experiment results summary
    """,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check API health and model status."""
    model_loaded = "lgbm" in model_store
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        n_features=len(model_store.get("features", [])),
        version="1.0.0",
    )

@app.get("/features", tags=["System"])
def get_features():
    """Return list of required sensor features."""
    if "features" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"features": model_store["features"], "n_features": len(model_store["features"])}

@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """
    Predict yield outcome for a single wafer.
    
    Returns probability of failure and binary prediction.
    Threshold tuned on validation set for best F1.
    """
    if "lgbm" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = model_store["features"]
    model    = model_store["lgbm"]

    # Build feature vector
    X = pd.DataFrame([{f: request.sensor_readings.get(f, 0.0) for f in features}])

    try:
        proba      = float(model.predict_proba(X)[0, 1])
        threshold  = request.threshold or 0.15
        prediction = int(proba >= threshold)
        confidence = proba if prediction == 1 else 1 - proba

        return PredictResponse(
            wafer_id=request.wafer_id,
            failure_probability=round(proba, 4),
            prediction=prediction,
            prediction_label="FAIL" if prediction == 1 else "PASS",
            confidence=round(confidence, 4),
            threshold_used=threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(request: BatchPredictRequest):
    """Batch prediction for multiple wafers."""
    if "lgbm" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features  = model_store["features"]
    model     = model_store["lgbm"]
    threshold = request.threshold or 0.15

    rows = [{f: w.sensor_readings.get(f, 0.0) for f in features}
            for w in request.wafers]
    X = pd.DataFrame(rows)

    try:
        probas      = model.predict_proba(X)[:, 1]
        predictions = (probas >= threshold).astype(int)

        results = []
        for i, w in enumerate(request.wafers):
            proba = float(probas[i])
            pred  = int(predictions[i])
            results.append(PredictResponse(
                wafer_id=w.wafer_id,
                failure_probability=round(proba, 4),
                prediction=pred,
                prediction_label="FAIL" if pred == 1 else "PASS",
                confidence=round(proba if pred == 1 else 1 - proba, 4),
                threshold_used=threshold,
            ))

        n_fail = int(predictions.sum())
        return BatchPredictResponse(
            predictions=results,
            n_wafers=len(results),
            n_predicted_fail=n_fail,
            n_predicted_pass=len(results) - n_fail,
            fail_rate=round(n_fail / len(results), 4),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/summary", tags=["Research"])
def get_results_summary():
    """Return experiment results summary for paper."""
    results_dir = PROJECT_ROOT / "paper" / "results"
    summary = {}
    for f in results_dir.glob("*.json"):
        try:
            with open(f) as fp:
                summary[f.stem] = json.load(fp)
        except Exception:
            pass
    return summary
