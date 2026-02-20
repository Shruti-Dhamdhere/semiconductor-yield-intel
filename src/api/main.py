"""
src/api/main.py
FastAPI REST API serving GCN + LightGBM ensemble for yield prediction.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import joblib
import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import torch.nn as nn
from pathlib import Path
from loguru import logger
from src.api.schemas import PredictRequest, PredictResponse, BatchPredictRequest, BatchPredictResponse, HealthResponse

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
MODELS_DIR    = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

model_store = {}

# ── Inline GCN definition (avoids import issues) ──────────────────────────────
class StableGCN(nn.Module):
    def __init__(self, in_ch=1, hidden=64, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.dropout = dropout
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        out = torch.cat([global_mean_pool(x, batch),
                         global_max_pool(x, batch)], dim=1)
        return self.head(out).squeeze(-1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...")
    try:
        # Load LightGBM
        model_store["lgbm"] = joblib.load(MODELS_DIR / "lgbm_model.pkl")

        # Load feature names
        with open(PROCESSED_DIR / "feature_names.json") as f:
            model_store["features"] = json.load(f)

        # Load GCN
        gnn = StableGCN(in_ch=1, hidden=64, dropout=0.3)
        gnn.load_state_dict(torch.load(
            MODELS_DIR / "gnn_best.pt", map_location="cpu"
        ))
        gnn.eval()
        model_store["gnn"] = gnn

        # Load graph edge index
        g = torch.load(PROCESSED_DIR / "graph.pt", map_location="cpu")
        model_store["edge_index"] = g["edge_index"]

        logger.success(
            f"All models loaded — Features: {len(model_store['features'])}, "
            f"GCN params: {sum(p.numel() for p in gnn.parameters()):,}"
        )
    except Exception as e:
        logger.error(f"Model loading error: {e}")
    yield
    model_store.clear()


app = FastAPI(
    title="Semiconductor Yield Prediction API",
    description="""
    REST API for wafer yield prediction using **GCN + LightGBM Ensemble**.

    ## Models
    - **GCN** (Graph Convolutional Network) — captures process graph structure
    - **LightGBM** — captures tabular feature importance
    - **Ensemble** — weighted combination of both

    ## Endpoints
    - **POST /predict** — single wafer prediction
    - **POST /predict/batch** — batch predictions
    - **GET /health** — API health check
    - **GET /features** — required sensor features
    - **GET /results/summary** — experiment results
    """,
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def get_lgbm_proba(sensor_readings: dict, features: list) -> float:
    X = pd.DataFrame([{f: sensor_readings.get(f, 0.0) for f in features}])
    return float(model_store["lgbm"].predict_proba(X)[0, 1])


def get_gnn_proba(sensor_readings: dict, features: list) -> float:
    vals = np.array([sensor_readings.get(f, 0.0) for f in features],
                    dtype=np.float32)
    # Normalise
    mu, sd = vals.mean(), vals.std() + 1e-8
    vals = (vals - mu) / sd
    x = torch.tensor(vals, dtype=torch.float).unsqueeze(1)  # (nodes, 1)
    edge_index = model_store["edge_index"]
    with torch.no_grad():
        logit = model_store["gnn"](x, edge_index)
        proba = torch.sigmoid(logit).item()
    return float(proba)


def ensemble_proba(lgbm_p: float, gnn_p: float,
                   lgbm_w: float = 0.6, gnn_w: float = 0.4) -> float:
    return lgbm_w * lgbm_p + gnn_w * gnn_p


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    lgbm_ok = "lgbm" in model_store
    gnn_ok  = "gnn"  in model_store
    status  = "healthy" if (lgbm_ok and gnn_ok) else "degraded"
    return HealthResponse(
        status=status,
        model_loaded=lgbm_ok and gnn_ok,
        n_features=len(model_store.get("features", [])),
        version="2.0.0",
    )


@app.get("/features", tags=["System"])
def get_features():
    if "features" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"features": model_store["features"],
            "n_features": len(model_store["features"])}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest):
    """Single wafer prediction using GCN + LightGBM ensemble."""
    if "lgbm" not in model_store or "gnn" not in model_store:
        raise HTTPException(status_code=503, detail="Models not loaded")

    features  = model_store["features"]
    threshold = request.threshold or 0.15

    try:
        lgbm_p = get_lgbm_proba(request.sensor_readings, features)
        gnn_p  = get_gnn_proba(request.sensor_readings, features)
        proba  = ensemble_proba(lgbm_p, gnn_p)
        pred   = int(proba >= threshold)

        return PredictResponse(
            wafer_id=request.wafer_id,
            failure_probability=round(proba, 4),
            prediction=pred,
            prediction_label="FAIL" if pred == 1 else "PASS",
            confidence=round(proba if pred == 1 else 1 - proba, 4),
            threshold_used=threshold,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(request: BatchPredictRequest):
    """Batch prediction for multiple wafers."""
    if "lgbm" not in model_store or "gnn" not in model_store:
        raise HTTPException(status_code=503, detail="Models not loaded")

    features  = model_store["features"]
    threshold = request.threshold or 0.15

    try:
        results = []
        for w in request.wafers:
            lgbm_p = get_lgbm_proba(w.sensor_readings, features)
            gnn_p  = get_gnn_proba(w.sensor_readings, features)
            proba  = ensemble_proba(lgbm_p, gnn_p)
            pred   = int(proba >= threshold)
            results.append(PredictResponse(
                wafer_id=w.wafer_id,
                failure_probability=round(proba, 4),
                prediction=pred,
                prediction_label="FAIL" if pred == 1 else "PASS",
                confidence=round(proba if pred == 1 else 1 - proba, 4),
                threshold_used=threshold,
            ))

        n_fail = sum(r.prediction for r in results)
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
    results_dir = PROJECT_ROOT / "paper" / "results"
    summary = {}
    for f in results_dir.glob("*.json"):
        try:
            with open(f) as fp:
                summary[f.stem] = json.load(fp)
        except Exception:
            pass
    return summary
