"""
src/models/ensemble.py
Hybrid ensemble: LightGBM + XGBoost + probability averaging.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, average_precision_score

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
RESULTS_DIR   = PROJECT_ROOT / "paper" / "results"

def load_split(split):
    X = pd.read_csv(PROCESSED_DIR / f"X_{split}.csv")
    y = pd.read_csv(PROCESSED_DIR / f"y_{split}.csv").squeeze()
    with open(PROCESSED_DIR / "feature_names.json") as f:
        feat = json.load(f)
    cols = [c for c in feat if c in X.columns]
    return X[cols], y

def compute_metrics(y_true, proba, t=0.5):
    pred = (proba >= t).astype(int)
    return {
        "auc": round(roc_auc_score(y_true, proba), 4),
        "f1":  round(f1_score(y_true, pred, zero_division=0), 4),
        "mcc": round(matthews_corrcoef(y_true, pred), 4),
        "ap":  round(average_precision_score(y_true, proba), 4),
    }

def find_best_threshold(y_true, proba):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t

def tune_weight(p1_val, p2_val, y_val):
    best_auc, best_w = 0, 0.5
    for w in np.arange(0.0, 1.05, 0.05):
        combined = w * p1_val + (1 - w) * p2_val
        auc = roc_auc_score(y_val, combined)
        if auc > best_auc:
            best_auc, best_w = auc, float(w)
    logger.info(f"Best LightGBM weight: {best_w:.2f} (Val AUC={best_auc:.4f})")
    return best_w

def run_ensemble():
    logger.info("Running ensemble: LightGBM + XGBoost...")

    X_val,  y_val  = load_split("val")
    X_test, y_test = load_split("test")

    lgbm = joblib.load(MODELS_DIR / "lgbm_model.pkl")
    xgb  = joblib.load(MODELS_DIR / "xgboost_model.pkl")

    lgbm_val  = lgbm.predict_proba(X_val)[:,1]
    lgbm_test = lgbm.predict_proba(X_test)[:,1]
    xgb_val   = xgb.predict_proba(X_val)[:,1]
    xgb_test  = xgb.predict_proba(X_test)[:,1]

    # Tune weight on validation
    best_w = tune_weight(lgbm_val, xgb_val, y_val.values)

    val_ens  = best_w * lgbm_val  + (1 - best_w) * xgb_val
    test_ens = best_w * lgbm_test + (1 - best_w) * xgb_test

    best_t = find_best_threshold(y_val.values, val_ens)
    val_m  = compute_metrics(y_val.values,  val_ens,  best_t)
    test_m = compute_metrics(y_test.values, test_ens, best_t)

    lgbm_m = compute_metrics(y_test.values, lgbm_test, best_t)
    xgb_m  = compute_metrics(y_test.values, xgb_test,  best_t)

    logger.info("── Full Model Comparison (Test Set) ────────────────")
    logger.info(f"XGBoost   — AUC: {xgb_m['auc']}, F1: {xgb_m['f1']}, MCC: {xgb_m['mcc']}")
    logger.info(f"LightGBM  — AUC: {lgbm_m['auc']}, F1: {lgbm_m['f1']}, MCC: {lgbm_m['mcc']}")
    logger.info(f"Ensemble  — AUC: {test_m['auc']}, F1: {test_m['f1']}, MCC: {test_m['mcc']}")
    logger.info("────────────────────────────────────────────────────")

    results = {
        "lgbm_weight": best_w,
        "xgb_weight":  1 - best_w,
        "val":  val_m,
        "test": test_m,
        "comparison": {
            "xgboost":  xgb_m,
            "lightgbm": lgbm_m,
            "ensemble": test_m,
        }
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "ensemble_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.success("Ensemble results saved.")
    return results

if __name__ == "__main__":
    run_ensemble()
