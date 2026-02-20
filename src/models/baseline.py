"""
src/models/baseline.py
XGBoost and LightGBM baseline models with MLflow tracking.
"""
import json
from pathlib import Path
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import yaml
import joblib
from loguru import logger
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "paper" / "results"

def load_params():
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)

def load_data():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_val   = pd.read_csv(PROCESSED_DIR / "X_val.csv")
    X_test  = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_val   = pd.read_csv(PROCESSED_DIR / "y_val.csv").squeeze()
    y_test  = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return X_train, X_val, X_test, y_train, y_val, y_test

def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "auc": round(roc_auc_score(y_true, y_pred_proba), 4),
        "f1":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        "mcc": round(matthews_corrcoef(y_true, y_pred), 4),
        "ap":  round(average_precision_score(y_true, y_pred_proba), 4),
    }

def find_best_threshold(y_true, y_pred_proba):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.5, 0.05):
        pred = (y_pred_proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    logger.info(f"Best threshold: {best_t:.2f} (F1={best_f1:.4f})")
    return best_t

def train_xgboost(X_train, y_train, X_val, y_val, params):
    logger.info("Training XGBoost...")
    # Compute class weight from data
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = neg / pos
    logger.info(f"scale_pos_weight set to {spw:.1f}")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        min_child_weight=5,
        early_stopping_rounds=30,
        eval_metric="auc",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    logger.success(f"XGBoost trained. Best iteration: {model.best_iteration}")
    return model

def train_lightgbm(X_train, y_train, X_val, y_val, params):
    logger.info("Training LightGBM...")
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    logger.info(f"Class ratio: {neg/pos:.1f}:1")
    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=5,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    logger.success("LightGBM trained.")
    return model

def run_baseline_training():
    params = load_params()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///mlflow_tracking/mlflow.db")
    mlflow.set_experiment(params["training"]["mlflow_experiment"])
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    logger.info(f"Train: {X_train.shape}, Fail rate: {y_train.mean()*100:.1f}%")
    all_results = {}

    with mlflow.start_run(run_name="xgboost_baseline"):
        model = train_xgboost(X_train, y_train, X_val, y_val, params)
        val_proba  = model.predict_proba(X_val)[:,1]
        test_proba = model.predict_proba(X_test)[:,1]
        best_t = find_best_threshold(y_val, val_proba)
        val_metrics  = compute_metrics(y_val,  val_proba,  threshold=best_t)
        test_metrics = compute_metrics(y_test, test_proba, threshold=best_t)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        joblib.dump(model, MODELS_DIR / "xgboost_model.pkl")
        logger.info(f"XGBoost — Val AUC: {val_metrics['auc']}, Test AUC: {test_metrics['auc']}")
        logger.info(f"XGBoost — Test F1: {test_metrics['f1']}, MCC: {test_metrics['mcc']}")
        all_results["xgboost"] = {"val": val_metrics, "test": test_metrics}

    with mlflow.start_run(run_name="lightgbm_baseline"):
        model = train_lightgbm(X_train, y_train, X_val, y_val, params)
        val_proba  = model.predict_proba(X_val)[:,1]
        test_proba = model.predict_proba(X_test)[:,1]
        best_t = find_best_threshold(y_val, val_proba)
        val_metrics  = compute_metrics(y_val,  val_proba,  threshold=best_t)
        test_metrics = compute_metrics(y_test, test_proba, threshold=best_t)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        joblib.dump(model, MODELS_DIR / "lgbm_model.pkl")
        logger.info(f"LightGBM — Val AUC: {val_metrics['auc']}, Test AUC: {test_metrics['auc']}")
        logger.info(f"LightGBM — Test F1: {test_metrics['f1']}, MCC: {test_metrics['mcc']}")
        all_results["lightgbm"] = {"val": val_metrics, "test": test_metrics}

    with open(RESULTS_DIR / "baseline_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.success("Baseline training complete. Results saved.")
    return all_results

if __name__ == "__main__":
    run_baseline_training()
