"""
src/explainability/shap_analysis.py
------------------------------------
SHAP analysis for LightGBM and GCN models.
Identifies which sensors most influence yield predictions.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from loguru import logger

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
FIG_DIR       = PROJECT_ROOT / "paper" / "figures"
RESULTS_DIR   = PROJECT_ROOT / "paper" / "results"

def load_test_data():
    X = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    with open(PROCESSED_DIR / "feature_names.json") as f:
        feat = json.load(f)
    cols = [c for c in feat if c in X.columns]
    return X[cols], y

def run_shap_lgbm():
    logger.info("Running SHAP analysis on LightGBM...")
    X_test, y_test = load_test_data()
    model = joblib.load(MODELS_DIR / "lgbm_model.pkl")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Use TreeExplainer — exact SHAP values for tree models
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, shap_values is a list [class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[1]   # class 1 = fail
    else:
        sv = shap_values

    # ── Top 20 features by mean absolute SHAP ────────────────────
    mean_shap = np.abs(sv).mean(axis=0)
    top20_idx = np.argsort(mean_shap)[-20:][::-1]
    top20_features = [X_test.columns[i] for i in top20_idx]
    top20_values   = mean_shap[top20_idx]

    logger.info("Top 10 most important sensors:")
    for feat, val in zip(top20_features[:10], top20_values[:10]):
        logger.info(f"  {feat}: {val:.4f}")

    # ── Summary bar plot ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#e74c3c" if i < 5 else "#3498db" for i in range(20)]
    bars = ax.barh(range(20), top20_values[::-1], color=colors[::-1],
                   edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(20))
    ax.set_yticklabels([f.replace("sensor_", "S") for f in top20_features[::-1]],
                       fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Top 20 Sensors by SHAP Importance\n(LightGBM — Yield Failure Prediction)",
                 fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_summary.png", bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("SHAP summary plot saved.")

    # ── Beeswarm plot for top 10 ──────────────────────────────────
    top10_cols = top20_features[:10]
    top10_idx_list = [list(X_test.columns).index(c) for c in top10_cols]
    X_top10 = X_test[top10_cols]
    sv_top10 = sv[:, top10_idx_list]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (col, shap_col) in enumerate(zip(top10_cols, sv_top10.T)):
        feat_vals = X_top10[col].values
        norm = (feat_vals - feat_vals.min()) / (feat_vals.ptp() + 1e-8)
        scatter = ax.scatter(shap_col,
                             np.random.normal(i, 0.1, len(shap_col)),
                             c=norm, cmap="RdBu_r", alpha=0.5, s=8)
    ax.set_yticks(range(10))
    ax.set_yticklabels([c.replace("sensor_", "S") for c in top10_cols], fontsize=9)
    ax.set_xlabel("SHAP Value (impact on failure prediction)", fontsize=11)
    ax.set_title("SHAP Beeswarm — Top 10 Sensors", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=1)
    plt.colorbar(scatter, ax=ax, label="Feature value (normalized)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_beeswarm.png", bbox_inches="tight", dpi=150)
    plt.close()
    logger.info("SHAP beeswarm plot saved.")

    # Save top features for causal discovery
    top_features_data = {
        "top20_features": top20_features,
        "top20_shap_values": top20_values.tolist(),
    }
    with open(RESULTS_DIR / "shap_top_features.json", "w") as f:
        json.dump(top_features_data, f, indent=2)

    logger.success("SHAP analysis complete.")
    return top20_features

if __name__ == "__main__":
    run_shap_lgbm()
