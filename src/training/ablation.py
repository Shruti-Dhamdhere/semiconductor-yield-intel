"""
src/training/ablation.py
------------------------
Systematic ablation study comparing:
- Graph type: v1 (Pearson) vs v2 (Multi-criterion)
- Model: GCN vs LightGBM vs Ensemble

Produces a clean results table for the paper.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from loguru import logger
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, average_precision_score
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
RESULTS_DIR   = PROJECT_ROOT / "paper" / "results"


# ── Inline model definition ───────────────────────────────────────────────────
class StableGCN(nn.Module):
    def __init__(self, in_ch=1, hidden=64, dropout=0.3):
        super().__init__()
        self.conv1   = GCNConv(in_ch, hidden)
        self.conv2   = GCNConv(hidden, hidden)
        self.dropout = dropout
        self.head    = nn.Linear(hidden * 2, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        batch = torch.zeros(x.size(0), dtype=torch.long)
        out   = torch.cat([global_mean_pool(x, batch),
                           global_max_pool(x, batch)], dim=1)
        return self.head(out).squeeze(-1)


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_split(split):
    X = pd.read_csv(PROCESSED_DIR / f"X_{split}.csv")
    y = pd.read_csv(PROCESSED_DIR / f"y_{split}.csv").squeeze()
    with open(PROCESSED_DIR / "feature_names.json") as f:
        feat = json.load(f)
    cols  = [c for c in feat if c in X.columns]
    X_np  = X[cols].values.astype(np.float32)
    mu, sd = X_np.mean(0), X_np.std(0) + 1e-8
    X_norm = (X_np - mu) / sd
    return torch.tensor(X_norm, dtype=torch.float), torch.tensor(y.values, dtype=torch.float), X[cols], y


def load_graph(version="v1"):
    fname = "graph.pt" if version == "v1" else "graph_v2.pt"
    path  = PROCESSED_DIR / fname
    if not path.exists():
        logger.warning(f"{fname} not found, falling back to graph.pt")
        path = PROCESSED_DIR / "graph.pt"
    g = torch.load(path, map_location="cpu")
    return g["edge_index"]


# ── Evaluation ────────────────────────────────────────────────────────────────
def compute_metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "auc":  round(roc_auc_score(y_true, proba), 4),
        "f1":   round(f1_score(y_true, pred, zero_division=0), 4),
        "mcc":  round(matthews_corrcoef(y_true, pred), 4),
        "ap":   round(average_precision_score(y_true, proba), 4),
    }

def find_threshold(y_true, proba):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


# ── GCN inference ─────────────────────────────────────────────────────────────
def get_gcn_probas(X_tensor, edge_index, model_path):
    model = StableGCN()
    if not model_path.exists():
        logger.warning(f"GCN model not found at {model_path}")
        return None
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    probas = []
    with torch.no_grad():
        for i in range(X_tensor.shape[0]):
            x_i  = X_tensor[i].unsqueeze(1)
            out  = model(x_i, edge_index)
            probas.append(torch.sigmoid(out).item())
    return np.array(probas)


# ── LightGBM inference ────────────────────────────────────────────────────────
def get_lgbm_probas(X_df):
    model = joblib.load(MODELS_DIR / "lgbm_model.pkl")
    return model.predict_proba(X_df)[:, 1]


# ── Main ablation ─────────────────────────────────────────────────────────────
def run_ablation():
    logger.info("Starting ablation study...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data once
    X_val_t,  y_val_t,  X_val_df,  y_val  = load_split("val")
    X_test_t, y_test_t, X_test_df, y_test = load_split("test")

    y_val_np  = y_val.values
    y_test_np = y_test.values

    results = []

    for graph_ver in ["v1", "v2"]:
        edge_index = load_graph(graph_ver)
        graph_label = "Pearson (v1)" if graph_ver == "v1" else "Multi-Criterion (v2)"

        # ── GCN ──────────────────────────────────────────────────────────────
        logger.info(f"Evaluating GCN with graph {graph_label}...")
        gcn_val  = get_gcn_probas(X_val_t,  edge_index, MODELS_DIR / "gnn_best.pt")
        gcn_test = get_gcn_probas(X_test_t, edge_index, MODELS_DIR / "gnn_best.pt")

        if gcn_val is not None:
            t = find_threshold(y_val_np, gcn_val)
            m = compute_metrics(y_test_np, gcn_test, t)
            results.append({
                "model": "GCN",
                "graph": graph_label,
                **m
            })
            logger.info(f"GCN {graph_label} -- AUC: {m['auc']}, F1: {m['f1']}, MCC: {m['mcc']}")

        # ── LightGBM (graph-independent) ──────────────────────────────────────
        if graph_ver == "v1":
            logger.info("Evaluating LightGBM...")
            lgbm_val  = get_lgbm_probas(X_val_df)
            lgbm_test = get_lgbm_probas(X_test_df)
            t = find_threshold(y_val_np, lgbm_val)
            m = compute_metrics(y_test_np, lgbm_test, t)
            results.append({
                "model": "LightGBM",
                "graph": "N/A (tabular)",
                **m
            })
            logger.info(f"LightGBM -- AUC: {m['auc']}, F1: {m['f1']}, MCC: {m['mcc']}")

        # ── Ensemble ──────────────────────────────────────────────────────────
        if gcn_val is not None:
            logger.info(f"Evaluating Ensemble with graph {graph_label}...")
            lgbm_val  = get_lgbm_probas(X_val_df)
            lgbm_test = get_lgbm_probas(X_test_df)
            ens_val   = 0.6 * lgbm_val  + 0.4 * gcn_val
            ens_test  = 0.6 * lgbm_test + 0.4 * gcn_test
            t = find_threshold(y_val_np, ens_val)
            m = compute_metrics(y_test_np, ens_test, t)
            results.append({
                "model": "GCN+LightGBM Ensemble",
                "graph": graph_label,
                **m
            })
            logger.info(f"Ensemble {graph_label} -- AUC: {m['auc']}, F1: {m['f1']}, MCC: {m['mcc']}")

    # ── Print table ───────────────────────────────────────────────────────────
    logger.info("── Ablation Study Results ──────────────────────────────────")
    logger.info(f"{'Model':<25} {'Graph':<22} {'AUC':>6} {'F1':>6} {'MCC':>6} {'AP':>6}")
    logger.info("-" * 75)
    for r in results:
        logger.info(
            f"{r['model']:<25} {r['graph']:<22} "
            f"{r['auc']:>6.3f} {r['f1']:>6.3f} "
            f"{r['mcc']:>6.3f} {r['ap']:>6.3f}"
        )
    logger.info("-" * 75)

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(RESULTS_DIR / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.success("Ablation study complete. Results saved.")
    return results


if __name__ == "__main__":
    run_ablation()
