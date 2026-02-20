"""
src/training/trainer.py - Stable GNN training loop.
"""
import json
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from loguru import logger
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, average_precision_score

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PARAMS_PATH   = PROJECT_ROOT / "params.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR    = PROJECT_ROOT / "models"
RESULTS_DIR   = PROJECT_ROOT / "paper" / "results"

import yaml
def load_params():
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)

# ── Lightweight GCN that actually converges ──────────────────────────────────
class StableGCN(nn.Module):
    def __init__(self, in_ch, hidden=64, layers=2, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.dropout = dropout
        self.head = nn.Linear(hidden * 2, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        # graph readout
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        out = torch.cat([global_mean_pool(x, batch),
                         global_max_pool(x, batch)], dim=1)
        return self.head(out).squeeze(-1)

# ── Data helpers ─────────────────────────────────────────────────────────────
def load_split(split):
    X = pd.read_csv(PROCESSED_DIR / f"X_{split}.csv")
    y = pd.read_csv(PROCESSED_DIR / f"y_{split}.csv").squeeze()
    with open(PROCESSED_DIR / "feature_names.json") as f:
        feat = json.load(f)
    cols = [c for c in feat if c in X.columns]
    X = X[cols]
    # Normalise each sample independently
    X_np = X.values.astype(np.float32)
    mu = X_np.mean(axis=0, keepdims=True)
    sd = X_np.std(axis=0, keepdims=True) + 1e-8
    X_np = (X_np - mu) / sd
    # shape: (n_samples, n_nodes)
    return torch.tensor(X_np, dtype=torch.float), torch.tensor(y.values, dtype=torch.float)

def load_graph():
    g = torch.load(PROCESSED_DIR / "graph.pt")
    return g["edge_index"]

# ── Train / eval ─────────────────────────────────────────────────────────────
def run_epoch(model, X, y, edge_index, optimizer, criterion, train=True):
    """X: (n_samples, n_nodes)  — one forward pass per sample."""
    model.train() if train else model.eval()
    ctx = torch.enable_grad() if train else torch.no_grad()
    losses, logits_list = [], []
    with ctx:
        for i in range(X.shape[0]):
            x_i = X[i].unsqueeze(1)          # (n_nodes, 1)
            out = model(x_i, edge_index)      # scalar logit
            losses.append(out)
        logits = torch.stack(losses).squeeze()
        loss = criterion(logits, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    proba = torch.sigmoid(logits.detach()).numpy()
    return loss.item(), proba

def compute_metrics(y_true, proba, t=0.5):
    pred = (proba >= t).astype(int)
    return {
        "auc": round(roc_auc_score(y_true, proba), 4),
        "f1":  round(f1_score(y_true, pred, zero_division=0), 4),
        "mcc": round(matthews_corrcoef(y_true, pred), 4),
        "ap":  round(average_precision_score(y_true, proba), 4),
    }

def best_threshold(y_true, proba):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.05, 0.95, 0.05):
        f1 = f1_score(y_true, (proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t

# ── Main training function ────────────────────────────────────────────────────
def train_gnn(model_type="GCN"):
    params = load_params()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")
    edge_index       = load_graph()

    n_nodes = X_train.shape[1]
    neg = (y_train == 0).sum().item()
    pos = (y_train == 1).sum().item()
    pos_weight = torch.tensor([neg / pos])
    logger.info(f"Nodes: {n_nodes} | pos_weight: {pos_weight.item():.1f}")

    model     = StableGCN(in_ch=1, hidden=64, layers=2, dropout=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {total:,}")

    mlflow.set_tracking_uri("sqlite:///mlflow_tracking/mlflow.db")
    mlflow.set_experiment("semiconductor-yield")

    best_val_auc, patience, best_path = 0, 0, MODELS_DIR / "gnn_best.pt"

    with mlflow.start_run(run_name=f"gcn_stable"):
        for epoch in range(1, 151):
            tr_loss, _         = run_epoch(model, X_train, y_train, edge_index, optimizer, criterion, train=True)
            val_loss, val_prob = run_epoch(model, X_val,   y_val,   edge_index, optimizer, criterion, train=False)
            scheduler.step()

            val_auc = roc_auc_score(y_val.numpy(), val_prob)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch:03d} | tr_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | val_auc {val_auc:.4f}")

            mlflow.log_metrics({"train_loss": tr_loss, "val_loss": val_loss, "val_auc": val_auc}, step=epoch)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience += 1
                if patience >= 30:
                    logger.info(f"Early stop at epoch {epoch}")
                    break

        # Final evaluation
        model.load_state_dict(torch.load(best_path))
        _, val_prob  = run_epoch(model, X_val,  y_val,  edge_index, optimizer, criterion, train=False)
        _, test_prob = run_epoch(model, X_test, y_test, edge_index, optimizer, criterion, train=False)

        t = best_threshold(y_val.numpy(), val_prob)
        val_m  = compute_metrics(y_val.numpy(),  val_prob,  t)
        test_m = compute_metrics(y_test.numpy(), test_prob, t)

        mlflow.log_metrics({f"final_val_{k}":  v for k, v in val_m.items()})
        mlflow.log_metrics({f"final_test_{k}": v for k, v in test_m.items()})

        logger.info("── GCN Final Results ───────────────────────────────")
        logger.info(f"Val  — AUC: {val_m['auc']}, F1: {val_m['f1']}, MCC: {val_m['mcc']}")
        logger.info(f"Test — AUC: {test_m['auc']}, F1: {test_m['f1']}, MCC: {test_m['mcc']}")
        logger.info("────────────────────────────────────────────────────")

        results = {"gcn": {"val": val_m, "test": test_m, "best_val_auc": best_val_auc}}
        with open(RESULTS_DIR / "gnn_metrics.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.success(f"Done. Best val AUC: {best_val_auc:.4f}")
        return results

if __name__ == "__main__":
    train_gnn()
