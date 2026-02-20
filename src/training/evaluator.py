"""
src/training/evaluator.py
Shared evaluation utilities used across all models.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.metrics import (
    roc_auc_score, f1_score, matthews_corrcoef,
    roc_curve, precision_recall_curve,
    confusion_matrix, average_precision_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR     = PROJECT_ROOT / "paper" / "figures"
RESULTS_DIR = PROJECT_ROOT / "paper" / "results"

def compute_all_metrics(y_true, y_pred_proba, threshold=0.5, model_name="model"):
    y_pred = (y_pred_proba >= threshold).astype(int)
    metrics = {
        "model":   model_name,
        "auc_roc": round(roc_auc_score(y_true, y_pred_proba), 4),
        "auc_pr":  round(average_precision_score(y_true, y_pred_proba), 4),
        "f1":      round(f1_score(y_true, y_pred, zero_division=0), 4),
        "mcc":     round(matthews_corrcoef(y_true, y_pred), 4),
    }
    logger.info(f"{model_name} — AUC-ROC: {metrics['auc_roc']}, F1: {metrics['f1']}, MCC: {metrics['mcc']}")
    return metrics

def find_best_threshold(y_true, y_pred_proba):
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1, best_thresh = 0, 0.5
    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    logger.info(f"Best threshold: {best_thresh:.2f} (F1={best_f1:.4f})")
    return best_thresh

def plot_roc_curves(models_dict, y_true, save=True):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
    for (name, proba), color in zip(models_dict.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / "roc_curves.png", bbox_inches="tight", dpi=150)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name, save=True):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)
    classes = ["Pass (0)", "Fail (1)"]
    ax.set_xticks([0,1]); ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks([0,1]); ax.set_yticklabels(classes, fontsize=11)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=14,
                    fontweight="bold", color="white" if cm[i,j] > thresh else "black")
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / f"cm_{model_name.lower().replace(' ','_')}.png", bbox_inches="tight", dpi=150)
    plt.show()
