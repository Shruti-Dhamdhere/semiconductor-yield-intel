"""
src/models/active_learning.py
Uncertainty-based active learning simulation for yield prediction.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR   = PROJECT_ROOT / "paper" / "results"

def load_data():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv").values.astype(np.float32)
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze().values
    X_test  = pd.read_csv(PROCESSED_DIR / "X_test.csv").values.astype(np.float32)
    y_test  = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze().values
    return X_train, y_train, X_test, y_test

def train_model(X, y):
    clf = LogisticRegression(
        class_weight="balanced", max_iter=500,
        random_state=42, solver="saga", n_jobs=1
    )
    clf.fit(X, y)
    return clf

def get_uncertainty(model, X, strategy="least_confidence"):
    proba = model.predict_proba(X)[:, 1]
    if strategy == "least_confidence":
        return 1 - np.abs(proba - 0.5) * 2
    elif strategy == "entropy":
        eps = 1e-10
        return -(proba * np.log(proba + eps) + (1-proba) * np.log(1-proba + eps))
    return 1 - np.abs(proba - 0.5) * 2

def run_active_learning_simulation(strategy="least_confidence"):
    logger.info(f"Active learning simulation — strategy: {strategy}")
    X_train, y_train, X_test, y_test = load_data()

    n_total   = len(X_train)
    init_size = max(int(n_total * 0.2), 30)
    query_size = 20
    n_queries  = 15

    # ── Active Learning ───────────────────────────────────────────
    np.random.seed(42)
    all_idx = np.arange(n_total)

    # Stratified init
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    n_pos_init = max(3, int(init_size * y_train.mean()))
    n_neg_init = init_size - n_pos_init

    labeled_idx = list(np.random.choice(pos_idx, n_pos_init, replace=False)) +                   list(np.random.choice(neg_idx, n_neg_init, replace=False))
    unlabeled_idx = [i for i in all_idx if i not in set(labeled_idx)]

    al_aucs, al_sizes = [], []

    for q in range(n_queries + 1):
        X_lab = X_train[labeled_idx]
        y_lab = y_train[labeled_idx]

        model = train_model(X_lab, y_lab)
        test_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, test_proba)
        al_aucs.append(round(auc, 4))
        al_sizes.append(len(labeled_idx))
        logger.info(f"AL  Query {q:02d} | Labeled: {len(labeled_idx):4d} | AUC: {auc:.4f}")

        if unlabeled_idx and q < n_queries:
            unc = get_uncertainty(model, X_train[unlabeled_idx], strategy)
            top = np.argsort(unc)[-min(query_size, len(unlabeled_idx)):]
            queried = [unlabeled_idx[i] for i in top]
            labeled_idx += queried
            unlabeled_idx = [i for i in unlabeled_idx if i not in set(queried)]

    # ── Random Baseline ───────────────────────────────────────────
    np.random.seed(99)
    labeled_r   = list(np.random.choice(pos_idx, n_pos_init, replace=False)) +                   list(np.random.choice(neg_idx, n_neg_init, replace=False))
    unlabeled_r = [i for i in all_idx if i not in set(labeled_r)]

    rand_aucs, rand_sizes = [], []

    for q in range(n_queries + 1):
        X_lab = X_train[labeled_r]
        y_lab = y_train[labeled_r]
        model = train_model(X_lab, y_lab)
        test_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, test_proba)
        rand_aucs.append(round(auc, 4))
        rand_sizes.append(len(labeled_r))
        logger.info(f"RND Query {q:02d} | Labeled: {len(labeled_r):4d} | AUC: {auc:.4f}")

        if unlabeled_r and q < n_queries:
            pick = np.random.choice(len(unlabeled_r), min(query_size, len(unlabeled_r)), replace=False)
            queried_r = [unlabeled_r[i] for i in pick]
            labeled_r += queried_r
            unlabeled_r = [i for i in unlabeled_r if i not in set(queried_r)]

    # ── Summary ───────────────────────────────────────────────────
    logger.info("── Active Learning Summary ─────────────────────────")
    logger.info(f"Final AL AUC:     {al_aucs[-1]:.4f} ({al_sizes[-1]} samples)")
    logger.info(f"Final Random AUC: {rand_aucs[-1]:.4f} ({rand_sizes[-1]} samples)")

    # Labeling efficiency: samples needed to reach 90% of final AUC
    target = 0.9 * al_aucs[-1]
    al_thresh  = next((s for s,a in zip(al_sizes, al_aucs)   if a >= target), al_sizes[-1])
    rnd_thresh = next((s for s,a in zip(rand_sizes, rand_aucs) if a >= target), rand_sizes[-1])
    saving = max(0, rnd_thresh - al_thresh)
    logger.info(f"Samples to reach 90% performance — AL: {al_thresh}, Random: {rnd_thresh}")
    logger.info(f"Labeling cost reduction: {saving} samples saved ({saving/rnd_thresh*100:.1f}%)")
    logger.info("────────────────────────────────────────────────────")

    results = {
        "strategy": strategy,
        "active_learning": {"sizes": al_sizes, "aucs": al_aucs},
        "random_baseline":  {"sizes": rand_sizes, "aucs": rand_aucs},
        "final_al_auc":     al_aucs[-1],
        "final_random_auc": rand_aucs[-1],
        "samples_saved":    saving,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "active_learning_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.success("Active learning complete.")
    return results

if __name__ == "__main__":
    run_active_learning_simulation()
