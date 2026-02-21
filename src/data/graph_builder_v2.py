"""
src/data/graph_builder_v2.py
----------------------------
Multi-criterion graph construction for GNN input.

Combines three edge criteria:
1. Pearson correlation  -- linear dependencies
2. Mutual information   -- non-linear dependencies
3. Partial correlation  -- direct dependencies (controls for confounders)

Each criterion produces a score in [0,1].
Final edge weight = weighted combination of all three.
Only edges above a combined threshold are kept.

This is the key methodological novelty over v1 (correlation-only).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from sklearn.feature_selection import mutual_info_classif
from scipy import stats

PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PARAMS_PATH   = PROJECT_ROOT / "params.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_params():
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def compute_pearson(X: np.ndarray) -> np.ndarray:
    """
    Pairwise absolute Pearson correlation matrix.
    Shape: (n_features, n_features)
    """
    logger.info("Computing Pearson correlation matrix...")
    corr = np.corrcoef(X.T)
    return np.abs(corr)


def compute_mutual_information(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Pairwise mutual information between features.
    MI(Xi, Xj) estimated by discretizing Xj into binary via median split,
    then using sklearn mutual_info_classif.

    Shape: (n_features, n_features)
    """
    logger.info("Computing mutual information matrix...")
    n_features = X.shape[1]
    mi_matrix = np.zeros((n_features, n_features))

    for j in range(n_features):
        # Discretize feature j by median split
        median_j = np.median(X[:, j])
        y_binary = (X[:, j] > median_j).astype(int)

        # Skip constant columns
        if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
            continue

        mi_scores = mutual_info_classif(
            X, y_binary, random_state=42, n_neighbors=3
        )
        mi_matrix[:, j] = mi_scores

    # Symmetrize
    mi_matrix = (mi_matrix + mi_matrix.T) / 2

    # Normalize to [0, 1]
    max_val = mi_matrix.max()
    if max_val > 0:
        mi_matrix = mi_matrix / max_val

    return mi_matrix


def compute_partial_correlation(X: np.ndarray) -> np.ndarray:
    """
    Partial correlation matrix via precision matrix inversion.
    Partial correlation controls for all other variables,
    capturing DIRECT dependencies between sensor pairs.

    Shape: (n_features, n_features)
    """
    logger.info("Computing partial correlation matrix...")
    n_features = X.shape[1]

    try:
        # Correlation matrix
        corr = np.corrcoef(X.T)

        # Add small regularization for numerical stability
        corr_reg = corr + np.eye(n_features) * 1e-6

        # Precision matrix = inverse of correlation matrix
        precision = np.linalg.inv(corr_reg)

        # Partial correlation from precision matrix
        partial_corr = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                if i == j:
                    partial_corr[i, j] = 1.0
                else:
                    denom = np.sqrt(precision[i, i] * precision[j, j])
                    if denom > 0:
                        partial_corr[i, j] = -precision[i, j] / denom

        return np.abs(partial_corr)

    except np.linalg.LinAlgError:
        logger.warning("Partial correlation failed (singular matrix). Using zeros.")
        return np.zeros((n_features, n_features))


def build_multicriteria_graph(
    X: np.ndarray,
    feature_names: list,
    pearson_w: float = 0.4,
    mi_w: float = 0.4,
    partial_w: float = 0.2,
    threshold: float = 0.5,
    max_neighbors: int = 10,
):
    """
    Build graph edges using weighted combination of three criteria.

    Args:
        X: Feature matrix (n_samples, n_features)
        feature_names: List of feature names
        pearson_w: Weight for Pearson correlation
        mi_w: Weight for mutual information
        partial_w: Weight for partial correlation
        threshold: Minimum combined score to create an edge
        max_neighbors: Maximum edges per node

    Returns:
        edge_index: (2, n_edges) tensor
        edge_weight: (n_edges,) tensor
        stats: dict of graph statistics
    """
    n_features = X.shape[1]
    logger.info(f"Building multi-criterion graph for {n_features} features...")

    # Compute all three criteria
    pearson_mat  = compute_pearson(X)
    mi_mat       = compute_mutual_information(X, None)
    partial_mat  = compute_partial_correlation(X)

    # Combined score
    combined = (pearson_w  * pearson_mat +
                mi_w       * mi_mat +
                partial_w  * partial_mat)

    # Zero out diagonal
    np.fill_diagonal(combined, 0)

    # Build edges
    edge_src, edge_dst, edge_weights = [], [], []

    for i in range(n_features):
        scores = combined[i].copy()
        scores[i] = 0  # no self-loops in edge list

        # Keep only above threshold
        above = np.where(scores > threshold)[0]

        # Keep top max_neighbors
        if len(above) > max_neighbors:
            top_idx = np.argsort(scores)[-max_neighbors:]
            above = top_idx[scores[top_idx] > threshold]

        for j in above:
            edge_src.append(i)
            edge_dst.append(int(j))
            edge_weights.append(float(scores[j]))

    # Add self-loops for isolated nodes
    degrees = np.zeros(n_features, dtype=int)
    for s in edge_src:
        degrees[s] += 1

    isolated = np.where(degrees == 0)[0]
    for i in isolated:
        edge_src.append(int(i))
        edge_dst.append(int(i))
        edge_weights.append(1.0)

    edge_index  = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    stats = {
        "n_nodes":         n_features,
        "n_edges":         len(edge_src),
        "n_isolated_fixed": len(isolated),
        "avg_degree":      len(edge_src) / n_features,
        "pearson_weight":  pearson_w,
        "mi_weight":       mi_w,
        "partial_weight":  partial_w,
        "threshold": threshold,
    }

    logger.info(f"Graph built -- Nodes: {n_features}, Edges: {len(edge_src)}, "
                f"Avg degree: {stats['avg_degree']:.1f}, "
                f"Isolated fixed: {len(isolated)}")

    return edge_index, edge_weight, stats


def run_graph_builder_v2():
    """Full v2 graph construction pipeline."""
    logger.info("Starting multi-criterion graph construction (v2)...")

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    with open(PROCESSED_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    available = [c for c in feature_names if c in X_train.columns]
    X = X_train[available].values.astype(np.float32)

    # Fill any remaining NaN with median
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    logger.info(f"Feature matrix: {X.shape}, NaNs: {np.isnan(X).sum()}")

    edge_index, edge_weight, stats = build_multicriteria_graph(
        X,
        feature_names=available,
        pearson_w=0.4,
        mi_w=0.4,
        partial_w=0.2,
        threshold=0.3,
        max_neighbors=10,
    )

    # Save graph
    graph_data = {
        "edge_index":    edge_index,
        "edge_weight":   edge_weight,
        "node_features": torch.tensor(X.T, dtype=torch.float),
        "feature_names": available,
        "num_nodes":     len(available),
        "num_edges":     edge_index.shape[1],
        "construction":  "multi_criterion_v2",
        "stats":         stats,
    }

    out_path = PROCESSED_DIR / "graph_v2.pt"
    torch.save(graph_data, out_path)

    logger.info("--- Multi-Criterion Graph Statistics ---")
    for k, v in stats.items():
        logger.info(f"  {k:<25} {v}")
    logger.info("----------------------------------------")
    logger.success(f"Graph v2 saved to {out_path}")
    return stats


if __name__ == "__main__":
    run_graph_builder_v2()
