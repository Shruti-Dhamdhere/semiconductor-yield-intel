"""
src/data/graph_builder.py
-------------------------
Constructs a process dependency graph from SECOM sensor data.

Each node = one sensor/feature
Each edge = strong correlation between two sensors (|r| > threshold)

This graph structure is what allows the GNN to capture
relational patterns between process steps that tabular
models completely ignore.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_params():
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def build_correlation_graph(X: pd.DataFrame, threshold: float, max_neighbors: int):
    """
    Build graph edges from Pearson correlation matrix.

    Two sensors are connected if |correlation| > threshold.
    Each node is limited to max_neighbors edges to avoid
    over-connected graphs.

    Args:
        X: Feature matrix (samples x features)
        threshold: Minimum |correlation| to create an edge
        max_neighbors: Max edges per node

    Returns:
        edge_index: Tensor of shape (2, num_edges)
        edge_weight: Tensor of shape (num_edges,)
    """
    logger.info(f"Computing correlation matrix for {X.shape[1]} features...")
    corr = X.corr().abs()

    edge_src = []
    edge_dst = []
    edge_weights = []

    n_features = len(corr)

    for i in range(n_features):
        # Get correlations for node i, exclude self
        node_corr = corr.iloc[i].copy()
        node_corr.iloc[i] = 0  # remove self-loop

        # Keep only top max_neighbors above threshold
        above_thresh = node_corr[node_corr > threshold]
        top_neighbors = above_thresh.nlargest(max_neighbors)

        for j, weight in top_neighbors.items():
            j_idx = corr.columns.get_loc(j)
            edge_src.append(i)
            edge_dst.append(j_idx)
            edge_weights.append(float(weight))

    if len(edge_src) == 0:
        logger.warning("No edges found! Lowering threshold recommended.")
        # Add self-loops as fallback so GNN doesn't crash
        edge_src = list(range(n_features))
        edge_dst = list(range(n_features))
        edge_weights = [1.0] * n_features

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)

    logger.info(
        f"Graph built — Nodes: {n_features}, "
        f"Edges: {len(edge_src)}, "
        f"Avg degree: {len(edge_src)/n_features:.1f}"
    )
    return edge_index, edge_weight


def build_node_features(X: pd.DataFrame):
    """
    Build node feature matrix.
    Each node (sensor) gets a feature vector = its values across all samples.
    Shape: (num_features, num_samples) — transposed from usual ML convention.
    """
    X_tensor = torch.tensor(X.values.T, dtype=torch.float)
    logger.info(f"Node feature matrix shape: {X_tensor.shape} (nodes x samples)")
    return X_tensor


def save_graph(edge_index, edge_weight, node_features, feature_names):
    """Save graph as a PyTorch file."""
    graph_data = {
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "node_features": node_features,
        "feature_names": feature_names,
        "num_nodes": len(feature_names),
        "num_edges": edge_index.shape[1],
    }
    out_path = PROCESSED_DIR / "graph.pt"
    torch.save(graph_data, out_path)
    logger.success(f"Graph saved to {out_path}")
    return out_path


def print_graph_stats(edge_index, feature_names):
    """Print graph statistics useful for the paper."""
    n_nodes = len(feature_names)
    n_edges = edge_index.shape[1]
    degrees = torch.bincount(edge_index[0], minlength=n_nodes)

    logger.info("─── Graph Statistics ──────────────────────────")
    logger.info(f"  Nodes (sensors)     : {n_nodes}")
    logger.info(f"  Edges               : {n_edges}")
    logger.info(f"  Avg degree          : {degrees.float().mean():.2f}")
    logger.info(f"  Max degree          : {degrees.max().item()}")
    logger.info(f"  Min degree          : {degrees.min().item()}")
    logger.info(f"  Graph density       : {n_edges/(n_nodes*(n_nodes-1)):.4f}")
    logger.info("───────────────────────────────────────────────")


def run_graph_builder():
    """Full graph construction pipeline."""
    logger.info("Starting graph construction pipeline...")
    params = load_params()
    graph_params = params["graph"]

    # Load training features (graph structure learned from train only)
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")

    with open(PROCESSED_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    # Align columns
    available_cols = [c for c in feature_names if c in X_train.columns]
    X_train = X_train[available_cols]

    logger.info(f"Building graph from {len(available_cols)} features, {len(X_train)} samples")

    # Build graph
    edge_index, edge_weight = build_correlation_graph(
        X_train,
        threshold=graph_params["correlation_threshold"],
        max_neighbors=graph_params["max_neighbors"],
    )

    # Build node features
    node_features = build_node_features(X_train)

    # Print stats
    print_graph_stats(edge_index, available_cols)

    # Save
    save_graph(edge_index, edge_weight, node_features, available_cols)
    logger.success("Graph construction complete.")


if __name__ == "__main__":
    run_graph_builder()
