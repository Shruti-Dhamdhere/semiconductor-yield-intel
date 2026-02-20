"""
src/models/gnn.py
-----------------
Graph Neural Network models for wafer yield prediction.
Implements GCN, GAT, and GraphSAGE using PyTorch Geometric.

Each node = one sensor/process feature
Each edge = strong correlation between sensors
Node features = sensor readings across all training samples
Task = binary classification (pass/fail) at graph level
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    global_mean_pool,
    global_max_pool,
)
from loguru import logger


class GCN(nn.Module):
    """
    Graph Convolutional Network (Kipf & Welling, 2017).
    Aggregates neighbor information via symmetric normalization.
    """
    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_weight)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level readout: concat mean + max pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        x_graph = torch.cat([x_mean, x_max], dim=1)

        return self.classifier(x_graph).squeeze(-1)


class GAT(nn.Module):
    """
    Graph Attention Network (Velickovic et al., 2018).
    Uses attention mechanism to weight neighbor contributions.
    This is our primary model — attention weights map to sensor importance.
    """
    def __init__(self, in_channels, hidden_channels, num_layers=3,
                 dropout=0.3, heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels,
                                  heads=heads, dropout=dropout, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                      heads=heads, dropout=dropout, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels * heads))

        # Final layer: single head, no concat
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                  heads=1, dropout=dropout, concat=False))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        x_graph = torch.cat([x_mean, x_max], dim=1)

        return self.classifier(x_graph).squeeze(-1)


class GraphSAGE(nn.Module):
    """
    GraphSAGE (Hamilton et al., 2017).
    Inductive learning via neighborhood sampling and aggregation.
    """
    def __init__(self, in_channels, hidden_channels, num_layers=3,
                 dropout=0.3, aggr="mean"):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        x_graph = torch.cat([x_mean, x_max], dim=1)

        return self.classifier(x_graph).squeeze(-1)


def build_model(model_type, in_channels, hidden_channels,
                num_layers, dropout, heads=4):
    """Factory function to build GNN model by type."""
    model_type = model_type.upper()
    if model_type == "GCN":
        return GCN(in_channels, hidden_channels, num_layers, dropout)
    elif model_type == "GAT":
        return GAT(in_channels, hidden_channels, num_layers, dropout, heads)
    elif model_type == "GRAPHSAGE":
        return GraphSAGE(in_channels, hidden_channels, num_layers, dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose GCN, GAT, or GRAPHSAGE.")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total:,} | Trainable: {trainable:,}")
    return trainable
