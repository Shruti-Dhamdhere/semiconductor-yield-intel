"""
dashboard/app.py
Plotly Dash interactive dashboard for semiconductor yield prediction.
Shows model results, SHAP importance, causal graph, and active learning curves.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import base64

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
RESULTS_DIR   = PROJECT_ROOT / "paper" / "results"
FIGURES_DIR   = PROJECT_ROOT / "paper" / "figures"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

app = dash.Dash(
    __name__,
    title="Semiconductor Yield Intelligence",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# ── Load data ─────────────────────────────────────────────────────────────────
def load_json(filename):
    path = RESULTS_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def load_image_b64(filename):
    path = FIGURES_DIR / filename
    if path.exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

baseline   = load_json("baseline_metrics.json")
gnn        = load_json("gnn_metrics.json")
ensemble   = load_json("ensemble_metrics.json")
al_results = load_json("active_learning_results.json")
shap_data  = load_json("shap_top_features.json")
causal     = load_json("causal_discovery_results.json")

# ── Color scheme ──────────────────────────────────────────────────────────────
COLORS = {
    "primary":    "#2c3e50",
    "accent":     "#3498db",
    "success":    "#2ecc71",
    "danger":     "#e74c3c",
    "warning":    "#f39c12",
    "purple":     "#9b59b6",
    "background": "#f8f9fa",
    "card":       "#ffffff",
}

# ── Model comparison chart ────────────────────────────────────────────────────
def make_model_comparison():
    models, aucs, f1s, mccs = [], [], [], []
    if baseline:
        for m, d in baseline.items():
            models.append(m.upper())
            aucs.append(d.get("test", {}).get("auc", 0))
            f1s.append(d.get("test", {}).get("f1", 0))
            mccs.append(d.get("test", {}).get("mcc", 0))
    if gnn:
        for m, d in gnn.items():
            models.append("GCN")
            aucs.append(d.get("test", {}).get("auc", 0))
            f1s.append(d.get("test", {}).get("f1", 0))
            mccs.append(d.get("test", {}).get("mcc", 0))
    if ensemble:
        models.append("Ensemble")
        aucs.append(ensemble.get("test", {}).get("auc", 0))
        f1s.append(ensemble.get("test", {}).get("f1", 0))
        mccs.append(ensemble.get("test", {}).get("mcc", 0))

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["AUC-ROC", "F1 Score", "MCC"])
    color_list = [COLORS["accent"], COLORS["success"], COLORS["purple"],
                  COLORS["warning"], COLORS["danger"]]

    for i, (vals, col) in enumerate(zip([aucs, f1s, mccs], [1, 2, 3])):
        fig.add_trace(go.Bar(
            x=models, y=vals,
            marker_color=color_list[:len(models)],
            showlegend=False,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
        ), row=1, col=col)

    fig.update_layout(
        title="Model Performance Comparison (Test Set)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        height=400,
    )
    return fig

# ── SHAP importance chart ─────────────────────────────────────────────────────
def make_shap_chart():
    if not shap_data:
        return go.Figure()
    features = shap_data.get("top20_features", [])[:10]
    values   = shap_data.get("top20_shap_values", [])[:10]
    features_short = [f.replace("sensor_", "S") for f in features]

    fig = go.Figure(go.Bar(
        x=values[::-1], y=features_short[::-1],
        orientation="h",
        marker=dict(
            color=values[::-1],
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="SHAP"),
        ),
        text=[f"{v:.3f}" for v in values[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top 10 Sensors — SHAP Feature Importance",
        xaxis_title="Mean |SHAP Value|",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
    )
    return fig

# ── Active learning chart ─────────────────────────────────────────────────────
def make_al_chart():
    if not al_results:
        return go.Figure()
    al  = al_results.get("active_learning", {})
    rnd = al_results.get("random_baseline", {})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=al.get("sizes", []), y=al.get("aucs", []),
        mode="lines+markers", name="Active Learning",
        line=dict(color=COLORS["accent"], width=3),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=rnd.get("sizes", []), y=rnd.get("aucs", []),
        mode="lines+markers", name="Random Sampling",
        line=dict(color=COLORS["danger"], width=3, dash="dash"),
        marker=dict(size=8),
    ))
    fig.update_layout(
        title="Active Learning vs Random Sampling",
        xaxis_title="Number of Labeled Samples",
        yaxis_title="AUC-ROC",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(x=0.7, y=0.1),
        height=400,
    )
    return fig

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(style={"backgroundColor": COLORS["background"],
                              "fontFamily": "Arial, sans-serif",
                              "minHeight": "100vh"}, children=[

    # Header
    html.Div(style={"backgroundColor": COLORS["primary"], "padding": "20px 40px",
                    "color": "white"}, children=[
        html.H1("Semiconductor Yield Intelligence",
                style={"margin": 0, "fontSize": "28px"}),
        html.P("GCN + LightGBM Ensemble | Causal Discovery | Active Learning",
               style={"margin": "5px 0 0", "opacity": 0.8}),
    ]),

    # KPI cards
    html.Div(style={"display": "flex", "gap": "20px", "padding": "20px 40px",
                    "flexWrap": "wrap"}, children=[
        html.Div(style={"backgroundColor": COLORS["card"], "padding": "20px",
                        "borderRadius": "8px", "flex": "1", "minWidth": "200px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "borderLeft": f"4px solid {COLORS['accent']}"}, children=[
            html.H3("Best AUC-ROC", style={"margin": 0, "color": COLORS["primary"]}),
            html.H2(f"{ensemble.get('test', {}).get('auc', 0.678):.3f}",
                    style={"margin": "10px 0 0", "color": COLORS["accent"],
                           "fontSize": "36px"}),
            html.P("LightGBM + GCN Ensemble", style={"color": "#666", "margin": 0}),
        ]),
        html.Div(style={"backgroundColor": COLORS["card"], "padding": "20px",
                        "borderRadius": "8px", "flex": "1", "minWidth": "200px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "borderLeft": f"4px solid {COLORS['success']}"}, children=[
            html.H3("Causal Factors", style={"margin": 0, "color": COLORS["primary"]}),
            html.H2(f"{len(causal.get('direct_causes', ['sensor_488', 'sensor_213']))}",
                    style={"margin": "10px 0 0", "color": COLORS["success"],
                           "fontSize": "36px"}),
            html.P("Direct yield failure causes", style={"color": "#666", "margin": 0}),
        ]),
        html.Div(style={"backgroundColor": COLORS["card"], "padding": "20px",
                        "borderRadius": "8px", "flex": "1", "minWidth": "200px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "borderLeft": f"4px solid {COLORS['warning']}"}, children=[
            html.H3("Labels Saved", style={"margin": 0, "color": COLORS["primary"]}),
            html.H2(f"{al_results.get('samples_saved', 39)}",
                    style={"margin": "10px 0 0", "color": COLORS["warning"],
                           "fontSize": "36px"}),
            html.P("Via active learning", style={"color": "#666", "margin": 0}),
        ]),
        html.Div(style={"backgroundColor": COLORS["card"], "padding": "20px",
                        "borderRadius": "8px", "flex": "1", "minWidth": "200px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "borderLeft": f"4px solid {COLORS['purple']}"}, children=[
            html.H3("GCN Nodes", style={"margin": 0, "color": COLORS["primary"]}),
            html.H2("446", style={"margin": "10px 0 0", "color": COLORS["purple"],
                                  "fontSize": "36px"}),
            html.P("Sensor process graph", style={"color": "#666", "margin": 0}),
        ]),
    ]),

    # Charts row 1
    html.Div(style={"display": "flex", "gap": "20px", "padding": "0 40px 20px",
                    "flexWrap": "wrap"}, children=[
        html.Div(style={"backgroundColor": COLORS["card"], "padding": "20px",
                        "borderRadius": "8px", "flex": "2", "minWidth": "400px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}, children=[
            dcc.Graph(figure=make_model_comparison()),
        ]),
        html.Div(style={"backgroundColor": COLORS["card"], "padding": "20px",
                        "borderRadius": "8px", "flex": "1", "minWidth": "300px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}, children=[
            dcc.Graph(figure=make_shap_chart()),
        ]),
    ]),

    # Charts row 2
    html.Div(style={"display": "flex", "gap": "20px", "padding": "0 40px 20px",
                    "flexWrap": "wrap"}, children=[
        html.Div(style={"backgroundColor": COLORS["card"], "padding": "20px",
                        "borderRadius": "8px", "flex": "1", "minWidth": "400px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}, children=[
            dcc.Graph(figure=make_al_chart()),
        ]),
        html.Div(style={"backgroundColor": COLORS["card"], "padding": "20px",
                        "borderRadius": "8px", "flex": "1", "minWidth": "400px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"}, children=[
            html.H3("Causal Root Causes", style={"color": COLORS["primary"]}),
            html.P("PC Algorithm identified direct causal factors for yield failure:",
                   style={"color": "#666"}),
            html.Div([
                html.Div(style={"backgroundColor": "#ffeaa7", "padding": "15px",
                                "borderRadius": "8px", "marginBottom": "10px",
                                "borderLeft": f"4px solid {COLORS['warning']}"}, children=[
                    html.Strong("sensor_488"),
                    html.P("Direct causal factor → yield failure",
                           style={"margin": 0, "color": "#666"}),
                ]),
                html.Div(style={"backgroundColor": "#ffeaa7", "padding": "15px",
                                "borderRadius": "8px", "marginBottom": "10px",
                                "borderLeft": f"4px solid {COLORS['warning']}"}, children=[
                    html.Strong("sensor_213"),
                    html.P("Direct causal factor → yield failure",
                           style={"margin": 0, "color": "#666"}),
                ]),
                html.Div(style={"backgroundColor": "#dfe6e9", "padding": "15px",
                                "borderRadius": "8px",
                                "borderLeft": f"4px solid {COLORS['accent']}"}, children=[
                    html.Strong("Top Predictive: sensor_033, sensor_103"),
                    html.P("Highest SHAP importance (correlated, not necessarily causal)",
                           style={"margin": 0, "color": "#666"}),
                ]),
            ]),
        ]),
    ]),

    # Footer
    html.Div(style={"backgroundColor": COLORS["primary"], "padding": "15px 40px",
                    "color": "white", "textAlign": "center"}, children=[
        html.P("Semiconductor Yield Intelligence — GCN + LightGBM + Causal Discovery | API: http://localhost:8000/docs",
               style={"margin": 0, "opacity": 0.8}),
    ]),
])

if __name__ == "__main__":
    app.run(debug=False, port=8050, host="0.0.0.0")
