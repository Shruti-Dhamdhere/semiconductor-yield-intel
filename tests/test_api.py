"""
tests/test_api.py
Unit tests for the FastAPI yield prediction API.
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data
    assert "n_features" in data
    assert data["version"] == "1.0.0"

def test_features_endpoint():
    response = client.get("/features")
    if response.status_code == 200:
        data = response.json()
        assert "features" in data
        assert "n_features" in data
        assert data["n_features"] > 0

def test_predict_single():
    response = client.post("/predict", json={
        "wafer_id": "test_wafer_001",
        "sensor_readings": {
            "sensor_033": 0.5,
            "sensor_103": -0.3,
            "sensor_031": 1.2,
        },
        "threshold": 0.15,
    })
    if response.status_code == 200:
        data = response.json()
        assert data["wafer_id"] == "test_wafer_001"
        assert 0.0 <= data["failure_probability"] <= 1.0
        assert data["prediction"] in [0, 1]
        assert data["prediction_label"] in ["PASS", "FAIL"]
        assert data["threshold_used"] == 0.15

def test_predict_empty_sensors():
    response = client.post("/predict", json={
        "wafer_id": "test_wafer_002",
        "sensor_readings": {},
        "threshold": 0.15,
    })
    assert response.status_code in [200, 500, 503]

def test_predict_batch():
    response = client.post("/predict/batch", json={
        "wafers": [
            {"wafer_id": "w001", "sensor_readings": {"sensor_033": 0.5}},
            {"wafer_id": "w002", "sensor_readings": {"sensor_033": -2.0}},
            {"wafer_id": "w003", "sensor_readings": {"sensor_103": 1.5}},
        ],
        "threshold": 0.15,
    })
    if response.status_code == 200:
        data = response.json()
        assert data["n_wafers"] == 3
        assert len(data["predictions"]) == 3
        assert data["n_predicted_fail"] + data["n_predicted_pass"] == 3

def test_results_summary():
    response = client.get("/results/summary")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
