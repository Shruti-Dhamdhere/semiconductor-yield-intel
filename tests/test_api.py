"""
tests/test_api.py
Unit tests for the FastAPI yield prediction API.
"""
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_schemas():
    from src.api.schemas import PredictRequest, PredictResponse
    req = PredictRequest(
        wafer_id="test_001",
        sensor_readings={"sensor_033": 0.5},
        threshold=0.15,
    )
    assert req.wafer_id == "test_001"
    assert req.threshold == 0.15


def test_predict_response_schema():
    from src.api.schemas import PredictResponse
    resp = PredictResponse(
        wafer_id="test_001",
        failure_probability=0.25,
        prediction=1,
        prediction_label="FAIL",
        confidence=0.25,
        threshold_used=0.15,
    )
    assert resp.prediction_label == "FAIL"


def test_loader_functions():
    from src.data.loader import binarize_labels, drop_high_missing
    y = pd.Series([-1, -1, 1, -1, 1])
    assert list(binarize_labels(y)) == [0, 0, 1, 0, 1]
    X = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
    X.iloc[:7, 0] = np.nan
    X_f, dropped = drop_high_missing(X, 0.5)
    assert "a" in dropped


def test_binarize_dtype():
    from src.data.loader import binarize_labels
    y = pd.Series([-1, 1, -1])
    assert binarize_labels(y).dtype == int
