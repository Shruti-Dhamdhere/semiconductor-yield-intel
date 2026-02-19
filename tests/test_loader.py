import numpy as np
import pandas as pd
import pytest
from src.data.loader import binarize_labels, drop_high_missing, drop_zero_variance, split_data


@pytest.fixture
def sample_X():
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 10),
                     columns=[f"sensor_{i:03d}" for i in range(10)])
    X.iloc[:60, 0] = np.nan
    X["constant"] = 1.0
    return X


@pytest.fixture
def sample_y():
    return pd.Series([0] * 93 + [1] * 7)


def test_binarize_labels():
    y = pd.Series([-1, -1, 1, -1, 1])
    assert list(binarize_labels(y)) == [0, 0, 1, 0, 1]


def test_drop_high_missing(sample_X):
    X_f, dropped = drop_high_missing(sample_X, 0.5)
    assert "sensor_000" in dropped
    assert "sensor_001" not in dropped


def test_drop_zero_variance(sample_X):
    X_filled = sample_X.fillna(0)
    _, dropped = drop_zero_variance(X_filled)
    assert "constant" in dropped


def test_split_no_leakage(sample_X, sample_y):
    X = sample_X.fillna(0).iloc[:, :5]
    splits = split_data(X, sample_y, 0.2, 0.1, 42)
    train_idx = set(splits["X_train"].index)
    test_idx = set(splits["X_test"].index)
    assert len(train_idx & test_idx) == 0
