"""
src/data/loader.py
SECOM dataset ingestion and cleaning pipeline.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_params():
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def download_secom(params):
    raw_x_path = RAW_DIR / "secom.csv"
    raw_y_path = RAW_DIR / "secom_labels.csv"

    if raw_x_path.exists() and raw_y_path.exists():
        logger.info("Loading SECOM data from local cache...")
        X = pd.read_csv(raw_x_path)
        labels_df = pd.read_csv(raw_y_path)
        return X, labels_df["label"]

    logger.info("Downloading SECOM data from UCI repository...")
    X = pd.read_csv(params["data"]["secom_url"], sep=" ", header=None, na_values="NaN")
    X.columns = [f"sensor_{i:03d}" for i in range(X.shape[1])]

    labels_df = pd.read_csv(params["data"]["secom_labels_url"], sep=" ", header=None)
    labels_df.columns = ["label", "timestamp"]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    X.to_csv(raw_x_path, index=False)
    labels_df.to_csv(raw_y_path, index=False)

    logger.success(f"Downloaded: X{X.shape}, labels{labels_df.shape}")
    return X, labels_df["label"]


def binarize_labels(y):
    return (y == 1).astype(int)


def drop_high_missing(X, threshold):
    missing_ratio = X.isnull().mean()
    high_missing = missing_ratio[missing_ratio > threshold].index.tolist()
    logger.info(f"Dropping {len(high_missing)} features with >{threshold*100:.0f}% missing")
    return X.drop(columns=high_missing), high_missing


def drop_zero_variance(X):
    std = X.std()
    zero_var_cols = std[std == 0].index.tolist()
    logger.info(f"Dropping {len(zero_var_cols)} zero-variance features")
    return X.drop(columns=zero_var_cols), zero_var_cols


def log_class_distribution(y):
    counts = y.value_counts()
    total = len(y)
    logger.info(
        f"Class distribution — Pass: {counts.get(0,0)} "
        f"({counts.get(0,0)/total*100:.1f}%), "
        f"Fail: {counts.get(1,0)} ({counts.get(1,0)/total*100:.1f}%)"
    )


def split_data(X, y, test_size, val_size, random_seed):
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    rel_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=rel_val, random_state=random_seed, stratify=y_tv
    )
    logger.info(f"Split — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return {"X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test}


def save_splits(splits, feature_names):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        data.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    with open(PROCESSED_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)
    logger.success("All splits saved.")


def run_ingestion():
    params = load_params()
    X_raw, y_raw = download_secom(params)
    y = binarize_labels(y_raw)
    log_class_distribution(y)
    fe = params["feature_engineering"]
    X_filtered, _ = drop_high_missing(X_raw, fe["missing_threshold"])
    X_clean, _ = drop_zero_variance(X_filtered)
    dp = params["data"]
    splits = split_data(X_clean, y, dp["test_size"], dp["val_size"], dp["random_seed"])
    save_splits(splits, X_clean.columns.tolist())
    logger.success("Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()
