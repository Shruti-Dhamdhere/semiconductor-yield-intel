"""
src/data/augmentation.py
------------------------
Feature engineering pipeline:
- Median imputation
- Variance filtering
- Correlation-based feature removal
- Standard scaling
- SMOTE oversampling for class imbalance
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_params():
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def impute_missing(X_train, X_val, X_test, strategy="median"):
    """Fit imputer on train only, transform all splits."""
    logger.info(f"Imputing missing values using strategy: {strategy}")
    imputer = SimpleImputer(strategy=strategy)
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )
    X_val_imp = pd.DataFrame(
        imputer.transform(X_val),
        columns=X_val.columns, index=X_val.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns, index=X_test.index
    )
    logger.success("Imputation complete. No missing values remain.")
    return X_train_imp, X_val_imp, X_test_imp, imputer


def remove_low_variance(X_train, X_val, X_test, threshold=0.01):
    """Remove features with variance below threshold (fit on train only)."""
    variances = X_train.var()
    low_var_cols = variances[variances < threshold].index.tolist()
    logger.info(f"Removing {len(low_var_cols)} low-variance features (threshold={threshold})")
    keep_cols = [c for c in X_train.columns if c not in low_var_cols]
    return (
        X_train[keep_cols],
        X_val[keep_cols],
        X_test[keep_cols],
        keep_cols,
    )


def remove_high_correlation(X_train, X_val, X_test, threshold=0.95):
    """
    Remove one feature from each highly correlated pair.
    Decision made on train set only.
    """
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    logger.info(
        f"Removing {len(to_drop)} highly correlated features (threshold={threshold})"
    )
    return (
        X_train.drop(columns=to_drop),
        X_val.drop(columns=to_drop),
        X_test.drop(columns=to_drop),
        to_drop,
    )


def scale_features(X_train, X_val, X_test):
    """Standard scaling — fit on train, transform all."""
    logger.info("Applying StandardScaler...")
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )
    X_val_sc = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns, index=X_val.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns, index=X_test.index
    )
    return X_train_sc, X_val_sc, X_test_sc, scaler


def apply_smote(X_train, y_train, random_seed=42):
    """
    Apply SMOTE to training set only to handle class imbalance.
    NEVER apply to val or test sets.
    """
    logger.info(f"Applying SMOTE. Before: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=random_seed)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    y_res = pd.Series(y_res, name=y_train.name)
    logger.success(f"After SMOTE: {y_res.value_counts().to_dict()}")
    return pd.DataFrame(X_res, columns=X_train.columns), y_res


def run_feature_engineering():
    """Full feature engineering pipeline."""
    logger.info("Starting feature engineering pipeline...")
    params = load_params()
    fe = params["feature_engineering"]

    # Load splits from loader output
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_val = pd.read_csv(PROCESSED_DIR / "X_val.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_val = pd.read_csv(PROCESSED_DIR / "y_val.csv").squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()

    logger.info(f"Loaded splits — Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Step 1: Impute
    X_train, X_val, X_test, _ = impute_missing(
        X_train, X_val, X_test, strategy=fe["imputation_strategy"]
    )

    # Step 2: Remove low variance
    X_train, X_val, X_test, kept_cols = remove_low_variance(
        X_train, X_val, X_test, threshold=fe["variance_threshold"]
    )

    # Step 3: Remove high correlation
    X_train, X_val, X_test, dropped_corr = remove_high_correlation(
        X_train, X_val, X_test, threshold=fe["correlation_threshold"]
    )

    # Step 4: Scale
    X_train, X_val, X_test, _ = scale_features(X_train, X_val, X_test)

    # Step 5: SMOTE on train only
    X_train_sm, y_train_sm = apply_smote(
        X_train, y_train, random_seed=params["data"]["random_seed"]
    )

    # Save engineered features (overwrite processed dir)
    X_train_sm.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_val.to_csv(PROCESSED_DIR / "X_val.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train_sm.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_val.to_csv(PROCESSED_DIR / "y_val.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    # Save final feature names
    final_features = X_train.columns.tolist()
    with open(PROCESSED_DIR / "feature_names.json", "w") as f:
        json.dump(final_features, f)

    logger.success(
        f"Feature engineering complete. "
        f"Final features: {len(final_features)} | "
        f"Train samples after SMOTE: {len(X_train_sm)}"
    )


if __name__ == "__main__":
    run_feature_engineering()
