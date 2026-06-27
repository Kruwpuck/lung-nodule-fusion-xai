"""Serial feature selection: ICC filtering -> mRMR -> LASSO with nested CV."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

FEATURE_COLS_EXCLUDE = {"patient_id", "nodule_idx", "label", "fold"}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return radiomic feature column names (excludes metadata cols)."""
    return [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]


def icc_filter(
    features_df: pd.DataFrame,
    perturbed_df: pd.DataFrame,
    feature_cols: list[str],
    icc_threshold: float = 0.75,
) -> list[str]:
    """Keep features with ICC > icc_threshold between two extraction runs.

    perturbed_df: features extracted with slight mask perturbation (±1 voxel dilation)
    to assess stability. Must have same row order as features_df.
    """
    from src.segmentation.consensus import compute_icc

    a = features_df[feature_cols].values
    b = perturbed_df[feature_cols].values
    icc_vals = compute_icc(a, b)

    stable_cols = [col for col, icc_val in zip(feature_cols, icc_vals)
                   if icc_val > icc_threshold]
    logger.info("ICC filter: %d/%d features retained (threshold=%.2f)",
                len(stable_cols), len(feature_cols), icc_threshold)
    return stable_cols


def mrmr_select(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_select: int = 50,
) -> list[str]:
    """mRMR (max-relevance min-redundancy) feature selection.

    Falls back to mutual-info top-k if pymrmr unavailable.
    """
    try:
        import pymrmr
        df_in = pd.DataFrame(X, columns=feature_names)
        df_in["target"] = y.astype(int)
        selected = pymrmr.mRMR(df_in, "MIQ", n_select)
        logger.info("mRMR selected %d features", len(selected))
        return selected
    except ImportError:
        logger.warning("pymrmr not installed; using mutual_info_classif fallback")
        from sklearn.feature_selection import mutual_info_classif, SelectKBest
        selector = SelectKBest(mutual_info_classif, k=min(n_select, X.shape[1]))
        selector.fit(X, y)
        selected = [feature_names[i] for i in selector.get_support(indices=True)]
        logger.info("MI-fallback selected %d features", len(selected))
        return selected


def lasso_select(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    cv: int = 5,
    random_state: int = 42,
) -> tuple[list[str], StandardScaler, LassoCV]:
    """LASSO-based feature selection with cross-validated alpha.

    Scaler is fit on training data only (pass back for test-time use).
    Returns selected feature names, fitted scaler, fitted LASSO model.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    lasso = LassoCV(cv=cv, random_state=random_state, max_iter=5000, n_jobs=-1)
    lasso.fit(X_scaled, y_train)

    selected_mask = lasso.coef_ != 0
    selected = [f for f, s in zip(feature_names, selected_mask) if s]
    logger.info("LASSO selected %d/%d features (alpha=%.4f)",
                len(selected), len(feature_names), lasso.alpha_)
    return selected, scaler, lasso


def full_feature_selection_pipeline(
    features_df: pd.DataFrame,
    train_mask: np.ndarray,
    icc_threshold: float = 0.75,
    mrmr_n: int = 50,
    perturbed_df: Optional[pd.DataFrame] = None,
    seed: int = 42,
) -> dict:
    """Run ICC -> mRMR -> LASSO pipeline on training fold only.

    Args:
        features_df: full feature DataFrame
        train_mask: boolean mask for training rows
        icc_threshold: ICC cutoff for stability filter
        mrmr_n: max features to keep after mRMR
        perturbed_df: optional second extraction for ICC; skips ICC if None
        seed: random seed

    Returns dict with keys: selected_features, scaler, lasso, icc_features, mrmr_features
    """
    feat_cols = get_feature_columns(features_df)

    # ICC filtering (optional)
    if perturbed_df is not None:
        icc_cols = icc_filter(features_df, perturbed_df, feat_cols, icc_threshold)
    else:
        logger.warning("No perturbed_df provided; skipping ICC filter")
        icc_cols = feat_cols

    # Train-only data for mRMR and LASSO
    train_df = features_df[train_mask]
    X_train = train_df[icc_cols].values
    y_train = train_df["label"].values

    # mRMR
    mrmr_cols = mrmr_select(X_train, y_train, icc_cols, n_select=mrmr_n)
    X_mrmr = train_df[mrmr_cols].values

    # LASSO
    selected, scaler, lasso = lasso_select(X_mrmr, y_train, mrmr_cols, seed=seed)

    return {
        "selected_features": selected,
        "scaler": scaler,
        "lasso": lasso,
        "icc_features": icc_cols,
        "mrmr_features": mrmr_cols,
    }
