"""Serial feature selection: ICC filtering -> relevance filter -> LASSO with nested CV.

The relevance filter is mRMR when `pymrmr` is installed and mutual information
(`mutual_info_classif`) otherwise. Which one ran is returned by mrmr_select and
must be persisted with the results; see FS_MRMR / FS_MI_FALLBACK below.
"""
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


#: Identifiers for the filter step that actually ran. These strings are written
#: into result CSVs (column ``fs_method``) so every reported number carries the
#: name of the method that produced it. Never write "mRMR" in a report without
#: checking this value: `pymrmr` needs a C++ toolchain and is absent on most
#: Windows setups, in which case FS_MI_FALLBACK is what really ran.
FS_MRMR = "mrmr"
FS_MI_FALLBACK = "mutual_info_classif"


def mrmr_select(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_select: int = 50,
    random_state: int = 42,
) -> tuple[list[str], str]:
    """Max-relevance min-redundancy filter, with a mutual-information fallback.

    Returns ``(selected_feature_names, method)`` where ``method`` is one of
    :data:`FS_MRMR` or :data:`FS_MI_FALLBACK`.

    The method is returned rather than only logged because `pymrmr` is an
    optional dependency that fails to build without a C++ toolchain. When it is
    missing, this function silently computes something else, and a warning in a
    log file is not enough to stop the wrong method being named in a write-up.
    Callers are expected to persist the returned value alongside their metrics.

    ``random_state`` is passed to ``mutual_info_classif``, whose k-nearest-
    neighbour estimator adds small random noise to break ties. Left unseeded,
    two runs of an otherwise identical pipeline select slightly different
    feature sets and report slightly different AUCs -- the measured spread was
    up to 0.0036 AUC on radiomics_only, an arm no code change had touched.
    Seeding makes the selection step reproducible. mRMR is deterministic, so
    this affects the fallback path only.
    """
    try:
        import pymrmr
        df_in = pd.DataFrame(X, columns=feature_names)
        df_in["target"] = y.astype(int)
        selected = pymrmr.mRMR(df_in, "MIQ", n_select)
        logger.info("mRMR selected %d features", len(selected))
        return selected, FS_MRMR
    except ImportError:
        logger.warning(
            "pymrmr not installed; feature selection is %s, NOT mRMR. "
            "Report this method name, not mRMR.", FS_MI_FALLBACK,
        )
        from functools import partial
        from sklearn.feature_selection import mutual_info_classif, SelectKBest
        scorer = partial(mutual_info_classif, random_state=random_state)
        selector = SelectKBest(scorer, k=min(n_select, X.shape[1]))
        selector.fit(X, y)
        selected = [feature_names[i] for i in selector.get_support(indices=True)]
        logger.info("%s selected %d features", FS_MI_FALLBACK, len(selected))
        return selected, FS_MI_FALLBACK


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
    """Run ICC -> relevance filter -> LASSO pipeline on training fold only.

    Args:
        features_df: full feature DataFrame
        train_mask: boolean mask for training rows
        icc_threshold: ICC cutoff for stability filter
        mrmr_n: max features to keep after the relevance filter
        perturbed_df: optional second extraction for ICC; skips ICC if None
        seed: random seed

    Returns dict with keys: selected_features, scaler, lasso, icc_features,
    mrmr_features, fs_method. ``fs_method`` names the filter that actually ran
    and must be reported alongside any metric derived from these features.
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

    # mRMR, or the mutual-information fallback when pymrmr is unavailable
    mrmr_cols, fs_method = mrmr_select(X_train, y_train, icc_cols, n_select=mrmr_n,
                                        random_state=seed)
    X_mrmr = train_df[mrmr_cols].values

    # LASSO
    selected, scaler, lasso = lasso_select(X_mrmr, y_train, mrmr_cols, random_state=seed)

    return {
        "selected_features": selected,
        "scaler": scaler,
        "lasso": lasso,
        "icc_features": icc_cols,
        "mrmr_features": mrmr_cols,
        "fs_method": fs_method,
    }
