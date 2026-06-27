"""External validation pipeline: run frozen model on NSCLC-Radiomics / NLST subset."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_external_inference(
    model,
    external_loader,
    device,
    is_fusion: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run frozen model on external dataset. Returns (y_true, y_prob)."""
    import torch
    from src.training.trainer import evaluate
    model.eval()
    return evaluate(model, external_loader, device, is_fusion)


def compute_external_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    internal_auc: float,
    output_csv: str = "results/external/external_validation.csv",
) -> pd.DataFrame:
    """Compute metrics on external set and report delta AUC vs internal."""
    from src.evaluation.metrics import compute_metrics, bootstrap_ci

    m = compute_metrics(y_true, y_prob)
    ci_lo, ci_hi = bootstrap_ci(y_true, y_prob)
    delta_auc = m["auc"] - internal_auc

    results = {
        **m,
        "auc_ci_lo": ci_lo,
        "auc_ci_hi": ci_hi,
        "internal_auc": internal_auc,
        "delta_auc_internal_to_external": delta_auc,
    }
    df = pd.DataFrame([results])

    os.makedirs(Path(output_csv).parent, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("External AUC: %.4f (%.3f, %.3f) | ΔAUC: %.4f",
                m["auc"], ci_lo, ci_hi, delta_auc)
    return df


def compute_route_b_icc_delta_auc(
    features_route_a: pd.DataFrame,
    features_route_b: pd.DataFrame,
    labels_df: pd.DataFrame,
    clf,               # trained classifier (XGBoost or similar)
    scaler,            # fitted StandardScaler
    selected_features: list[str],
    output_csv: str = "results/external/route_b_icc.csv",
) -> pd.DataFrame:
    """Compare Route A vs Route B: per-feature ICC and delta AUC.

    Both feature DataFrames must have same row order and same feature columns.
    """
    from src.segmentation.consensus import compute_icc
    from sklearn.metrics import roc_auc_score

    feat_cols = selected_features
    a_vals = features_route_a[feat_cols].values
    b_vals = features_route_b[feat_cols].values
    icc_per_feature = compute_icc(a_vals, b_vals)

    # AUC comparison
    y_true = labels_df["label"].values
    X_a = scaler.transform(features_route_a[feat_cols].values)
    X_b = scaler.transform(features_route_b[feat_cols].values)

    prob_a = clf.predict_proba(X_a)[:, 1]
    prob_b = clf.predict_proba(X_b)[:, 1]
    auc_a = roc_auc_score(y_true, prob_a)
    auc_b = roc_auc_score(y_true, prob_b)

    icc_df = pd.DataFrame({
        "feature": feat_cols,
        "icc": icc_per_feature,
        "feature_class": [f.split("_")[1] if "_" in f else "other" for f in feat_cols],
    })

    # summary per feature class
    summary = icc_df.groupby("feature_class")["icc"].agg(["mean", "std", "count"])
    summary["auc_route_a"] = auc_a
    summary["auc_route_b"] = auc_b
    summary["delta_auc"] = auc_b - auc_a

    os.makedirs(Path(output_csv).parent, exist_ok=True)
    icc_df.to_csv(output_csv.replace(".csv", "_per_feature.csv"), index=False)
    summary.to_csv(output_csv, index=True)
    logger.info("Route B vs A: AUC %.4f → %.4f (ΔAUC %.4f)", auc_a, auc_b, auc_b - auc_a)
    return summary
