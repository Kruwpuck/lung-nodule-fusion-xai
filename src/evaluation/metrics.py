"""Classification metrics: AUC, sensitivity, specificity, F1, calibration, Brier score."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, precision_score,
    f1_score, brier_score_loss, confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute full classification metric set from binary labels and probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-10)  # recall for positive class
    specificity = tn / (tn + fp + 1e-10)

    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn=roc_auc_score,
    n_iterations: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a scalar metric. Returns (lower, upper)."""
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)

    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_prob[idx]))

    alpha = (1 - ci) / 2
    return float(np.percentile(scores, alpha * 100)), float(np.percentile(scores, (1 - alpha) * 100))


def aggregate_fold_results(
    fold_results: list[dict],
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    """Aggregate per-fold results into summary table with mean ± std and 95% CI.

    Each element of fold_results: {"fold": int, "y_true": array, "y_prob": array}
    """
    rows = []
    for fr in fold_results:
        m = compute_metrics(fr["y_true"], fr["y_prob"])
        m["fold"] = fr["fold"]
        rows.append(m)

    df = pd.DataFrame(rows)

    # pooled CI (concatenate all folds)
    y_true_all = np.concatenate([fr["y_true"] for fr in fold_results])
    y_prob_all = np.concatenate([fr["y_prob"] for fr in fold_results])
    ci_lo, ci_hi = bootstrap_ci(y_true_all, y_prob_all, n_iterations=n_bootstrap)

    summary = df.drop(columns=["fold"]).agg(["mean", "std"])
    summary.loc["auc_ci_lo"] = {c: ci_lo if c == "auc" else np.nan for c in summary.columns}
    summary.loc["auc_ci_hi"] = {c: ci_hi if c == "auc" else np.nan for c in summary.columns}
    return summary


def build_calibration_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return calibration curve data (fraction_positive vs mean_predicted_prob per bin)."""
    from sklearn.calibration import calibration_curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return pd.DataFrame({
        "mean_predicted": mean_predicted_value,
        "fraction_positive": fraction_of_positives,
    })
