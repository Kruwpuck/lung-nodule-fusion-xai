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


def ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, Quadratic Weighted Kappa, and within-1-grade accuracy for a 1-5 ordinal target.

    y_true/y_pred are raw (possibly fractional) ratings; rounded to the nearest
    integer grade in [1, 5] for QWK and within-1 accuracy.
    """
    from sklearn.metrics import cohen_kappa_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.abs(y_true - y_pred).mean())
    t = np.clip(np.round(y_true), 1, 5).astype(int)
    p = np.clip(np.round(y_pred), 1, 5).astype(int)
    return {
        "mae": mae,
        "qwk": float(cohen_kappa_score(t, p, weights="quadratic")),
        "acc_within_1": float((np.abs(t - p) <= 1).mean()),
    }


def derive_binary(y_rating_true: np.ndarray, y_rating_pred: np.ndarray) -> dict[str, float]:
    """Binary AUC/metrics derived from ordinal predictions, on the median!=3 subset only.

    This is the endpoint directly comparable to the existing binary baseline
    (arm A), since that baseline also excludes median==3 nodules.
    """
    y_rating_true = np.asarray(y_rating_true, dtype=float)
    y_rating_pred = np.asarray(y_rating_pred, dtype=float)
    mask = y_rating_true != 3.0
    y_true_bin = (y_rating_true[mask] > 3).astype(int)
    # compute_metrics expects a [0,1] probability; ordinal predictions live on
    # the 1-5 rating scale, so rescale linearly (rating=3 -> prob=0.5, the
    # same boundary used for y_true_bin above).
    y_prob = np.clip((y_rating_pred[mask] - 1.0) / 4.0, 0.0, 1.0)
    return compute_metrics(y_true_bin, y_prob)


def derive_grade3(
    y_rating_true: np.ndarray,
    y_rating_pred: np.ndarray,
    lo: float = 2.75,
    hi: float = 3.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin ordinal ratings into 3 classes: 0=benign, 1=indeterminate, 2=malignant."""
    def to3(v: np.ndarray) -> np.ndarray:
        return np.where(v < lo, 0, np.where(v > hi, 2, 1))

    return to3(np.asarray(y_rating_true, dtype=float)), to3(np.asarray(y_rating_pred, dtype=float))


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


def grade4_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Macro-AUC (OvR), accuracy, macro-F1 for the 4-class
    no-nodule/benign/indeterminate/malignant task (arm D)."""
    from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

    y_pred = np.argmax(y_prob, axis=1)
    return {
        "auc_macro": float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def grade4_nodule_only_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Benign-vs-malignant metrics on the nodule-only subset of arm D (grade4).

    Excludes class 0 (no-nodule) and class 2 (indeterminate) so a strong 4-class
    macro-AUC/accuracy can't be inflated by the trivially separable no-nodule
    negatives. This is the headline metric for arm D, per the anti-inflation gate.
    """
    y_true = np.asarray(y_true)
    mask = np.isin(y_true, [1, 3])
    if mask.sum() == 0:
        return {"auc_nodule_only": float("nan"), "accuracy_nodule_only": float("nan"),
                "sensitivity_nodule_only": float("nan"), "specificity_nodule_only": float("nan"),
                "precision_nodule_only": float("nan"), "f1_nodule_only": float("nan"),
                "balanced_accuracy_nodule_only": float("nan"), "brier_score_nodule_only": float("nan"),
                "n_nodule_only": 0}

    y_true_bin = (y_true[mask] == 3).astype(int)
    p_benign = y_prob[mask, 1]
    p_malignant = y_prob[mask, 3]
    p_bin = p_malignant / (p_benign + p_malignant + 1e-10)

    m = compute_metrics(y_true_bin, p_bin)
    out = {f"{k}_nodule_only": v for k, v in m.items() if k not in ("tp", "tn", "fp", "fn")}
    out["n_nodule_only"] = int(mask.sum())
    return out
