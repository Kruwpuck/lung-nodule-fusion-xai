"""DeLong test for paired AUC comparison and ablation table generation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _auc_variance(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute AUC and the structural components needed for DeLong variance."""
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    pos_scores = y_prob[pos_idx]
    neg_scores = y_prob[neg_idx]

    # placement values
    v10 = np.array([
        np.mean((neg_scores < p) + 0.5 * (neg_scores == p))
        for p in pos_scores
    ])
    v01 = np.array([
        np.mean((pos_scores > n) + 0.5 * (pos_scores == n))
        for n in neg_scores
    ])
    auc = v10.mean()
    return auc, v10, v01


def delong_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
) -> tuple[float, float, float]:
    """DeLong test for paired AUC comparison.

    Returns (z_stat, p_value, delta_auc).
    H0: AUC_A == AUC_B.
    """
    auc_a, v10_a, v01_a = _auc_variance(y_true, y_prob_a)
    auc_b, v10_b, v01_b = _auc_variance(y_true, y_prob_b)

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    # structural variance (covariance between two AUCs on same dataset)
    s10 = np.cov(v10_a, v10_b)  # 2x2 covariance matrix
    s01 = np.cov(v01_a, v01_b)

    var_a = s10[0, 0] / n_pos + s01[0, 0] / n_neg
    var_b = s10[1, 1] / n_pos + s01[1, 1] / n_neg
    cov_ab = s10[0, 1] / n_pos + s01[0, 1] / n_neg

    var_diff = var_a + var_b - 2 * cov_ab
    if var_diff <= 0:
        return 0.0, 1.0, auc_a - auc_b

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p_value), float(auc_a - auc_b)


def build_ablation_table(
    model_results: dict[str, dict],
    reference_model: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Build ablation comparison table.

    model_results: {model_name: {"y_true": array, "y_prob": array, **metrics}}
    reference_model: model name used as baseline for DeLong test
    """
    ref = model_results[reference_model]
    rows = []

    for name, res in model_results.items():
        from src.evaluation.metrics import compute_metrics, bootstrap_ci
        from sklearn.metrics import roc_auc_score

        m = compute_metrics(res["y_true"], res["y_prob"])
        ci_lo, ci_hi = bootstrap_ci(res["y_true"], res["y_prob"])

        if name != reference_model:
            z, p_val, delta_auc = delong_test(
                ref["y_true"], ref["y_prob"], res["y_prob"]
            )
        else:
            z, p_val, delta_auc = 0.0, 1.0, 0.0

        rows.append({
            "model": name,
            "auc": m["auc"],
            "auc_ci_95": f"({ci_lo:.3f}, {ci_hi:.3f})",
            "accuracy": m["accuracy"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "f1": m["f1"],
            "brier": m["brier_score"],
            "delta_auc_vs_ref": delta_auc,
            "delong_z": z,
            "delong_p": p_val,
            "significant": p_val < alpha,
        })

    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
