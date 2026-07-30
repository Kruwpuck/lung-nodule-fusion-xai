"""DeLong test for paired AUC comparison and ablation table generation."""
from __future__ import annotations

from itertools import combinations

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


def holm_correction(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction.

    Standard textbook algorithm: sort ascending, multiply p_i by (n - rank),
    then enforce monotonicity by a running cumulative max, capped at 1.
    Returns adjusted p-values in the original input order.
    """
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((n - rank) * p[idx], 1.0)
        running_max = max(running_max, val)
        adjusted_sorted[rank] = running_max
    out = np.empty(n)
    out[order] = adjusted_sorted
    return out.tolist()


def brown_forsythe_test(*groups: np.ndarray) -> tuple[float, float]:
    """Brown-Forsythe test for equality of variances (Levene centered on the
    median). Primary variance test here because it tolerates skewed,
    non-normal AUC distributions better than Bartlett or mean-centered Levene.

    Returns (statistic, p_value).
    """
    stat, p = stats.levene(*groups, center="median")
    return float(stat), float(p)


def levene_test(*groups: np.ndarray) -> tuple[float, float]:
    """Classic Levene test for equality of variances (mean-centered)."""
    stat, p = stats.levene(*groups, center="mean")
    return float(stat), float(p)


def pairwise_variance_tests(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    test: str = "brown-forsythe",
) -> pd.DataFrame:
    """All pairwise variance-equality tests across levels of group_col, with
    Holm correction over the pairwise family.

    test: "brown-forsythe" (median-centered, primary) or "levene" (mean-centered).
    """
    if test == "brown-forsythe":
        test_fn = brown_forsythe_test
    elif test == "levene":
        test_fn = levene_test
    else:
        raise ValueError(f"unknown test: {test}")

    levels = sorted(df[group_col].dropna().unique())
    samples = {lvl: df.loc[df[group_col] == lvl, value_col].dropna().to_numpy(dtype=float)
               for lvl in levels}

    rows = []
    for a, b in combinations(levels, 2):
        stat, p = test_fn(samples[a], samples[b])
        rows.append({"group_a": a, "group_b": b, "n_a": len(samples[a]), "n_b": len(samples[b]),
                     "statistic": stat, "p_value": p})

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_holm"] = holm_correction(result["p_value"].tolist())
        result["significant_holm_05"] = result["p_holm"] < 0.05
    result.attrs["test"] = test
    return result


def _ols_rss(y: np.ndarray, design: np.ndarray) -> float:
    """Residual sum of squares of y regressed on `design` (incl. intercept col)."""
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return float(resid @ resid)


def _dummy_block(series: pd.Series) -> np.ndarray:
    """One-hot columns for a categorical factor, one level dropped (reference
    level) so the block is full rank alongside an intercept column."""
    dummies = pd.get_dummies(series.astype(str), drop_first=True)
    return dummies.to_numpy(dtype=float)


def factorial_anova(
    df: pd.DataFrame,
    factors: list[str],
    value_col: str,
) -> pd.DataFrame:
    """Main-effects factorial ANOVA (Type II sums of squares) over the given
    categorical factors, reporting eta-squared (SS_factor / SS_total) per
    factor plus the residual.

    No interaction terms: the Rev1 ask is eta-squared *per factor*, and Type
    II SS for a main-effects-only model is exactly "RSS of the model without
    this factor, minus RSS of the full model" — valid for unbalanced designs
    too, so a missing/failed run in the sweep doesn't break it. Implemented
    directly on dummy-coded design matrices (numpy least squares) rather than
    statsmodels, which needs a C/C++ toolchain this box doesn't have.
    """
    clean = df[factors + [value_col]].dropna().copy()
    n = len(clean)
    y = clean[value_col].to_numpy(dtype=float)
    grand_mean = y.mean()
    ss_total = float(((y - grand_mean) ** 2).sum())

    intercept = np.ones((n, 1))
    blocks = {f: _dummy_block(clean[f]) for f in factors}
    full_design = np.hstack([intercept] + [blocks[f] for f in factors])
    rss_full = _ols_rss(y, full_design)
    df_full = n - full_design.shape[1]

    rows = []
    for f in factors:
        reduced_design = np.hstack([intercept] + [blocks[g] for g in factors if g != f])
        rss_reduced = _ols_rss(y, reduced_design)
        ss_f = max(rss_reduced - rss_full, 0.0)
        df_f = blocks[f].shape[1]
        ms_f = ss_f / df_f if df_f else float("nan")
        ms_resid = rss_full / df_full if df_full > 0 else float("nan")
        f_stat = ms_f / ms_resid if ms_resid else float("nan")
        p_value = float(stats.f.sf(f_stat, df_f, df_full)) if df_full > 0 else float("nan")
        rows.append({
            "term": f, "sum_sq": ss_f, "df": df_f, "F": f_stat, "p_value": p_value,
            "eta_sq": ss_f / ss_total if ss_total else float("nan"),
        })

    rows.append({
        "term": "Residual", "sum_sq": rss_full, "df": df_full, "F": float("nan"),
        "p_value": float("nan"), "eta_sq": rss_full / ss_total if ss_total else float("nan"),
    })

    return pd.DataFrame(rows)
