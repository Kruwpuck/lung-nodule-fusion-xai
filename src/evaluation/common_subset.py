"""Rev1 task 2: common-subset collapsed-binary evaluation across label arms.

Scores every label arm (A binary, B ordinal 1-5, C 3-class, D 4-class with a
no-nodule class) on the identical set of binary-eligible nodules per fold, so
any AUC delta between arms is attributable to the training label scheme and
not to which nodules got scored. Every arm collapses to P(malignant) by
probability renormalization only -- never by argmax, since argmax throws
away the ranking AUC needs.

Arm D additionally restricts to the benign/malignant nodule subset BEFORE
renormalizing (`grade4_nodule_only_metrics`). That restriction is the
defense against the easy-negative AUC inflation a strong no-nodule/nodule
separator would otherwise buy for free, so it is not optional.

Self-check enforced inline: arm A collapsed (identity renormalization,
p / (p + (1-p)) == p) must equal arm A binary's raw probability.

Predictions are read from the existing `artifacts/results/preds/*.npz`
(written by stage_04_evaluate.py); this module only scores them, it does
not retrain or re-run inference.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    compute_metrics, derive_binary, grade4_nodule_only_metrics, ordinal_metrics,
)

ARMS = ("A_binary", "B_ordinal", "C_grade3", "D_grade4")


def _load(preds_dir: str, model: str, arm_suffix: str, fold: int) -> dict:
    suffix = f"_{arm_suffix}" if arm_suffix else ""
    path = os.path.join(preds_dir, f"{model}{suffix}_fold{fold}.npz")
    return np.load(path)


def score_arm_a(preds_dir: str, model: str, fold: int) -> tuple[dict, int, np.ndarray]:
    """Arm A binary: P(malignant) used directly, no collapsing needed."""
    d = _load(preds_dir, model, "", fold)
    y_true, y_prob = d["y_true"].astype(int), d["y_prob"].astype(float)
    return compute_metrics(y_true, y_prob), len(y_true), y_prob


def score_arm_b(preds_dir: str, model: str, fold: int) -> tuple[dict, int]:
    """Arm B ordinal 1-5: P(rating > 3), via derive_binary (median==3 dropped)."""
    d = _load(preds_dir, model, "ordinal", fold)
    y_true_rating, y_pred_rating = d["y_true"].astype(float), d["y_out"].astype(float)
    m = derive_binary(y_true_rating, y_pred_rating)
    n = int((y_true_rating != 3.0).sum())
    return m, n


def score_arm_c(preds_dir: str, model: str, fold: int) -> tuple[dict, int]:
    """Arm C 3-class [benign=0, indeterminate=1, malignant=2]: renormalize
    P(malignant) / (P(benign) + P(malignant)); indeterminate rows dropped to
    match arm A's binary-eligible subset."""
    d = _load(preds_dir, model, "grade3", fold)
    y_true, y_out = d["y_true"].astype(int), d["y_out"].astype(float)
    mask = y_true != 1
    y_true_bin = (y_true[mask] == 2).astype(int)
    p_benign, p_malignant = y_out[mask, 0], y_out[mask, 2]
    y_prob = p_malignant / (p_benign + p_malignant + 1e-10)
    return compute_metrics(y_true_bin, y_prob), int(mask.sum())


def score_arm_d(preds_dir: str, model: str, fold: int) -> tuple[dict, int]:
    """Arm D 4-class [no-nodule=0, benign=1, indeterminate=2, malignant=3]:
    restrict to the benign/malignant nodule subset first, then renormalize
    as arm C (grade4_nodule_only_metrics already implements exactly this)."""
    d = _load(preds_dir, model, "grade4", fold)
    y_true, y_out = d["y_true"].astype(int), d["y_out"].astype(float)
    m = grade4_nodule_only_metrics(y_true, y_out)
    return m, m["n_nodule_only"]


def collapse_arm_arrays(preds_dir: str, model: str, arm: str, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true_bin, y_prob) for one arm on its common-subset cases.

    Mirrors the exact masking/renormalization each `score_arm_*` above already
    performs, but returns the aligned arrays instead of aggregate metrics --
    needed for paired DeLong (task 4), which operates on per-case scores, not
    per-fold AUC summaries.
    """
    if arm == "A_binary":
        d = _load(preds_dir, model, "", fold)
        return d["y_true"].astype(int), d["y_prob"].astype(float)

    if arm == "B_ordinal":
        d = _load(preds_dir, model, "ordinal", fold)
        y_true_rating = d["y_true"].astype(float)
        y_pred_rating = d["y_out"].astype(float)
        mask = y_true_rating != 3.0
        y_true_bin = (y_true_rating[mask] > 3).astype(int)
        y_prob = np.clip((y_pred_rating[mask] - 1.0) / 4.0, 0.0, 1.0)
        return y_true_bin, y_prob

    if arm == "C_grade3":
        d = _load(preds_dir, model, "grade3", fold)
        y_true, y_out = d["y_true"].astype(int), d["y_out"].astype(float)
        mask = y_true != 1
        y_true_bin = (y_true[mask] == 2).astype(int)
        p_benign, p_malignant = y_out[mask, 0], y_out[mask, 2]
        y_prob = p_malignant / (p_benign + p_malignant + 1e-10)
        return y_true_bin, y_prob

    if arm == "D_grade4":
        d = _load(preds_dir, model, "grade4", fold)
        y_true, y_out = d["y_true"].astype(int), d["y_out"].astype(float)
        mask = np.isin(y_true, [1, 3])
        y_true_bin = (y_true[mask] == 3).astype(int)
        p_benign, p_malignant = y_out[mask, 1], y_out[mask, 3]
        y_prob = p_malignant / (p_benign + p_malignant + 1e-10)
        return y_true_bin, y_prob

    raise ValueError(f"unknown arm: {arm}")


def discover_models(preds_dir: str) -> list[str]:
    """Models with an arm-A (plain) prediction file, from the legacy naming."""
    models = set()
    for f in glob.glob(os.path.join(preds_dir, "*_fold0.npz")):
        b = os.path.basename(f)[: -len("_fold0.npz")]
        if b.endswith(("_ordinal", "_grade3", "_grade4")):
            continue
        models.add(b)
    return sorted(models)


def score_common_subset(preds_dir: str, models: list[str] | None = None,
                         n_folds: int = 5) -> pd.DataFrame:
    """Per-model, per-fold, per-arm collapsed-binary AUC on identical cases."""
    models = models or discover_models(preds_dir)
    rows = []
    for model in models:
        for fold in range(n_folds):
            m_a, n_a, y_prob_a = score_arm_a(preds_dir, model, fold)

            # self-check: arm A needs no collapsing, so the identity
            # renormalization must reproduce the raw probability exactly.
            y_prob_a_collapsed = y_prob_a / (y_prob_a + (1.0 - y_prob_a) + 1e-10)
            if not np.allclose(y_prob_a_collapsed, y_prob_a, atol=1e-6):
                raise AssertionError(
                    f"self-check failed: arm A collapsed != arm A binary "
                    f"({model} fold{fold})"
                )

            m_b, n_b = score_arm_b(preds_dir, model, fold)
            m_c, n_c = score_arm_c(preds_dir, model, fold)
            m_d, n_d = score_arm_d(preds_dir, model, fold)

            for arm, m, n in (("A_binary", m_a, n_a), ("B_ordinal", m_b, n_b),
                              ("C_grade3", m_c, n_c), ("D_grade4", m_d, n_d)):
                auc_key = "auc" if "auc" in m else "auc_nodule_only"
                rows.append({"model": model, "fold": fold, "arm": arm,
                             "n": n, "auc": m[auc_key]})
    return pd.DataFrame(rows)


def arm_auc_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rev1 task 3: per-arm mean +/- std of collapsed-binary AUC across every
    scored model/fold row of the task-2 common-subset table."""
    g = df.groupby("arm")["auc"].agg(auc_mean="mean", auc_std="std", n_runs="count")
    return g.reindex(list(ARMS)).reset_index()


def score_arm_b_ordinal_native(preds_dir: str, model: str, fold: int) -> dict:
    """Rev1 task 3: ordinal-native metrics for arm B -- QWK, MAE, one-off
    (adjacent) accuracy -- computed on the full 1-5 rating scale.

    Unlike `score_arm_b`, this does NOT restrict to the median!=3 binary
    subset: that restriction exists only to make arm B's AUC comparable to
    arm A's binary task, whereas QWK/MAE/one-off accuracy evaluate the
    ordinal task on its own terms, including the ambiguous rating==3 cases.
    """
    d = _load(preds_dir, model, "ordinal", fold)
    return ordinal_metrics(d["y_true"], d["y_out"])


def ordinal_native_summary(preds_dir: str, models: list[str] | None = None,
                            n_folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-model-per-fold ordinal-native metrics for arm B, plus their
    mean/std across all scored runs. Returns (per_run, summary)."""
    models = models or discover_models(preds_dir)
    rows = []
    for model in models:
        for fold in range(n_folds):
            m = score_arm_b_ordinal_native(preds_dir, model, fold)
            rows.append({"model": model, "fold": fold, **m})
    per_run = pd.DataFrame(rows)
    summary = per_run[["qwk", "mae", "acc_within_1"]].agg(["mean", "std"])
    return per_run, summary


def run_task3(preds_dir: str = "artifacts/results/preds",
              common_subset_csv: str = "artifacts/results/common_subset_auc.csv",
              out_arm_auc: str = "artifacts/results/arm_auc_summary.csv",
              out_ordinal: str = "artifacts/results/ordinal_native_metrics.csv") -> None:
    """Rev1 task 3: per-arm collapsed-binary AUC mean/std, plus QWK/MAE/one-off
    accuracy for the ordinal arm. Reads the task-2 CSV rather than rescoring."""
    df = pd.read_csv(common_subset_csv)
    arm_summary = arm_auc_summary(df)
    os.makedirs(os.path.dirname(out_arm_auc), exist_ok=True)
    arm_summary.to_csv(out_arm_auc, index=False)
    print(f"[DONE] {out_arm_auc}\n{arm_summary}")

    per_run, ord_summary = ordinal_native_summary(preds_dir)
    per_run.to_csv(out_ordinal, index=False)
    print(f"[DONE] {out_ordinal} ({len(per_run)} rows)\n{ord_summary}")


def run(preds_dir: str = "artifacts/results/preds",
        out_csv: str = "artifacts/results/common_subset_auc.csv") -> pd.DataFrame:
    df = score_common_subset(preds_dir)

    # every arm must have scored the identical number of nodules per
    # model/fold, or the "common subset" claim is void.
    counts = df.pivot_table(index=["model", "fold"], columns="arm", values="n")
    mismatched = counts[counts.nunique(axis=1) != 1]
    if not mismatched.empty:
        raise AssertionError(f"arm case counts diverge for:\n{mismatched}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[DONE] {out_csv} ({len(df)} rows: {df['model'].nunique()} models "
          f"x {df['fold'].nunique()} folds x {len(ARMS)} arms)")
    return df


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--preds-dir", default="artifacts/results/preds")
    p.add_argument("--out", default="artifacts/results/common_subset_auc.csv")
    p.add_argument("--task3", action="store_true",
                    help="also emit arm_auc_summary.csv and ordinal_native_metrics.csv")
    args = p.parse_args()
    run(args.preds_dir, args.out)
    if args.task3:
        run_task3(args.preds_dir, args.out)


if __name__ == "__main__":
    main()
