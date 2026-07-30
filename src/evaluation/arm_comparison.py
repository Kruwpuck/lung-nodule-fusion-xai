"""Rev1 task 4: statistical comparison of label arms on the common subset.

Two independent questions, both answered on the task-2 common subset (same
binary-eligible cases scored under every arm):

1. Paired DeLong, per model, per arm pair, on cases pooled across the 5 CV
   folds (folds are disjoint case sets, so pooling is the same trick
   `aggregate_fold_results` already uses for bootstrap CI). Reuses
   `delong_test` from `statistical_tests.py` -- not reimplemented here.
2. Friedman omnibus + Nemenyi post-hoc across arms, with each (model, fold)
   pair as one block and its collapsed-binary AUC as the unit -- exactly the
   rows already sitting in `common_subset_auc.csv`.

Paired DeLong is only valid because the arms share cases. Before every test
this module re-verifies (does not assume) that the two arms being compared
have identical sample counts for that model; a mismatch is reported and the
comparison skipped, never silently run on unequal N.
"""
from __future__ import annotations

import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from src.evaluation.common_subset import ARMS, collapse_arm_arrays, discover_models
from src.evaluation.statistical_tests import delong_test


def delong_by_model(preds_dir: str, models: list[str] | None = None,
                     n_folds: int = 5) -> tuple[pd.DataFrame, list[str]]:
    """Paired DeLong for every arm pair, per model, pooling folds.

    Returns (results_df, skipped_notes). A pair is skipped (and noted, not
    run) whenever the pooled sample counts for the two arms differ.
    """
    models = models or discover_models(preds_dir)
    rows = []
    skipped = []

    for model in models:
        pooled = {arm: ([], []) for arm in ARMS}
        for fold in range(n_folds):
            for arm in ARMS:
                y_true, y_prob = collapse_arm_arrays(preds_dir, model, arm, fold)
                pooled[arm][0].append(y_true)
                pooled[arm][1].append(y_prob)
        pooled = {arm: (np.concatenate(yt), np.concatenate(yp))
                  for arm, (yt, yp) in pooled.items()}

        for arm_a, arm_b in combinations(ARMS, 2):
            y_true_a, y_prob_a = pooled[arm_a]
            y_true_b, y_prob_b = pooled[arm_b]
            if len(y_true_a) != len(y_true_b):
                skipped.append(
                    f"{model}: {arm_a} (n={len(y_true_a)}) vs {arm_b} "
                    f"(n={len(y_true_b)}) -- sample counts differ, not run"
                )
                continue
            if not np.array_equal(y_true_a, y_true_b):
                skipped.append(
                    f"{model}: {arm_a} vs {arm_b} -- pooled labels don't "
                    f"align case-for-case, not run"
                )
                continue

            z, p, delta = delong_test(y_true_a, y_prob_a, y_prob_b)
            rows.append({
                "model": model, "arm_a": arm_a, "arm_b": arm_b,
                "n": len(y_true_a),
                "auc_a": _auc(y_true_a, y_prob_a), "auc_b": _auc(y_true_b, y_prob_b),
                "delta_auc": delta, "delong_z": z, "delong_p": p,
                "significant_p05": p < 0.05,
            })

    return pd.DataFrame(rows), skipped


def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_prob))


def friedman_omnibus(auc_csv_df: pd.DataFrame) -> tuple[float, float, pd.DataFrame]:
    """Friedman test across arms, block = (model, fold), unit = collapsed AUC.

    Returns (chi2_stat, p_value, ranks_df) where ranks_df has one row per arm
    with its mean rank (1 = best AUC in that block) across all blocks.
    """
    wide = auc_csv_df.pivot_table(index=["model", "fold"], columns="arm", values="auc")
    wide = wide[list(ARMS)].dropna()

    chi2, p = stats.friedmanchisquare(*[wide[arm].to_numpy() for arm in ARMS])

    # rank 1 = best (highest AUC) within each block
    ranks = wide.apply(lambda row: stats.rankdata(-row.to_numpy()), axis=1, result_type="expand")
    ranks.columns = ARMS
    mean_ranks = ranks.mean(axis=0)
    ranks_df = pd.DataFrame({"arm": ARMS, "mean_rank": [mean_ranks[a] for a in ARMS]})
    ranks_df.attrs["n_blocks"] = len(wide)
    return float(chi2), float(p), ranks_df


def nemenyi_posthoc(ranks_df: pd.DataFrame, n_blocks: int) -> pd.DataFrame:
    """Nemenyi pairwise post-hoc from Friedman mean ranks.

    Standard formula (Demsar 2006): under H0 each arm's mean rank has known
    variance k(k+1)/(12N), so the standardized rank difference for a pair is
    compared against the Studentized range distribution directly --
    `scipy.stats.studentized_range`, no extra dependency needed.
    """
    k = len(ranks_df)
    se = np.sqrt(k * (k + 1) / (12.0 * n_blocks))
    mean_rank = dict(zip(ranks_df["arm"], ranks_df["mean_rank"]))

    rows = []
    for arm_a, arm_b in combinations(ranks_df["arm"], 2):
        diff = abs(mean_rank[arm_a] - mean_rank[arm_b])
        q = diff / se
        p = float(stats.studentized_range.sf(q, k, np.inf))
        rows.append({"arm_a": arm_a, "arm_b": arm_b, "rank_diff": diff,
                      "q_stat": q, "p_value": p, "significant_p05": p < 0.05})
    return pd.DataFrame(rows)


def run(auc_csv: str = "artifacts/results/common_subset_auc.csv",
        preds_dir: str = "artifacts/results/preds",
        out_dir: str = "artifacts/results") -> dict:
    df = pd.read_csv(auc_csv)

    delong_df, skipped = delong_by_model(preds_dir)
    chi2, p_friedman, ranks_df = friedman_omnibus(df)
    nemenyi_df = nemenyi_posthoc(ranks_df, n_blocks=ranks_df.attrs["n_blocks"])

    os.makedirs(out_dir, exist_ok=True)
    delong_path = os.path.join(out_dir, "delong_arms_common_subset.csv")
    friedman_path = os.path.join(out_dir, "friedman_ranks_common_subset.csv")
    nemenyi_path = os.path.join(out_dir, "nemenyi_arms_common_subset.csv")

    delong_df.to_csv(delong_path, index=False)
    ranks_df.assign(friedman_chi2=chi2, friedman_p=p_friedman).to_csv(friedman_path, index=False)
    nemenyi_df.to_csv(nemenyi_path, index=False)

    print(f"[DONE] {delong_path} ({len(delong_df)} rows, {len(skipped)} skipped)")
    for note in skipped:
        print(f"[SKIPPED] {note}")
    print(f"[DONE] {friedman_path} (chi2={chi2:.3f}, p={p_friedman:.4g})")
    print(f"[DONE] {nemenyi_path}")
    return {"delong": delong_df, "friedman_chi2": chi2, "friedman_p": p_friedman,
            "ranks": ranks_df, "nemenyi": nemenyi_df, "skipped": skipped}


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--auc-csv", default="artifacts/results/common_subset_auc.csv")
    p.add_argument("--preds-dir", default="artifacts/results/preds")
    p.add_argument("--out-dir", default="artifacts/results")
    args = p.parse_args()
    run(args.auc_csv, args.preds_dir, args.out_dir)


if __name__ == "__main__":
    main()
