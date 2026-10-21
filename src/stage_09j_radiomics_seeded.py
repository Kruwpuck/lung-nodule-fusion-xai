"""Stage 09j: refit radiomics_only once, with the seed pinned, and write the number down.

Table I lists `radiomics_only` seven times, once per backbone, spanning 0.9301 to 0.9343.
That arm never touches a CNN, so the seven values cannot differ because of the backbone --
they differ because each row was fitted under a different seed. Published as seven rows it
reads as seven measurements of seven things; it is one measurement plus seed noise.

This stage refits that arm exactly once at `cfg["seed"]`, through the same
`_select_fold_features` -> `train_early_fusion_xgboost` path `stage_03b_fusion` uses for it,
so Table I can quote a single value with a file behind it. The published spread stays
quotable as a noise estimate, which is what it actually is.

The output is its own CSV rather than a row appended to `size_baseline.csv`, so nothing
already written is reopened for writing.

Run:
    .venv/Scripts/python.exe -m src.stage_09j_radiomics_seeded
    .venv/Scripts/python.exe -m src.stage_09j_radiomics_seeded --self-check

Remember `git add -f`: artifacts/ is gitignored in full, so the output will not appear in a
plain `git status`.
"""
from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from src.fusion.early_fusion import train_early_fusion_xgboost
from src.stage_03b_fusion import _load_merged, _select_fold_features
from src.utils.io import cached

RUN_ID = "2026-10-16-radiomics-seeded"
CONFIG_PATH = "configs/config.yaml"
OUT_DIR = "artifacts/results/size_baseline"
OUT_CSV = os.path.join(OUT_DIR, "radiomics_only_seeded.csv")
# Per-case probabilities. Without these a collaborator can read the AUC but cannot redo
# Holm, a bootstrap, or a paired test against a fusion arm -- which is the whole point of
# pinning the comparator in the first place.
OUT_PROBS = os.path.join(OUT_DIR, "radiomics_only_seeded_probs.csv")

# The seven per-backbone radiomics_only means in artifacts/results/fusion/ablation_summary.csv
# (35 rows, 7 backbones x 5 folds). Carried here only so the single seeded value can be
# reported next to the spread it is meant to replace.
PUBLISHED = [0.9301, 0.9318, 0.9321, 0.9322, 0.9322, 0.9340, 0.9343]


def _commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _refit(merged: pd.DataFrame, feat_cols: list[str], seed: int) -> pd.DataFrame:
    rows = []
    for fold in sorted(merged["fold"].unique()):
        train_df, val_df = merged[merged["fold"] != fold], merged[merged["fold"] == fold]
        X_tr, X_val, selected, fs_method = _select_fold_features(train_df, val_df, feat_cols)
        clf = train_early_fusion_xgboost(X_tr, train_df["label"].values, seed=seed)
        prob = clf.predict_proba(X_val)[:, 1]
        rows.append({
            "fold": int(fold),
            "auc": float(roc_auc_score(val_df["label"].values, prob)),
            "n_selected": len(selected),
            "fs_method": fs_method,
            "y_true": val_df["label"].values,
            "y_prob": prob,
            "patient_id": val_df["patient_id"].values.astype(str),
            "nodule_idx": val_df["nodule_idx"].values,
        })
    return pd.DataFrame(rows)


def run(cfg: dict) -> pd.DataFrame | None:
    if cached(OUT_CSV) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {OUT_CSV} sudah ada")
        return None

    merged, feat_cols = _load_merged(cfg)
    seed = cfg.get("seed", 42)
    per_fold = _refit(merged, feat_cols, seed)

    pooled = roc_auc_score(
        np.concatenate(per_fold["y_true"].tolist()),
        np.concatenate(per_fold["y_prob"].tolist()),
    )
    aucs = per_fold["auc"].tolist()
    row = {
        "run_id": RUN_ID,
        "commit_sha": _commit_sha(),
        "input_size": cfg.get("tracks", {}).get("track1", {}).get("input_size"),
        "checkpoint_mtime": "none",
        "arm": "radiomics_only",
        "seed": seed,
        "n_features_available": len(feat_cols),
        "n_selected_min": int(per_fold["n_selected"].min()),
        "n_selected_max": int(per_fold["n_selected"].max()),
        "fs_method": "|".join(sorted(per_fold["fs_method"].unique())),
        "n_nodules": len(merged),
        "n_patients": merged["patient_id"].nunique(),
        "auc_mean_fold": float(np.mean(aucs)),
        "auc_sd_fold": float(np.std(aucs)),
        "auc_pooled": float(pooled),
        "published_spread_min": min(PUBLISHED),
        "published_spread_max": max(PUBLISHED),
        "published_spread_mean": float(np.mean(PUBLISHED)),
    }
    for i, auc in enumerate(aucs):
        row[f"auc_fold{i}"] = float(auc)

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(OUT_CSV, index=False)

    probs = pd.DataFrame({
        "run_id": RUN_ID,
        "commit_sha": row["commit_sha"],
        "patient_id": np.concatenate(per_fold["patient_id"].tolist()),
        "nodule_idx": np.concatenate(per_fold["nodule_idx"].tolist()),
        "fold": np.concatenate([np.full(len(p), f) for f, p in
                                zip(per_fold["fold"], per_fold["y_prob"])]),
        "label": np.concatenate(per_fold["y_true"].tolist()),
        "rad_prob": np.concatenate(per_fold["y_prob"].tolist()),
    })
    probs.to_csv(OUT_PROBS, index=False)

    # Re-derive both headline numbers from the file just written, not from the in-memory
    # arrays. A dump that cannot reproduce the AUC it claims to explain is worse than no
    # dump, because the collaborator would build a paired test on top of it.
    rt = pd.read_csv(OUT_PROBS)
    rt_pooled = roc_auc_score(rt["label"], rt["rad_prob"])
    rt_mean = float(np.mean([roc_auc_score(g["label"], g["rad_prob"])
                             for _, g in rt.groupby("fold")]))
    assert abs(rt_pooled - row["auc_pooled"]) < 1e-12, (
        f"probs CSV pooled {rt_pooled} != summary {row['auc_pooled']}")
    assert abs(rt_mean - row["auc_mean_fold"]) < 1e-12, (
        f"probs CSV mean-fold {rt_mean} != summary {row['auc_mean_fold']}")
    assert len(rt) == row["n_nodules"], f"probs CSV {len(rt)} baris != {row['n_nodules']} nodul"

    print(f"seeded radiomics_only : mean-fold {row['auc_mean_fold']:.4f} "
          f"+/- {row['auc_sd_fold']:.4f} | pooled {row['auc_pooled']:.4f}")
    print(f"per fold              : {' '.join(f'{a:.4f}' for a in aucs)}")
    print(f"selected per fold     : {row['n_selected_min']}..{row['n_selected_max']} "
          f"of {row['n_features_available']} ({row['fs_method']})")
    print(f"published spread      : {min(PUBLISHED):.4f}..{max(PUBLISHED):.4f} "
          f"(mean {np.mean(PUBLISHED):.4f}, range {max(PUBLISHED) - min(PUBLISHED):.4f})")
    print(f"re-read dari probs CSV : pooled {rt_pooled:.5f} | mean-fold {rt_mean:.5f} "
          f"({len(rt)} baris) -- cocok")
    print(f"\n[DITULIS] {OUT_CSV}  (run_id={RUN_ID}, commit={row['commit_sha']})")
    print(f"[DITULIS] {OUT_PROBS}  ({len(probs)} baris)")
    print(f"  kohort: {row['n_nodules']} nodul, {row['n_patients']} pasien")
    print(f"  jangan lupa: git add -f {OUT_CSV}")
    return df


def self_check() -> None:
    """Assert the pooled AUC really pools, instead of averaging the folds a second time.

    The two differ whenever the folds rank differently, so a pooling bug would otherwise
    surface as a plausible number rather than as a failure.
    """
    y = [np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])]
    # Fold 0 ranks perfectly, fold 1 is inverted. Mean-fold is 0.5; pooled is not.
    p = [np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.6, 0.7, 0.3, 0.4])]
    mean_fold = float(np.mean([roc_auc_score(yi, pi) for yi, pi in zip(y, p)]))
    pooled = roc_auc_score(np.concatenate(y), np.concatenate(p))
    assert abs(mean_fold - 0.5) < 1e-9, f"mean-fold harus 0.5, dapat {mean_fold}"
    assert abs(pooled - mean_fold) > 0.05, (
        f"pooled tidak boleh sama dengan mean-fold pada data ini: {pooled}"
    )
    print(f"[OK] self-check lolos (mean-fold {mean_fold:.4f} vs pooled {pooled:.4f})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--self-check", action="store_true", help="jalankan assert saja, tanpa menulis")
    args = p.parse_args()

    if args.self_check:
        self_check()
        return

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)


if __name__ == "__main__":
    main()
