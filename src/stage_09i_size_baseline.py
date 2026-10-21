"""Stage 09i: how much of the radiomics AUC is nodule size alone.

Track 1 reports `radiomics_only` at 0.9324 mean-fold AUC over 1130 PyRadiomics features
and `cnn_only` at 0.9018. Neither number says how much of that is texture and wavelet
information and how much is the fact that malignant nodules are larger. Baltatzis et al.
2021 put a diameter threshold alone near 85% accuracy on LIDC, so the question is not
hypothetical, and a reviewer who asks it and gets no answer will assume the worst.

This stage answers it on the same cohort, the same folds and the same metric as
`stage_03b_fusion`, by reusing that module's `_load_merged` rather than re-deriving the
merge. Two kinds of baseline are scored:

  raw     - a single shape feature used directly as the score. Nothing is fitted, so there
            is no train/test split to get wrong. Per-fold values are reported only so the
            numbers line up with the other arms, not because anything was learned.
  xgboost - the same feature(s) through the `stage_03b` recipe, which is what a reader
            will assume was done unless told otherwise.

Both are written. On a single monotone feature the raw score beats the fitted one -- 400
trees of depth 4 can only approximate a ruler -- and reporting just the higher of the two
would read as picking the convenient number.

No checkpoint is loaded and no model is trained from pixels, so this runs on CPU in
seconds. `checkpoint_mtime` is carried as "none" rather than dropped, so the column set
still matches every other result CSV.

Run:
    .venv/Scripts/python.exe -m src.stage_09i_size_baseline
    .venv/Scripts/python.exe -m src.stage_09i_size_baseline --self-check

Remember `git add -f`: artifacts/ is gitignored in full, so the output will not appear in
a plain `git status`.
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
from src.stage_03b_fusion import _load_merged
from src.utils.io import cached

RUN_ID = "2026-10-12-size-baseline"
CONFIG_PATH = "configs/config.yaml"
OUT_DIR = "artifacts/results/size_baseline"
OUT_CSV = os.path.join(OUT_DIR, "size_baseline.csv")

DIAMETER = "original_shape_Maximum3DDiameter"
VOLUME = "original_shape_MeshVolume"

# Feature sets, shallowest claim first. "shape_all" is every original_shape_* column and is
# resolved at run time because the exact set depends on configs/radiomics_params.yaml.
FEATURE_SETS: dict[str, list[str] | None] = {
    "diameter": [DIAMETER],
    "volume": [VOLUME],
    "diameter_volume": [DIAMETER, VOLUME],
    "shape_all": None,
}

COLUMNS = [
    "run_id", "commit_sha", "input_size", "checkpoint_mtime",
    "feature_set", "estimator", "n_features", "n_nodules", "n_patients",
    "auc_mean_fold", "auc_sd_fold", "auc_pooled",
    "auc_fold0", "auc_fold1", "auc_fold2", "auc_fold3", "auc_fold4",
]


def _commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _score_raw(merged: pd.DataFrame, col: str) -> tuple[list[float], float]:
    """Use the feature itself as the score. Nothing is fitted."""
    per_fold = [
        roc_auc_score(merged.loc[merged["fold"] == f, "label"], merged.loc[merged["fold"] == f, col])
        for f in sorted(merged["fold"].unique())
    ]
    return per_fold, roc_auc_score(merged["label"], merged[col])


def _score_xgboost(merged: pd.DataFrame, cols: list[str], seed: int) -> tuple[list[float], float]:
    """Same recipe as stage_03b's radiomics_only arm, restricted to `cols`."""
    per_fold, pooled_y, pooled_p = [], [], []
    for f in sorted(merged["fold"].unique()):
        train_df, val_df = merged[merged["fold"] != f], merged[merged["fold"] == f]
        clf = train_early_fusion_xgboost(
            train_df[cols].values, train_df["label"].values, seed=seed
        )
        prob = clf.predict_proba(val_df[cols].values)[:, 1]
        per_fold.append(roc_auc_score(val_df["label"].values, prob))
        pooled_y.append(val_df["label"].values)
        pooled_p.append(prob)
    return per_fold, roc_auc_score(np.concatenate(pooled_y), np.concatenate(pooled_p))


def _row(merged, sha, input_size, feature_set, estimator, cols, per_fold, pooled) -> dict:
    row = {
        "run_id": RUN_ID, "commit_sha": sha, "input_size": input_size,
        "checkpoint_mtime": "none",
        "feature_set": feature_set, "estimator": estimator, "n_features": len(cols),
        "n_nodules": len(merged), "n_patients": merged["patient_id"].nunique(),
        "auc_mean_fold": float(np.mean(per_fold)), "auc_sd_fold": float(np.std(per_fold)),
        "auc_pooled": float(pooled),
    }
    for i, auc in enumerate(per_fold):
        row[f"auc_fold{i}"] = float(auc)
    return row


def run(cfg: dict) -> pd.DataFrame | None:
    if cached(OUT_CSV) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {OUT_CSV} sudah ada")
        return None

    merged, feat_cols = _load_merged(cfg)
    shape_cols = [c for c in feat_cols if c.startswith("original_shape")]
    sha = _commit_sha()
    # Recorded for provenance parity with the other CSVs. Nothing here reads pixels, so the
    # value cannot affect a number in this file -- it only pins which config produced it.
    input_size = cfg.get("tracks", {}).get("track1", {}).get("input_size")
    seed = cfg.get("seed", 42)

    rows = []
    for name, cols in FEATURE_SETS.items():
        cols = shape_cols if cols is None else cols
        missing = [c for c in cols if c not in merged.columns]
        if missing:
            print(f"[LEWAT] {name}: kolom hilang {missing}")
            continue
        per_fold, pooled = _score_xgboost(merged, cols, seed)
        rows.append(_row(merged, sha, input_size, name, "xgboost", cols, per_fold, pooled))
        if len(cols) == 1:
            per_fold, pooled = _score_raw(merged, cols[0])
            rows.append(_row(merged, sha, input_size, name, "raw", cols, per_fold, pooled))

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(OUT_CSV, index=False)

    show = ["feature_set", "estimator", "n_features", "auc_mean_fold", "auc_sd_fold", "auc_pooled"]
    print(df[show].to_string(index=False))
    print(f"\n[DITULIS] {OUT_CSV}  ({len(df)} baris, run_id={RUN_ID}, commit={sha})")
    print(f"  kohort: {len(merged)} nodul, {merged['patient_id'].nunique()} pasien")
    print("  pembanding stage_03b: radiomics_only 0.9324 | cnn_only 0.9018 | fusion_late 0.9349")
    print(f"  jangan lupa: git add -f {OUT_CSV}")
    return df


def self_check() -> None:
    """Assert the scorer behaves, on synthetic data with a known answer.

    Catches the failures that would quietly produce a plausible wrong number: a scorer with
    its label polarity inverted, or one that reports signal where there is none.
    """
    rng = np.random.default_rng(0)
    n = 400
    label = rng.integers(0, 2, n)
    # Separable by construction: positives are larger. AUC must come out high, not ~0.5.
    size = label * 3.0 + rng.normal(0, 1.0, n)
    df = pd.DataFrame({
        "label": label, DIAMETER: size, "fold": np.tile(np.arange(5), n // 5),
        "patient_id": [f"p{i}" for i in range(n)],
    })

    per_fold, pooled = _score_raw(df, DIAMETER)
    assert pooled > 0.9, f"raw scorer gagal pada data terpisah: {pooled}"
    assert len(per_fold) == 5, f"harus 5 fold, dapat {len(per_fold)}"

    # Polarity: flipping the labels must flip the AUC about 0.5, not leave it alone.
    flipped = df.assign(label=1 - df["label"])
    _, pooled_flipped = _score_raw(flipped, DIAMETER)
    assert pooled_flipped < 0.1, f"polaritas label tidak terdeteksi: {pooled_flipped}"

    # Pure noise must land near chance, or the scorer is manufacturing signal.
    noise = df.assign(**{DIAMETER: rng.normal(0, 1.0, n)})
    _, pooled_noise = _score_raw(noise, DIAMETER)
    assert 0.35 < pooled_noise < 0.65, f"derau tidak mendarat di sekitar kebetulan: {pooled_noise}"

    print("[OK] self-check lolos")


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
