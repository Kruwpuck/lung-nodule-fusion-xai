"""Stage 09k: split-half check on explanation-site selection, plus provenance repair
of posthoc_mcnemar_bootstrap.csv.

POST HOC, AT REVIEWER REQUEST -- NOT PRE-REGISTERED. Descriptive re-analysis of
existing per-sample results; no model is run and no new performance search is opened.

Why this stage exists. Table I of the Track 2 draft reports, for each backbone, the
best of 6-12 candidate sites scored on the same 60 nodules it was selected on, so the
best-site value and the best-minus-default gap are optimistically biased. This stage
repeats the selection on a random half of the 60 nodules and scores the selected site
and the default site on the other half, over many random splits.

Inputs (run04, unchanged): track2rev/depth_sweep_12_persample.csv, cam_12.csv,
depth_curve_summary.csv.

Outputs (artifacts/ is gitignored -- stage with `git add -f`):
    track2rev/split_half_site.csv              one row per backbone
    track2rev/posthoc_mcnemar_bootstrap.csv    same numeric columns; input_size and
                                               checkpoint_mtime filled from cam_12.csv

Run (needs only the CSVs above, no data/ or GPU):
    python -m src.stage_09k_split_half_site
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
import pandas as pd

RUN_ID = "2026-10-11-run06-posthoc"
N_SPLITS = 2000
SPLIT_SEED = 20261011
DIR = os.path.join("artifacts", "results", "track2rev")

# Model input size per backbone: crop is 64x64; seven backbones resize to 96 inside the
# model (configs/config.yaml tracks.track1.input_size), ViT-Base to 224 (_MIN_INPUT).
INPUT_SIZE = {
    "mobilenetv3_small": 64, "efficientnet_b0": 64, "resnet50": 64, "vgg16": 64,
    "densenet121": 96, "densenet201": 96, "googlenet": 96, "inceptionv3": 96,
    "xception": 96, "convnext_tiny": 96, "inception_resnet_v2": 96, "vit_base": 224,
}


def _commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def split_half(per: pd.DataFrame, default_site: dict[str, str]) -> pd.DataFrame:
    rng = np.random.default_rng(SPLIT_SEED)
    rows = []
    for bb, g in per.groupby("backbone", sort=False):
        hits = g.pivot(index="sample_idx", columns="module_path", values="pointing_hit")
        sites = list(hits.columns)
        h = hits.to_numpy(dtype=float)
        n = h.shape[0]
        d_col = sites.index(default_site[bb])
        sel_scores, def_scores, gaps, beats, same = [], [], [], [], []
        for _ in range(N_SPLITS):
            perm = rng.permutation(n)
            a, b = perm[: n // 2], perm[n // 2:]
            pick = int(np.argmax(h[a].mean(axis=0)))  # first max on ties, deterministic
            s_sel = h[b, pick].mean()
            s_def = h[b, d_col].mean()
            sel_scores.append(s_sel)
            def_scores.append(s_def)
            gaps.append(s_sel - s_def)
            beats.append(s_sel > s_def)
            same.append(pick == d_col)
        gaps = np.asarray(gaps)
        rows.append({
            "backbone": bb, "n": n, "n_sites": len(sites), "n_splits": N_SPLITS,
            "default_site": default_site[bb],
            "insample_best": float(h.mean(axis=0).max()),
            "insample_default": float(h[:, d_col].mean()),
            "heldout_selected_mean": float(np.mean(sel_scores)),
            "heldout_default_mean": float(np.mean(def_scores)),
            "heldout_gap_mean": float(gaps.mean()),
            "heldout_gap_p2_5": float(np.percentile(gaps, 2.5)),
            "heldout_gap_p97_5": float(np.percentile(gaps, 97.5)),
            "frac_selected_beats_default": float(np.mean(beats)),
            "frac_selected_is_default": float(np.mean(same)),
        })
    return pd.DataFrame(rows)


def run() -> None:
    sha = _commit_sha()
    per = pd.read_csv(os.path.join(DIR, "depth_sweep_12_persample.csv"))
    cam = pd.read_csv(os.path.join(DIR, "cam_12.csv"))
    default_site = dict(zip(cam["backbone"], cam["module_path"]))
    mtime = dict(zip(cam["backbone"], cam["checkpoint_mtime"]))

    out = split_half(per, default_site)
    out.insert(0, "checkpoint_mtime", out["backbone"].map(mtime))
    out.insert(0, "input_size", out["backbone"].map(INPUT_SIZE))
    out.insert(0, "commit_sha", sha)
    out.insert(0, "run_id", RUN_ID)
    out["note"] = "post_hoc_reviewer_request"
    out.to_csv(os.path.join(DIR, "split_half_site.csv"), index=False)

    mc_path = os.path.join(DIR, "posthoc_mcnemar_bootstrap.csv")
    # Read every column as text so the numeric columns are rewritten byte-identical.
    mc = pd.read_csv(mc_path, dtype=str, keep_default_na=False)
    mc["input_size"] = mc["backbone"].map(INPUT_SIZE).astype(str)
    mc["checkpoint_mtime"] = mc["backbone"].map(mtime)
    mc.to_csv(mc_path, index=False)

    cols = ["backbone", "insample_best", "insample_default", "heldout_selected_mean",
            "heldout_default_mean", "heldout_gap_mean", "heldout_gap_p2_5",
            "heldout_gap_p97_5", "frac_selected_beats_default"]
    print(out[cols].round(3).to_string(index=False))
    print("[DONE] remember: git add -f artifacts/results/track2rev/split_half_site.csv "
          "artifacts/results/track2rev/posthoc_mcnemar_bootstrap.csv")


if __name__ == "__main__":
    run()
