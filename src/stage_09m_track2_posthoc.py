"""Stage 09m: post hoc descriptive checks for the Track 2 manuscript.

POST HOC, AT REVIEWER REQUEST -- NOT PRE-REGISTERED. Re-analysis of existing per-site results; no model
is run and no new performance search is opened.

P1  Grid parity. Pointing accuracy on centered crops is high for a map whose peak lies at the crop
    center. A map with an odd number of cells has a middle cell centered on the crop center; a map with
    an even number does not. We tabulate mean pointing accuracy by feature-map size and parity over the
    candidate sites of the eleven CNN backbones (ViT-Base excluded: its grid is always 14 x 14).
P2  Holm correction of the nine best-versus-canonical McNemar tests of posthoc_mcnemar_bootstrap.csv.

Inputs (read only): artifacts/results/track2rev/depth_sweep_12.csv
                    artifacts/results/track2rev/posthoc_mcnemar_bootstrap.csv
Outputs (artifacts/ is gitignored -- stage with `git add -f`):
    artifacts/results/track2rev/grid_parity.csv
    artifacts/results/track2rev/mcnemar_holm.csv

Every row carries run_id, commit_sha, input_size, checkpoint_mtime (not recorded in the sources).

Run:  python -m src.stage_09m_track2_posthoc
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
import pandas as pd

RUN_ID = "2026-10-17-run08-posthoc"
DIR = os.path.join("artifacts", "results", "track2rev")
NOTE = "post_hoc_reviewer_request"


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _prov() -> dict:
    return {"run_id": RUN_ID, "commit_sha": SHA, "input_size": "64/96 (CNN backbones)",
            "checkpoint_mtime": "not_recorded_in_source", "note": NOTE}


def grid_parity() -> pd.DataFrame:
    d = pd.read_csv(os.path.join(DIR, "depth_sweep_12.csv"))
    d = d[(d["backbone"] != "vit_base") & (d["spatial_h"] > 1)].copy()
    d["parity"] = np.where(d["spatial_h"] % 2 == 1, "odd", "even")
    rows = []

    def add(label: str, g: pd.DataFrame, parity: str) -> None:
        rows.append({**_prov(), "group": label, "parity": parity, "n_sites": len(g),
                     "mean_pointing": float(g["pointing_acc"].mean()),
                     "min_pointing": float(g["pointing_acc"].min()),
                     "max_pointing": float(g["pointing_acc"].max())})

    for h in sorted(d["spatial_h"].unique()):
        if h <= 6:
            g = d[d["spatial_h"] == h]
            add(f"size_{int(h)}", g, "odd" if h % 2 else "even")
    small = d[d["spatial_h"] <= 6]
    for p in ["odd", "even"]:
        add("size_le_6", small[small["parity"] == p], p)
    for p in ["odd", "even"]:
        add("all_sizes_gt_1", d[d["parity"] == p], p)
    return pd.DataFrame(rows)


def mcnemar_holm() -> pd.DataFrame:
    m = pd.read_csv(os.path.join(DIR, "posthoc_mcnemar_bootstrap.csv"), dtype=str, keep_default_na=False)
    m["p"] = m["mcnemar_p"].astype(float)
    order = np.argsort(m["p"].to_numpy())
    k = len(m)
    adj = np.empty(k)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (k - rank) * m["p"].iloc[i])
        adj[i] = min(1.0, run)
    out = pd.DataFrame({**{key: v for key, v in _prov().items()},
                        "backbone": m["backbone"], "n_tests": k, "mcnemar_p": m["p"], "holm_p": adj,
                        "significant_holm_005": adj < 0.05})
    return out.sort_values("holm_p").reset_index(drop=True)


SHA = _sha()


def run() -> None:
    gp, mh = grid_parity(), mcnemar_holm()
    gp.to_csv(os.path.join(DIR, "grid_parity.csv"), index=False)
    mh.to_csv(os.path.join(DIR, "mcnemar_holm.csv"), index=False)
    pd.set_option("display.width", 200)
    print(gp[["group", "parity", "n_sites", "mean_pointing", "min_pointing", "max_pointing"]].round(3).to_string(index=False))
    print(mh[["backbone", "mcnemar_p", "holm_p", "significant_holm_005"]].to_string(index=False))
    print("git add -f artifacts/results/track2rev/grid_parity.csv artifacts/results/track2rev/mcnemar_holm.csv")


if __name__ == "__main__":
    run()
