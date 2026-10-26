"""Stage 09r: patient-cluster bootstrap of best-minus-canonical pointing accuracy (Track 2, post hoc, descriptive).

POST HOC, AT REVIEWER REQUEST. No model is run. The 60 fold-0 localization nodules come from 46 patients (35 contribute one
nodule, 8 two, 3 three), so nodules are not independent; the exact McNemar test and the nodule-level bootstrap assume they are.
This stage resamples the 46 PATIENTS (10,000 times, percentile intervals) and recomputes the per-backbone difference between
the pointing accuracy at the best site and at the canonical site, using the stored per-nodule hits.

Inputs  (read only): artifacts/results/track2rev/depth_curve_summary.csv   (best_site, canonical_site per backbone)
                     artifacts/results/track2rev/depth_sweep_12_persample.csv   (per-nodule pointing_hit per module)
Output  (artifacts/ is gitignored; stage with `git add -f`):
    artifacts/results/track2rev/cluster_bootstrap_best_vs_canonical.csv   (with provenance columns)

Run:  python src/stage_09r_cluster_bootstrap_site.py
"""
from __future__ import annotations

import os
import subprocess

import numpy as np
import pandas as pd

T2 = os.path.join("artifacts", "results", "track2rev")
N_BOOT = 10_000
SEED = 20261022


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    summ = pd.read_csv(os.path.join(T2, "depth_curve_summary.csv"))
    ps = pd.read_csv(os.path.join(T2, "depth_sweep_12_persample.csv"))
    rows = []
    for r in summ.itertuples():
        g = ps[ps["backbone"] == r.backbone]
        hb = g[g["module_path"] == r.best_site].sort_values("sample_idx")
        hc = g[g["module_path"] == r.canonical_site].sort_values("sample_idx")
        if len(hb) != 60 or len(hc) != 60:
            rows.append({"backbone": r.backbone, "note": f"missing per-sample rows (best {len(hb)}, canonical {len(hc)})"})
            continue
        if not np.array_equal(hb["sample_idx"].values, hc["sample_idx"].values):
            raise SystemExit(f"sample order mismatch for {r.backbone}")
        pat = hb["patient_id"].astype(str).values
        d = hb["pointing_hit"].values.astype(float) - hc["pointing_hit"].values.astype(float)
        point = float(d.mean())
        patients = np.unique(pat)
        idx_by = {p: np.flatnonzero(pat == p) for p in patients}
        rng = np.random.default_rng(SEED)
        boots = np.empty(N_BOOT)
        for b in range(N_BOOT):
            pick = rng.choice(patients, size=len(patients), replace=True)
            boots[b] = d[np.concatenate([idx_by[p] for p in pick])].mean()
        rows.append({
            "run_id": "2026-10-22-cluster-bootstrap-posthoc", "commit_sha": _sha(), "input_size": "64/96/224 (see backbone)",
            "checkpoint_mtime": "not_recorded_in_source", "note": "post_hoc_reviewer_request",
            "backbone": r.backbone, "best_site": r.best_site, "canonical_site": r.canonical_site,
            "same_module": r.best_site == r.canonical_site, "n_nodules": len(d), "n_patients": len(patients),
            "delta_point": point, "cluster_ci95_lo": float(np.percentile(boots, 2.5)), "cluster_ci95_hi": float(np.percentile(boots, 97.5)),
            "cluster_ci90_lo": float(np.percentile(boots, 5)), "cluster_ci90_hi": float(np.percentile(boots, 95)),
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(T2, "cluster_bootstrap_best_vs_canonical.csv"), index=False)
    pd.set_option("display.width", 200)
    print(out[["backbone", "best_site", "canonical_site", "same_module", "n_patients", "delta_point", "cluster_ci95_lo", "cluster_ci95_hi"]].round(3).to_string(index=False))
    print("git add -f artifacts/results/track2rev/cluster_bootstrap_best_vs_canonical.csv src/stage_09r_cluster_bootstrap_site.py")


if __name__ == "__main__":
    main()
