"""Stage 12d: re-score Table II / Fig. 2 of the Track 1 manuscript against ONE fixed radiomics model.

POST HOC, AT REVIEWER REQUEST -- NOT PRE-REGISTERED. No CNN is trained or loaded. For the three backbones with
stored per-case probabilities (run02: Selected / Final-epoch, run03: Nested) late fusion is recomputed as the
mean of the stored CNN probability and a *replacement* radiomics probability, and compared with that same
radiomics model by DeLong, TOST (+/-0.02) and a patient-cluster bootstrap. Holm correction is reported for the
nine-test family of Table II and for the three Nested tests alone.

Why: the radiomics comparator behind the current Table II / Fig. 2 has pooled AUC 0.93362 (stored `rad` of the
DenseNet121 run), whereas the seed-fixed refit reported in Sec. IV-A has pooled AUC 0.93195. This stage swaps the
comparator without touching anything else.

Modes
    (default)             --rad stored : uses the stored `rad` column. Reproduces the current Table II and is the
                                         sanity check for this script.
    --rad-seeded PATH.csv replaces `rad` by the seed-fixed per-case radiomics probabilities. PATH needs columns
                          patient_id, nodule_idx, rad_prob (one row per nodule of the 1366-nodule cohort; fold and
                          label optional but checked when present).

Inputs  (read only): artifacts/results/run02/probs/{backbone}.npz, artifacts/results/run03/probs/{backbone}.npz
Outputs (artifacts/ is gitignored; stage with `git add -f`):
    artifacts/results/track1rev/refswap_{tag}.csv           (one row per regime x backbone, with provenance)
    artifacts/results/track1rev/refswap_{tag}_holm.csv      (Holm over 9 tests and over the 3 Nested tests)

Run:
    python -m src.stage_12d_table2_reference_swap
    python -m src.stage_12d_table2_reference_swap --rad-seeded results/rad_seeded_percase.csv
"""
from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.evaluation.statistical_tests import delong_test, tost_auc

RES = os.path.join("artifacts", "results")
OUT = os.path.join(RES, "track1rev")
BACKBONES = ["convnext_tiny", "densenet201", "densenet121"]
REGIMES = {
    "Selected": (os.path.join(RES, "run02", "probs"), "cnn_best"),
    "Final-epoch": (os.path.join(RES, "run02", "probs"), "cnn_last"),
    "Nested": (os.path.join(RES, "run03", "probs"), "cnn_best"),
}
MARGIN = 0.02
N_BOOT = 10_000
BOOT_SEED = 20261011


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def auc(y, p) -> float:
    y = np.asarray(y, dtype=int)
    r = rankdata(np.asarray(p, dtype=float))
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def holm(p: dict) -> dict:
    keys = sorted(p, key=lambda k: p[k])
    m, run, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * p[k]))
        out[k] = run
    return out


def load(folder: str, cnn_key: str, bb: str):
    z = np.load(os.path.join(folder, f"{bb}.npz"), allow_pickle=True)
    return {
        "y": z["y_true"].astype(int), "fold": z["fold"].astype(int), "pat": z["patient_id"].astype(str),
        "nod": z["nodule_idx"].astype(int), "cnn": z[cnn_key].astype(float), "rad": z["rad"].astype(float),
        "late": (z["late_best"] if cnn_key == "cnn_best" else z["late_last"]).astype(float),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rad-seeded", default=None)
    args = ap.parse_args()
    tag = "seeded" if args.rad_seeded else "stored"
    seeded = None
    if args.rad_seeded:
        seeded = pd.read_csv(args.rad_seeded)
        need = {"patient_id", "nodule_idx", "rad_prob"}
        if not need <= set(seeded.columns):
            raise SystemExit(f"{args.rad_seeded} needs columns {sorted(need)}")
        if seeded.duplicated(["patient_id", "nodule_idx"]).any():
            raise SystemExit("duplicate (patient_id, nodule_idx) keys in the seeded file")
    sha = _sha()
    rows, p_all = [], {}
    for regime, (folder, cnn_key) in REGIMES.items():
        for bb in BACKBONES:
            d = load(folder, cnn_key, bb)
            y, pat = d["y"], d["pat"]
            # stored late must equal the mean of the stored branches (sanity of the recomputation rule)
            gap = float(np.max(np.abs(d["late"] - 0.5 * (d["cnn"] + d["rad"]))))
            if seeded is not None:
                key = pd.DataFrame({"patient_id": pat, "nodule_idx": d["nod"]})
                m = key.merge(seeded[["patient_id", "nodule_idx", "rad_prob"]], on=["patient_id", "nodule_idx"], how="left")
                if m["rad_prob"].isna().any():
                    raise SystemExit(f"{int(m['rad_prob'].isna().sum())} nodules of {bb}/{regime} missing in the seeded file")
                if "label" in seeded.columns:
                    lab = key.merge(seeded[["patient_id", "nodule_idx", "label"]], on=["patient_id", "nodule_idx"], how="left")["label"].values
                    if not np.array_equal(lab.astype(int), y):
                        raise SystemExit(f"label mismatch for {bb}/{regime}")
                rad = m["rad_prob"].values.astype(float)
                late = 0.5 * (d["cnn"] + rad)
            else:
                rad, late = d["rad"], d["late"]
            _, p_dl, delta = delong_test(y, late, rad)
            t = tost_auc(y, late, rad, MARGIN)
            patients = np.unique(pat)
            idx_by = {p: np.flatnonzero(pat == p) for p in patients}
            rng = np.random.default_rng(BOOT_SEED)
            clus = []
            for _ in range(N_BOOT):
                pick = rng.choice(patients, size=len(patients), replace=True)
                idx = np.concatenate([idx_by[p] for p in pick])
                if 0 < y[idx].sum() < len(idx):
                    clus.append(auc(y[idx], late[idx]) - auc(y[idx], rad[idx]))
                _ = rng.integers(0, len(y), size=len(y))  # keep the stream aligned with stage_12a
            clus = np.asarray(clus)
            p_all[(regime, bb)] = float(p_dl)
            rows.append({
                "run_id": f"2026-10-refswap-{tag}", "commit_sha": sha, "input_size": 96,
                "checkpoint_mtime": "not_recorded_in_source", "note": "post_hoc_reviewer_request",
                "regime": regime, "backbone": bb, "radiomics_source": tag, "n_nodules": len(y), "n_patients": len(patients),
                "stored_late_minus_mean_branches_max": gap,
                "auc_late": t["auc_a"], "auc_rad": t["auc_b"], "delta": delta, "delong_p": float(p_dl),
                "tost_p": t["p_tost"], "equivalent": bool(t["equivalent"]),
                "nodule_ci90_lo": t["ci90_low"], "nodule_ci90_hi": t["ci90_high"],
                "cluster_ci90_lo": float(np.percentile(clus, 5)), "cluster_ci90_hi": float(np.percentile(clus, 95)),
                "cluster_ci95_lo": float(np.percentile(clus, 2.5)), "cluster_ci95_hi": float(np.percentile(clus, 97.5)),
            })
    h9 = holm(p_all)
    h3 = holm({k: v for k, v in p_all.items() if k[0] == "Nested"})
    df = pd.DataFrame(rows)
    df["holm_p_9tests"] = [h9[(r.regime, r.backbone)] for r in df.itertuples()]
    df["holm_p_3nested"] = [h3.get((r.regime, r.backbone), np.nan) for r in df.itertuples()]
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, f"refswap_{tag}.csv"), index=False)
    pd.DataFrame([{"regime": k[0], "backbone": k[1], "delong_p": p_all[k], "holm_9": h9[k], "holm_3_nested_only": h3.get(k, np.nan)}
                  for k in sorted(p_all, key=lambda k: p_all[k])]).to_csv(os.path.join(OUT, f"refswap_{tag}_holm.csv"), index=False)
    pd.set_option("display.width", 200)
    print(df[["regime", "backbone", "auc_late", "auc_rad", "delta", "delong_p", "equivalent", "cluster_ci90_lo", "cluster_ci90_hi", "holm_p_9tests", "holm_p_3nested"]].round(5).to_string(index=False))
    print("max |stored late - mean(cnn, rad)|:", float(df["stored_late_minus_mean_branches_max"].max()))
    print(f"git add -f artifacts/results/track1rev/refswap_{tag}*.csv src/stage_12d_table2_reference_swap.py")


if __name__ == "__main__":
    main()
