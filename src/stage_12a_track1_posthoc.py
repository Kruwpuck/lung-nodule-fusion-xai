"""Stage 12a: post hoc analyses for the Track 1 (fusion) manuscript.

POST HOC, AT REVIEWER REQUEST -- NOT PRE-REGISTERED. Descriptive re-analysis of
predictions and p-values that already exist on disk. No model is trained, no
checkpoint is loaded, and no new performance search is opened (run03
pre-registration section 5 stays closed).

Analyses
    A1  Patient-cluster bootstrap of the late-fusion minus radiomics-only AUC
        difference under the three checkpoint regimes (Selected, Unselected,
        Nested), next to an ordinary nodule-level bootstrap. Nodules from the same
        patient are not independent; resampling patients shows whether Table III
        of the manuscript survives that dependence.
    A2  Holm-adjusted p-values for the 21 fusion-versus-radiomics DeLong tests of
        the main ablation, so significance can be checked per cell.
    A3  Both AUC estimators -- mean of the five fold AUCs and the AUC of the
        predictions pooled over folds -- for CNN-only, radiomics-only and late
        fusion in each regime.
    A4  Case-level complementarity at a 0.5 threshold and at a Youden threshold
        chosen per branch on the other four folds (no threshold is fitted on the
        fold it is applied to). Only the three backbones with per-case
        predictions on disk are covered.

Inputs (read only)
    artifacts/results/run02/probs/{backbone}.npz   Selected (cnn_best, late_best)
                                                    and Unselected (cnn_last, late_last)
    artifacts/results/run03/probs/{backbone}.npz   Nested (cnn_best, late_best)
    artifacts/results/fusion/delong_fusion.csv      main-ablation DeLong p-values

Outputs (artifacts/ is gitignored -- stage with `git add -f`)
    artifacts/results/track1rev/a1_cluster_bootstrap.csv
    artifacts/results/track1rev/a2_holm.csv
    artifacts/results/track1rev/a3_estimators.csv
    artifacts/results/track1rev/a4_complementarity.csv

Every row carries run_id, commit_sha, input_size and checkpoint_mtime. The probs
files do not record checkpoint modification times, so that column is
"not_recorded_in_source" and the probs file's own mtime is given alongside.

Run (CSV/NPZ only, no data/ or GPU needed):
    python -m src.stage_12a_track1_posthoc
"""
from __future__ import annotations

import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import rankdata

RUN_ID = "2026-10-11-run07-posthoc"
RES = os.path.join("artifacts", "results")
OUT_DIR = os.path.join(RES, "track1rev")
BACKBONES = ["convnext_tiny", "densenet201", "densenet121"]
INPUT_SIZE = 96
N_BOOT = 10_000
BOOT_SEED = 20261011
NOTE = "post_hoc_reviewer_request"

# regime -> (probs file, cnn key, late key)
REGIMES = {
    "Selected": (os.path.join(RES, "run02", "probs"), "cnn_best", "late_best"),
    "Unselected": (os.path.join(RES, "run02", "probs"), "cnn_last", "late_last"),
    "Nested": (os.path.join(RES, "run03", "probs"), "cnn_best", "late_best"),
}


def _commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _mtime(path: str) -> str:
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    y = np.asarray(y, dtype=int)
    ranks = rankdata(np.asarray(p, dtype=float))  # average ranks for ties
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _load(regime: str, backbone: str):
    folder, cnn_key, late_key = REGIMES[regime]
    path = os.path.join(folder, f"{backbone}.npz")
    z = np.load(path, allow_pickle=True)
    return {
        "y": z["y_true"].astype(int), "fold": z["fold"].astype(int),
        "patient": z["patient_id"].astype(str),
        "cnn": z[cnn_key].astype(float), "rad": z["rad"].astype(float),
        "late": z[late_key].astype(float), "path": path,
    }


def _prov(path: str | None = None) -> dict:
    return {
        "run_id": RUN_ID, "commit_sha": SHA, "input_size": INPUT_SIZE,
        "checkpoint_mtime": "not_recorded_in_source",
        "source_file_mtime": _mtime(path) if path else "",
        "note": NOTE,
    }


def a1_cluster_bootstrap() -> pd.DataFrame:
    rows = []
    for regime in REGIMES:
        for bb in BACKBONES:
            d = _load(regime, bb)
            y, late, rad, pat = d["y"], d["late"], d["rad"], d["patient"]
            point = auc(y, late) - auc(y, rad)
            patients = np.unique(pat)
            idx_by_patient = {p: np.flatnonzero(pat == p) for p in patients}
            rng = np.random.default_rng(BOOT_SEED)
            clus, nod = [], []
            for _ in range(N_BOOT):
                pick = rng.choice(patients, size=len(patients), replace=True)
                idx = np.concatenate([idx_by_patient[p] for p in pick])
                if 0 < y[idx].sum() < len(idx):
                    clus.append(auc(y[idx], late[idx]) - auc(y[idx], rad[idx]))
                idx2 = rng.integers(0, len(y), size=len(y))
                if 0 < y[idx2].sum() < len(idx2):
                    nod.append(auc(y[idx2], late[idx2]) - auc(y[idx2], rad[idx2]))
            clus, nod = np.asarray(clus), np.asarray(nod)
            rows.append({
                **_prov(d["path"]), "regime": regime, "backbone": bb,
                "n_nodules": len(y), "n_patients": len(patients),
                "delta_pooled_auc": point,
                "cluster_ci95_lo": np.percentile(clus, 2.5), "cluster_ci95_hi": np.percentile(clus, 97.5),
                "cluster_ci90_lo": np.percentile(clus, 5), "cluster_ci90_hi": np.percentile(clus, 95),
                "cluster_se": clus.std(ddof=1),
                "cluster_frac_delta_below_0": float((clus < 0).mean()),
                "cluster_within_pm002_ci90": bool(np.percentile(clus, 5) > -0.02 and np.percentile(clus, 95) < 0.02),
                "nodule_ci95_lo": np.percentile(nod, 2.5), "nodule_ci95_hi": np.percentile(nod, 97.5),
                "nodule_se": nod.std(ddof=1),
                "se_ratio_cluster_over_nodule": clus.std(ddof=1) / nod.std(ddof=1),
                "n_boot_valid": len(clus),
            })
    return pd.DataFrame(rows)


def a2_holm() -> pd.DataFrame:
    path = os.path.join(RES, "fusion", "delong_fusion.csv")
    d = pd.read_csv(path)
    d = d[d["fusion_arm"].isin(["fusion_late", "fusion_early", "fusion_intermediate"])].reset_index(drop=True)
    p = d["delong_p"].to_numpy()
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    out = d[["backbone", "fusion_arm", "fusion_auc", "best_single_auc", "delong_p"]].copy()
    out["holm_p"] = adj
    out["significant_holm_005"] = adj < 0.05
    prov = _prov(path)
    prov["checkpoint_mtime"] = "not_recorded_in_source"
    for k, v in reversed(list(prov.items())):
        out.insert(0, k, v)
    return out.sort_values("holm_p").reset_index(drop=True)


def a3_estimators() -> pd.DataFrame:
    rows = []
    for regime in REGIMES:
        for bb in BACKBONES:
            d = _load(regime, bb)
            for arm in ["cnn", "rad", "late"]:
                fold_aucs = [auc(d["y"][d["fold"] == k], d[arm][d["fold"] == k]) for k in range(5)]
                rows.append({
                    **_prov(d["path"]), "regime": regime, "backbone": bb, "arm": arm,
                    "auc_mean_of_folds": float(np.mean(fold_aucs)),
                    "auc_sd_of_folds": float(np.std(fold_aucs, ddof=1)),
                    "auc_pooled": auc(d["y"], d[arm]),
                })
    df = pd.DataFrame(rows)
    df["pooled_minus_mean"] = df["auc_pooled"] - df["auc_mean_of_folds"]
    return df


def _youden(y: np.ndarray, p: np.ndarray) -> float:
    best_j, best_t = -1.0, 0.5
    for t in np.unique(p):
        pred = p >= t
        tpr = (pred & (y == 1)).sum() / max(1, (y == 1).sum())
        fpr = (pred & (y == 0)).sum() / max(1, (y == 0).sum())
        if tpr - fpr > best_j:
            best_j, best_t = tpr - fpr, float(t)
    return best_t


def a4_complementarity() -> pd.DataFrame:
    rows = []
    for bb in BACKBONES:
        d = _load("Selected", bb)
        y, f = d["y"], d["fold"]
        for rule in ["fixed_0.5", "youden_other_folds"]:
            per_fold = []
            for k in range(5):
                te = f == k
                if rule == "fixed_0.5":
                    tc = tr = 0.5
                else:
                    trn = ~te
                    tc = _youden(y[trn], d["cnn"][trn])
                    tr = _youden(y[trn], d["rad"][trn])
                cc = (d["cnn"][te] >= tc).astype(int) == y[te]
                rc = (d["rad"][te] >= tr).astype(int) == y[te]
                per_fold.append({
                    "n": int(te.sum()),
                    "cnn_only_correct": int((cc & ~rc).sum()),
                    "rad_only_correct": int((rc & ~cc).sum()),
                    "both_correct": int((cc & rc).sum()),
                    "both_wrong": int((~cc & ~rc).sum()),
                })
            pf = pd.DataFrame(per_fold)
            n = pf["n"].mean()
            rows.append({
                **_prov(d["path"]), "regime": "Selected", "backbone": bb, "threshold_rule": rule,
                "mean_n_per_fold": n,
                "cnn_only_correct_per_fold": pf["cnn_only_correct"].mean(),
                "rad_only_correct_per_fold": pf["rad_only_correct"].mean(),
                "both_wrong_per_fold": pf["both_wrong"].mean(),
                "agree_pct": 100.0 * (1 - (pf["cnn_only_correct"].mean() + pf["rad_only_correct"].mean()) / n),
                "rad_over_cnn_ratio": pf["rad_only_correct"].mean() / pf["cnn_only_correct"].mean(),
                "pearson_r": float(np.corrcoef(d["cnn"], d["rad"])[0, 1]),
            })
    return pd.DataFrame(rows)


SHA = _commit_sha()


def run() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    outputs = {
        "a2_holm.csv": a2_holm(),
        "a3_estimators.csv": a3_estimators(),
        "a4_complementarity.csv": a4_complementarity(),
        "a1_cluster_bootstrap.csv": a1_cluster_bootstrap(),
    }
    pd.set_option("display.width", 250)
    for name, df in outputs.items():
        df.to_csv(os.path.join(OUT_DIR, name), index=False)
        print(f"== {name}")
        print(df.drop(columns=["run_id", "commit_sha", "input_size", "checkpoint_mtime",
                               "source_file_mtime", "note"], errors="ignore").round(4).to_string(index=False))
    print("[DONE] artifacts/ is gitignored: git add -f artifacts/results/track1rev/*.csv")


if __name__ == "__main__":
    run()
