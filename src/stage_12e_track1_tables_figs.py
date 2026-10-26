"""Stage 12e: operating-point table, confusion-matrix figure, and cohort-by-fold table for the Track 1 manuscript.

POST HOC, DESCRIPTIVE. No model is run. Every number is computed from existing fold-level outputs:
  - artifacts/results/fusion/ablation_summary.csv (or --ablation): per arm, backbone and fold: accuracy, sensitivity,
    specificity, precision, F1, Brier score, and tp/tn/fp/fn at the 0.5 threshold of src/evaluation/metrics.py
  - results/rad_seeded_percase.csv: patient, nodule, fold, label of the 1366-nodule cohort

Outputs (artifacts/ is gitignored; stage with `git add -f`):
    artifacts/results/track1rev/t1_metrics_by_arm.csv         mean and SD over 35 backbone-fold cells (ddof=1)
    artifacts/results/track1rev/t1_confusion_by_arm.csv       summed over folds; mean per backbone
    artifacts/results/track1rev/t1_cohort_by_fold.csv         nodules, patients, benign, malignant per fold
    artifacts/results/track1rev/fig_t1_confusion_print.{png,pdf}
    artifacts/results/track1rev/tab_t1_metrics.tex, tab_t1_cohort.tex   LaTeX bodies generated from the CSVs
Every CSV carries run_id, commit_sha, input_size, checkpoint_mtime.

Run:  python src/stage_12e_track1_tables_figs.py [--ablation PATH]
"""
from __future__ import annotations

import argparse
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

T1 = os.path.join("artifacts", "results", "track1rev")
ARMS = [("cnn_only", "CNN-only"), ("radiomics_only", "Radiomics-only"), ("fusion_early", "Early fusion"),
        ("fusion_intermediate", "Intermediate fusion"), ("fusion_late", "Late fusion")]
METRICS = [("accuracy", "Accuracy"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
           ("precision", "Precision"), ("f1", "F1"), ("brier_score", "Brier")]


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default=os.path.join("artifacts", "results", "fusion", "ablation_summary.csv"))
    args = ap.parse_args()
    a = pd.read_csv(args.ablation)
    if "condition" in a:
        a = a[a["condition"].fillna("plain").isin(["plain", ""])]
    a = a[a["arm"].isin([k for k, _ in ARMS])]
    assert a.groupby(["backbone", "arm"]).size().eq(5).all(), "expected 5 folds per backbone and arm"
    n_bb = a["backbone"].nunique()
    sha = _sha()
    prov = {"run_id": "2026-10-22-t1-tables-figs-posthoc", "commit_sha": sha, "input_size": 96,
            "checkpoint_mtime": "not_recorded_in_source", "note": f"post_hoc_descriptive; source={os.path.basename(args.ablation)}"}

    # ---- metrics
    rows = []
    for key, lab in ARMS:
        g = a[a["arm"] == key]
        row = {**prov, "arm": lab, "n_cells": len(g)}
        for m, _ in METRICS:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_sd"] = g[m].std(ddof=1)
        rows.append(row)
    met = pd.DataFrame(rows)
    met.to_csv(os.path.join(T1, "t1_metrics_by_arm.csv"), index=False)
    body = []
    for r in met.itertuples():
        cells = " & ".join(f"${getattr(r, m + '_mean'):.3f}\\pm{getattr(r, m + '_sd'):.3f}$" for m, _ in METRICS)
        body.append(f"{r.arm} & {cells} \\\\")
    open(os.path.join(T1, "tab_t1_metrics.tex"), "w", encoding="utf-8").write("\n".join(body) + "\n")

    # ---- confusion matrices (summed over folds, mean per backbone)
    crow = []
    mats = {}
    for key, lab in ARMS:
        g = a[a["arm"] == key]
        tp, tn, fp, fn = (g[c].sum() / n_bb for c in ("tp", "tn", "fp", "fn"))
        mats[lab] = np.array([[tn, fp], [fn, tp]])
        crow.append({**prov, "arm": lab, "n_backbones": n_bb, "tn_mean_per_backbone": tn, "fp_mean_per_backbone": fp,
                     "fn_mean_per_backbone": fn, "tp_mean_per_backbone": tp,
                     "n_nodules": int(g[["tp", "tn", "fp", "fn"]].sum().sum() / n_bb)})
    pd.DataFrame(crow).to_csv(os.path.join(T1, "t1_confusion_by_arm.csv"), index=False)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.75))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.82, bottom=0.2, wspace=0.12)
    vmax = max(m.max() for m in mats.values())
    for ax, (lab, m) in zip(axes, mats.items()):
        ax.imshow(m, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if m[i, j] > 0.55 * vmax else "black")
        ax.set_title(lab, fontsize=7.5, pad=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["benign", "malignant"], fontsize=7)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["benign", "malignant"] if ax is axes[0] else [], fontsize=7, rotation=90, va="center")
        ax.tick_params(length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
    axes[0].set_ylabel("True class", fontsize=7.5, labelpad=1)
    fig.text(0.54, 0.015, "Predicted class", ha="center", fontsize=7.5)
    fig.savefig(os.path.join(T1, "fig_t1_confusion_print.png"), dpi=300)
    fig.savefig(os.path.join(T1, "fig_t1_confusion_print.pdf"))

    # ---- cohort by fold
    p = pd.read_csv(os.path.join("results", "rad_seeded_percase.csv"))
    crows = []
    for f, g in p.groupby("fold"):
        crows.append({**prov, "fold": int(f) + 1, "nodules": len(g), "patients": g["patient_id"].nunique(),
                      "benign": int((g["label"] == 0).sum()), "malignant": int((g["label"] == 1).sum())})
    coh = pd.DataFrame(crows)
    tot = {**prov, "fold": "Total", "nodules": int(coh["nodules"].sum()), "patients": p["patient_id"].nunique(),
           "benign": int(coh["benign"].sum()), "malignant": int(coh["malignant"].sum())}
    pd.concat([coh, pd.DataFrame([tot])], ignore_index=True).to_csv(os.path.join(T1, "t1_cohort_by_fold.csv"), index=False)
    body = [f"{int(r.fold)} & {r.nodules} & {r.patients} & {r.benign} & {r.malignant} \\\\" for r in coh.itertuples()]
    body.append("\\midrule")
    body.append(f"Total & {tot['nodules']} & {tot['patients']} & {tot['benign']} & {tot['malignant']} \\\\")
    open(os.path.join(T1, "tab_t1_cohort.tex"), "w", encoding="utf-8").write("\n".join(body) + "\n")
    print(met[["arm"] + [f"{m}_mean" for m, _ in METRICS]].round(3).to_string(index=False))
    print(pd.DataFrame(crow)[["arm", "tn_mean_per_backbone", "fp_mean_per_backbone", "fn_mean_per_backbone", "tp_mean_per_backbone"]].round(1).to_string(index=False))
    print(pd.concat([coh, pd.DataFrame([tot])])[["fold", "nodules", "patients", "benign", "malignant"]].to_string(index=False))
    print("git add -f artifacts/results/track1rev/t1_*.csv artifacts/results/track1rev/tab_t1_*.tex artifacts/results/track1rev/fig_t1_confusion_print.* src/stage_12e_track1_tables_figs.py")


if __name__ == "__main__":
    main()
