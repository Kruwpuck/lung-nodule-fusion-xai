"""Stage 12c: AUC of every arm at every backbone, with the size-only references (Track 1 manuscript figure).

Post hoc, descriptive; no model is run. Reads the fold-level AUCs of the main ablation and draws, for the
five arms (CNN-only, early, intermediate, late fusion, radiomics-only), one dot per backbone (mean of the
five fold AUCs) and the mean over the 35 backbone-fold cells (no SD bar: fold variation is shared by all arms), next to three horizontal references:
the seed-fixed radiomics-only model, mesh volume used directly as the score, and maximum 3-D diameter used
directly as the score.

Inputs  (read only): --ablation (default artifacts/results/fusion/ablation_summary.csv)
                     results/radiomics_only_seeded.csv, results/size_baseline.csv
Outputs (artifacts/ is gitignored; stage with `git add -f`):
    artifacts/results/track1rev/fig_t1_arms_print.png   (300 dpi)
    artifacts/results/track1rev/fig_t1_arms_print.pdf   (vector, TrueType fonts)
    artifacts/results/track1rev/fig_t1_arms_source.csv  (plotted values, with provenance columns)

After the P0-3 rerun lands, regenerate with:
    python src/stage_12c_fig_arms.py --ablation artifacts/results/rerun_seeded/fusion/ablation_summary.csv
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
BLUE, VERM, GRAY = "#0072B2", "#D55E00", "#555555"
ARMS = [("cnn_only", "CNN-only", GRAY), ("fusion_early", "Early", BLUE), ("fusion_intermediate", "Inter-\nmediate", BLUE),
        ("fusion_late", "Late", BLUE), ("radiomics_only", "Radiomics-\nonly", GRAY)]


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", default=os.path.join("artifacts", "results", "fusion", "ablation_summary.csv"))
    args = ap.parse_args()
    a = pd.read_csv(args.ablation)
    per_bb = a.groupby(["backbone", "arm"])["auc"].mean().unstack()
    seeded = float(pd.read_csv(os.path.join("results", "radiomics_only_seeded.csv"))["auc_mean_fold"].iloc[0])
    sz = pd.read_csv(os.path.join("results", "size_baseline.csv"))
    vol = float(sz[(sz["feature_set"] == "volume") & (sz["estimator"] == "raw")]["auc_mean_fold"].iloc[0])
    dia = float(sz[(sz["feature_set"] == "diameter") & (sz["estimator"] == "raw")]["auc_mean_fold"].iloc[0])

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.linewidth": 0.6, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.major.size": 0, "ytick.major.size": 2.5, "ytick.major.width": 0.5,
        "axes.grid": True, "grid.color": "#D0D0D0", "grid.linewidth": 0.4, "grid.linestyle": ":",
        "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    fig.subplots_adjust(left=0.17, right=0.985, top=0.975, bottom=0.36)
    for y, lab, ls, col in ((seeded, "radiomics, seeded", "-.", VERM), (vol, "volume alone", (0, (1, 1.5)), VERM),
                            (dia, "diameter alone", "--", VERM)):
        ax.axhline(y, color=col, lw=0.9, ls=ls, zorder=1)
    rng = np.random.default_rng(0)
    rows = []
    for i, (key, lab, col) in enumerate(ARMS):
        v = per_bb[key].dropna()
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v.values, s=11, color=col, alpha=0.55, lw=0, zorder=3)
        cells = a[a["arm"] == key]["auc"]
        ax.scatter(i, cells.mean(), marker="D", s=26, facecolors="white", edgecolors="black", lw=0.9, zorder=5)
        for bb, val in v.items():
            rows.append(dict(arm=key, backbone=bb, auc_mean_fold=val, arm_mean_35cells=cells.mean(),
                             arm_sd_35cells=cells.std()))
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([l for _, l, _ in ARMS])
    ax.set_xlim(-0.5, len(ARMS) - 0.5)
    ax.set_ylim(0.872, 0.946)
    ax.set_yticks([0.88, 0.90, 0.92, 0.94])
    ax.set_ylabel("AUC (mean of five folds)", labelpad=1.5)
    ax.grid(axis="x", visible=False)
    hs = [plt.Line2D([], [], marker="o", color=BLUE, alpha=0.7, lw=0, ms=3.5, label="backbone, fusion arm"),
          plt.Line2D([], [], marker="o", color=GRAY, alpha=0.7, lw=0, ms=3.5, label="backbone, single modality"),
          plt.Line2D([], [], marker="D", mfc="white", mec="black", color="black", lw=0, ms=4.5, label="mean of 35 cells"),
          plt.Line2D([], [], color=VERM, lw=0.9, ls="-.", label=f"radiomics, seeded ({seeded:.3f})"),
          plt.Line2D([], [], color=VERM, lw=0.9, ls=(0, (1, 1.5)), label=f"volume alone ({vol:.3f})"),
          plt.Line2D([], [], color=VERM, lw=0.9, ls="--", label=f"diameter alone ({dia:.3f})")]
    fig.legend(handles=hs, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.01),
               columnspacing=1.2, handletextpad=0.4, labelspacing=0.25, fontsize=7)
    fig.savefig(os.path.join(T1, "fig_t1_arms_print.png"), dpi=300)
    fig.savefig(os.path.join(T1, "fig_t1_arms_print.pdf"))
    src = pd.DataFrame(rows)
    src.insert(0, "run_id", "2026-10-21-fig-arms-posthoc")
    src.insert(1, "commit_sha", _sha())
    src.insert(2, "input_size", "96")
    src.insert(3, "checkpoint_mtime", "not_recorded_in_source")
    src.insert(4, "note", f"post_hoc_figure; source={os.path.basename(os.path.dirname(os.path.dirname(args.ablation)))}/ablation_summary.csv has no provenance columns")
    src.to_csv(os.path.join(T1, "fig_t1_arms_source.csv"), index=False)
    print("[DONE] rows:", len(src), "| radiomics seeded:", round(seeded, 4), "| volume:", round(vol, 4), "| diameter:", round(dia, 4))
    print("git add -f artifacts/results/track1rev/fig_t1_arms_print.* artifacts/results/track1rev/fig_t1_arms_source.csv src/stage_12c_fig_arms.py")


if __name__ == "__main__":
    main()
