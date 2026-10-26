"""Stage 12b: forest plot of late fusion minus radiomics-only under three checkpoint regimes.

Track 1 manuscript figure (post hoc, descriptive). No model is run: every value is read from the
existing patient-cluster bootstrap output of stage_12a (artifacts/results/track1rev/a1_cluster_bootstrap.csv).

For each of three backbones and each checkpoint regime (Selected, Unselected, Nested) it draws the
pooled AUC difference (late fusion minus the seed-fixed radiomics-only model) with its patient-level
bootstrap 90% interval (10,000 resamples), against the +/-0.02 AUC equivalence margin used in the TOST.

Inputs  (read only): artifacts/results/track1rev/a1_cluster_bootstrap.csv
Outputs (artifacts/ is gitignored; stage with `git add -f`):
    artifacts/results/track1rev/fig_t1_regimes_print.png   (300 dpi)
    artifacts/results/track1rev/fig_t1_regimes_print.pdf   (vector, TrueType fonts)
    artifacts/results/track1rev/fig_t1_regimes_source.csv  (plotted values, with provenance columns)

Run:  python src/stage_12b_fig_regimes.py
"""
from __future__ import annotations

import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

T1 = os.path.join("artifacts", "results", "track1rev")
SRC = os.path.join(T1, "a1_cluster_bootstrap.csv")
MARGIN = 0.02
BLUE, VERM, GRAY = "#0072B2", "#D55E00", "#555555"
NAMES = {"convnext_tiny": "ConvNeXt-Tiny", "densenet201": "DenseNet201", "densenet121": "DenseNet121"}
ORDER = ["convnext_tiny", "densenet201", "densenet121"]
REGIMES = [("Selected", BLUE, "o", -0.24), ("Unselected", GRAY, "s", 0.0), ("Nested", VERM, "D", 0.24)]


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_source(use_refswap: bool) -> pd.DataFrame:
    """Default: a1 (comparator = stored radiomics of the DenseNet121 run). --refswap: stage_12d output with the
    seed-fixed radiomics model as the single comparator (regime 'Final-epoch' is drawn under the key 'Unselected')."""
    if not use_refswap:
        return pd.read_csv(SRC)
    d = pd.read_csv(os.path.join(T1, "refswap_seeded.csv"))
    d["regime"] = d["regime"].replace({"Final-epoch": "Unselected"})
    d = d.rename(columns={"delta": "delta_pooled_auc"})
    d["cluster_within_pm002_ci90"] = (d["cluster_ci90_lo"] > -MARGIN) & (d["cluster_ci90_hi"] < MARGIN)
    return d


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--refswap", action="store_true", help="use artifacts/results/track1rev/refswap_seeded.csv")
    use_refswap = ap.parse_args().refswap
    d = _load_source(use_refswap)
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 2.5, "ytick.major.size": 0,
        "xtick.major.width": 0.5, "axes.grid": True, "grid.color": "#D0D0D0", "grid.linewidth": 0.4,
        "grid.linestyle": ":", "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    fig.subplots_adjust(left=0.235, right=0.985, top=0.975, bottom=0.27)
    ax.axvspan(-MARGIN, MARGIN, color="#EDEDED", zorder=0)
    ax.axvline(0, color="black", lw=0.7, zorder=1)
    for x in (-MARGIN, MARGIN):
        ax.axvline(x, color=GRAY, lw=0.8, ls="--", zorder=1)
    rows = []
    for yi, bb in enumerate(ORDER):
        for reg, col, mk, off in REGIMES:
            r = d[(d["regime"] == reg) & (d["backbone"] == bb)].iloc[0]
            y = yi + off
            ax.plot([r["cluster_ci90_lo"], r["cluster_ci90_hi"]], [y, y], color=col, lw=1.1, zorder=3,
                    solid_capstyle="butt")
            ax.scatter(r["delta_pooled_auc"], y, marker=mk, s=18, color=col, lw=0, zorder=4)
            rows.append(dict(regime=reg, backbone=bb, delta_pooled_auc=r["delta_pooled_auc"],
                             ci90_lo=r["cluster_ci90_lo"], ci90_hi=r["cluster_ci90_hi"],
                             n_patients=r["n_patients"], inside_margin=bool(r["cluster_within_pm002_ci90"])))
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([NAMES[b] for b in ORDER])
    ax.set_ylim(len(ORDER) - 0.45, -0.5)
    ax.set_xlim(-0.032, 0.032)
    ax.set_xticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03])
    ax.set_xticklabels(["-0.03", "-0.02", "-0.01", "0", "0.01", "0.02", "0.03"], fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"AUC(late fusion) $-$ AUC(radiomics-only)", labelpad=1.5)
    ax.text(0.0, len(ORDER) - 0.5, "equivalence margin", ha="center", va="bottom", fontsize=7, color=GRAY)
    ha = [plt.Line2D([], [], color=c, marker=m, ms=4, lw=1.1, label={"Unselected": "Final-epoch"}.get(r, r)) for r, c, m, _ in REGIMES]
    fig.legend(handles=ha, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, -0.01),
               columnspacing=1.2, handletextpad=0.4, handlelength=1.6, fontsize=7.5)

    fig.savefig(os.path.join(T1, "fig_t1_regimes_print.png"), dpi=300)
    fig.savefig(os.path.join(T1, "fig_t1_regimes_print.pdf"))
    src = pd.DataFrame(rows)
    src.insert(0, "run_id", "2026-10-18-fig-regimes-posthoc")
    src.insert(1, "commit_sha", _sha())
    src.insert(2, "input_size", "96")
    src.insert(3, "checkpoint_mtime", "not_recorded_in_source")
    src.insert(4, "note", "post_hoc_figure_from_refswap_seeded" if use_refswap else "post_hoc_figure_from_a1_cluster_bootstrap")
    src.to_csv(os.path.join(T1, "fig_t1_regimes_source.csv"), index=False)
    print("[DONE] rows:", len(src))
    print("git add -f artifacts/results/track1rev/fig_t1_regimes_print.* artifacts/results/track1rev/fig_t1_regimes_source.csv src/stage_12b_fig_regimes.py")


if __name__ == "__main__":
    main()
