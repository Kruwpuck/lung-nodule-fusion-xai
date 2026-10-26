"""Stage 09o: two-panel summary figure for the Track 2 manuscript (post hoc, descriptive).

Replaces the twelve-panel depth-sweep figure in the main text. No model is run; every number is
read from existing artifacts.

  (a) Canonical vs best-site pointing accuracy per backbone, with the paired difference and its
      bootstrap interval, against the center-map level.
  (b) Every CNN site with a spatial map (n > 1) by feature-map size and parity; the canonical
      site of each backbone is marked with the same star as in (a), which links the two panels:
      most canonical sites sit on the small even-sized grids where scores are lowest.

The protocol-shift result (original vs retrained, three backbones) is not drawn: it rests on three
backbones and one fold and is reported in the text.

Encoding: star = canonical site, filled circle = best site (a); triangle / diamond = odd / even
feature-map size (b); blue = candidate or best site, vermilion = canonical site, gray dotted line =
center map, gray dashed line = random level (same in both panels).

Inputs  (read only): artifacts/results/track2rev/depth_sweep_12.csv, depth_curve_summary.csv,
                     mcnemar_holm.csv, posthoc_mcnemar_bootstrap.csv, grid_parity.csv
                     results/center_baseline.csv
Outputs (artifacts/ is gitignored; stage with `git add -f`):
    artifacts/results/track2rev/fig_t2_summary_print.png   (300 dpi)
    artifacts/results/track2rev/fig_t2_summary_print.pdf   (vector, TrueType fonts)
    artifacts/results/track2rev/fig_t2_summary_source.csv  (plotted values, with provenance columns)

Run:  python src/stage_09o_fig_summary.py
"""
from __future__ import annotations

import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

T2 = os.path.join("artifacts", "results", "track2rev")
SWEEP = os.path.join(T2, "depth_sweep_12.csv")
SUMMARY = os.path.join(T2, "depth_curve_summary.csv")
HOLM = os.path.join(T2, "mcnemar_holm.csv")
BOOT = os.path.join(T2, "posthoc_mcnemar_bootstrap.csv")
PARITY = os.path.join(T2, "grid_parity.csv")
CENTER = os.path.join("results", "center_baseline.csv")

NAMES = {
    "mobilenetv3_small": "MobileNetV3-S", "efficientnet_b0": "EfficientNet-B0", "resnet50": "ResNet50",
    "vgg16": "VGG16", "densenet121": "DenseNet121", "densenet201": "DenseNet201",
    "googlenet": "GoogLeNet", "inceptionv3": "InceptionV3", "xception": "Xception",
    "convnext_tiny": "ConvNeXt-Tiny", "inception_resnet_v2": "Inc.-ResNet-v2", "vit_base": "ViT-Base",
}
BLUE, VERM, GRAY, LGRAY = "#0072B2", "#D55E00", "#555555", "#999999"
RANDOM_LEVEL = 0.019
YTICKS = [0, 0.25, 0.5, 0.75, 1.0]
YLAB = ["0", "0.25", "0.50", "0.75", "1.00"]
CENTER_LS = (0, (1, 1.5))


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    sweep = pd.read_csv(SWEEP)
    summ = pd.read_csv(SUMMARY).set_index("backbone")
    holm = pd.read_csv(HOLM).set_index("backbone")
    boot = pd.read_csv(BOOT).set_index("backbone")
    c = pd.read_csv(CENTER)
    center = float(c[(c["sample_set"] == "fold0_60") & (c["baseline"] == "center_gaussian")]["pointing_acc"].iloc[0])

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "axes.grid": True, "grid.color": "#D0D0D0", "grid.linewidth": 0.4, "grid.linestyle": ":",
        "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42})

    fig = plt.figure(figsize=(7.16, 2.7))
    B, H = 0.215, 0.765
    axa = fig.add_axes([0.125, B, 0.215, H])
    axd = fig.add_axes([0.345, B, 0.175, H])
    axb = fig.add_axes([0.605, B, 0.385, H])
    rows = []

    def tag(ax, text, x=0.03, y=0.97):
        ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")

    # ---------------------------------------------------------------- (a)
    s = summ.copy()
    s["gap"] = s["best_score"] - s["canonical_score"]
    s = s.sort_values("gap")
    y = np.arange(len(s))
    axa.axvline(center, color=GRAY, lw=0.9, ls=CENTER_LS, zorder=1)
    axa.axvline(RANDOM_LEVEL, color=LGRAY, lw=0.8, ls="--", zorder=1)
    for yi, (bb, r) in zip(y, s.iterrows()):
        axa.plot([r["canonical_score"], r["best_score"]], [yi, yi], color="#B8B8B8", lw=1.0, zorder=2)
        axa.scatter(r["best_score"], yi, marker="o", s=20, color=BLUE, lw=0, zorder=4)
        axa.scatter(r["canonical_score"], yi, marker="*", s=50, color=VERM, lw=0, zorder=5)
        sig = bool(holm["significant_holm_005"].get(bb, False))
        has = bb in boot.index and r["gap"] > 0
        bt = boot.loc[bb] if has else pd.Series({"diff": 0.0, "ci_lo": np.nan, "ci_hi": np.nan})
        txt = f"{bt['diff']:.2f} [{bt['ci_lo']:.2f}, {bt['ci_hi']:.2f}]" if has else "0"
        axd.text(0.05, yi, txt + ("*" if sig else ""), ha="left", va="center", fontsize=7.5,
                 transform=axd.get_yaxis_transform())
        rows.append(dict(panel="a", backbone=bb, canonical=r["canonical_score"], best=r["best_score"],
                         diff=bt["diff"], ci_lo=bt["ci_lo"], ci_hi=bt["ci_hi"], holm_significant=sig))
    axa.set_yticks(y)
    axa.set_yticklabels([NAMES[b] for b in s.index])
    axa.set_xlim(-0.04, 1.05)
    axa.set_xticks(YTICKS)
    axa.set_xticklabels(YLAB)
    axa.set_ylim(-0.6, len(s) + 0.9)
    axa.set_xlabel("Pointing accuracy", labelpad=1.5)
    axa.grid(axis="y", visible=False)
    axa.text(center - 0.03, 3.0, "center map", ha="right", va="center", rotation=90, fontsize=7.5)
    tag(axa, "(a)", x=0.075, y=0.995)
    axd.set_ylim(axa.get_ylim())
    axd.set_xlim(0, 1)
    axd.set_xticks([])
    axd.set_yticks([])
    axd.grid(False)
    for sp in axd.spines.values():
        sp.set_visible(False)
    axd.text(0.05, len(s) + 0.1, r"$\Delta$ [95% CI]", ha="left", va="center", fontsize=7.5,
             transform=axd.get_yaxis_transform())

    # ---------------------------------------------------------------- (b)
    cnn = sweep[(sweep["backbone"] != "vit_base") & (sweep["spatial_h"] > 1)].copy()
    cnn["odd"] = cnn["spatial_h"].astype(int) % 2 == 1
    rng = np.random.default_rng(0)
    cnn["xj"] = np.log2(cnn["spatial_h"].astype(float)) + rng.uniform(-0.035, 0.035, len(cnn))
    ev, od = ~cnn["odd"].values, cnn["odd"].values
    axb.axhline(center, color=GRAY, lw=0.9, ls=CENTER_LS, zorder=1)
    axb.axhline(RANDOM_LEVEL, color=LGRAY, lw=0.8, ls="--", zorder=1)
    axb.scatter(cnn["xj"][ev], cnn["pointing_acc"][ev], s=16, marker="D", facecolors="none", edgecolors=BLUE,
                lw=0.7, alpha=0.9, zorder=3)
    axb.scatter(cnn["xj"][od], cnn["pointing_acc"][od], s=20, marker="^", color=BLUE, lw=0, alpha=0.9, zorder=3)
    can = cnn[cnn["is_canonical_pick"] == 1]
    axb.scatter(can["xj"], can["pointing_acc"], s=62, marker="*", color=VERM, lw=0, zorder=6)
    gp = pd.read_csv(PARITY)
    gp = gp[gp["group"] == "size_le_6"].set_index("parity")
    for par, ls in (("odd", "-"), ("even", (0, (3, 1.5)))):
        m = float(gp.loc[par, "mean_pointing"])
        axb.plot([np.log2(1.7), np.log2(6.4)], [m, m], color=GRAY, lw=1.1, ls=ls, zorder=2)
        if par == "odd":
            axb.text(np.log2(6.8), m, f"mean {m:.2f}", color="black", fontsize=7.5, va="center", ha="left")
        else:
            axb.annotate(f"mean {m:.2f}", xy=(np.log2(2.1), m), xytext=(np.log2(1.7), 0.19), fontsize=7.5,
                         va="center", ha="left", arrowprops=dict(arrowstyle="-", lw=0.5, color="black",
                                                                  shrinkA=1, shrinkB=0))
    major = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
    axb.set_xticks([np.log2(t) for t in major])
    axb.set_xticklabels([str(t) for t in major], fontsize=7.5)
    axb.set_xticks([np.log2(t) for t in sorted(set(cnn["spatial_h"].astype(int)))], minor=True)
    axb.set_xlim(np.log2(1.6), np.log2(60))
    axb.set_ylim(-0.04, 1.04)
    axb.set_yticks(YTICKS)
    axb.set_yticklabels(YLAB)
    axb.set_xlabel("Feature-map size of the site (cells per side)", labelpad=1.5)
    axb.set_ylabel("Pointing accuracy", labelpad=1.5)
    axb.text(0.985, 0.925, "center map", transform=axb.transAxes, ha="right", va="top", fontsize=7.5)
    axb.legend(handles=[plt.Line2D([], [], marker="^", color=BLUE, lw=0, ms=4, label="odd size"),
                        plt.Line2D([], [], marker="D", mfc="none", mec=BLUE, color=BLUE, lw=0, ms=3.5,
                                   label="even size")],
               loc="upper right", frameon=False, handletextpad=0.1, borderpad=0.1, labelspacing=0.2,
               handlelength=0.8, bbox_to_anchor=(1.0, 0.84))
    tag(axb, "(b)", x=0.03, y=0.935)
    for _, r in cnn.iterrows():
        rows.append(dict(panel="b", backbone=r["backbone"], spatial_h=r["spatial_h"], pointing=r["pointing_acc"],
                         parity="odd" if r["odd"] else "even", canonical=bool(r["is_canonical_pick"])))

    ha = [plt.Line2D([], [], color=VERM, marker="*", ms=7, lw=0, label="canonical site"),
          plt.Line2D([], [], color=BLUE, marker="o", ms=4, lw=0, label="best site (a) / candidate site (b)"),
          plt.Line2D([], [], color=GRAY, lw=0.9, ls=CENTER_LS, label=f"center map ({center:.3f})"),
          plt.Line2D([], [], color=LGRAY, lw=0.9, ls="--", label=f"random level ({RANDOM_LEVEL:.3f})"),
          plt.Line2D([], [], color="none", marker="$*$", mfc="black", mec="black", ms=5,
                     label=r"Holm $p<0.005$")]
    fig.legend(handles=ha, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02),
               columnspacing=1.0, handletextpad=0.3, fontsize=7.5)

    fig.savefig(os.path.join(T2, "fig_t2_summary_print.png"), dpi=300)
    fig.savefig(os.path.join(T2, "fig_t2_summary_print.pdf"))
    src = pd.DataFrame(rows)
    src.insert(0, "run_id", "2026-10-18-fig-summary-posthoc")
    src.insert(1, "commit_sha", _sha())
    src.insert(2, "input_size", "64/96/224 (see backbone)")
    src.insert(3, "checkpoint_mtime", "not_recorded_in_source")
    src.insert(4, "note", "post_hoc_figure_from_existing_artifacts")
    src.to_csv(os.path.join(T2, "fig_t2_summary_source.csv"), index=False)
    print("[DONE] panel a:", len(s), "| panel b sites:", len(cnn), "| canonical stars:", len(can),
          "| center:", round(center, 4))
    print("git add -f artifacts/results/track2rev/fig_t2_summary_print.* artifacts/results/track2rev/fig_t2_summary_source.csv src/stage_09o_fig_summary.py")


if __name__ == "__main__":
    main()
