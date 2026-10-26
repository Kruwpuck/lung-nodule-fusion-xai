"""Stage 09l: print-ready depth-sweep figure for the Track 2 manuscript.

Redraws pointing accuracy against explanation site for all twelve backbones from the existing
run04 sweep (track2rev/depth_sweep_12.csv). No model is run. The figure is sized for a full-width
IEEE two-column float (7.16 in) with text of at least 7 pt, uses marker shapes in addition to
colour so it survives grayscale printing, and draws the center-map level next to the random level.

Inputs  (read only): artifacts/results/track2rev/depth_sweep_12.csv
                     results/center_baseline.csv   (center Gaussian pointing accuracy, fold0_60)
Outputs (artifacts/ is gitignored; stage with `git add -f`):
    artifacts/results/track2rev/fig_t2_2_depth_sweep_print.png   (300 dpi)
    artifacts/results/track2rev/fig_t2_2_depth_sweep_print.pdf   (vector)

Run:  python -m src.stage_09l_fig_depth_sweep
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SWEEP = os.path.join("artifacts", "results", "track2rev", "depth_sweep_12.csv")
CENTER = os.path.join("results", "center_baseline.csv")
OUT_DIR = os.path.join("artifacts", "results", "track2rev")

ORDER = [
    ("mobilenetv3_small", "MobileNetV3-Small"), ("efficientnet_b0", "EfficientNet-B0"),
    ("resnet50", "ResNet50"), ("vgg16", "VGG16"),
    ("densenet121", "DenseNet121"), ("densenet201", "DenseNet201"),
    ("googlenet", "GoogLeNet"), ("inceptionv3", "InceptionV3"),
    ("xception", "Xception"), ("convnext_tiny", "ConvNeXt-Tiny"),
    ("inception_resnet_v2", "Inception-ResNet-v2"), ("vit_base", "ViT-Base"),
]
BLUE, VERM, ORANGE, GRAY = "#0072B2", "#D55E00", "#E69F00", "#555555"
RANDOM_LEVEL = 0.019


def main() -> None:
    d = pd.read_csv(SWEEP).sort_values(["backbone", "site_rank"])
    c = pd.read_csv(CENTER)
    center = float(c[(c["sample_set"] == "fold0_60") & (c["baseline"] == "center_gaussian")]["pointing_acc"].iloc[0])

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 7,
                         "font.family": "DejaVu Sans", "axes.linewidth": 0.6})
    fig, axes = plt.subplots(3, 4, figsize=(7.16, 5.0), sharey=True)
    for ax, (key, title) in zip(axes.ravel(), ORDER):
        g = d[d["backbone"] == key].reset_index(drop=True)
        x = list(range(len(g)))
        ax.axhline(center, color=GRAY, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
        ax.axhline(RANDOM_LEVEL, color="#999999", lw=0.7, ls="--", zorder=1)
        ax.plot(x, g["pointing_acc"], "-o", color=BLUE, lw=1.1, ms=3, zorder=2)
        b = g[g["is_band_pick"] == 1]
        k = g[g["is_canonical_pick"] == 1]
        ax.scatter(b.index, b["pointing_acc"], marker="s", s=44, facecolors="none",
                   edgecolors=ORANGE, linewidths=1.2, zorder=3)
        ax.scatter(k.index, k["pointing_acc"], marker="*", s=62, color=VERM, zorder=4)
        ax.set_title(title, pad=2)
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", ".25", ".5", ".75", "1"])
        if key == "vit_base":
            ticks = [0, 3, 7, 11]
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(i + 1) for i in ticks])
            ax.set_xlabel("Encoder block", labelpad=1)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(h)) for h in g["spatial_h"]])
            ax.set_xlabel("Feature-map size (px)", labelpad=1)
        ax.tick_params(length=2, width=0.5, pad=1.5)
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_xlim(-0.4, len(g) - 0.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("Pointing accuracy", labelpad=1)

    handles = [
        plt.Line2D([], [], color=BLUE, marker="o", ms=3, lw=1.1, label="Candidate site"),
        plt.Line2D([], [], color=VERM, marker="*", ms=8, lw=0, label="Canonical site"),
        plt.Line2D([], [], color=ORANGE, marker="s", ms=6, mfc="none", mew=1.2, lw=0, label="Band-rule site"),
        plt.Line2D([], [], color=GRAY, lw=0.9, ls=(0, (1, 1.5)), label=f"Center map ({center:.3f})"),
        plt.Line2D([], [], color="#999999", lw=0.8, ls="--", label=f"Random level ({RANDOM_LEVEL:.3f})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.005), columnspacing=1.2, handletextpad=0.4)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.965, bottom=0.135, wspace=0.08, hspace=0.62)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "fig_t2_2_depth_sweep_print.png"), dpi=300)
    fig.savefig(os.path.join(OUT_DIR, "fig_t2_2_depth_sweep_print.pdf"))
    print("[DONE] sites plotted:", len(d), "| center level:", round(center, 4))
    print("git add -f artifacts/results/track2rev/fig_t2_2_depth_sweep_print.*")


if __name__ == "__main__":
    main()
