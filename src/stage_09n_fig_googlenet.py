"""Stage 09n: print-ready GoogLeNet diagnosis figure for the Track 2 manuscript.

Panel (a): pointing accuracy at each GoogLeNet explanation site (same weights, same 60 nodules).
Panel (b): four CAM methods at features.13. Drawn from existing run04 CSVs; no model is run.

Inputs (read only): artifacts/results/track2rev/googlenet_layer_sweep.csv
                    artifacts/results/track2rev/googlenet_cam_methods.csv
                    results/center_baseline.csv
Outputs (artifacts/ is gitignored -- stage with `git add -f`):
    artifacts/results/track2rev/fig_t2_1_googlenet_print.png (300 dpi)
    artifacts/results/track2rev/fig_t2_1_googlenet_print.pdf

Run:  python -m src.stage_09n_fig_googlenet
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DIR = os.path.join("artifacts", "results", "track2rev")
BLUE, VERM, GRAY = "#0072B2", "#D55E00", "#555555"
LABEL = {"features.3": "24", "features.12": "6", "features.13": "3\n(pool)",
         "features.15": "3", "features.16": "1\n(avg)", "features.17": "1\n(drop)"}
METHOD = [("layercam", "Layer-\nCAM"), ("hirescam", "HiRes-\nCAM"), ("gradcam", "Grad-\nCAM"),
          ("scorecam", "Score-\nCAM")]


def main() -> None:
    s = pd.read_csv(os.path.join(DIR, "googlenet_layer_sweep.csv"))
    s = s[s["backbone"] == "googlenet"].copy()
    s["order"] = s["layer_path"].str.replace("features.", "", regex=False).astype(int)
    s = s.sort_values("order").reset_index(drop=True)
    m = pd.read_csv(os.path.join(DIR, "googlenet_cam_methods.csv")).set_index("method")
    c = pd.read_csv(os.path.join("results", "center_baseline.csv"))
    center = float(c[(c["sample_set"] == "fold0_60") & (c["baseline"] == "center_gaussian")]["pointing_acc"].iloc[0])
    chance = float(s["chance_pointing"].iloc[0])

    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 7,
                         "font.family": "DejaVu Sans", "axes.linewidth": 0.6})
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 2.5), gridspec_kw={"width_ratios": [1.6, 1]}, sharey=True)
    x = list(range(len(s)))
    for ax in (a, b):
        ax.axhline(center, color=GRAY, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
        ax.axhline(chance, color="#999999", lw=0.7, ls="--", zorder=1)
        ax.set_ylim(-0.04, 1.04)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(length=2, width=0.5, pad=1.5)
    a.plot(x, s["pointing_acc"], "-o", color=BLUE, lw=1.1, ms=3, zorder=2)
    zero = s["frac_zero_cam"] >= 1.0
    a.scatter([i for i in x if zero[i]], s.loc[zero, "pointing_acc"], marker="x", s=40,
              color=VERM, linewidths=1.4, zorder=4)
    a.set_xticks(x)
    a.set_xticklabels([LABEL[p] for p in s["layer_path"]])
    a.set_xlabel("Site: map size (px)", labelpad=1)
    a.set_ylabel("Pointing accuracy", labelpad=1)
    a.set_title("(a) By site", pad=2)
    a.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    a.set_yticklabels(["0", ".25", ".5", ".75", "1"])
    xm = list(range(len(METHOD)))
    b.bar(xm, [float(m.loc[k, "pointing_acc"]) for k, _ in METHOD], color=BLUE, width=0.65, zorder=2)
    b.set_xticks(xm)
    b.set_xticklabels([n for _, n in METHOD])
    b.set_title("(b) At features.13", pad=2)
    b.tick_params(axis="x", labelsize=6.5)
    handles = [
        plt.Line2D([], [], color=VERM, marker="x", lw=0, ms=5, mew=1.4, label="all 60 maps zero"),
        plt.Line2D([], [], color=GRAY, lw=0.9, ls=(0, (1, 1.5)), label=f"center map ({center:.3f})"),
        plt.Line2D([], [], color="#999999", lw=0.8, ls="--", label=f"random level ({chance:.3f})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005),
               columnspacing=0.8, handletextpad=0.3, fontsize=6.5)
    fig.subplots_adjust(left=0.14, right=0.995, top=0.92, bottom=0.36, wspace=0.08)
    os.makedirs(DIR, exist_ok=True)
    fig.savefig(os.path.join(DIR, "fig_t2_1_googlenet_print.png"), dpi=300)
    fig.savefig(os.path.join(DIR, "fig_t2_1_googlenet_print.pdf"))
    print("[DONE]", s[["layer_path", "spatial_size", "pointing_acc", "frac_zero_cam"]].round(3).to_dict("records"))
    print("git add -f artifacts/results/track2rev/fig_t2_1_googlenet_print.*")


if __name__ == "__main__":
    main()
