"""Stage 09q: formal-style GoogLeNet diagnosis figure for the Track 2 manuscript (replaces stage 09n output in the draft).

Same data as stage 09n (existing run04 CSVs; no model is run), redrawn in the style of the other manuscript
figures: serif text of at least 7.5 pt, TrueType fonts, panel tags inside the axes, two stacked panels so that the four
method names fit on one line, reference lines labelled in the legend with the same styles as the summary figure.

  (a) pointing accuracy at each GoogLeNet explanation site (same weights, same 60 nodules);
      crosses mark sites whose 60 maps are all zero.
  (b) four CAM methods at features.13.

Inputs  (read only): artifacts/results/track2rev/googlenet_layer_sweep.csv, googlenet_cam_methods.csv
                     results/center_baseline.csv
Outputs (artifacts/ is gitignored; stage with `git add -f`):
    artifacts/results/track2rev/fig_t2_1_googlenet_formal.png   (300 dpi)
    artifacts/results/track2rev/fig_t2_1_googlenet_formal.pdf   (vector, TrueType fonts)

Run:  python src/stage_09q_fig_googlenet_formal.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DIR = os.path.join("artifacts", "results", "track2rev")
BLUE, VERM, GRAY, LGRAY = "#0072B2", "#D55E00", "#555555", "#999999"
LABEL = {"features.3": "24", "features.12": "6", "features.13": "3 (pool)",
         "features.15": "3", "features.16": "1 (avg)", "features.17": "1 (drop)"}
METHOD = [("layercam", "Layer-CAM"), ("hirescam", "HiResCAM"), ("gradcam", "Grad-CAM"), ("scorecam", "Score-CAM")]
CENTER_LS = (0, (1, 1.5))


def main() -> None:
    s = pd.read_csv(os.path.join(DIR, "googlenet_layer_sweep.csv"))
    s = s[s["backbone"] == "googlenet"].copy()
    s["order"] = s["layer_path"].str.replace("features.", "", regex=False).astype(int)
    s = s.sort_values("order").reset_index(drop=True)
    m = pd.read_csv(os.path.join(DIR, "googlenet_cam_methods.csv")).set_index("method")
    c = pd.read_csv(os.path.join("results", "center_baseline.csv"))
    center = float(c[(c["sample_set"] == "fold0_60") & (c["baseline"] == "center_gaussian")]["pointing_acc"].iloc[0])
    chance = float(s["chance_pointing"].iloc[0])

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "axes.grid": True, "grid.color": "#D0D0D0", "grid.linewidth": 0.4,
        "grid.linestyle": ":", "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(3.5, 3.35))
    a = fig.add_axes([0.15, 0.575, 0.835, 0.405])
    b = fig.add_axes([0.15, 0.215, 0.835, 0.255])
    yt, yl = [0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1.00"]
    for ax in (a, b):
        ax.axhline(center, color=GRAY, lw=0.9, ls=CENTER_LS, zorder=1)
        ax.axhline(chance, color=LGRAY, lw=0.8, ls="--", zorder=1)
        ax.set_ylim(-0.05, 1.06)
        ax.set_yticks(yt)
        ax.set_yticklabels(yl)
        ax.set_ylabel("Pointing accuracy", labelpad=1.5)
        ax.grid(axis="x", visible=False)
    x = list(range(len(s)))
    a.plot(x, s["pointing_acc"], "-o", color=BLUE, lw=1.1, ms=3.2, zorder=2)
    zero = s["frac_zero_cam"] >= 1.0
    a.scatter([i for i in x if zero[i]], s.loc[zero, "pointing_acc"], marker="x", s=42, color=VERM, linewidths=1.5, zorder=4)
    a.set_xticks(x)
    a.set_xticklabels([LABEL[p] for p in s["layer_path"]])
    a.set_xlim(-0.4, len(s) - 0.6)
    a.set_xlabel("Explanation site: feature-map size (cells per side)", labelpad=1.5)
    a.text(0.015, 0.915, "(a)", transform=a.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")
    xm = list(range(len(METHOD)))
    b.bar(xm, [float(m.loc[k, "pointing_acc"]) for k, _ in METHOD], color=BLUE, width=0.55, zorder=2)
    b.set_xticks(xm)
    b.set_xticklabels([n for _, n in METHOD])
    b.set_xlim(-0.6, len(METHOD) - 0.4)
    b.set_xlabel(r"CAM method at \texttt{features.13}".replace(r"\texttt{features.13}", "features.13"), labelpad=1.5)
    b.text(0.015, 0.90, "(b)", transform=b.transAxes, ha="left", va="top", fontsize=8, fontweight="bold")
    handles = [
        plt.Line2D([], [], color=VERM, marker="x", lw=0, ms=5, mew=1.5, label="all 60 maps zero"),
        plt.Line2D([], [], color=GRAY, lw=0.9, ls=CENTER_LS, label=f"center map ({center:.3f})"),
        plt.Line2D([], [], color=LGRAY, lw=0.9, ls="--", label=f"random level ({chance:.3f})"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0),
               columnspacing=0.9, handletextpad=0.3, fontsize=7.2)
    os.makedirs(DIR, exist_ok=True)
    fig.savefig(os.path.join(DIR, "fig_t2_1_googlenet_formal.png"), dpi=300)
    fig.savefig(os.path.join(DIR, "fig_t2_1_googlenet_formal.pdf"))
    print("[DONE]", s[["layer_path", "spatial_size", "pointing_acc", "frac_zero_cam"]].round(3).to_dict("records"))
    print("git add -f artifacts/results/track2rev/fig_t2_1_googlenet_formal.* src/stage_09q_fig_googlenet_formal.py")


if __name__ == "__main__":
    main()
