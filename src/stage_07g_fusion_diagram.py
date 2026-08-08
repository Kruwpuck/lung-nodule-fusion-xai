"""Stage 07g: fusion architecture diagram (Fig 11).

Static box+arrow schematic of the 5 ablation arms (cnn_only, radiomics_only,
fusion_early, fusion_intermediate, fusion_late). No model / GPU / data needed --
purely draws the design from known FusionNet dimensions.

`fusion_late` digambar sebagai arm utama karena itulah klaim utama Track 1
(docs/laporan/LAPORAN_TRACK1_FUSION_XAI.md sec 6.3). Empat arm lain tetap
digambar sebagai pembanding yang diuji, bukan dihapus -- ablasinya memang lima
arm. Semua dimensi/nama layer diparameterkan di blok konstanta di bawah.

Output: artifacts/results/figures/fusion_architecture.png
"""
import argparse
import logging
import os

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- parameter desain (edit di sini kalau wiring final berubah) ---------------
BACKBONE_NAME = "DenseNet201"   # model utama Track 1 (sec 6.3.2)
BACKBONE_OUT = 1920      # src/models/backbones.py -- densenet201 feature dim
EMB_DIM = 256            # FusionNet.img_proj  -> src/models/fusion_net.py
RAD_DIM = 128            # FusionNet.rad_branch
FUSION_DIM = 128         # FusionNet.classifier hidden
DROPOUT = 0.3
N_SLICES = 3
PATCH_XY = 64            # patch native
INPUT_XY = 96            # resolusi seragam saat latih/evaluasi (sec 3.3)
N_CLASSES = 2
CONCAT_DIM = EMB_DIM + RAD_DIM  # 384
W_CNN = 0.5              # bobot fusion_late -> src/fusion/late_fusion.py

# warna cabang
C_CNN = "#4a7fb5"        # steelblue -- aliran CNN
C_RAD = "#4f9d69"        # seagreen  -- aliran radiomics
C_INPUT = "#e8e8e8"
C_ARM = "#fbf3d9"
C_ARM_PRIMARY = "#f6dd8a"  # arm utama (fusion_late)
# -----------------------------------------------------------------------------


def _box(ax, x, y, w, h, text, fc, ec="#333333", fs=8, weight="normal"):
    """Rounded box centered at (x, y). Returns (left, right, top, bottom) anchors."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=1.1, edgecolor=ec, facecolor=fc, mutation_scale=1,
    ))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, weight=weight)
    return {"l": (x - w / 2, y), "r": (x + w / 2, y),
            "t": (x, y + h / 2), "b": (x, y - h / 2), "c": (x, y)}


def _arrow(ax, p0, p1, color):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5,
                                shrinkA=2, shrinkB=2))


def run(cfg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(cfg["paths"]["results"], "figures")
    out_path = os.path.join(out_dir, "fusion_architecture.png")
    from src.utils.io import cached
    if cached(out_path) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {out_path}")
        return
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 96, "Arsitektur Fusion CNN + Radiomics (ablasi 5 arm)",
            ha="center", va="center", fontsize=13, weight="bold")
    ax.text(50, 91.5,
            "fusion_late = arm utama Track 1; empat arm lain = pembanding yang diuji",
            ha="center", va="center", fontsize=9, style="italic", color="#555555")

    # --- kolom input (kiri) ---
    ct = _box(ax, 12, 72, 20, 11,
              f"Input 2.5D\n{N_SLICES}x{PATCH_XY}x{PATCH_XY} -> {INPUT_XY}x{INPUT_XY}",
              C_INPUT, fs=8.5)
    rad = _box(ax, 12, 26, 20, 11,
               "Radiomics vector\nPyRadiomics -> MI-50 -> LASSO", C_INPUT, fs=7.5)

    # --- extractor (tengah) ---
    cnn = _box(ax, 38, 72, 22, 12,
               f"CNN backbone\n{BACKBONE_NAME}\n({BACKBONE_OUT}-d)", C_CNN, fs=8.5,
               ec=C_CNN)
    ax.text(38, 72, "", ha="center")  # anchor placeholder
    _arrow(ax, ct["r"], cnn["l"], C_CNN)

    # radiomics tetap sebagai vektor (tidak ada extractor terpisah)
    _arrow(ax, rad["r"], (rad["r"][0] + 6, rad["r"][1]), C_RAD)
    rad_hub = (rad["r"][0] + 6, rad["r"][1])
    cnn_hub = cnn["r"]

    # --- kolom arm (kanan) ---
    ax_x = 80
    ax_w = 34
    # arm utama digambar paling atas dan disorot; sisanya pembanding yang diuji
    arms = [
        (85, "fusion_late  [ARM UTAMA]",
         f"p = {W_CNN:g}*p_cnn + {1 - W_CNN:g}*p_radiomics\n"
         "(fusi tingkat keputusan, tanpa jaringan baru)", [C_CNN, C_RAD], True),
        (68, "cnn_only",
         f"CNN -> softmax ({N_CLASSES})", [C_CNN], False),
        (51, "radiomics_only",
         "radiomics -> XGBoost", [C_RAD], False),
        (34, "fusion_intermediate",
         f"img_proj->{EMB_DIM}  (+)  rad_branch->{RAD_DIM}\n"
         f"concat {CONCAT_DIM} -> dense {FUSION_DIM} -> {N_CLASSES}",
         [C_CNN, C_RAD], False),
        (17, "fusion_early",
         f"concat [CNN emb {EMB_DIM} || radiomics] -> XGBoost",
         [C_CNN, C_RAD], False),
    ]

    for y, name, desc, srcs, primary in arms:
        b = _box(ax, ax_x, y, ax_w, 12, f"{name}\n{desc}",
                 C_ARM_PRIMARY if primary else C_ARM,
                 ec="#8a6d1f" if primary else "#333333",
                 fs=8 if primary else 7.5, weight="bold")
        for c in srcs:
            hub = cnn_hub if c == C_CNN else rad_hub
            _arrow(ax, hub, b["l"], c)

    # legenda
    ax.plot([6, 11], [8, 8], color=C_CNN, lw=2.2)
    ax.text(12, 8, "aliran CNN", va="center", fontsize=8)
    ax.plot([30, 35], [8, 8], color=C_RAD, lw=2.2)
    ax.text(36, 8, "aliran radiomics", va="center", fontsize=8)
    ax.text(70, 8,
            f"dropout={DROPOUT} pada tiap proyeksi (FusionNet)",
            va="center", fontsize=7.5, color="#555555", style="italic")

    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[DONE] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
