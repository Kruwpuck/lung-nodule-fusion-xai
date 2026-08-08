"""Fig 14 run02: satu figure yang menyandingkan penjelasan spasial dan penjelasan fitur.

Ini bukti visual untuk baris terakhir tabel keunggulan gabungan ("penjelasan spasial
DAN fitur sekaligus", lihat docs/laporan/LAPORAN_TRACK1_FUSION_XAI.md sec 6.3.1).
Panel atas menunjukkan Layer-CAM cabang citra `fusion_late` pada keenam nodul tetap
`artifacts/xai/fixed_display_samples.json`; panel bawah menunjukkan beeswarm SHAP
cabang radiomik dari model yang sama. Keduanya menjelaskan satu prediksi yang sama,
lewat dua jenis bukti yang tidak bisa saling menggantikan.

Satu figure, bukan tiga. Cabang radiomik `fusion_late` nol masukan dari backbone,
jadi beeswarm-nya identik untuk ketiga backbone (`identical_across_backbones=True` di
shap_provenance.csv). Menerbitkan tiga salinan gambar yang sama akan menyesatkan.
Backbone yang digambar adalah DenseNet201, model utama Track 1 (sec 6.3.2).

Peta CAM dihitung terhadap kelas keputusan `fusion_late` (argmax late_prob), sama
seperti stage_08b_run02_xai. Pada keenam nodul ini `n_disagree` nol, jadi peta ini
identik dengan peta `cnn_only` -- itu fakta arsitektural yang justru menjadi isi
klaim (fusi tidak merusak lokalisasi), dan dicetak di figure supaya tidak bisa
disalahbaca sebagai penjelasan baru yang dihasilkan fusi.

Keluaran: artifacts/results/run02/fig14_spatial_and_feature.png
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import yaml

from src.stage_08b_run02_xai import (
    CAM_METHOD,
    OUT_DIR,
    RUN_ID,
    _commit_sha,
    _fixed_samples,
    _load_patch_tensor,
    _prob_lookup,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKBONE = "densenet201"
SHAP_PNG = os.path.join(OUT_DIR, f"shap_beeswarm_{BACKBONE}.png")
OUT_PNG = os.path.join(OUT_DIR, "fig14_spatial_and_feature.png")


def _cam_panels(cfg: dict) -> tuple[list[dict], int]:
    """Satu entri per nodul tetap: patch, mask, cam, label, kelas keputusan."""
    import torch

    from src.models.registry import _NAME_MAP, build_model
    from src.xai.gradcam_utils import compute_gradcam

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)

    probs = _prob_lookup(BACKBONE)
    ckpt = os.path.join(cfg["paths"]["checkpoints"], BACKBONE, "fold0_best.pt")
    model = build_model(BACKBONE, cfg, task="binary").to(device)
    state = torch.load(ckpt, weights_only=True, map_location="cpu")
    model.load_state_dict(state["model_state"] if isinstance(state, dict)
                          and "model_state" in state else state)
    model.eval()
    internal = _NAME_MAP.get(BACKBONE, BACKBONE)

    panels, n_disagree = [], 0
    for _, row in _fixed_samples().iterrows():
        key = (str(row["patient_id"]), int(row["nodule_idx"]))
        if key not in probs:
            raise KeyError(
                f"{key} tidak ada di probs/{BACKBONE}.npz fold 0. Sampel tetap tidak "
                "boleh diganti; jalankan ulang stage_08a_run02_probs.")
        cnn_p, late_p = probs[key]
        cls_cnn, cls_late = int(cnn_p > 0.5), int(late_p > 0.5)
        n_disagree += int(cls_cnn != cls_late)

        img = _load_patch_tensor(row["patch_path"], n_slices, patch_xy).to(device)
        mask_full = np.load(row["mask_path"]).astype(np.float32)
        cam = compute_gradcam(model, img, backbone_name=internal,
                              target_class=cls_late, method=CAM_METHOD)
        panels.append({
            "slot": row["slot"],
            "patient_id": row["patient_id"],
            "patch": img[0, n_slices // 2].detach().cpu().numpy(),
            "mask": mask_full[mask_full.shape[0] // 2],
            "cam": cam,
            "label": int(row["label"]),
            "pred": cls_late,
            "prob": late_p,
        })
    return panels, n_disagree


def _trim_white(img: np.ndarray, tol: float = 0.99) -> np.ndarray:
    """Buang bingkai putih PNG SHAP supaya panel (b) tidak tenggelam jadi kecil."""
    gray = img[..., :3].mean(axis=2) if img.ndim == 3 else img
    ink = gray < tol
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def run(cfg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    if not os.path.exists(SHAP_PNG):
        raise FileNotFoundError(
            f"{SHAP_PNG} belum ada. Jalankan: python -m src.stage_08b_run02_xai --only shap")
    if os.path.exists(OUT_PNG) and not cfg.get("force_rerun", False):
        print(f"[LEWAT] {OUT_PNG}")
        return

    panels, n_disagree = _cam_panels(cfg)

    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, len(panels), height_ratios=[1.0, 2.1], figure=fig,
                           hspace=0.32, wspace=0.06, top=0.88, bottom=0.09)

    fig.suptitle(
        "Fig 14. Satu prediksi, dua jenis bukti: penjelasan spasial dan penjelasan fitur",
        fontsize=13, weight="bold", y=0.975)
    fig.text(0.5, 0.947,
             f"arm fusion_late, backbone {BACKBONE}, fold 0, {CAM_METHOD}; "
             f"{len(panels)} nodul tetap dari fixed_display_samples.json",
             ha="center", fontsize=9, color="#555555")
    fig.text(0.085, 0.912,
             "(a) cabang citra -- di MANA bukti berada  (kontur putih = mask radiolog)",
             fontsize=10, weight="bold")

    im = None
    for j, p in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(p["patch"], cmap="gray", interpolation="nearest")
        im = ax.imshow(p["cam"], cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0,
                       interpolation="bilinear")
        # kontur mask radiolog: pembanding kebenaran, bukan hiasan
        ax.contour(p["mask"], levels=[0.5], colors="#ffffff", linewidths=1.2)
        ok = "benar" if p["pred"] == p["label"] else "salah"
        ax.set_title(f"{p['slot']}  p={p['prob']:.2f}\n{ok}", fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])

    cax = fig.add_axes([0.92, 0.66, 0.012, 0.18])
    fig.colorbar(im, cax=cax).set_label("aktivasi Layer-CAM (ternormalisasi)", fontsize=7.5)

    ax_shap = fig.add_subplot(gs[1, :])
    ax_shap.imshow(_trim_white(plt.imread(SHAP_PNG)), aspect="auto",
                   interpolation="bilinear")
    ax_shap.axis("off")
    fig.text(0.085, 0.615,
             "(b) cabang radiomik -- FITUR apa yang mendorong keputusan (SHAP)",
             fontsize=10, weight="bold")

    fig.text(0.5, 0.028,
             f"Pada keenam nodul ini kelas keputusan fusion_late sama dengan cnn_only "
             f"(n_disagree={n_disagree}), sehingga panel (a) identik dengan peta cnn_only: "
             "menambah modalitas radiomik tidak mengubah lokalisasi.\n"
             "Panel (b) tidak punya padanan pada cnn_only, dan panel (a) mustahil secara "
             "struktural pada radiomics_only. Hanya fusion_late memiliki keduanya.",
             ha="center", fontsize=8.5, color="#333333")

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SELESAI] {OUT_PNG}  (run {RUN_ID}, commit {_commit_sha()}, "
          f"n_disagree={n_disagree})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--force", action="store_true", help="tulis ulang figure yang sudah ada")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.force:
        cfg["force_rerun"] = True
    run(cfg)


if __name__ == "__main__":
    main()
