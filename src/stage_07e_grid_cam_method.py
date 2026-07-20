"""Stage 07e: Grid B -- CAM method comparison figure (VISUALIZATION_FIGURES_PLAN.md,
Bagian 1.3 Grid B). Same fixed sample set, single best backbone (densenet121 --
best mask overlap in Grid A), rows = CAM method instead of backbone.

Uses the 5 methods available in src/xai/gradcam_utils.py's cam_classes
(gradcam, gradcampp, hirescam, layercam, scorecam). Ablation-CAM from the plan
isn't wired in pytorch-grad-cam's cam_classes here, so eigencam is used as the
5th method instead -- reported as-is, not silently renamed.

Output: artifacts/results/figures_grid/grid_cam_method.png
"""
import argparse
import json
import logging
import os

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_METHODS = ["gradcam", "gradcampp", "hirescam", "layercam", "eigencam"]
_BEST_BACKBONE = "densenet121"


def _load_img_and_mask(patch_path: str, mask_path: str, n_slices: int, hu_min=-1000.0, hu_max=400.0):
    vol = np.load(patch_path).astype(np.float32)
    vol = np.clip(vol, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min)
    Z = vol.shape[0]
    cz = Z // 2
    half = n_slices // 2
    slices = [vol[max(0, min(Z - 1, cz + off))] for off in range(-half, half + 1)]
    img_chw = np.stack(slices, axis=0)

    mask_vol = np.load(mask_path)
    mask2d = mask_vol[mask_vol.shape[0] // 2].astype(bool)
    return img_chw, mask2d


def run(cfg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from src.models.registry import build_model
    from src.xai.gradcam_utils import compute_gradcam

    out_dir = os.path.join(cfg["paths"]["results"], "figures_grid")
    out_path = os.path.join(out_dir, "grid_cam_method.png")
    from src.utils.io import cached
    if cached(out_path) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {out_path}")
        return
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join("artifacts", "xai", "fixed_display_samples.json")) as f:
        fixed = json.load(f)
    samples = fixed["samples"]
    fold = fixed["fold"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_slices = cfg["data"].get("n_slices", 3)

    loaded = []
    for s in samples:
        img_chw, mask2d = _load_img_and_mask(s["patch_path"], s["mask_path"], n_slices)
        loaded.append((s, img_chw, mask2d))

    ckpt = os.path.join(cfg["paths"]["checkpoints"], _BEST_BACKBONE, f"fold{fold}_best.pt")
    model = build_model(_BEST_BACKBONE, cfg, task="binary").to(device)
    state = torch.load(ckpt, weights_only=True, map_location="cpu")
    model.load_state_dict(state["model_state"] if isinstance(state, dict) and "model_state" in state else state)
    model.eval()

    n_rows = len(_METHODS) + 1
    n_cols = len(samples)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.6 * n_rows))

    for col, (s, img_chw, mask2d) in enumerate(loaded):
        ct = img_chw[n_slices // 2]
        ax = axes[0, col]
        ax.imshow(ct, cmap="gray")
        ax.contour(mask2d, colors="lime", linewidths=1.2)
        ax.set_title(s["slot"], fontsize=11, fontweight="bold")
        ax.axis("off")

    for row, method in enumerate(_METHODS, start=1):
        for col, (s, img_chw, mask2d) in enumerate(loaded):
            ct = img_chw[n_slices // 2]
            img_t = torch.from_numpy(img_chw).unsqueeze(0).float().to(device)
            cam = compute_gradcam(model, img_t, backbone_name=_BEST_BACKBONE,
                                    target_class=s["label"], method=method)

            ax = axes[row, col]
            ax.imshow(ct, cmap="gray")
            im = ax.imshow(cam, alpha=0.5, cmap="jet", vmin=0.0, vmax=1.0)
            ax.contour(mask2d, colors="lime", linewidths=1.0)
            ax.axis("off")
            if col == 0:
                ax.text(-0.15, 0.5, method, transform=ax.transAxes, fontsize=10,
                         fontweight="bold", rotation=90, va="center", ha="center")
        logger.info("[%s] grid row done", method)

    fig.suptitle(f"backbone={_BEST_BACKBONE}", fontsize=12)
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="CAM activation")

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
