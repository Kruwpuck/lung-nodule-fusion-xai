"""Stage 07f: cross-model comparability grid for the Rev1 XAI revision (task 8).

Fixes the defect in stage_05_xai.py: every backbone there picked its OWN top-6
true-positive samples (sorted by that backbone's predicted probability), so no two
backbones' panels showed the same nodule and heat intensity was auto-scaled per
panel -- neither is comparable across models.

This script instead:
  - reuses the ALREADY fixed, deterministic sample set from
    artifacts/xai/fixed_display_samples.json (built once by stage_07b with no
    randomness -- picks are sorted by probability/size, not sampled), so every
    row below shows the identical nodules;
  - puts backbones on rows, fixed samples on columns (stage_07d's layout, reused
    via its `_load_img_and_mask` helper);
  - renders ONE shared colorbar (vmin=0, vmax=1, single "jet" colormap, alpha=0.5)
    across every panel, so heat intensity means the same thing everywhere;
  - overlays the ground-truth nodule mask contour on every panel, including the
    CT-only reference row;
  - deliberately keeps mobilenetv3_small, vit_base and googlenet as rows even
    though their fold-0 pointing accuracy is 0.0 -- that null result next to
    densenet121/convnext_tiny's 0.7167 IS the failure-mode evidence the Rev1
    report needs, not a reason to drop them;
  - states the CAM upsampling method and the actual spatial size of the chosen
    convolutional stage per backbone in the figure caption.

Output: artifacts/results/figures_grid/grid_comparability.png
"""
import argparse
import json
import logging
import os

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rows requested for Rev1 task 8: the high-pointing pair first, then the
# zero-pointing group -- adjacent rows make the divergence visible at a glance.
BACKBONES = ["densenet121", "convnext_tiny", "mobilenetv3_small", "vit_base", "googlenet"]
CAM_METHOD = "layercam"

# Fold-0 pointing accuracy read from artifacts/results/track2rev/cam_12.csv, the table
# measured under the same resolver the maps in this figure are drawn with. Deliberately
# not artifacts/results/xai/xai_metrics.csv: that table predates commit f81ba0d and
# reports the band rule's sites, so quoting it beside maps drawn at the canonical sites
# would caption the figure with numbers from a different layer.
CAM_12_CSV = os.path.join("artifacts", "results", "track2rev", "cam_12.csv")


def _pointing_acc() -> dict:
    import pandas as pd

    if not os.path.exists(CAM_12_CSV):
        return {}
    col = pd.read_csv(CAM_12_CSV).set_index("backbone")["pointing_acc"]
    return {b: float(col[b]) for b in BACKBONES if b in col.index}


def _target_layer_spatial_size(model, backbone_internal: str, dummy) -> tuple[int, int] | None:
    """Forward-hook the CAM target layer once to report its (H, W) for the caption.

    Resolved exactly as `compute_gradcam` resolves it, so the caption names the layer the
    figure was actually drawn from. Before this, the size came from `_auto_target_layer`,
    the band rule that commit f81ba0d replaced, while the maps came from the canonical
    rule -- so the caption reported GoogLeNet at 1x1 beside a map that is visibly
    localized, which is the tooling-not-the-network failure this project reports on
    others.
    """
    import torch

    from src.xai.gradcam_utils import _get_target_layer, _last_spatial_target_layer

    is_vit = "vit" in backbone_internal.lower()
    layer = None if is_vit else _last_spatial_target_layer(model, dummy)
    if layer is None:
        layer = _get_target_layer(model, backbone_internal)

    size: dict = {}

    def hook(_mod, _inp, out):
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            size["hw"] = (int(out.shape[-2]), int(out.shape[-1]))

    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(dummy)
    finally:
        handle.remove()
    return size.get("hw")


def _caption(spatial_sizes: dict) -> str:
    size_bits = ", ".join(
        f"{b}={s[0]}x{s[1]}" if s else f"{b}=n/a" for b, s in spatial_sizes.items()
    )
    acc = _pointing_acc()
    acc_bits = ", ".join(f"{b}={acc[b]:.4f}" for b in BACKBONES if b in acc) or "unavailable"
    return (
        f"CAM method: {CAM_METHOD} (Layer-CAM), target layer resolved per backbone as the deepest "
        "feature map that still has spatial extent, upsampled to the model's input resolution via "
        "pytorch-grad-cam's bilinear interpolation. ViT-Base has no 4-D intermediate and is "
        f"explained through a reshape transform instead. Resolved stage sizes: {size_bits}. "
        f"Fold-0 pointing accuracy ({CAM_12_CSV}): {acc_bits}. A low row is a localization failure "
        "mode at the explanation site, not evidence of a worse classifier."
    )


def run(cfg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from src.models.registry import build_model, _NAME_MAP
    from src.stage_07d_grid_backbone import _load_img_and_mask
    from src.xai.gradcam_utils import compute_gradcam
    from src.utils.tracks import track_input_size

    out_dir = os.path.join(cfg["paths"]["results"], "figures_grid")
    out_path = os.path.join(out_dir, "grid_comparability.png")
    from src.utils.io import cached
    if cached(out_path) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {out_path}")
        return
    os.makedirs(out_dir, exist_ok=True)

    fixed_json = os.path.join("artifacts", "xai", "fixed_display_samples.json")
    if not os.path.exists(fixed_json):
        raise FileNotFoundError(
            f"{fixed_json} not found -- run `python -m src.stage_07b_fixed_samples` first "
            "to pick the fixed, seeded sample set this figure reuses."
        )
    with open(fixed_json) as f:
        fixed = json.load(f)
    samples = fixed["samples"]
    fold = fixed["fold"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)

    loaded = []
    for s in samples:
        img_chw, mask2d = _load_img_and_mask(s["patch_path"], s["mask_path"], n_slices, patch_xy)
        loaded.append((s, img_chw, mask2d))

    n_rows = len(BACKBONES) + 1
    n_cols = len(samples)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.6 * n_rows))
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for col, (s, img_chw, mask2d) in enumerate(loaded):
        ct = img_chw[n_slices // 2]
        ax = axes[0, col]
        ax.imshow(ct, cmap="gray")
        ax.contour(mask2d, colors="lime", linewidths=1.2)
        ax.set_title(s["slot"], fontsize=11, fontweight="bold")
        ax.axis("off")
    axes[0, 0].text(-0.15, 0.5, "CT + mask", transform=axes[0, 0].transAxes, fontsize=10,
                     fontweight="bold", rotation=90, va="center", ha="center")

    im = None
    spatial_sizes: dict = {}
    for row, backbone in enumerate(BACKBONES, start=1):
        ckpt = os.path.join(cfg["paths"]["checkpoints"], backbone, f"fold{fold}_best.pt")
        if not os.path.exists(ckpt):
            logger.warning("missing checkpoint %s, blank row for %s", ckpt, backbone)
            for col in range(n_cols):
                axes[row, col].axis("off")
            spatial_sizes[backbone] = None
            continue

        model = build_model(backbone, cfg, task="binary").to(device)
        state = torch.load(ckpt, weights_only=True, map_location="cpu")
        model.load_state_dict(state["model_state"] if isinstance(state, dict) and "model_state" in state else state)
        model.eval()

        backbone_internal = _NAME_MAP.get(backbone, backbone)
        input_size = track_input_size(cfg, backbone) or patch_xy
        dummy = torch.zeros(1, n_slices, input_size, input_size, device=device)
        spatial_sizes[backbone] = _target_layer_spatial_size(model, backbone_internal, dummy)

        for col, (s, img_chw, mask2d) in enumerate(loaded):
            ct = img_chw[n_slices // 2]
            img_t = torch.from_numpy(img_chw).unsqueeze(0).float().to(device)
            cam = compute_gradcam(model, img_t, backbone_name=backbone_internal,
                                    target_class=s["label"], method=CAM_METHOD)

            ax = axes[row, col]
            ax.imshow(ct, cmap="gray")
            im = ax.imshow(cam, alpha=0.5, cmap="jet", vmin=0.0, vmax=1.0)
            ax.contour(mask2d, colors="lime", linewidths=1.0)
            ax.axis("off")
            if col == 0:
                ax.text(-0.15, 0.5, backbone, transform=ax.transAxes, fontsize=10,
                         fontweight="bold", rotation=90, va="center", ha="center")

        logger.info("[%s] comparability row done", backbone)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fig.subplots_adjust(right=0.90, bottom=0.12)
    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label=f"{CAM_METHOD} activation (shared vmin=0, vmax=1)")

    fig.text(0.02, 0.02, _caption(spatial_sizes), fontsize=7.5, wrap=True)

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
