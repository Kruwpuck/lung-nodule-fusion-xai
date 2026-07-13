"""Stage 05: Grad-CAM on Track 1 backbone best checkpoint (resumable)."""
import argparse
import logging
import os
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

N_SAMPLES = 10


def run(cfg: dict) -> None:
    from src.utils.io import cached
    xai_dir = os.path.join(cfg["paths"]["results"], "xai")
    sentinel = os.path.join(xai_dir, "done.txt")
    if cached(sentinel) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {sentinel}")
        return

    import pandas as pd
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from src.models.registry import build_model, _NAME_MAP
    from src.training.dataset import NoduleDataset2_5D
    from src.xai.gradcam_utils import compute_gradcam

    backbone = cfg.get("track1_fusion", {}).get("backbone", "mobilenetv3_small")
    backbone_internal = _NAME_MAP.get(backbone, backbone)
    fold = 0
    best_pt = os.path.join(cfg["paths"]["checkpoints"], backbone, f"fold{fold}_best.pt")

    if not os.path.exists(best_pt):
        logger.warning("Checkpoint not found: %s — run stage_03 first", best_pt)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(backbone, cfg).to(device)

    try:
        state = torch.load(best_pt, weights_only=True, map_location="cpu")
    except (TypeError, __import__("pickle").UnpicklingError):
        state = torch.load(best_pt, weights_only=False, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()

    labels_path = os.path.join(cfg["paths"]["interim"], "labels.csv")
    df = pd.read_csv(labels_path)
    val_df = df[df["fold"] == fold].reset_index(drop=True).head(N_SAMPLES)

    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)
    val_ds = NoduleDataset2_5D(val_df, patch_size=patch_xy, n_slices=n_slices, augment=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    os.makedirs(xai_dir, exist_ok=True)
    for i, (img, label) in enumerate(val_loader):
        img = img.to(device)
        cam = compute_gradcam(model, img, backbone_internal)
        _save_cam_png(img[0].cpu().numpy(), cam, label.item(), i, xai_dir)

    with open(sentinel, "w") as f:
        f.write("done")
    print(f"[DONE] {xai_dir}  ({N_SAMPLES} CAMs)")


def _save_cam_png(img_chw, cam, label: int, idx: int, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mid = img_chw.shape[0] // 2
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(img_chw[mid], cmap="gray")
        axes[0].set_title(f"label={label}")
        axes[1].imshow(img_chw[mid], cmap="gray")
        axes[1].imshow(cam, alpha=0.5, cmap="jet")
        axes[1].set_title("Grad-CAM")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"sample_{idx:03d}.png"), dpi=80)
        plt.close()
    except Exception as e:
        logger.warning("Could not save CAM PNG %d: %s", idx, e)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
