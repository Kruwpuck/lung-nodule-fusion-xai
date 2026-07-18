"""Stage 05: Grad-CAM on all Track 2 backbones (arm A binary, fold 0). Resumable."""
import argparse
import logging
import os
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

N_SAMPLES_METRICS = 60   # cap per backbone for metric computation (fold-0 valid samples)
N_TP_FIGURE = 6          # curated true-positive malignant samples shown in the figure
N_FAIL_FIGURE = 2        # instructive failure cases (FP/FN) shown separately

def run(cfg: dict) -> None:
    from src.utils.io import cached
    xai_dir = os.path.join(cfg["paths"]["results"], "xai")
    sentinel = os.path.join(xai_dir, "done.txt")
    if cached(sentinel) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {sentinel}")
        return

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F

    from src.models.registry import build_model, _NAME_MAP
    from src.xai.gradcam_utils import (
        compute_gradcam,
        dice_iou,
        pointing_hit,
        energy_pointing_game,
    )

    os.makedirs(xai_dir, exist_ok=True)

    backbones = cfg["models"]["lightweight"] + cfg["models"]["heavyweight"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels_path = os.path.join(cfg["paths"]["interim"], "labels.csv")
    df = pd.read_csv(labels_path)
    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)

    val_df = df[(df["fold"] == 0) & (df["label"] != -1)].reset_index(drop=True)
    if len(val_df) > N_SAMPLES_METRICS:
        val_df = val_df.sample(n=N_SAMPLES_METRICS, random_state=42).reset_index(drop=True)

    hu_min, hu_max = -1000.0, 400.0

    def load_patch_tensor(patch_path: str) -> "torch.Tensor":
        vol = np.load(patch_path).astype(np.float32)
        vol = np.clip(vol, hu_min, hu_max)
        vol = (vol - hu_min) / (hu_max - hu_min)
        Z, H, W = vol.shape
        cz = Z // 2
        half = n_slices // 2
        cy, cx = H // 2, W // 2
        y0, y1 = max(0, cy - patch_xy // 2), min(H, cy + patch_xy // 2)
        x0, x1 = max(0, cx - patch_xy // 2), min(W, cx + patch_xy // 2)
        slices = []
        for offset in range(-half, half + 1):
            z = max(0, min(Z - 1, cz + offset))
            sl = np.zeros((patch_xy, patch_xy), dtype=np.float32)
            crop = vol[z, y0:y1, x0:x1]
            sl[: crop.shape[0], : crop.shape[1]] = crop
            slices.append(sl)
        arr = np.stack(slices, axis=0)
        return torch.from_numpy(arr).unsqueeze(0)  # (1, n_slices, H, W)

    metrics_rows = []

    for backbone in backbones:
        best_pt = os.path.join(cfg["paths"]["checkpoints"], backbone, "fold0_best.pt")
        if not os.path.exists(best_pt):
            logger.warning("Checkpoint not found: %s — skip", best_pt)
            continue

        model = build_model(backbone, cfg, task="binary").to(device)
        try:
            state = torch.load(best_pt, weights_only=True, map_location="cpu")
        except TypeError:
            state = torch.load(best_pt, map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
        model.eval()

        backbone_internal = _NAME_MAP.get(backbone, backbone)

        # Self-test: target layer must produce a spatial (H, W > 1) activation map.
        try:
            dummy = torch.zeros(1, n_slices, patch_xy, patch_xy, device=device)
            _ = compute_gradcam(model, dummy, backbone_name=backbone_internal, method="gradcam")
        except Exception as e:
            logger.warning("Target-layer self-test failed for %s: %s — skip", backbone, e)
            continue

        tp_candidates = []   # (prob, idx, img_chw, cam, mask2d, label, pred)
        fail_candidates = []
        dices, ious, hits, energies = [], [], [], []

        for i, row in val_df.iterrows():
            img = load_patch_tensor(row["patch_path"]).to(device)
            with torch.no_grad():
                logits = model(img)
                prob = F.softmax(logits, dim=1)[0, 1].item()
                pred = int(logits.argmax(dim=1).item())
            label = int(row["label"])

            cam = compute_gradcam(model, img, backbone_name=backbone_internal, method="gradcam")

            mask_full = np.load(row["mask_path"]).astype(np.float32)
            mid = mask_full.shape[0] // 2
            mask2d = mask_full[mid]

            d, iou = dice_iou(cam, mask2d)
            hit = pointing_hit(cam, mask2d)
            energy = energy_pointing_game(cam, mask2d)
            dices.append(d); ious.append(iou); hits.append(hit); energies.append(energy)

            img_chw = img[0].detach().cpu().numpy()
            entry = (prob, i, img_chw, cam, mask2d, label, pred)
            if label == 1 and pred == 1:
                tp_candidates.append(entry)
            elif label != pred:
                fail_candidates.append(entry)

        n = len(val_df)
        metrics_rows.append({
            "backbone": backbone,
            "n": n,
            "dice": float(np.mean(dices)) if dices else float("nan"),
            "iou": float(np.mean(ious)) if ious else float("nan"),
            "pointing_acc": float(np.mean(hits)) if hits else float("nan"),
            "energy_mean": float(np.mean(energies)) if energies else float("nan"),
            "threshold_pct": 0.80,
        })

        tp_candidates.sort(key=lambda t: t[0], reverse=True)
        curated = tp_candidates[:N_TP_FIGURE]
        failures = fail_candidates[:N_FAIL_FIGURE]
        _save_figure(backbone, curated, failures, xai_dir)

        logger.info("[%s] dice=%.3f iou=%.3f pointing_acc=%.3f energy=%.3f",
                    backbone, metrics_rows[-1]["dice"], metrics_rows[-1]["iou"],
                    metrics_rows[-1]["pointing_acc"], metrics_rows[-1]["energy_mean"])

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(xai_dir, "xai_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    with open(sentinel, "w") as f:
        f.write("done")
    print(f"[DONE] {xai_dir}  ({len(metrics_rows)} backbones)")

def _save_figure(backbone: str, curated: list, failures: list, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = curated + failures
        if not rows:
            logger.warning("No samples to plot for %s", backbone)
            return

        n_rows = len(rows)
        fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, 3)

        for r, (prob, idx, img_chw, cam, mask2d, label, pred) in enumerate(rows):
            mid = img_chw.shape[0] // 2
            ct = img_chw[mid]
            is_failure = r >= len(curated)
            tag = "FAIL " if is_failure else ""

            axes[r, 0].imshow(ct, cmap="gray")
            axes[r, 0].set_title(f"{tag}label={label} pred={pred} p={prob:.2f}")

            axes[r, 1].imshow(ct, cmap="gray")
            axes[r, 1].contour(mask2d, colors="lime", linewidths=1.0)
            axes[r, 1].set_title("nodule mask")

            axes[r, 2].imshow(ct, cmap="gray")
            axes[r, 2].imshow(cam, alpha=0.5, cmap="jet")
            axes[r, 2].set_title("Grad-CAM (predicted class)")

            for c in range(3):
                axes[r, c].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"xai_{backbone}.png"), dpi=100)
        plt.close()
    except Exception as e:
        logger.warning("Could not save XAI figure for %s: %s", backbone, e)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg)

if __name__ == "__main__":
    main()
