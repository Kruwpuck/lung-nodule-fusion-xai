"""Stage 07b: select the FIXED display sample set for all downstream XAI figures
(VISUALIZATION_FIGURES_PLAN.md, Bagian 1.1-1.2).

Controlled-comparison principle: every backbone/CAM-method grid must show the SAME
nodules, otherwise a difference in heatmap could come from the nodule, not the
method. This script runs all 6 arm-A (binary) backbones once on the fold-0
validation split, then picks a fixed slot set:

  S1-S3  TP malignant, high-confidence (all 6 backbones agree), diverse nodule size
  S4     TP malignant, high-confidence, smallest nodule (resolution stress test)
  S5     True benign, high-confidence (all 6 backbones agree)
  S6     Failure case: high cross-backbone disagreement (std of predicted prob)

All picks are verified to have a non-empty, roughly-centered mask (centroid within
6px of the 64x64 patch center) before being accepted.

Output: artifacts/xai/fixed_display_samples.json
"""
import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_MAX_CENTER_OFFSET_PX = 6.0


def _load_patch_tensor(patch_path: str, n_slices: int, patch_xy: int, hu_min=-1000.0, hu_max=400.0):
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
    return np.stack(slices, axis=0)  # (n_slices, H, W)


def _mask_center_offset(mask_path: str) -> tuple[float, bool]:
    """Returns (offset_px from patch center, mask_nonempty)."""
    mask_vol = np.load(mask_path)
    mid = mask_vol.shape[0] // 2
    mask2d = mask_vol[mid].astype(bool)
    if not mask2d.any():
        return float("inf"), False
    H, W = mask2d.shape
    ys, xs = np.nonzero(mask2d)
    cy, cx = ys.mean(), xs.mean()
    offset = float(np.hypot(cy - H / 2, cx - W / 2))
    return offset, True


def run(cfg: dict, fold: int = 0) -> None:
    import torch
    import torch.nn.functional as F

    from src.models.registry import build_model

    out_json = os.path.join("artifacts", "xai", "fixed_display_samples.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    from src.utils.io import cached
    if cached(out_json) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {out_json}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)
    backbones = cfg["models"]["lightweight"] + cfg["models"]["heavyweight"]

    labels_path = os.path.join(cfg["paths"]["interim"], "labels.csv")
    df = pd.read_csv(labels_path)
    val_df = df[(df["fold"] == fold) & (df["label"].isin([0, 1]))].reset_index(drop=True)
    logger.info("fold %d val (binary usable): %d rows", fold, len(val_df))

    # nodule diameter, for size-diversity picks
    rad_path = cfg["paths"]["features"]
    rad = pd.read_parquet(rad_path, columns=["patient_id", "nodule_idx", "original_shape_Maximum3DDiameter"])
    val_df = val_df.merge(rad, on=["patient_id", "nodule_idx"], how="left")

    probs = np.zeros((len(backbones), len(val_df)), dtype=np.float32)
    models_loaded = []
    for bi, backbone in enumerate(backbones):
        ckpt = os.path.join(cfg["paths"]["checkpoints"], backbone, f"fold{fold}_best.pt")
        if not os.path.exists(ckpt):
            logger.warning("missing checkpoint %s, skip backbone %s", ckpt, backbone)
            probs[bi, :] = np.nan
            continue
        model = build_model(backbone, cfg, task="binary").to(device)
        state = torch.load(ckpt, weights_only=True, map_location="cpu")
        model.load_state_dict(state["model_state"] if isinstance(state, dict) and "model_state" in state else state)
        model.eval()
        models_loaded.append(backbone)

        with torch.no_grad():
            for i, row in val_df.iterrows():
                img_chw = _load_patch_tensor(row["patch_path"], n_slices, patch_xy)
                img_t = torch.from_numpy(img_chw).unsqueeze(0).float().to(device)
                logits = model(img_t)
                probs[bi, i] = F.softmax(logits, dim=1)[0, 1].item()
        logger.info("[%s] inference done", backbone)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_prob = np.nanmean(probs, axis=0)
    std_prob = np.nanstd(probs, axis=0)
    n_valid_backbones = np.sum(~np.isnan(probs), axis=0)

    y = val_df["label"].values
    all_agree_malignant = np.all(np.where(np.isnan(probs), 0.5, probs) > 0.5, axis=0)
    all_agree_benign = np.all(np.where(np.isnan(probs), 0.5, probs) < 0.5, axis=0)

    def _passes_centering(idx: int) -> bool:
        offset, nonempty = _mask_center_offset(val_df.iloc[idx]["mask_path"])
        return nonempty and offset <= _MAX_CENTER_OFFSET_PX

    tp_mask = (y == 1) & all_agree_malignant
    tp_idx = [i for i in np.where(tp_mask)[0] if _passes_centering(i)]
    tp_idx.sort(key=lambda i: -mean_prob[i])

    tn_mask = (y == 0) & all_agree_benign
    tn_idx = [i for i in np.where(tn_mask)[0] if _passes_centering(i)]
    tn_idx.sort(key=lambda i: mean_prob[i])  # lowest prob = most confidently benign

    disagree_idx = [i for i in range(len(val_df)) if _passes_centering(i)]
    disagree_idx.sort(key=lambda i: -std_prob[i])

    picks: dict[str, int] = {}
    if len(tp_idx) >= 3:
        by_size = sorted(tp_idx[: max(20, len(tp_idx) // 2)],
                          key=lambda i: val_df.iloc[i]["original_shape_Maximum3DDiameter"])
        picks["S1"] = by_size[0]
        picks["S2"] = by_size[len(by_size) // 2]
        picks["S3"] = by_size[-1]
    elif tp_idx:
        for k, i in zip(["S1", "S2", "S3"], tp_idx):
            picks[k] = i

    remaining_tp = [i for i in tp_idx if i not in picks.values()]
    if remaining_tp:
        picks["S4"] = min(remaining_tp, key=lambda i: val_df.iloc[i]["original_shape_Maximum3DDiameter"])

    if tn_idx:
        picks["S5"] = tn_idx[0]

    remaining_disagree = [i for i in disagree_idx if i not in picks.values() and y[i] != -1]
    for i in remaining_disagree:
        pred_majority = int(np.nanmean(probs[:, i]) > 0.5)
        if pred_majority != y[i] or std_prob[i] > 0.15:
            picks["S6"] = i
            break

    fixed_set = []
    slot_desc = {
        "S1": "TP malignant, high-conf, small nodule",
        "S2": "TP malignant, high-conf, medium nodule",
        "S3": "TP malignant, high-conf, large nodule",
        "S4": "TP malignant, high-conf, smallest nodule (resolution stress test)",
        "S5": "True benign, high-conf (all backbones agree)",
        "S6": "Disagreement / failure case across backbones",
    }
    for slot, idx in picks.items():
        row = val_df.iloc[idx]
        fixed_set.append({
            "slot": slot,
            "description": slot_desc[slot],
            "patient_id": row["patient_id"],
            "scan_id": int(row["scan_id"]),
            "nodule_idx": int(row["nodule_idx"]),
            "fold": int(row["fold"]),
            "label": int(row["label"]),
            "patch_path": row["patch_path"],
            "mask_path": row["mask_path"],
            "diameter_mm": float(row["original_shape_Maximum3DDiameter"]),
            "mean_prob_malignant": float(mean_prob[idx]),
            "std_prob_malignant": float(std_prob[idx]),
            "per_backbone_prob": {b: (float(probs[bi, idx]) if not np.isnan(probs[bi, idx]) else None)
                                   for bi, b in enumerate(backbones)},
        })
        logger.info("%s -> patient=%s nodule=%d diam=%.1fmm mean_prob=%.3f",
                    slot, row["patient_id"], row["nodule_idx"],
                    row["original_shape_Maximum3DDiameter"], mean_prob[idx])

    with open(out_json, "w") as f:
        json.dump({"fold": fold, "backbones": backbones, "samples": fixed_set}, f, indent=2)
    print(f"[DONE] {out_json} ({len(fixed_set)} slots)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--fold", type=int, default=0)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg, args.fold)


if __name__ == "__main__":
    main()
