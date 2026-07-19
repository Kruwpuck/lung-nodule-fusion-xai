"""Stage 05c: side-by-side Grad-CAM (CNN branch) + SHAP (radiomics branch) for the
SAME two nodules picked by stage_05b (highest-confidence TP malignant + TP benign,
fold 0 val split). This is the Track 1 XAI level-1 cross-modality figure.

Also does a lightweight level-2 spatial cross-check: for these two nodules, does the
CNN's Grad-CAM high-activation region actually fall inside the nodule mask (pointing
hit)? If the CNN is looking at the right place *and* the radiomics SHAP explanation is
dominated by shape/texture features (not spurious ones), that's mutually reinforcing
evidence; if either disagrees, it's flagged in the figure title rather than hidden.
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run(cfg: dict, fold: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    import torch

    from src.stage_03b_fusion import _load_merged, _select_fold_features
    from src.fusion.early_fusion import train_early_fusion_xgboost
    from src.models.registry import build_model, _NAME_MAP
    from src.xai.gradcam_utils import compute_gradcam, overlay_cam_on_image, pointing_hit

    out_dir = os.path.join(cfg["paths"]["results"], "xai_track1")
    sentinel = os.path.join(out_dir, "sidebyside_done.txt")
    from src.utils.io import cached
    if cached(sentinel) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {sentinel}")
        return
    os.makedirs(out_dir, exist_ok=True)

    merged, feat_cols = _load_merged(cfg)
    train_df = merged[merged["fold"] != fold].reset_index(drop=True)
    val_df = merged[merged["fold"] == fold].reset_index(drop=True)

    X_train_sel, X_val_sel, selected = _select_fold_features(train_df, val_df, feat_cols)
    clf = train_early_fusion_xgboost(X_train_sel, train_df["label"].values, cfg.get("xgboost", {}))
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_val_sel)

    y_val = val_df["label"].values
    prob = clf.predict_proba(X_val_sel)[:, 1]
    tp_malignant = np.where((y_val == 1) & (prob > 0.5))[0]
    tp_benign = np.where((y_val == 0) & (prob < 0.5))[0]
    picks = []
    if len(tp_malignant):
        picks.append(("malignant", int(tp_malignant[np.argmax(prob[tp_malignant])])))
    if len(tp_benign):
        picks.append(("benign", int(tp_benign[np.argmin(prob[tp_benign])])))

    model_name = cfg["track1_fusion"].get("backbone", "mobilenetv3_small")
    backbone_internal = _NAME_MAP.get(model_name, model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.path.join(cfg["paths"]["checkpoints"], model_name, f"fold{fold}_best.pt")
    from src.models.backbones import BackboneClassifier
    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)
    cnn_model = BackboneClassifier(backbone_internal, n_input_channels=n_slices, n_classes=2, pretrained=True).to(device)
    state = torch.load(ckpt, weights_only=True, map_location="cpu")
    cnn_model.load_state_dict(state["model_state"] if isinstance(state, dict) and "model_state" in state else state)
    cnn_model.eval()

    hu_min, hu_max = -1000.0, 400.0

    for label_name, idx in picks:
        row = val_df.iloc[idx]
        volume = np.load(row["patch_path"]).astype(np.float32)
        volume = np.clip(volume, hu_min, hu_max)
        volume = (volume - hu_min) / (hu_max - hu_min)
        Z, H, W = volume.shape
        cz = Z // 2
        half = n_slices // 2
        slices = [volume[max(0, min(Z - 1, cz + off))] for off in range(-half, half + 1)]
        img_chw = np.stack(slices, axis=0)
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0).float().to(device)

        cam = compute_gradcam(cnn_model, img_tensor, backbone_name=backbone_internal, target_class=int(y_val[idx]))

        mask_path = row.get("mask_path")
        hit = None
        if isinstance(mask_path, str) and os.path.exists(mask_path):
            mask_vol = np.load(mask_path)
            mz = mask_vol.shape[0] // 2
            mask2d = mask_vol[mz].astype(bool)
            if mask2d.shape != cam.shape:
                import cv2
                mask2d = cv2.resize(mask2d.astype(np.uint8), cam.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            hit = pointing_hit(cam, mask2d)

        center_slice_rgb = np.stack([img_chw[n_slices // 2]] * 3, axis=-1)
        overlay = overlay_cam_on_image(center_slice_rgb, cam)

        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        axes[0].imshow(overlay)
        axes[0].set_title(f"Grad-CAM (CNN branch)\npointing_hit={hit}")
        axes[0].axis("off")

        exp = shap.Explanation(
            values=shap_values[idx], base_values=explainer.expected_value,
            data=X_val_sel[idx], feature_names=selected,
        )
        plt.sca(axes[1])
        shap.plots.waterfall(exp, show=False, max_display=12)
        axes[1].set_title("SHAP (radiomics branch)")

        fig.suptitle(f"{label_name} nodule (fold {fold}, idx {idx}) — prob={prob[idx]:.3f}")
        plt.tight_layout()
        fname = f"sidebyside_{label_name}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info("%s saved (pointing_hit=%s)", fname, hit)

    with open(sentinel, "w") as f:
        f.write("done")
    print(f"[DONE] {out_dir} (sidebyside)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--fold", type=int, default=0)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg, args.fold)


if __name__ == "__main__":
    main()
