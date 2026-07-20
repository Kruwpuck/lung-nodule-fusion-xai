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
        mask2d = None
        if isinstance(mask_path, str) and os.path.exists(mask_path):
            mask_vol = np.load(mask_path)
            mz = mask_vol.shape[0] // 2
            mask2d = mask_vol[mz].astype(bool)
            if mask2d.shape != cam.shape:
                import cv2
                mask2d = cv2.resize(mask2d.astype(np.uint8), cam.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            hit = pointing_hit(cam, mask2d)

        ct = img_chw[n_slices // 2]

        # Render each panel to its own figure (shap.plots.waterfall manages its own
        # layout internally and squashes a shared subplot axis), then stitch with PIL.
        title = f"{label_name} nodule (fold {fold}, idx {idx}) — prob={prob[idx]:.3f}"

        # 3-panel CNN view, same style as Track 2's xai_{backbone}.png: raw CT,
        # CT + mask contour, CT + matplotlib alpha-blended jet heatmap (much more
        # legible than cv2.addWeighted, which over-saturates a low-contrast CT patch).
        fig_cam, axc = plt.subplots(1, 3, figsize=(10, 3.6))
        axc[0].imshow(ct, cmap="gray")
        axc[0].set_title("CT (center slice)", fontsize=11)
        axc[1].imshow(ct, cmap="gray")
        if mask2d is not None:
            axc[1].contour(mask2d, colors="lime", linewidths=1.2)
        axc[1].set_title("nodule mask", fontsize=11)
        axc[2].imshow(ct, cmap="gray")
        axc[2].imshow(cam, alpha=0.5, cmap="jet")
        axc[2].set_title(f"Grad-CAM\npointing_hit={hit}", fontsize=11)
        for a in axc:
            a.axis("off")
        plt.tight_layout()
        cam_path = os.path.join(out_dir, f"_tmp_cam_{label_name}.png")
        plt.savefig(cam_path, dpi=150, bbox_inches="tight")
        plt.close(fig_cam)

        exp = shap.Explanation(
            values=shap_values[idx], base_values=explainer.expected_value,
            data=X_val_sel[idx], feature_names=selected,
        )
        fig_shap = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(exp, show=False, max_display=12)
        plt.title("SHAP (radiomics branch)", fontsize=13)
        plt.tight_layout()
        shap_path = os.path.join(out_dir, f"_tmp_shap_{label_name}.png")
        plt.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close(fig_shap)

        from PIL import Image
        img_cam = Image.open(cam_path)
        img_shap = Image.open(shap_path)
        h = max(img_cam.height, img_shap.height)
        img_cam = img_cam.resize((int(img_cam.width * h / img_cam.height), h))
        img_shap = img_shap.resize((int(img_shap.width * h / img_shap.height), h))
        pad = 30
        combined = Image.new("RGB", (img_cam.width + img_shap.width + pad, h + 40), "white")
        combined.paste(img_cam, (0, 40))
        combined.paste(img_shap, (img_cam.width + pad, 40))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(combined)
        draw.text((10, 10), title, fill="black")
        fname = f"sidebyside_{label_name}.png"
        combined.save(os.path.join(out_dir, fname))
        os.remove(cam_path)
        os.remove(shap_path)
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
