"""TreeSHAP and DeepSHAP utilities for radiomic and fusion model explanation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_tree_shap(
    clf,                        # fitted XGBoost / LightGBM classifier
    X: np.ndarray,
    feature_names: list[str],
    output_dir: str = "results/xai",
) -> tuple[np.ndarray, object]:
    """Compute TreeSHAP values for tabular classifier.

    Returns (shap_values, explainer). shap_values shape: (N, n_features).
    Saves global beeswarm plot to output_dir/shap_beeswarm.png.
    """
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("shap not installed. Run: pip install shap") from e

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    # For binary XGBoost: shap_values can be (N, F) or list[(N,F), (N,F)]
    if isinstance(shap_values, list):
        shap_vals_class1 = shap_values[1]
    else:
        shap_vals_class1 = shap_values

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Global beeswarm
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_vals_class1,
        X,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP beeswarm to %s/shap_beeswarm.png", output_dir)

    return shap_vals_class1, explainer


def plot_shap_waterfall(
    explainer,
    X: np.ndarray,
    feature_names: list[str],
    sample_idx: int = 0,
    output_path: str = "results/xai/shap_waterfall.png",
) -> None:
    """Local waterfall plot for a single sample."""
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("shap not installed.") from e

    explanation = explainer(pd.DataFrame(X, columns=feature_names))
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation[sample_idx], show=False)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved waterfall plot to %s", output_path)


def get_top_shap_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    n_top: int = 10,
) -> pd.DataFrame:
    """Return top-n features by mean absolute SHAP value."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).head(n_top).reset_index(drop=True)


def spatial_cross_check(
    shap_values: np.ndarray,
    feature_names: list[str],
    cam_maps: list[np.ndarray],        # one (H, W) array per test nodule
    nodule_masks_2d: list[np.ndarray], # one (H, W) binary mask per test nodule
    n_top_features: int = 5,
    cam_threshold: float = 0.5,
) -> dict:
    """Cross-validate SHAP texture dominance with Grad-CAM spatial activation.

    Checks: if top SHAP features are texture-class, do CAM maps focus inside nodule?

    Returns dict with:
        top_features: list of top SHAP feature names
        texture_dominated: bool (majority of top features are texture-class)
        mean_cam_in_nodule: fraction of high-activation CAM inside nodule mask
        spurious_flag: True if cam_in_nodule < 0.70
    """
    from src.xai.gradcam_utils import cam_in_nodule_fraction

    top_df = get_top_shap_features(shap_values, feature_names, n_top_features)
    top_feat_names = top_df["feature"].tolist()

    texture_keywords = ("glcm", "glrlm", "glszm", "gldm", "ngtdm")
    n_texture = sum(
        any(kw in f.lower() for kw in texture_keywords)
        for f in top_feat_names
    )
    texture_dominated = n_texture >= (n_top_features // 2 + 1)

    cam_fractions = [
        cam_in_nodule_fraction(cam, mask, cam_threshold)
        for cam, mask in zip(cam_maps, nodule_masks_2d)
    ]
    mean_cam_in_nodule = float(np.mean(cam_fractions))

    return {
        "top_features": top_feat_names,
        "texture_dominated": texture_dominated,
        "n_texture_in_top": n_texture,
        "mean_cam_in_nodule": mean_cam_in_nodule,
        "cam_fractions_per_nodule": cam_fractions,
        "spurious_flag": mean_cam_in_nodule < 0.70,
    }
