"""Grad-CAM, Grad-CAM++, Score-CAM via pytorch-grad-cam library."""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_target_layer(model: Any, backbone_name: str) -> Any:
    """Resolve last convolutional block for given backbone architecture."""
    name = backbone_name.lower()

    if "mobilenet_v3" in name:
        if hasattr(model, "cnn_branch"):
            return model.cnn_branch[-1]  # FusionNet
        return model.features[0][-1]  # BackboneClassifier: features = Sequential(backbone_features, pool, flatten)

    if "efficientnet" in name:
        # features[-1] is the last MBConv stage
        if hasattr(model, "cnn_branch"):
            return model.cnn_branch[0][-1]
        return model.features[-1]

    if "resnet" in name and "3d" not in name:
        # model.layer4[-1] for ResNet-50
        if hasattr(model, "features"):
            return list(model.features.children())[-2]  # before avgpool
        return model.layer4[-1]

    if "densenet" in name:
        if hasattr(model, "cnn_branch"):
            return model.cnn_branch[0][-1]
        return model.features[-1]

    if "convnext" in name:
        if hasattr(model, "cnn_branch"):
            return model.cnn_branch[0][-1]
        return model.features[-1]

    raise ValueError(f"Cannot resolve target layer for backbone: {backbone_name}")


def compute_gradcam(
    model: Any,
    img_tensor: Any,
    backbone_name: str,
    target_class: int = 1,
    method: str = "gradcam",
) -> np.ndarray:
    """Compute saliency map using pytorch-grad-cam.

    Args:
        model: trained model (BackboneClassifier or FusionNet.cnn_branch part)
        img_tensor: (1, C, H, W) input image tensor
        backbone_name: used to resolve target layer
        target_class: class index to explain (1 = malignant)
        method: one of 'gradcam', 'gradcampp', 'scorecam', 'eigencam'

    Returns:
        grayscale_cam: (H, W) numpy array, values in [0, 1]
    """
    try:
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError as e:
        raise ImportError(
            "pytorch-grad-cam not installed. Run: pip install grad-cam"
        ) from e

    cam_classes = {
        "gradcam": GradCAM,
        "gradcampp": GradCAMPlusPlus,
        "scorecam": ScoreCAM,
        "eigencam": EigenCAM,
    }
    if method not in cam_classes:
        raise ValueError(f"method must be one of {list(cam_classes)}")

    target_layer = _get_target_layer(model, backbone_name)
    targets = [ClassifierOutputTarget(target_class)]

    with cam_classes[method](model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)

    return grayscale_cam[0]  # (H, W)


def overlay_cam_on_image(
    image: np.ndarray,
    cam_map: np.ndarray,
    colormap: int = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay CAM heatmap on original image slice. Returns RGB uint8 array."""
    import cv2

    if colormap is None:
        colormap = cv2.COLORMAP_JET

    cam_uint8 = (cam_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)

    # normalize image to 0-255
    img_norm = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    if img_norm.ndim == 2:
        img_rgb = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_norm

    overlaid = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlaid


def cam_in_nodule_fraction(
    cam_map: np.ndarray,
    nodule_mask_2d: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Fraction of high-activation CAM region that falls inside nodule mask.

    Used for spatial cross-validation: higher fraction = less spurious activation.
    """
    high_activation = (cam_map >= threshold).astype(bool)
    inside = (high_activation & nodule_mask_2d.astype(bool)).sum()
    total = high_activation.sum()
    if total == 0:
        return 0.0
    return float(inside / total)
