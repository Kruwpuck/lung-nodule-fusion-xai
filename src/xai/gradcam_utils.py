"""Grad-CAM, Grad-CAM++, Score-CAM via pytorch-grad-cam library."""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def vit_reshape_transform(tensor: Any) -> Any:
    """Drop CLS token and reshape ViT tokens (B, 1+N, C) -> (B, C, H, W)."""
    import torch

    n = tensor.shape[1] - 1
    s = int(round(n ** 0.5))
    result = tensor[:, 1:, :].reshape(tensor.size(0), s, s, tensor.size(2))
    return result.permute(0, 3, 1, 2)


def _get_target_layer(model: Any, backbone_name: str) -> Any:
    """Resolve last spatial feature layer for given backbone architecture."""
    name = backbone_name.lower()

    if "mobilenet_v3" in name:
        if hasattr(model, "cnn_branch"):
            return model.cnn_branch[-1]  # FusionNet
        # BackboneClassifier: features is Sequential(model.features, AvgPool, Flatten)
        return model.features[0][-1]

    if "efficientnet" in name:
        if hasattr(model, "cnn_branch"):
            return model.cnn_branch[0][-1]
        return model.features[0][-1]

    if "resnet" in name and "3d" not in name:
        if hasattr(model, "cnn_branch"):
            return list(model.cnn_branch.children())[-2]  # before avgpool/flatten
        # BackboneClassifier.features = Sequential(conv1..layer4, avgpool, Flatten)
        # [-1]=Flatten, [-2]=avgpool, [-3]=layer4 (last spatial block)
        return model.features[-3]

    if "densenet" in name:
        if hasattr(model, "cnn_branch"):
            return model.cnn_branch[0][-1]
        return model.features[0][-1]

    if "convnext" in name:
        if hasattr(model, "cnn_branch"):
            return model.cnn_branch[0][-1]
        return model.features[0][-1]

    if "vgg" in name:
        if hasattr(model, "cnn_branch"):
            return model.cnn_branch[0][-1]
        return model.features[0][-1]

    if "vit" in name:
        # BackboneClassifier.features = the full torchvision ViT (heads=Identity)
        return model.features.encoder.layers[-1].ln_1

    raise ValueError(f"Cannot resolve target layer for backbone: {backbone_name}")


def compute_gradcam(
    model: Any,
    img_tensor: Any,
    backbone_name: str,
    target_class: Optional[int] = None,
    method: str = "gradcam",
) -> np.ndarray:
    """Compute saliency map using pytorch-grad-cam.

    Args:
        model: trained model (BackboneClassifier or FusionNet.cnn_branch part)
        img_tensor: (1, C, H, W) input image tensor
        backbone_name: used to resolve target layer
        target_class: class index to explain. None (default) explains the
            model's predicted (top-1) class — the standard diagnostic default.
            Pass an explicit int to force a fixed class (e.g. 1 = malignant),
            but note this produces an empty/near-zero map on samples the
            model does not associate with that class — expected Grad-CAM
            behavior (ReLU zeroes out unsupported classes), not a bug.
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
    targets = None if target_class is None else [ClassifierOutputTarget(target_class)]

    reshape_transform = vit_reshape_transform if "vit" in backbone_name.lower() else None

    with cam_classes[method](
        model=model, target_layers=[target_layer], reshape_transform=reshape_transform
    ) as cam:
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


def dice_iou(cam_map: np.ndarray, mask: np.ndarray, pct: float = 0.80) -> tuple[float, float]:
    """Dice and IoU of the top-(1-pct) activation region vs. the lesion mask."""
    thr = np.quantile(cam_map, pct)
    s = cam_map >= thr
    g = mask.astype(bool)
    inter = np.logical_and(s, g).sum()
    dice = 2 * inter / (s.sum() + g.sum() + 1e-7)
    iou = inter / (np.logical_or(s, g).sum() + 1e-7)
    return float(dice), float(iou)


def pointing_hit(cam_map: np.ndarray, mask: np.ndarray) -> bool:
    """Pointing game: does the CAM's single max-activation pixel fall inside the mask?"""
    y, x = np.unravel_index(np.argmax(cam_map), cam_map.shape)
    return bool(mask[y, x])


def energy_pointing_game(cam_map: np.ndarray, mask: np.ndarray) -> float:
    """Energy-based pointing game (Score-CAM style): fraction of CAM energy inside mask."""
    cam = cam_map - cam_map.min()
    total = cam.sum()
    if total <= 1e-7:
        return 0.0
    return float((cam * mask).sum() / (total + 1e-7))
