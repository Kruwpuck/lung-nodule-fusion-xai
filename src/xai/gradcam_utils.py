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

    if "googlenet" in name:
        if hasattr(model, "cnn_branch"):
            return list(model.cnn_branch.children())[-2]
        return model.features[-2]

    if "inception_resnet_v2" in name:
        # timm inception_resnet_v2: last spatial conv before global pool
        m = model.cnn_branch if hasattr(model, "cnn_branch") else model.features
        return m.conv2d_7b

    if "inception_v3" in name:
        if hasattr(model, "cnn_branch"):
            return list(model.cnn_branch.children())[-2]
        return model.features[-2]

    if "xception" in name:
        # timm legacy_xception: last block before global pool
        m = model.cnn_branch if hasattr(model, "cnn_branch") else model.features
        return m.conv4 if hasattr(m, "conv4") else list(m.children())[-2]

    raise ValueError(f"Cannot resolve target layer for backbone: {backbone_name}")


def _last_spatial_target_layer(model: Any, sample_input: Any) -> Any:
    """Pick the deepest-executed module whose 4-D output still has spatial extent.

    This is the canonical Grad-CAM rule (Selvaraju et al.): explain at the last layer
    that still carries a spatial map, i.e. the deepest module whose output is 4-D with
    both height and width greater than 1. Everything after that point has been through
    global pooling and carries a 1x1 map, which min-max normalises to identically zero
    and makes every localisation metric degenerate.

    Three details are resolved deliberately here rather than left to hook ordering:

    * A module that runs more than once in a forward pass (a shared ReLU inside a
      residual block, for instance) fires at several resolutions. pytorch-grad-cam pairs
      `target_layers[i]` with `activations[i]`, so a reused module hands the CAM its
      FIRST execution, which need not be the execution whose shape was observed here.
      Modules that fire more than once are therefore not eligible at all, so the shape
      recorded is always the shape the CAM receives.
    * A module sitting inside a block that has already collapsed the map to 1x1 is not
      the last spatial layer in any useful sense: it is one parallel branch of a block
      that pools at its exit (an Inception block whose stride-2 branches meet a 1x1
      concat, for instance). Explaining a single branch hides the class evidence flowing
      through its siblings, and the CAM can come out identically zero as a result --
      measured on inception_resnet_v2, 23 of 60 samples. Candidates enclosed by a
      collapsed block are dropped, which steps the choice out to the last block that
      still emitted a spatial map.
    * When the winner is a plain `nn.Sequential`, the tensor was really produced by its
      last child, and nothing happens between that child returning and the container
      returning. We descend to the child so the resolved layer names the operation that
      made the map instead of the wrapper around it. Descending is only safe for
      `Sequential`: any other container may combine or mutate its children's outputs
      after they return (a residual add, for example), which would make a child's
      activation differ from the block output the CAM should explain.

    Returns None when no module qualifies -- ViT has no 4-D intermediate outputs at all --
    and the caller then falls back to `_get_target_layer`.
    """
    import torch
    from torch import nn

    fired: list[tuple[Any, int, int]] = []   # (module, h, w) in execution order, 4-D only
    n_calls: dict[int, int] = {}

    def hook(mod: Any, inp: Any, out: Any) -> None:
        n_calls[id(mod)] = n_calls.get(id(mod), 0) + 1
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            fired.append((mod, int(out.shape[2]), int(out.shape[3])))

    handles = [m.register_forward_hook(hook) for m in model.modules()]
    try:
        with torch.no_grad():
            model(sample_input)
    finally:
        for h in handles:
            h.remove()

    paths = {id(m): name for name, m in model.named_modules()}
    collapsed = [paths[id(m)] for m, h, w in fired if h == 1 and w == 1 and paths[id(m)]]

    def _inside_collapsed(path: str) -> bool:
        return any(path.startswith(c + ".") for c in collapsed)

    spatial = [(m, h, w) for m, h, w in fired
               if h > 1 and w > 1 and n_calls[id(m)] == 1
               and not _inside_collapsed(paths[id(m)])]
    if not spatial:
        return None

    target, h, w = spatial[-1]
    shapes = {id(m): (mh, mw) for m, mh, mw in spatial}
    while isinstance(target, nn.Sequential) and len(target) and shapes.get(id(target[-1])) == (h, w):
        target = target[-1]
    return target


def _auto_target_layer(model: Any, sample_input: Any, lo: int = 7, hi: int = 10) -> Any:
    """Pick the deepest spatial (4D) submodule whose feature-map height falls in [lo, hi].

    Superseded and no longer on the CAM path. The premise behind the band -- that an
    ~8x8 map localises a small lesion better than the last spatial layer -- was measured
    and is false (see artifacts/results/track2rev/googlenet_layer_sweep.csv: a 6x6 map
    scored worse than a 3x3 map on both backbones tested), and on backbones whose heights
    never enter the band the search returned None and pushed resolution into the
    hand-written fallback table, which is how GoogLeNet ended up explaining a 1x1
    Dropout. Kept only so `src/stage_09a_target_layer_audit.py` can still reproduce the
    audit that established all of that.
    """
    import torch

    candidates: list[tuple[int, Any]] = []
    order = 0

    def make_hook(module: Any):
        nonlocal order

        def hook(mod: Any, inp: Any, out: Any) -> None:
            nonlocal order
            if isinstance(out, torch.Tensor) and out.dim() == 4:
                h = out.shape[2]
                if lo <= h <= hi:
                    candidates.append((order, mod))
            order += 1

        return hook

    handles = [m.register_forward_hook(make_hook(m)) for m in model.modules()]
    try:
        with torch.no_grad():
            model(sample_input)
    finally:
        for h in handles:
            h.remove()

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]  # deepest (last-executed) module in the target resolution band


def compute_gradcam(
    model: Any,
    img_tensor: Any,
    backbone_name: str,
    target_class: Optional[int] = None,
    method: str = "layercam",
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
        method: one of 'layercam', 'hirescam', 'gradcam', 'gradcampp', 'scorecam',
            'eigencam'. Default 'layercam', which weights each activation position
            separately and so keeps more detail than Grad-CAM's globally pooled
            weights on the small feature maps these inputs produce.

    Target layer: for non-ViT backbones it is resolved by `_last_spatial_target_layer`,
    the canonical rule -- the last layer that still has spatial extent. At a 64x64 input
    that map can be as small as 2x2 on a stride-32 backbone; it is upsampled to the input
    size, so the CAM is coarse but honest. ViT has no 4-D intermediate outputs, so it
    skips that search and resolves through `_get_target_layer` to the last encoder
    block's `ln_1`, read back into a grid by `vit_reshape_transform`.

    Returns:
        grayscale_cam: (H, W) numpy array, values in [0, 1]
    """
    try:
        from pytorch_grad_cam import (
            GradCAM,
            GradCAMPlusPlus,
            ScoreCAM,
            EigenCAM,
            LayerCAM,
            HiResCAM,
        )
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
        "layercam": LayerCAM,
        "hirescam": HiResCAM,
    }
    if method not in cam_classes:
        raise ValueError(f"method must be one of {list(cam_classes)}")

    is_vit = "vit" in backbone_name.lower()
    target_layer = None
    if not is_vit:
        target_layer = _last_spatial_target_layer(model, img_tensor)
    if target_layer is None:
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


def dice_size_matched(cam_map: np.ndarray, mask: np.ndarray) -> float:
    """Dice using a size-matched threshold: top-k CAM pixels, k = nodule area in pixels.

    Avoids the IoU/Dice ceiling imposed by a fixed top-20% threshold when the nodule
    occupies a much smaller fraction of the patch (e.g. <10%).
    """
    g = mask.astype(bool)
    k = int(g.sum())
    if k == 0:
        return 0.0
    flat = cam_map.ravel()
    k = min(k, flat.size)
    thr_idx = np.argpartition(flat, -k)[-k:]
    s = np.zeros_like(flat, dtype=bool)
    s[thr_idx] = True
    s = s.reshape(cam_map.shape)
    inter = np.logical_and(s, g).sum()
    return float(2 * inter / (s.sum() + g.sum() + 1e-7))


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
