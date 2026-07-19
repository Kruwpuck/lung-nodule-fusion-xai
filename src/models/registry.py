"""Model registry: map config names to BackboneClassifier."""
from __future__ import annotations

_NAME_MAP = {
    "mobilenetv3_small": "mobilenet_v3_small",
    "mobilenetv3_large": "mobilenet_v3_large",
    "efficientnet_b0":   "efficientnet_b0",
    "densenet121":       "densenet121",
    "convnext_tiny":     "convnext_tiny",
    "resnet50":          "resnet50",
    "vgg16":             "vgg16",
    "vit_base":          "vit_b_16",
}


# task -> n_classes for BackboneClassifier's final Linear head.
# "binary"  -> 2 (existing, CrossEntropyLoss)
# "ordinal" -> 1 (scalar regression on median_rating, SmoothL1Loss)
# "grade4"  -> 4 (no-nodule/benign/indeterminate/malignant, CrossEntropyLoss)
_TASK_N_CLASSES = {"binary": 2, "ordinal": 1, "grade3": 3, "grade4": 4}


def build_model(name: str, cfg: dict, task: str = "binary"):
    from src.models.backbones import BackboneClassifier
    backbone = _NAME_MAP.get(name, name)
    n_slices = cfg.get("data", {}).get("n_slices", 3)
    if task not in _TASK_N_CLASSES:
        raise ValueError(f"Unknown task: {task!r}. Expected one of {list(_TASK_N_CLASSES)}")
    n_classes = _TASK_N_CLASSES[task]
    return BackboneClassifier(backbone, n_input_channels=n_slices, n_classes=n_classes, pretrained=True)
