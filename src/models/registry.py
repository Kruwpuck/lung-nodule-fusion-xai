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


def build_model(name: str, cfg: dict):
    from src.models.backbones import BackboneClassifier
    backbone = _NAME_MAP.get(name, name)
    n_slices = cfg.get("data", {}).get("n_slices", 3)
    return BackboneClassifier(backbone, n_input_channels=n_slices, pretrained=True)
