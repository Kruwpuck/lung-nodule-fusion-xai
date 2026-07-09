"""CNN backbone wrappers for 2.5D and 3D classification."""
from __future__ import annotations

import torch
import torch.nn as nn


def build_2d_backbone(name: str, n_input_channels: int = 3, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Build 2D/2.5D backbone. Returns (feature_extractor, embedding_dim).

    feature_extractor outputs (B, embedding_dim) after adaptive pooling.
    n_input_channels = n_slices for 2.5D (default 3).
    """
    import torchvision.models as tvm

    if name == "mobilenet_v3_large":
        weights = "IMAGENET1K_V2" if pretrained else None
        model = tvm.mobilenet_v3_large(weights=weights)
        # patch first conv to accept n_input_channels
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            n_input_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=old_conv.bias is not None,
        )
        if pretrained and n_input_channels != 3:
            # average pretrained weights across channel dim
            with torch.no_grad():
                model.features[0][0].weight.data = \
                    old_conv.weight.data.mean(dim=1, keepdim=True).repeat(1, n_input_channels, 1, 1) / n_input_channels
        features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        emb_dim = 960

    elif name == "efficientnet_b0":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = tvm.efficientnet_b0(weights=weights)
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            n_input_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=old_conv.bias is not None,
        )
        features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        emb_dim = 1280

    elif name == "resnet50":
        weights = "IMAGENET1K_V2" if pretrained else None
        model = tvm.resnet50(weights=weights)
        model.conv1 = nn.Conv2d(
            n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        backbone = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        features = backbone
        emb_dim = 2048

    elif name == "densenet121":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = tvm.densenet121(weights=weights)
        model.features.conv0 = nn.Conv2d(
            n_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        emb_dim = 1024

    elif name == "convnext_tiny":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = tvm.convnext_tiny(weights=weights)
        # ConvNeXt first stage
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            n_input_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding,
        )
        features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        emb_dim = 768

    elif name == "mobilenet_v3_small":
        weights = "IMAGENET1K_V1" if pretrained else None
        model = tvm.mobilenet_v3_small(weights=weights)
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            n_input_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=old_conv.bias is not None,
        )
        if pretrained and n_input_channels != 3:
            with torch.no_grad():
                model.features[0][0].weight.data = \
                    old_conv.weight.data.mean(dim=1, keepdim=True).repeat(1, n_input_channels, 1, 1) / n_input_channels
        features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        emb_dim = 576

    elif name == "vgg16":
        # ponytail: 224×224 input recommended; works at 64×64 but accuracy degrades
        weights = "IMAGENET1K_V1" if pretrained else None
        model = tvm.vgg16(weights=weights)
        old_conv = model.features[0]
        model.features[0] = nn.Conv2d(
            n_input_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=old_conv.bias is not None,
        )
        features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        emb_dim = 512

    elif name == "vit_b_16":
        # ponytail: 224×224 input required for ViT patch tokenisation
        weights = "IMAGENET1K_V1" if pretrained else None
        model = tvm.vit_b_16(weights=weights)
        old_proj = model.conv_proj
        model.conv_proj = nn.Conv2d(
            n_input_channels, old_proj.out_channels,
            kernel_size=old_proj.kernel_size, stride=old_proj.stride,
        )
        model.heads = nn.Identity()
        features = model
        emb_dim = 768

    else:
        raise ValueError(f"Unknown backbone: {name}")

    return features, emb_dim


def build_3d_backbone(name: str = "resnet3d_10") -> tuple[nn.Module, int]:
    """Build 3D backbone via MONAI. Returns (feature_extractor, embedding_dim)."""
    try:
        from monai.networks.nets import ResNet, resnet10
    except ImportError as e:
        raise ImportError("monai not installed. Run: pip install monai") from e

    if name == "resnet3d_10":
        model = resnet10(spatial_dims=3, n_input_channels=1, num_classes=2)
        # remove final FC, expose avgpool output
        emb_dim = 512
        feature_extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    else:
        raise ValueError(f"Unknown 3D backbone: {name}")

    return feature_extractor, emb_dim


class BackboneClassifier(nn.Module):
    """Standalone CNN classifier (Phase 1 benchmark, no fusion)."""

    def __init__(
        self,
        backbone_name: str,
        n_input_channels: int = 3,
        n_classes: int = 2,
        pretrained: bool = True,
        mode: str = "2_5d",
    ) -> None:
        super().__init__()
        if mode in ("2d", "2_5d"):
            self.features, emb_dim = build_2d_backbone(
                backbone_name, n_input_channels, pretrained
            )
        elif mode == "3d":
            self.features, emb_dim = build_3d_backbone(backbone_name)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.classifier = nn.Linear(emb_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
