"""Intermediate (joint) fusion model: CNN embedding + radiomic vector -> shared dense layers."""
from __future__ import annotations

import torch
import torch.nn as nn
from src.models.backbones import build_2d_backbone, build_3d_backbone


class FusionNet(nn.Module):
    """End-to-end intermediate fusion: CNN branch + radiomic branch -> classifier.

    CNN produces embedding of size emb_dim.
    Radiomic branch projects n_radiomic features to rad_dim.
    Both are concatenated and passed through shared dense layers.
    """

    def __init__(
        self,
        n_radiomic: int,
        backbone_name: str = "mobilenet_v3_large",
        n_input_channels: int = 3,
        n_classes: int = 2,
        emb_dim: int = 256,
        rad_dim: int = 128,
        fusion_dim: int = 128,
        dropout: float = 0.3,
        pretrained: bool = True,
        mode: str = "2_5d",
    ) -> None:
        super().__init__()

        if mode in ("2d", "2_5d"):
            backbone, backbone_out = build_2d_backbone(
                backbone_name, n_input_channels, pretrained
            )
        elif mode == "3d":
            backbone, backbone_out = build_3d_backbone(backbone_name)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.cnn_branch = backbone
        self.img_proj = nn.Sequential(
            nn.Linear(backbone_out, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.rad_branch = nn.Sequential(
            nn.Linear(n_radiomic, rad_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim + rad_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, n_classes),
        )

    def forward(self, img: torch.Tensor, rad: torch.Tensor) -> torch.Tensor:
        img_emb = self.img_proj(self.cnn_branch(img))
        rad_emb = self.rad_branch(rad)
        return self.classifier(torch.cat([img_emb, rad_emb], dim=1))

    def get_cnn_embedding(self, img: torch.Tensor) -> torch.Tensor:
        return self.img_proj(self.cnn_branch(img))

    def get_radiomic_embedding(self, rad: torch.Tensor) -> torch.Tensor:
        return self.rad_branch(rad)
