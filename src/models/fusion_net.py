"""Intermediate (joint) fusion model: CNN embedding + radiomic vector -> shared dense layers."""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from src.models.backbones import build_2d_backbone, build_3d_backbone, _MIN_INPUT

logger = logging.getLogger(__name__)

FUSION_ARMS = ("concat", "branch_norm", "gmu")


class FusionNet(nn.Module):
    """End-to-end intermediate fusion: CNN branch + radiomic branch -> classifier.

    CNN produces embedding of size emb_dim.
    Radiomic branch projects n_radiomic features to rad_dim.

    fusion_arm="concat" (default): unchanged legacy behavior — CNN embedding at
    emb_dim (256) concatenated with the radiomic embedding at rad_dim (128).

    fusion_arm="branch_norm" (Rev1 task 5a): the CNN embedding is down-projected
    to proj_dim (~32) instead of emb_dim, and both branches are L2-normalized
    (per-sample) before concatenation. This addresses the magnitude/dimensionality
    imbalance where a 256-dim CNN embedding dominates or drowns out the much
    smaller radiomic branch. Per-branch norms are recorded in
    ``self.last_branch_norms`` after every forward call so the rebalancing is
    observable rather than assumed.

    fusion_arm="gmu" (Rev1 task 5b): Gated Multimodal Unit (Arevalo et al. 2017).
    Each modality gets its own tanh feature transform, h_v = tanh(W_v v) and
    h_t = tanh(W_t t), computed from the *raw* branch inputs (CNN backbone
    features and the radiomic vector). A sigmoid gate z = sigmoid(W_z [v, t])
    is computed from both raw modalities and mixes the two transforms
    elementwise: h = z*h_v + (1-z)*h_t. Unlike concat/branch_norm, this
    replaces concatenation entirely — the classifier consumes h directly, so
    the gate itself (not a fixed architecture choice) decides how much each
    modality contributes, including down-weighting a weak CNN branch to
    near zero. Gate activations are recorded in ``self.last_branch_norms``
    (keys "img_gate"/"rad_gate" = mean(z), mean(1-z)) after every forward call.

    modality_dropout / aux_loss_weight (Rev1 task 5c): regularizers, not a
    fourth fusion topology, so they compose with any of the three arms above
    instead of adding a new fusion_arm value. modality_dropout (a rate in
    [0, 1)) zeroes each branch's embedding independently per-sample during
    training only (self.training), forcing the classifier not to rely on one
    modality always being present. aux_loss_weight > 0 adds a small linear
    head per branch (aux_img_head / aux_rad_head) that must also classify the
    label from that branch's own embedding alone; the weighted sum of both
    auxiliary losses is added to the main loss by the training loop
    (train_fusion_epoch in src/fusion/intermediate_fusion.py), which reads
    model.aux_loss_weight and calls model.aux_logits(). Both default to 0,
    so the default config path is unaffected.
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
        input_size: int | None = None,
        fusion_arm: str = "concat",
        proj_dim: int = 32,
        gmu_dim: int = 128,
        modality_dropout: float = 0.0,
        aux_loss_weight: float = 0.0,
        n_classes_aux: int | None = None,
    ) -> None:
        super().__init__()

        if fusion_arm not in FUSION_ARMS:
            raise ValueError(f"Unknown fusion_arm: {fusion_arm!r}. Expected one of {FUSION_ARMS}")
        if not (0.0 <= modality_dropout < 1.0):
            raise ValueError(f"modality_dropout must be in [0, 1), got {modality_dropout!r}")
        self.fusion_arm = fusion_arm
        self.modality_dropout = modality_dropout
        self.aux_loss_weight = aux_loss_weight
        self.last_branch_norms: dict[str, float] = {}
        self.last_img_emb: torch.Tensor | None = None
        self.last_rad_emb: torch.Tensor | None = None

        if mode in ("2d", "2_5d"):
            backbone, backbone_out = build_2d_backbone(
                backbone_name, n_input_channels, pretrained
            )
        elif mode == "3d":
            backbone, backbone_out = build_3d_backbone(backbone_name)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.cnn_branch = backbone
        self._resize_to: int | None = input_size if input_size is not None else _MIN_INPUT.get(backbone_name)

        if fusion_arm == "gmu":
            # GMU consumes the raw branch inputs directly (CNN backbone features,
            # radiomic vector) — no img_proj/rad_branch, no concatenation. The gate
            # replaces concatenation as the fusion mechanism itself.
            self.gmu_img_transform = nn.Linear(backbone_out, gmu_dim)
            self.gmu_rad_transform = nn.Linear(n_radiomic, gmu_dim)
            self.gmu_gate = nn.Linear(backbone_out + n_radiomic, gmu_dim)
            classifier_in_dim = gmu_dim
            img_emb_dim = rad_emb_dim = gmu_dim
        else:
            # branch_norm down-projects the CNN embedding to proj_dim so it is
            # dimensionally comparable to the ~20-dim radiomic vector; concat keeps
            # today's emb_dim (256) unchanged so the default config path is byte-identical.
            img_out_dim = proj_dim if fusion_arm == "branch_norm" else emb_dim
            self.img_proj = nn.Sequential(
                nn.Linear(backbone_out, img_out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.rad_branch = nn.Sequential(
                nn.Linear(n_radiomic, rad_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            classifier_in_dim = img_out_dim + rad_dim
            img_emb_dim, rad_emb_dim = img_out_dim, rad_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, n_classes),
        )

        # Rev1 task 5c: auxiliary per-branch classification heads, only built
        # when actually used (aux_loss_weight > 0) so the default config path
        # allocates no extra parameters at all.
        if self.aux_loss_weight > 0:
            n_cls_aux = n_classes_aux if n_classes_aux is not None else n_classes
            self.aux_img_head = nn.Linear(img_emb_dim, n_cls_aux)
            self.aux_rad_head = nn.Linear(rad_emb_dim, n_cls_aux)

    def _maybe_resize(self, x: torch.Tensor) -> torch.Tensor:
        if self._resize_to is not None and x.shape[-1] != self._resize_to:
            x = torch.nn.functional.interpolate(
                x, size=(self._resize_to, self._resize_to),
                mode="bilinear", align_corners=False,
            )
        return x

    def _apply_modality_dropout(self, img_emb: torch.Tensor, rad_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rev1 task 5c: zero each branch independently, per-sample, with prob
        modality_dropout — training only, so eval/inference is unaffected."""
        if not self.training or self.modality_dropout <= 0.0:
            return img_emb, rad_emb
        keep_img = (torch.rand(img_emb.shape[0], 1, device=img_emb.device) >= self.modality_dropout).to(img_emb.dtype)
        keep_rad = (torch.rand(rad_emb.shape[0], 1, device=rad_emb.device) >= self.modality_dropout).to(rad_emb.dtype)
        return img_emb * keep_img, rad_emb * keep_rad

    def _branch_embeddings(self, img: torch.Tensor, rad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img_emb = self.img_proj(self.cnn_branch(self._maybe_resize(img)))
        rad_emb = self.rad_branch(rad)
        if self.fusion_arm == "branch_norm":
            img_emb = nn.functional.normalize(img_emb, p=2, dim=1)
            rad_emb = nn.functional.normalize(rad_emb, p=2, dim=1)
        self.last_branch_norms = {
            "img_norm": img_emb.detach().norm(dim=1).mean().item(),
            "rad_norm": rad_emb.detach().norm(dim=1).mean().item(),
        }
        logger.debug(
            "fusion_arm=%s branch norms: img=%.4f rad=%.4f",
            self.fusion_arm, self.last_branch_norms["img_norm"], self.last_branch_norms["rad_norm"],
        )
        # aux heads (5c) classify from the pre-dropout embedding — each branch
        # must learn to predict the label from itself alone.
        self.last_img_emb, self.last_rad_emb = img_emb, rad_emb
        return self._apply_modality_dropout(img_emb, rad_emb)

    def _gmu_fuse(self, img: torch.Tensor, rad: torch.Tensor) -> torch.Tensor:
        v = self.cnn_branch(self._maybe_resize(img))
        h_v = torch.tanh(self.gmu_img_transform(v))
        h_t = torch.tanh(self.gmu_rad_transform(rad))
        z = torch.sigmoid(self.gmu_gate(torch.cat([v, rad], dim=1)))
        self.last_branch_norms = {
            "img_gate": z.detach().mean().item(),
            "rad_gate": (1 - z).detach().mean().item(),
        }
        logger.debug(
            "fusion_arm=gmu gate activations: img=%.4f rad=%.4f",
            self.last_branch_norms["img_gate"], self.last_branch_norms["rad_gate"],
        )
        self.last_img_emb, self.last_rad_emb = h_v, h_t
        h_v, h_t = self._apply_modality_dropout(h_v, h_t)
        return z * h_v + (1 - z) * h_t

    def aux_logits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Rev1 task 5c: per-branch classification logits from the last forward's
        stored (pre-dropout) embeddings. Only valid when aux_loss_weight > 0
        (that's when the aux heads exist) and after a forward() call."""
        if self.aux_loss_weight <= 0.0:
            raise RuntimeError("aux_logits() requires aux_loss_weight > 0 at construction")
        if self.last_img_emb is None or self.last_rad_emb is None:
            raise RuntimeError("aux_logits() called before any forward() pass")
        return self.aux_img_head(self.last_img_emb), self.aux_rad_head(self.last_rad_emb)

    def forward(self, img: torch.Tensor, rad: torch.Tensor) -> torch.Tensor:
        if self.fusion_arm == "gmu":
            return self.classifier(self._gmu_fuse(img, rad))
        img_emb, rad_emb = self._branch_embeddings(img, rad)
        return self.classifier(torch.cat([img_emb, rad_emb], dim=1))

    def get_cnn_embedding(self, img: torch.Tensor) -> torch.Tensor:
        v = self.cnn_branch(self._maybe_resize(img))
        if self.fusion_arm == "gmu":
            return torch.tanh(self.gmu_img_transform(v))
        emb = self.img_proj(v)
        if self.fusion_arm == "branch_norm":
            emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb

    def get_radiomic_embedding(self, rad: torch.Tensor) -> torch.Tensor:
        if self.fusion_arm == "gmu":
            return torch.tanh(self.gmu_rad_transform(rad))
        emb = self.rad_branch(rad)
        if self.fusion_arm == "branch_norm":
            emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb
