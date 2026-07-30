"""Intermediate fusion training utilities (end-to-end CNN + radiomics)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.fusion_net import FusionNet

logger = logging.getLogger(__name__)


class RadiomicDataset(Dataset):
    """Dataset yielding (image_tensor, radiomic_tensor, label) for fusion training."""

    def __init__(
        self,
        image_dataset,          # NoduleDataset2_5D or 3D — must return (img, label)
        radiomic_features: np.ndarray,  # (N, n_features) already scaled
    ) -> None:
        assert len(image_dataset) == len(radiomic_features)
        self.image_dataset = image_dataset
        self.radiomic_features = torch.from_numpy(radiomic_features.astype(np.float32))

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, idx: int):
        img, label = self.image_dataset[idx]
        rad = self.radiomic_features[idx]
        return img, rad, label


def train_fusion_epoch(
    model: FusionNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,  # torch.cuda.amp.GradScaler or None
) -> float:
    """One training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    aux_weight = getattr(model, "aux_loss_weight", 0.0)

    def _loss(img, rad, labels):
        # Rev1 task 5c: auxiliary per-branch losses, config-driven via
        # model.aux_loss_weight (0.0 -> identical to pre-5c behavior).
        logits = model(img, rad)
        loss = criterion(logits, labels)
        if aux_weight > 0:
            aux_img_logits, aux_rad_logits = model.aux_logits()
            loss = loss + aux_weight * (criterion(aux_img_logits, labels) + criterion(aux_rad_logits, labels))
        return loss

    for img, rad, labels in loader:
        img = img.to(device)
        rad = rad.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device.type):
                loss = _loss(img, rad, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = _loss(img, rad, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_fusion(
    model: FusionNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate fusion model. Returns (y_true, y_prob_malignant)."""
    model.eval()
    all_probs, all_labels = [], []

    for img, rad, labels in loader:
        img, rad = img.to(device), rad.to(device)
        logits = model(img, rad)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_probs)
