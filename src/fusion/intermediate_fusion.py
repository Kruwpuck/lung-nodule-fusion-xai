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

    for img, rad, labels in loader:
        img = img.to(device)
        rad = rad.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device.type):
                logits = model(img, rad)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(img, rad)
            loss = criterion(logits, labels)
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
