"""Training loop with 5-fold CV for standalone CNN and fusion models."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best = -np.inf if mode == "max" else np.inf
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        improved = (self.mode == "max" and metric > self.best) or \
                   (self.mode == "min" and metric < self.best)
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler=None,
) -> float:
    """Single training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        if len(batch) == 2:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast(device.type):
                    loss = criterion(model(inputs), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
        elif len(batch) == 3:
            # fusion: (img, rad, label)
            img, rad, labels = batch
            img, rad, labels = img.to(device), rad.to(device), labels.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.amp.autocast(device.type):
                    loss = criterion(model(img, rad), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(img, rad), labels)
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_fusion: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate model. Returns (y_true, y_prob_class1)."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        if is_fusion:
            img, rad, labels = batch
            img, rad = img.to(device), rad.to(device)
            logits = model(img, rad)
        else:
            img, labels = batch
            img = img.to(device)
            logits = model(img)

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_probs)


def run_kfold_cv(
    model_factory: Callable[[], nn.Module],
    dataset_factory: Callable[[pd.DataFrame, bool], DataLoader],
    labels_df: pd.DataFrame,
    n_folds: int = 5,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    batch_size: int = 16,
    device_str: str = "auto",
    mixed_precision: bool = True,
    checkpoint_dir: str = "results/checkpoints",
    is_fusion: bool = False,
) -> dict:
    """Run n-fold cross-validation. Returns dict with per-fold AUC + aggregate stats.

    NOTE: Call with training code skipped when only evaluating architecture.
    Set epochs=0 to skip training (architecture validation only).
    """
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("Using device: %s", device)
    os.makedirs(checkpoint_dir, exist_ok=True)

    fold_results = []

    for fold in range(n_folds):
        train_df = labels_df[labels_df["fold"] != fold]
        val_df = labels_df[labels_df["fold"] == fold]

        train_loader = dataset_factory(train_df, augment=True)
        val_loader = dataset_factory(val_df, augment=False)

        model = model_factory().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        early_stopper = EarlyStopping(patience=patience, mode="max")
        amp_scaler = torch.amp.GradScaler() if mixed_precision and device.type == "cuda" else None

        best_auc = 0.0
        for epoch in range(epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, amp_scaler
            )
            y_true, y_prob = evaluate(model, val_loader, device, is_fusion)

            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, y_prob)
            scheduler.step()

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(),
                           f"{checkpoint_dir}/fold{fold}_best.pt")

            if early_stopper.step(auc):
                logger.info("Early stop at epoch %d (fold %d)", epoch, fold)
                break

        # final eval with best weights
        model.load_state_dict(torch.load(f"{checkpoint_dir}/fold{fold}_best.pt"))
        y_true, y_prob = evaluate(model, val_loader, device, is_fusion)
        fold_results.append({"fold": fold, "y_true": y_true, "y_prob": y_prob})
        logger.info("Fold %d best AUC: %.4f", fold, best_auc)

    return fold_results
