"""PyTorch Dataset for 2.5D and 3D nodule patches."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class NoduleDataset2_5D(Dataset):
    """2.5D nodule dataset: n_slices adjacent axial slices stacked as channels.

    Expects labels_df with columns: patch_path, label.
    patch_path should point to .npy files with shape (Z, Y, X) in HU.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        patch_size: int = 64,
        n_slices: int = 3,
        augment: bool = False,
        hu_min: float = -1000.0,
        hu_max: float = 400.0,
        target_col: str = "label",
        target_dtype: str = "long",
    ) -> None:
        """target_col/target_dtype select the training target per task:
        "label" (long, binary) | "median_rating" (float, ordinal) | "grade3"/"grade4" (long).
        """
        self.df = df.reset_index(drop=True)
        self.patch_size = patch_size
        self.n_slices = n_slices
        self.augment = augment
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.target_col = target_col
        self.target_dtype = target_dtype

    def __len__(self) -> int:
        return len(self.df)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        arr = np.clip(arr, self.hu_min, self.hu_max)
        return (arr - self.hu_min) / (self.hu_max - self.hu_min)

    def _extract_2_5d(self, volume: np.ndarray) -> np.ndarray:
        """Extract center slice ± half_slices, return (n_slices, H, W).

        Patches are pre-centered and pre-sized by lidc_loader — just stack slices.
        """
        Z, H, W = volume.shape
        cz = Z // 2
        half = self.n_slices // 2
        slices = []
        for offset in range(-half, half + 1):
            z = max(0, min(Z - 1, cz + offset))
            slices.append(volume[z])
        return np.stack(slices, axis=0)  # (n_slices, H, W)

    def _augment(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.5:
            tensor = TF.hflip(tensor)
        if random.random() > 0.5:
            tensor = TF.vflip(tensor)
        angle = random.uniform(-15, 15)
        tensor = TF.rotate(tensor, angle)
        return tensor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        volume = np.load(row["patch_path"]).astype(np.float32)
        volume = self._normalize(volume)
        patch_2_5d = self._extract_2_5d(volume)  # (n_slices, H, W)
        tensor = torch.from_numpy(patch_2_5d)
        if self.augment:
            tensor = self._augment(tensor)
        if self.target_dtype == "float":
            target = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        else:
            target = torch.tensor(int(row[self.target_col]), dtype=torch.long)
        return tensor, target


class NoduleDataset3D(Dataset):
    """3D nodule dataset: full volumetric patches for 3D CNN."""

    def __init__(
        self,
        df: pd.DataFrame,
        patch_size: int = 64,
        augment: bool = False,
        hu_min: float = -1000.0,
        hu_max: float = 400.0,
        target_col: str = "label",
        target_dtype: str = "long",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.patch_size = patch_size
        self.augment = augment
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.target_col = target_col
        self.target_dtype = target_dtype

    def __len__(self) -> int:
        return len(self.df)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        arr = np.clip(arr, self.hu_min, self.hu_max)
        return (arr - self.hu_min) / (self.hu_max - self.hu_min)

    def _center_crop_3d(self, volume: np.ndarray) -> np.ndarray:
        p = self.patch_size
        out = np.zeros((p, p, p), dtype=np.float32)
        Z, Y, X = volume.shape
        cz, cy, cx = Z // 2, Y // 2, X // 2
        half = p // 2
        z0, z1 = max(0, cz - half), min(Z, cz + half)
        y0, y1 = max(0, cy - half), min(Y, cy + half)
        x0, x1 = max(0, cx - half), min(X, cx + half)
        crop = volume[z0:z1, y0:y1, x0:x1]
        out[:crop.shape[0], :crop.shape[1], :crop.shape[2]] = crop
        return out

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        volume = np.load(row["patch_path"]).astype(np.float32)
        volume = self._normalize(volume)
        patch = self._center_crop_3d(volume)  # (D, H, W)
        tensor = torch.from_numpy(patch).unsqueeze(0)  # (1, D, H, W)
        if self.target_dtype == "float":
            target = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        else:
            target = torch.tensor(int(row[self.target_col]), dtype=torch.long)
        return tensor, target
