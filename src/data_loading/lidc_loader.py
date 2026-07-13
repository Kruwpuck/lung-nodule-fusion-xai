"""LIDC-IDRI data loading: DICOM -> NIfTI, consensus masks, binary labels, 5-fold CV splits."""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
# pylidc imported lazily inside functions that need it
# from pylidc.utils import consensus  (lazy)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def _median_label(anns: list, exclude_score: int = 3) -> Optional[int]:
    """Binary label from median of annotation malignancy ratings.

    Returns None for median == exclude_score (ambiguous).
    Returns 0 for benign, 1 for malignant.
    """
    ratings = [a.malignancy for a in anns]
    med = np.median(ratings)
    if med == exclude_score:
        return None
    return int(med > exclude_score)


def build_nodule_dataset(
    lidc_path: str = "data/raw/LIDC-IDRI",
    interim_path: str = "data/interim",
    consensus_level: float = 0.5,
    exclude_score: int = 3,
    min_annotations: int = 1,
    pad: list = [(20, 20), (20, 20), (0, 0)],
) -> pd.DataFrame:
    """Query all LIDC scans, extract nodules with consensus masks and binary labels.

    Returns DataFrame with columns:
        patient_id, scan_id, nodule_idx, label, mask_path, patch_path,
        median_rating, n_annotations, centroid_x, centroid_y, centroid_z
    """
    import pylidc as pl  # lazy: only needed at LIDC query time
    from pylidc.utils import consensus

    os.makedirs(interim_path, exist_ok=True)
    records = []

    scans = pl.query(pl.Scan).all()
    logger.info("Total scans: %d", len(scans))

    for scan in scans:
        pid = scan.patient_id
        try:
            vol = scan.to_volume(verbose=False)  # (Z, Y, X) numpy array in HU
        except Exception as e:
            logger.warning("Failed to load volume for %s: %s", pid, e)
            continue

        nodule_groups = scan.cluster_annotations()

        for nidx, anns in enumerate(nodule_groups):
            if len(anns) < min_annotations:
                continue

            label = _median_label(anns, exclude_score)
            if label is None:
                continue

            try:
                cmask, cbbox, _ = consensus(anns, clevel=consensus_level, pad=pad)
            except Exception as e:
                logger.warning("Consensus failed %s nodule %d: %s", pid, nidx, e)
                continue

            patch = vol[cbbox]
            centroid = np.array(cmask.nonzero()).mean(axis=1)

            nodule_dir = Path(interim_path) / pid / f"nodule_{nidx:03d}"
            nodule_dir.mkdir(parents=True, exist_ok=True)
            mask_path = str(nodule_dir / "mask.npy")
            patch_path = str(nodule_dir / "patch.npy")
            np.save(mask_path, cmask.astype(np.uint8))
            np.save(patch_path, patch.astype(np.float32))

            records.append({
                "patient_id": pid,
                "scan_id": scan.id,
                "nodule_idx": nidx,
                "label": label,
                "mask_path": mask_path,
                "patch_path": patch_path,
                "median_rating": float(np.median([a.malignancy for a in anns])),
                "n_annotations": len(anns),
                "centroid_z": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_x": float(centroid[2]),
            })

    df = pd.DataFrame(records)
    logger.info(
        "Dataset: %d nodules | %d malignant | %d benign",
        len(df), (df.label == 1).sum(), (df.label == 0).sum(),
    )
    return df


def add_kfold_splits(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Patient-level stratified k-fold. All nodules from one patient stay in same fold."""
    patient_df = (
        df.groupby("patient_id")
        .agg(patient_label=("label", lambda x: int(x.mode()[0])))
        .reset_index()
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    patient_df["fold"] = -1

    for fold_idx, (_, val_idx) in enumerate(
        skf.split(patient_df["patient_id"], patient_df["patient_label"])
    ):
        patient_df.loc[val_idx, "fold"] = fold_idx

    return df.merge(patient_df[["patient_id", "fold"]], on="patient_id", how="left")


def load_and_split(
    lidc_path: str = "data/raw/LIDC-IDRI",
    interim_path: str = "data/interim",
    processed_path: str = "data/processed",
    n_folds: int = 5,
    seed: int = 42,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Full pipeline: build nodule dataset + k-fold splits. Caches to processed/labels.csv."""
    labels_csv = Path(processed_path) / "labels.csv"

    if labels_csv.exists() and not force_rebuild:
        logger.info("Loading cached labels from %s", labels_csv)
        return pd.read_csv(labels_csv)

    df = build_nodule_dataset(lidc_path=lidc_path, interim_path=interim_path)
    df = add_kfold_splits(df, n_folds=n_folds, seed=seed)

    os.makedirs(processed_path, exist_ok=True)
    df.to_csv(labels_csv, index=False)
    logger.info("Saved labels to %s", labels_csv)
    return df
