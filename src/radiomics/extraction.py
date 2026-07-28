"""PyRadiomics feature extraction from CT patches and consensus masks."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

logger = logging.getLogger(__name__)

FEATURE_PREFIX_KEEP = ("original_", "log-", "wavelet-")
INFO_PREFIX_DROP = "diagnostics_"


def _load_extractor(params_yaml: str = "configs/radiomics_params.yaml"):
    """Load PyRadiomics extractor. Lazy import to avoid hard dependency at module level."""
    try:
        from radiomics import featureextractor
    except ImportError as e:
        raise ImportError("pyradiomics not installed. Run: pip install pyradiomics") from e
    return featureextractor.RadiomicsFeatureExtractor(params_yaml)


def extract_features_from_arrays(
    patch: np.ndarray,
    mask: np.ndarray,
    params_yaml: str = "configs/radiomics_params.yaml",
) -> dict[str, float]:
    """Extract radiomic features from numpy patch and mask arrays.

    Both arrays must be in (Z, Y, X) order and float32/int dtype.
    Mask must be binary (0/1) with label value 1 marking the nodule.
    """
    extractor = _load_extractor(params_yaml)

    image_sitk = sitk.GetImageFromArray(patch.astype(np.float32))
    mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
    # Explicit spacing required — GetImageFromArray defaults to (1,1,1) regardless
    # of true voxel size, making resampledPixelSpacing config a silent no-op.
    image_sitk.SetSpacing((1.0, 1.0, 1.0))
    mask_sitk.SetSpacing((1.0, 1.0, 1.0))

    result = extractor.execute(image_sitk, mask_sitk)

    # filter out diagnostic/info keys; keep only feature values
    features = {
        k: float(v)
        for k, v in result.items()
        if k.startswith(FEATURE_PREFIX_KEEP) and not k.startswith(INFO_PREFIX_DROP)
    }
    return features


def extract_dataset_features(
    labels_df: pd.DataFrame,
    params_yaml: str = "configs/radiomics_params.yaml",
    output_parquet: str = "data/processed/radiomic_features.parquet",
    force_rebuild: bool = False,
    incremental: bool = False,
) -> pd.DataFrame:
    """Extract features for all nodules in labels_df.

    Reads patch_path and mask_path columns from labels_df.
    Saves/loads cache from output_parquet.

    incremental: if True and a cache exists, only extract features for
        (patient_id, nodule_idx) pairs not already present in the cache, then
        append and re-save — avoids re-running PyRadiomics on nodules whose
        patch/mask didn't change (e.g. the ~1391 nodules extracted previously).
        Ignored if force_rebuild is True.

    Returns DataFrame with one row per nodule, columns = feature names.
    """
    cache_exists = Path(output_parquet).exists()

    if cache_exists and not force_rebuild and not incremental:
        logger.info("Loading cached features from %s", output_parquet)
        return pd.read_parquet(output_parquet)

    cached_df = None
    todo_df = labels_df
    if cache_exists and incremental and not force_rebuild:
        cached_df = pd.read_parquet(output_parquet)
        done_keys = set(zip(cached_df["patient_id"], cached_df["nodule_idx"]))
        todo_mask = ~labels_df.apply(
            lambda r: (r["patient_id"], r["nodule_idx"]) in done_keys, axis=1
        )
        todo_df = labels_df[todo_mask]
        logger.info(
            "Incremental extraction: %d cached, %d new to extract",
            len(cached_df), len(todo_df),
        )

    extractor = _load_extractor(params_yaml)
    rows = []

    for _, row in todo_df.iterrows():
        patch = np.load(row["patch_path"])
        mask = np.load(row["mask_path"])

        try:
            feats = extract_features_from_arrays(patch, mask, params_yaml)
        except Exception as e:
            logger.warning("Feature extraction failed for %s nodule %s: %s",
                           row["patient_id"], row["nodule_idx"], e)
            continue

        feats["patient_id"] = row["patient_id"]
        feats["nodule_idx"] = row["nodule_idx"]
        feats["label"] = row["label"]
        feats["fold"] = row["fold"]
        rows.append(feats)

    new_df = pd.DataFrame(rows)
    df = pd.concat([cached_df, new_df], ignore_index=True) if cached_df is not None else new_df
    os.makedirs(Path(output_parquet).parent, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    logger.info("Saved %d feature rows to %s", len(df), output_parquet)
    return df
