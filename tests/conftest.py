"""Shared pytest fixtures."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_labels_df():
    """Small labels DataFrame for unit tests (no LIDC data needed)."""
    rows = []
    for i in range(20):
        pid = f"LIDC-{i // 4:04d}"
        rows.append({
            "patient_id": pid,
            "scan_id": i,
            "nodule_idx": i % 4,
            "label": i % 2,
            "mask_path": f"/tmp/mask_{i}.npy",
            "patch_path": f"/tmp/patch_{i}.npy",
            "median_rating": float(1 + i % 4),
            "n_annotations": 4,
            "centroid_z": 32.0,
            "centroid_y": 32.0,
            "centroid_x": 32.0,
            "fold": i % 5,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def dummy_radiomic_features(dummy_labels_df):
    """Synthetic radiomic feature DataFrame aligned to dummy_labels_df."""
    n = len(dummy_labels_df)
    feat_names = [f"original_glcm_feat_{i}" for i in range(10)] + \
                 [f"original_shape_feat_{i}" for i in range(5)]
    feat_data = np.random.randn(n, len(feat_names))
    df = pd.DataFrame(feat_data, columns=feat_names)
    df["patient_id"] = dummy_labels_df["patient_id"].values
    df["nodule_idx"] = dummy_labels_df["nodule_idx"].values
    df["label"] = dummy_labels_df["label"].values
    df["fold"] = dummy_labels_df["fold"].values
    return df
