"""Route A: LIDC expert consensus masks via pylidc."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def get_consensus_mask(
    anns: list,
    clevel: float = 0.5,
    pad: tuple = ((20, 20), (20, 20), (0, 0)),
) -> tuple[np.ndarray, tuple, list]:
    """Compute 50% consensus mask from list of pylidc Annotation objects.

    Returns:
        cmask: boolean 3D consensus mask
        cbbox: bounding box slices into the full CT volume
        individual_masks: list of per-annotation boolean masks (same bbox)
    """
    from pylidc.utils import consensus  # lazy: pylidc only needed at runtime
    cmask, cbbox, masks = consensus(anns, clevel=clevel, pad=pad)
    return cmask.astype(np.uint8), cbbox, masks


def staple_consensus(anns: list, threshold: float = 0.5) -> np.ndarray:
    """STAPLE probabilistic consensus (alternative to simple majority vote).

    Requires SimpleITK. Returns binary mask at given probability threshold.
    """
    import SimpleITK as sitk

    # Build individual binary masks from first annotation's bbox
    ref_ann = anns[0]
    ref_mask_sitk = sitk.GetImageFromArray(ref_ann.boolean_mask().astype(np.uint8))

    mask_list = []
    for ann in anns:
        m = ann.boolean_mask().astype(np.uint8)
        mask_sitk = sitk.GetImageFromArray(m)
        mask_list.append(mask_sitk)

    staple_filter = sitk.STAPLEImageFilter()
    staple_filter.SetForegroundValue(1)
    prob_map = staple_filter.Execute(mask_list)
    prob_array = sitk.GetArrayFromImage(prob_map)
    return (prob_array >= threshold).astype(np.uint8)


def compute_icc(features_a: np.ndarray, features_b: np.ndarray) -> np.ndarray:
    """Intraclass correlation coefficient (ICC 2,1) between two feature sets.

    features_a, features_b: (n_nodules, n_features) arrays.
    Returns ICC per feature column (n_features,).
    """
    assert features_a.shape == features_b.shape
    n, k = features_a.shape[0], 2  # n subjects, k=2 raters

    # Stack rater measurements: shape (n, 2, F)
    X = np.stack([features_a, features_b], axis=1)
    grand_mean = X.mean(axis=(0, 1))  # (F,)
    subject_mean = X.mean(axis=1)     # (n, F)

    # Sum of squares
    ss_between = k * np.sum((subject_mean - grand_mean) ** 2, axis=0)
    ss_error = np.sum((X - subject_mean[:, None, :]) ** 2, axis=(0, 1))

    ms_between = ss_between / (n - 1)
    ms_error = ss_error / (n * (k - 1))

    # ICC(2,1): two-way random, absolute agreement, single measures
    icc = (ms_between - ms_error) / (ms_between + (k - 1) * ms_error + 1e-10)
    return icc


def dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Dice similarity coefficient between two binary masks."""
    intersection = (mask_a & mask_b).sum()
    union = mask_a.sum() + mask_b.sum()
    if union == 0:
        return 1.0
    return 2.0 * intersection / union
