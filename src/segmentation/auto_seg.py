"""Route B: automated segmentation via lungmask / nnU-Net."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def segment_lungs_lungmask(ct_path: str) -> np.ndarray:
    """Lung mask via lungmask (fast, no training required).

    Returns binary lung mask (Z, Y, X) as uint8.
    """
    try:
        from lungmask import LMInferer
        import SimpleITK as sitk
    except ImportError as e:
        raise ImportError("lungmask not installed. Run: pip install lungmask") from e

    inferer = LMInferer()
    image = sitk.ReadImage(ct_path)
    seg = inferer.apply(image)  # returns numpy array (Z, Y, X)
    return (seg > 0).astype(np.uint8)


def segment_nodule_nnunet(
    ct_path: str,
    model_folder: str,
    output_path: str,
    folds: tuple[int, ...] = (0,),
) -> np.ndarray:
    """Nodule segmentation via nnU-Net v2 (requires trained model).

    Uses nnUNetv2 predict API. Returns binary nodule mask (Z, Y, X).
    """
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        import SimpleITK as sitk
    except ImportError as e:
        raise ImportError("nnunetv2 not installed. Run: pip install nnunetv2") from e

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device="cuda",
        verbose=False,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=folds,
        checkpoint_name="checkpoint_best.pth",
    )
    predictor.predict_from_files(
        [[ct_path]],
        [output_path],
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )

    import SimpleITK as sitk
    seg = sitk.GetArrayFromImage(sitk.ReadImage(output_path))
    return (seg > 0).astype(np.uint8)


def compare_route_a_b(
    route_a_mask: np.ndarray,
    route_b_mask: np.ndarray,
) -> dict[str, float]:
    """Compute Dice and volume difference between Route A and B masks."""
    from src.segmentation.consensus import dice_score

    dice = dice_score(route_a_mask, route_b_mask)
    vol_a = route_a_mask.sum()
    vol_b = route_b_mask.sum()
    vol_diff_pct = abs(vol_a - vol_b) / (vol_a + 1e-6) * 100

    return {
        "dice": float(dice),
        "vol_a_voxels": int(vol_a),
        "vol_b_voxels": int(vol_b),
        "vol_diff_pct": float(vol_diff_pct),
    }
