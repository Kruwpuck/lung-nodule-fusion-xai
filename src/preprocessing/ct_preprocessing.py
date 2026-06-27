"""CT preprocessing: HU windowing, isotropic resampling, 2.5D patch extraction."""
from __future__ import annotations

import numpy as np
# SimpleITK imported lazily inside functions that need it


HU_WINDOW_LUNG = (-1000, 400)  # standard lung window


def clip_and_normalize_hu(
    volume: np.ndarray,
    hu_min: float = HU_WINDOW_LUNG[0],
    hu_max: float = HU_WINDOW_LUNG[1],
) -> np.ndarray:
    """Clip HU values to lung window and normalize to [0, 1]."""
    volume = np.clip(volume, hu_min, hu_max)
    return (volume - hu_min) / (hu_max - hu_min)


def resample_to_isotropic(
    image,
    new_spacing: list = [1.0, 1.0, 1.0],
    interpolator=None,
):
    """Resample SimpleITK image to isotropic voxel spacing."""
    import SimpleITK as sitk  # lazy: only needed when called with a sitk.Image
    if interpolator is None:
        interpolator = sitk.sitkBSpline

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)


def extract_patch_3d(
    volume: np.ndarray,
    centroid: tuple[float, float, float],
    patch_size: tuple[int, int, int] = (64, 64, 64),
) -> np.ndarray:
    """Crop 3D patch of patch_size centered at centroid (z, y, x)."""
    z, y, x = [int(round(c)) for c in centroid]
    dz, dy, dx = [s // 2 for s in patch_size]
    vol_shape = volume.shape

    # compute slice bounds with clamping
    z0, z1 = max(0, z - dz), min(vol_shape[0], z + dz)
    y0, y1 = max(0, y - dy), min(vol_shape[1], y + dy)
    x0, x1 = max(0, x - dx), min(vol_shape[2], x + dx)

    patch = np.zeros(patch_size, dtype=volume.dtype)
    pz0 = dz - (z - z0)
    py0 = dy - (y - y0)
    px0 = dx - (x - x0)
    patch[pz0:pz0 + (z1 - z0), py0:py0 + (y1 - y0), px0:px0 + (x1 - x0)] = \
        volume[z0:z1, y0:y1, x0:x1]
    return patch


def extract_2_5d_patch(
    volume: np.ndarray,
    centroid_z: int,
    centroid_y: int,
    centroid_x: int,
    patch_xy: int = 64,
    n_slices: int = 3,
) -> np.ndarray:
    """Extract 2.5D patch: n_slices adjacent axial slices stacked as channels.

    Returns array of shape (n_slices, patch_xy, patch_xy).
    """
    half = patch_xy // 2
    half_s = n_slices // 2
    H, W = volume.shape[1], volume.shape[2]

    y0 = max(0, centroid_y - half)
    x0 = max(0, centroid_x - half)
    y1 = min(H, y0 + patch_xy)
    x1 = min(W, x0 + patch_xy)

    slices = []
    for offset in range(-half_s, half_s + 1):
        z = centroid_z + offset
        z = max(0, min(volume.shape[0] - 1, z))
        sl = np.zeros((patch_xy, patch_xy), dtype=volume.dtype)
        sl[:y1 - y0, :x1 - x0] = volume[z, y0:y1, x0:x1]
        slices.append(sl)

    return np.stack(slices, axis=0)  # (n_slices, H, W)
