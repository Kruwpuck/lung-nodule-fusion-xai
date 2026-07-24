"""LIDC-IDRI data loading: DICOM -> fixed-window patches, ordinal malignancy
targets (median_rating, binary label, grade3), 5-fold CV splits.

Critical note: pylidc.Scan.to_volume() returns (Y, X, Z) — Z is the LAST axis.
All functions here transpose to (Z, Y, X) before processing.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def _malignancy_targets(anns: list) -> dict:
    """All malignancy targets for a nodule, derived from its annotation group.

    No nodule is ever dropped here (median==3 kept, label=-1 marks it as
    indeterminate for consumers that need to mask it out of binary loss/eval).
    """
    ratings = [float(a.malignancy) for a in anns]
    med = float(np.median(ratings))
    return {
        "median_rating": med,                                   # ordinal target, 1.0-5.0
        "label": int(med > 3) if med != 3 else -1,               # binary; -1 = indeterminate
        "grade3": 0 if med < 3 else (1 if med == 3 else 2),      # 0=benign 1=indeterminate 2=malignant
        "grade4": (1 if med < 3 else (2 if med == 3 else 3)),    # 0=no-nodule(reserved) 1=benign 2=indeterminate 3=malignant
        "n_annotations": len(anns),
        "rating_std": float(np.std(ratings)),
    }


def _crop_fixed_window(
    vol: np.ndarray,
    center_zyx: Tuple[int, int, int],
    half_vox: Tuple[int, int, int],
    fill_value: float = -1000.0,
    dtype=np.float32,
) -> np.ndarray:
    """Crop fixed-size window from vol (Z,Y,X). OOB filled with fill_value."""
    Z, Y, X = vol.shape
    cz, cy, cx = center_zyx
    hz, hy, hx = half_vox
    out = np.full((2 * hz, 2 * hy, 2 * hx), fill_value, dtype=dtype)

    sz0, sz1 = cz - hz, cz + hz
    sy0, sy1 = cy - hy, cy + hy
    sx0, sx1 = cx - hx, cx + hx

    vs_z0, vs_z1 = max(0, sz0), min(Z, sz1)
    vs_y0, vs_y1 = max(0, sy0), min(Y, sy1)
    vs_x0, vs_x1 = max(0, sx0), min(X, sx1)

    dz0 = vs_z0 - sz0;  dz1 = dz0 + (vs_z1 - vs_z0)
    dy0 = vs_y0 - sy0;  dy1 = dy0 + (vs_y1 - vs_y0)
    dx0 = vs_x0 - sx0;  dx1 = dx0 + (vs_x1 - vs_x0)

    if vs_z1 > vs_z0 and vs_y1 > vs_y0 and vs_x1 > vs_x0:
        out[dz0:dz1, dy0:dy1, dx0:dx1] = vol[vs_z0:vs_z1, vs_y0:vs_y1, vs_x0:vs_x1].astype(dtype)
    return out


def _paste_local(dst: np.ndarray, src: np.ndarray, offset_zyx: Tuple[int, int, int]) -> None:
    """Paste src into dst at offset_zyx, clipping to dst bounds (in-place)."""
    oz, oy, ox = offset_zyx
    DZ, DY, DX = dst.shape
    SZ, SY, SX = src.shape

    sz0 = max(0, -oz);   sz1 = min(SZ, DZ - oz)
    sy0 = max(0, -oy);   sy1 = min(SY, DY - oy)
    sx0 = max(0, -ox);   sx1 = min(SX, DX - ox)
    dz0 = max(0, oz);    dz1 = min(DZ, oz + SZ)
    dy0 = max(0, oy);    dy1 = min(DY, oy + SY)
    dx0 = max(0, ox);    dx1 = min(DX, ox + SX)

    if sz1 > sz0 and sy1 > sy0 and sx1 > sx0:
        dst[dz0:dz1, dy0:dy1, dx0:dx1] = src[sz0:sz1, sy0:sy1, sx0:sx1]


def _crop_and_resample_nodule(
    vol_zyx: np.ndarray,
    cmask_yxz: np.ndarray,
    cbbox_yxz: tuple,
    native_spacing_zyx: Tuple[float, float, float],
    window_mm: Tuple[float, float, float] = (64.0, 64.0, 16.0),
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop fixed physical window centered on nodule, resample to target_spacing.

    Args:
        vol_zyx: full scan volume (Z, Y, X) in HU
        cmask_yxz: consensus mask in pylidc bbox coords (Y, X, Z)
        cbbox_yxz: tuple of 3 slices (y_slice, x_slice, z_slice) — pylidc order
        native_spacing_zyx: (sz, sy, sx) in mm/voxel
        window_mm: (y_mm, x_mm, z_mm) physical crop size
        target_spacing: (sz, sy, sx) mm/voxel after resample

    Returns:
        (patch_zyx, mask_zyx) both exactly (tz, ty, tx) voxels
    """
    from scipy.ndimage import zoom

    sz, sy, sx = native_spacing_zyx

    # Reorder mask and bbox from pylidc (Y,X,Z) to (Z,Y,X)
    cmask_zyx = cmask_yxz.transpose(2, 0, 1)
    cbbox_zyx = (cbbox_yxz[2], cbbox_yxz[0], cbbox_yxz[1])

    # Centroid of mask in global (Z, Y, X) voxel coords
    nz = np.array(cmask_zyx.nonzero())  # (3, N)
    if nz.shape[1] == 0:
        raise ValueError("Empty mask — no nonzero voxels")
    centroid_in_bbox = nz.mean(axis=1)
    cz_g = cbbox_zyx[0].start + centroid_in_bbox[0]
    cy_g = cbbox_zyx[1].start + centroid_in_bbox[1]
    cx_g = cbbox_zyx[2].start + centroid_in_bbox[2]
    center_zyx = (int(round(cz_g)), int(round(cy_g)), int(round(cx_g)))

    # Half-window in native voxels — window_mm order is (y, x, z)
    wy_mm, wx_mm, wz_mm = window_mm
    hz_vox = max(1, int(round(wz_mm / 2 / sz)))
    hy_vox = max(1, int(round(wy_mm / 2 / sy)))
    hx_vox = max(1, int(round(wx_mm / 2 / sx)))
    half_vox = (hz_vox, hy_vox, hx_vox)

    # Crop patch (fill OOB with -1000 HU = air)
    patch_native = _crop_fixed_window(vol_zyx, center_zyx, half_vox, fill_value=-1000.0)

    # Place mask in same crop window
    mask_native = np.zeros(patch_native.shape, dtype=np.float32)
    bbox_start_zyx = (cbbox_zyx[0].start, cbbox_zyx[1].start, cbbox_zyx[2].start)
    crop_start_zyx = (center_zyx[0] - hz_vox, center_zyx[1] - hy_vox, center_zyx[2] - hx_vox)
    offset_zyx = tuple(int(bbox_start_zyx[i] - crop_start_zyx[i]) for i in range(3))
    _paste_local(mask_native, cmask_zyx.astype(np.float32), offset_zyx)

    # Resample to target spacing
    zoom_factors = (sz / target_spacing[0], sy / target_spacing[1], sx / target_spacing[2])
    patch_rs = zoom(patch_native, zoom_factors, order=3)
    mask_rs  = zoom(mask_native,  zoom_factors, order=0)

    # Target voxel counts
    tz = int(round(wz_mm / target_spacing[0]))   # 16
    ty = int(round(wy_mm / target_spacing[1]))   # 64
    tx = int(round(wx_mm / target_spacing[2]))   # 64

    # Center-crop / pad resampled volume to exact target shape
    crs = tuple(s // 2 for s in patch_rs.shape)
    patch_out = _crop_fixed_window(patch_rs, crs, (tz // 2, ty // 2, tx // 2), fill_value=-1000.0)
    mask_out  = _crop_fixed_window(mask_rs,  crs, (tz // 2, ty // 2, tx // 2), fill_value=0.0)

    return patch_out[:tz, :ty, :tx], (mask_out[:tz, :ty, :tx] > 0.5).astype(np.uint8)


def crop_and_resample_point(
    vol_zyx: np.ndarray,
    center_zyx: Tuple[int, int, int],
    native_spacing_zyx: Tuple[float, float, float],
    window_mm: Tuple[float, float, float] = (64.0, 64.0, 16.0),
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Crop+resample the same fixed physical window as `_crop_and_resample_nodule`,
    but centered on an arbitrary voxel coordinate instead of a consensus mask
    centroid. Used for no-nodule negatives (LUNA16 candidates) so the CT patch
    preprocessing is byte-for-byte identical between positive and negative
    classes — only the crop *center* differs, not the pipeline.

    Returns patch_zyx of exactly (tz, ty, tx) voxels (no mask — negatives get
    a synthetic ROI for radiomics, built by the caller).
    """
    from scipy.ndimage import zoom

    sz, sy, sx = native_spacing_zyx
    wy_mm, wx_mm, wz_mm = window_mm

    hz_vox = max(1, int(round(wz_mm / 2 / sz)))
    hy_vox = max(1, int(round(wy_mm / 2 / sy)))
    hx_vox = max(1, int(round(wx_mm / 2 / sx)))
    half_vox = (hz_vox, hy_vox, hx_vox)

    patch_native = _crop_fixed_window(vol_zyx, center_zyx, half_vox, fill_value=-1000.0)

    zoom_factors = (sz / target_spacing[0], sy / target_spacing[1], sx / target_spacing[2])
    patch_rs = zoom(patch_native, zoom_factors, order=3)

    tz = int(round(wz_mm / target_spacing[0]))
    ty = int(round(wy_mm / target_spacing[1]))
    tx = int(round(wx_mm / target_spacing[2]))

    crs = tuple(s // 2 for s in patch_rs.shape)
    patch_out = _crop_fixed_window(patch_rs, crs, (tz // 2, ty // 2, tx // 2), fill_value=-1000.0)
    return patch_out[:tz, :ty, :tx]


def build_nodule_dataset(
    lidc_path: str = "data/raw/LIDC-IDRI",
    interim_path: str = "data/interim",
    consensus_level: float = 0.5,
    include_indeterminate: bool = True,
    min_annotations: int = 1,
    window_mm: Tuple[float, float, float] = (64.0, 64.0, 16.0),
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    skip_existing: bool = False,
) -> pd.DataFrame:
    """Query all LIDC scans, extract nodules with fixed-window crop and ordinal targets.

    include_indeterminate: if True (default), median==3 nodules are kept with
        label=-1 (indeterminate) instead of being dropped. Set False to
        reproduce the old binary-only behavior (median==3 excluded).
    skip_existing: if True, nodules whose patch.npy/mask.npy already exist on
        disk are not re-cropped (record is rebuilt from the saved mask +
        annotation targets only) — used to add newly-included median==3
        nodules without re-processing the ~1391 nodules already extracted.

    Returns DataFrame with columns:
        patient_id, scan_id, nodule_idx, label, mask_path, patch_path,
        median_rating, grade3, n_annotations, rating_std, centroid_z/y/x, spacing_z/y/x
    All patches saved as (Z, Y, X) numpy arrays of shape (16, 64, 64).
    """
    import pylidc as pl
    from pylidc.utils import consensus

    os.makedirs(interim_path, exist_ok=True)
    records = []

    scans = pl.query(pl.Scan).all()
    logger.info("Total scans: %d", len(scans))

    for scan in scans:
        pid = scan.patient_id
        vol = None  # lazily loaded — skip_existing nodules don't need it

        def _get_vol():
            nonlocal vol
            if vol is None:
                vol_yxz = scan.to_volume(verbose=False)  # (Y, X, Z) — Z LAST
                vol = vol_yxz.transpose(2, 0, 1)
            return vol

        ps = float(scan.pixel_spacing)
        sz = float(getattr(scan, "slice_spacing", None) or getattr(scan, "slice_thickness", 1.0))
        native_spacing = (sz, ps, ps)

        try:
            nodule_groups = scan.cluster_annotations()
        except Exception as e:
            logger.warning("cluster_annotations failed for %s: %s", pid, e)
            continue

        for nidx, anns in enumerate(nodule_groups):
            if len(anns) < min_annotations:
                continue

            targets = _malignancy_targets(anns)
            if not include_indeterminate and targets["label"] == -1:
                continue

            nodule_dir = Path(interim_path) / pid / f"nodule_{nidx:03d}"
            mask_path  = str(nodule_dir / "mask.npy")
            patch_path = str(nodule_dir / "patch.npy")

            if skip_existing and os.path.exists(mask_path) and os.path.exists(patch_path):
                mask_out = np.load(mask_path)
                nz_local = np.array(mask_out.nonzero()).mean(axis=1)
                records.append({
                    "patient_id": pid, "scan_id": scan.id, "nodule_idx": nidx,
                    "mask_path": mask_path, "patch_path": patch_path,
                    "centroid_z": float(nz_local[0]), "centroid_y": float(nz_local[1]),
                    "centroid_x": float(nz_local[2]),
                    "spacing_z": native_spacing[0], "spacing_y": native_spacing[1],
                    "spacing_x": native_spacing[2],
                    **targets,
                })
                continue

            try:
                # pad=0 — we handle cropping ourselves with fixed window
                cmask, cbbox, _ = consensus(
                    anns, clevel=consensus_level,
                    pad=0,  # pylidc 0.2.2 rejects tuple-of-tuples; 0 == no pad
                )
            except Exception as e:
                logger.warning("Consensus failed %s nodule %d: %s", pid, nidx, e)
                continue

            if cmask.sum() == 0:
                logger.warning("Empty mask %s nodule %d — skip", pid, nidx)
                continue

            try:
                patch, mask_out = _crop_and_resample_nodule(
                    _get_vol(), cmask, cbbox, native_spacing, window_mm, target_spacing
                )
            except Exception as e:
                logger.warning("Crop/resample failed %s nodule %d: %s", pid, nidx, e)
                continue

            if mask_out.sum() == 0:
                logger.warning("Empty mask after resample %s nodule %d — skip", pid, nidx)
                continue

            nodule_dir.mkdir(parents=True, exist_ok=True)
            np.save(mask_path,  mask_out.astype(np.uint8))
            np.save(patch_path, patch.astype(np.float32))

            # Centroid in local patch coords (center of output volume)
            nz_local = np.array(mask_out.nonzero()).mean(axis=1)

            records.append({
                "patient_id":    pid,
                "scan_id":       scan.id,
                "nodule_idx":    nidx,
                "mask_path":     mask_path,
                "patch_path":    patch_path,
                "centroid_z":    float(nz_local[0]),
                "centroid_y":    float(nz_local[1]),
                "centroid_x":    float(nz_local[2]),
                "spacing_z":     native_spacing[0],
                "spacing_y":     native_spacing[1],
                "spacing_x":     native_spacing[2],
                **targets,
            })

    df = pd.DataFrame(records)
    logger.info(
        "Dataset: %d nodules | %d malignant | %d benign | %d indeterminate",
        len(df), (df.label == 1).sum(), (df.label == 0).sum(), (df.label == -1).sum(),
    )
    return df


def add_kfold_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    freeze_from: Optional[str] = None,
) -> pd.DataFrame:
    """Patient-level stratified k-fold. All nodules from one patient stay in same fold.

    freeze_from: path to an old labels.csv. Patients present there keep their
    original fold (so AUC on old data stays comparable across runs); only
    newly-added patients are freshly stratified, using grade3 mode (not the
    binary label mode, since a new patient could be all-indeterminate).
    """
    frozen: dict = {}
    if freeze_from and os.path.exists(freeze_from):
        old = pd.read_csv(freeze_from)
        frozen = dict(old.drop_duplicates("patient_id")[["patient_id", "fold"]].values)

    known_mask = df["patient_id"].isin(frozen)
    known = df[known_mask].copy()
    known["fold"] = known["patient_id"].map(frozen).astype(int)

    new = df[~known_mask].copy()
    if len(new):
        patient_df = (
            new.groupby("patient_id")
            .agg(patient_grade=("grade3", lambda x: int(x.mode()[0])))
            .reset_index()
        )
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        patient_df["fold"] = -1
        for fold_idx, (_, val_idx) in enumerate(
            skf.split(patient_df["patient_id"], patient_df["patient_grade"])
        ):
            patient_df.loc[val_idx, "fold"] = fold_idx
        new = new.merge(patient_df[["patient_id", "fold"]], on="patient_id", how="left")

    return pd.concat([known, new], ignore_index=True) if len(new) else known


def load_and_split(
    lidc_path: str = "data/raw/LIDC-IDRI",
    interim_path: str = "data/interim",
    processed_path: str = "data/processed",
    n_folds: int = 5,
    seed: int = 42,
    force_rebuild: bool = False,
    include_indeterminate: bool = True,
    skip_existing: bool = False,
    freeze_from: Optional[str] = None,
) -> pd.DataFrame:
    """Full pipeline: build nodule dataset + k-fold splits. Caches to processed/labels.csv."""
    labels_csv = Path(processed_path) / "labels.csv"
    if labels_csv.exists() and not force_rebuild:
        logger.info("Loading cached labels from %s", labels_csv)
        return pd.read_csv(labels_csv)

    df = build_nodule_dataset(
        lidc_path=lidc_path, interim_path=interim_path,
        include_indeterminate=include_indeterminate, skip_existing=skip_existing,
    )
    df = add_kfold_splits(df, n_folds=n_folds, seed=seed, freeze_from=freeze_from)
    os.makedirs(processed_path, exist_ok=True)
    df.to_csv(labels_csv, index=False)
    logger.info("Saved labels to %s", labels_csv)
    return df
