"""LUNA16 data loading: MHD/RAW patches + malignancy labels linked via pylidc."""
from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

DRIVE_BASE = "/content/drive/MyDrive/Kanker Kanker apa yg Kanker"
LUNA16_PART1 = f"{DRIVE_BASE}/luna16 part 1"
LUNA16_PART2 = f"{DRIVE_BASE}/luna16 part 2"
METADATA_CSV = f"{DRIVE_BASE}/metadata/metadata.csv"


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def extract_subsets(
    output_dir: str = "/content/luna16",
    subset_ids: Optional[list] = None,
    drive_part1: str = LUNA16_PART1,
    drive_part2: str = LUNA16_PART2,
) -> str:
    """Unzip LUNA16 subset*.zip files from Drive to output_dir.

    Args:
        output_dir: Local Colab path to extract into.
        subset_ids: List of ints (0-9). None = all available.
        drive_part1: Drive folder with subset0-6.
        drive_part2: Drive folder with subset7-9.

    Returns:
        output_dir path string.
    """
    os.makedirs(output_dir, exist_ok=True)
    subset_ids = subset_ids if subset_ids is not None else list(range(10))

    for sid in subset_ids:
        fname = f"subset{sid}.zip"
        part = drive_part1 if sid <= 6 else drive_part2
        src = Path(part) / fname
        if not src.exists():
            logger.warning("subset%d.zip not found at %s — skipping", sid, src)
            continue
        dest = Path(output_dir) / f"subset{sid}"
        if dest.exists() and any(dest.iterdir()):
            logger.info("subset%d already extracted, skipping", sid)
            continue
        logger.info("Extracting %s → %s …", fname, output_dir)
        with zipfile.ZipFile(src, "r") as zf:
            zf.extractall(output_dir)
        logger.info("subset%d done", sid)

    return output_dir


def extract_lung_masks(
    output_dir: str = "/content/luna16_seg",
    drive_part1: str = LUNA16_PART1,
) -> str:
    """Extract seg-lungs-LUNA16.zip (lung segmentation masks)."""
    os.makedirs(output_dir, exist_ok=True)
    src = Path(drive_part1) / "seg-lungs-LUNA16.zip"
    if not src.exists():
        raise FileNotFoundError(f"seg-lungs-LUNA16.zip not found at {src}")
    logger.info("Extracting lung masks → %s …", output_dir)
    with zipfile.ZipFile(src, "r") as zf:
        zf.extractall(output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------

def load_candidates(candidates_csv: str, positive_only: bool = True) -> pd.DataFrame:
    """Load LUNA16 candidates.csv.

    Columns: seriesuid, coordX, coordY, coordZ, class (0=FP, 1=nodule).
    """
    df = pd.read_csv(candidates_csv)
    if positive_only:
        df = df[df["class"] == 1].reset_index(drop=True)
    return df


def load_metadata_mapping(metadata_csv: str = METADATA_CSV) -> pd.DataFrame:
    """Load TCIA metadata.csv; extracts SeriesInstanceUID → PatientID mapping."""
    df = pd.read_csv(metadata_csv)
    # TCIA column names vary; try common variants
    uid_col = next((c for c in df.columns if "series" in c.lower() and "uid" in c.lower()), None)
    pid_col = next((c for c in df.columns if "patient" in c.lower()), None)
    if uid_col is None or pid_col is None:
        raise ValueError(
            f"Cannot find SeriesUID/PatientID columns in metadata.csv. "
            f"Columns: {list(df.columns)}"
        )
    mapping = df[[uid_col, pid_col]].copy()
    mapping.columns = ["series_uid", "patient_id"]
    mapping["patient_id"] = mapping["patient_id"].astype(str).str.strip()
    return mapping.drop_duplicates("series_uid").set_index("series_uid")


# ---------------------------------------------------------------------------
# MHD/RAW I/O
# ---------------------------------------------------------------------------

def load_mhd_volume(mhd_path: str):
    """Load MHD/RAW file. Returns (sitk_image, volume_np, origin, spacing, direction)."""
    import SimpleITK as sitk
    img = sitk.ReadImage(mhd_path)
    volume = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    origin = np.array(img.GetOrigin())    # (X, Y, Z)
    spacing = np.array(img.GetSpacing())  # (X, Y, Z)
    direction = np.array(img.GetDirection()).reshape(3, 3)
    return img, volume, origin, spacing, direction


def world_to_voxel(coord_world: np.ndarray, origin: np.ndarray,
                   spacing: np.ndarray) -> np.ndarray:
    """Convert world (X,Y,Z) mm coordinates to voxel (Z,Y,X) indices."""
    voxel_xyz = (coord_world - origin) / spacing  # (X,Y,Z)
    return np.array([voxel_xyz[2], voxel_xyz[1], voxel_xyz[0]])  # → (Z,Y,X)


def make_sphere_mask(shape: tuple, center_zyx: np.ndarray,
                     radius_vox: float) -> np.ndarray:
    """Binary sphere mask of given radius (voxels) at center."""
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    cz, cy, cx = center_zyx
    dist2 = (z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2
    return (dist2 <= radius_vox ** 2).astype(np.uint8)


def extract_luna16_patch(
    mhd_path: str,
    coord_world_xyz: tuple,
    diameter_mm: float,
    patch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (patch, mask) from MHD/RAW at world coordinates.

    Returns:
        patch: (patch_size, patch_size, patch_size) float32 HU array
        mask:  (patch_size, patch_size, patch_size) uint8 sphere mask
    """
    from src.preprocessing.ct_preprocessing import extract_patch_3d

    _, volume, origin, spacing, _ = load_mhd_volume(mhd_path)
    coord_w = np.array(coord_world_xyz)
    center_zyx = world_to_voxel(coord_w, origin, spacing)

    patch = extract_patch_3d(volume, center_zyx, patch_size=(patch_size,) * 3)

    # sphere mask centred in patch
    half = patch_size // 2
    radius_vox = (diameter_mm / 2.0) / float(spacing.mean())
    local_center = np.array([half, half, half], dtype=float)
    mask = make_sphere_mask((patch_size,) * 3, local_center, radius_vox)

    return patch.astype(np.float32), mask


# ---------------------------------------------------------------------------
# Malignancy label lookup via pylidc
# ---------------------------------------------------------------------------

def _find_mhd(subset_dir: str, series_uid: str) -> Optional[str]:
    """Search for .mhd file matching series_uid in subset_dir tree."""
    for root, _, files in os.walk(subset_dir):
        for f in files:
            if f.endswith(".mhd") and series_uid in f:
                return os.path.join(root, f)
    return None


def get_malignancy_from_lidc(
    patient_id: str,
    coord_world_xyz: tuple,
    lidc_path: str,
    tol_mm: float = 3.0,
    exclude_score: int = 3,
) -> Optional[int]:
    """Return binary malignancy label (0/1) for a nodule using pylidc.

    Finds the scan matching patient_id, then the closest annotation cluster
    within tol_mm of coord_world_xyz (XYZ in mm).

    Returns None if ambiguous (median score == exclude_score).
    """
    import pylidc as pl

    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
    if not scans:
        logger.warning("pylidc: no scan found for patient_id=%s", patient_id)
        return None

    scan = scans[0]
    coord_w = np.array(coord_world_xyz)

    best_label = None
    best_dist = tol_mm + 1.0

    for cluster in scan.cluster_annotations():
        if not cluster:
            continue
        # centroid of cluster (average annotation centroids)
        centroids = []
        for ann in cluster:
            bbox = ann.bbox()
            cz = (bbox[0].start + bbox[0].stop) / 2
            cy = (bbox[1].start + bbox[1].stop) / 2
            cx = (bbox[2].start + bbox[2].stop) / 2
            # convert to mm using scan spacing
            spacing = np.array([scan.pixel_spacing, scan.pixel_spacing,
                                 scan.slice_spacing])
            centroids.append(np.array([cx, cy, cz]) * spacing)

        centroid_mean = np.mean(centroids, axis=0)
        dist = float(np.linalg.norm(coord_w - centroid_mean))

        if dist < best_dist:
            ratings = [a.malignancy for a in cluster]
            med = float(np.median(ratings))
            if med == exclude_score:
                label = None
            else:
                label = int(med > exclude_score)
            best_dist = dist
            best_label = label

    if best_dist > tol_mm:
        logger.debug("No LIDC annotation within %.1f mm for patient %s coord %s",
                     tol_mm, patient_id, coord_world_xyz)
        return None

    return best_label


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def build_luna16_dataset(
    subset_dir: str,
    candidates_csv: str,
    metadata_csv: str = METADATA_CSV,
    lidc_path: str = f"{DRIVE_BASE}/lidc_idri",
    output_dir: str = "/content/drive/MyDrive/lung_nodule_interim_luna16",
    patch_size: int = 64,
    tol_mm: float = 3.0,
    subset_ids: Optional[list] = None,
) -> pd.DataFrame:
    """Build nodule DataFrame compatible with lidc_loader output.

    Schema: patient_id, label, patch_path, mask_path, fold, diameter_mm,
            centroid_x, centroid_y, centroid_z, series_uid.

    Steps:
        1. Load candidates (positive only from candidates.csv)
        2. Map seriesuid → patient_id via metadata.csv
        3. Find .mhd file for each candidate
        4. Extract 3D patch + sphere mask
        5. Lookup malignancy label via pylidc
        6. Save .npy files to output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    candidates = load_candidates(candidates_csv, positive_only=True)
    meta = load_metadata_mapping(metadata_csv)

    # filter to requested subsets
    if subset_ids is not None:
        candidates = candidates[
            candidates["seriesuid"].apply(
                lambda uid: any(
                    f"subset{i}" in str(_find_mhd(subset_dir, uid) or "")
                    for i in subset_ids
                )
            )
        ].reset_index(drop=True)

    records = []
    total = len(candidates)
    logger.info("Processing %d LUNA16 nodule candidates …", total)

    for i, row in candidates.iterrows():
        uid = row["seriesuid"]
        coord_w = (float(row["coordX"]), float(row["coordY"]), float(row["coordZ"]))
        diameter = float(row.get("diameter_mm", 6.0))

        # map to patient_id
        if uid not in meta.index:
            logger.debug("seriesuid %s not in metadata, skipping", uid)
            continue
        patient_id = str(meta.loc[uid, "patient_id"])

        # find .mhd file
        mhd_path = _find_mhd(subset_dir, uid)
        if mhd_path is None:
            logger.debug("MHD not found for %s, skipping", uid)
            continue

        # malignancy label via pylidc
        label = get_malignancy_from_lidc(
            patient_id, coord_w, lidc_path=lidc_path, tol_mm=tol_mm
        )
        if label is None:
            continue

        # extract patch + mask
        try:
            patch, mask = extract_luna16_patch(mhd_path, coord_w, diameter, patch_size)
        except Exception as e:
            logger.warning("Patch extraction failed for %s: %s", uid, e)
            continue

        # save
        nodule_dir = Path(output_dir) / patient_id / f"nodule_{i:06d}"
        nodule_dir.mkdir(parents=True, exist_ok=True)
        patch_path = str(nodule_dir / "patch.npy")
        mask_path = str(nodule_dir / "mask.npy")
        np.save(patch_path, patch)
        np.save(mask_path, mask)

        records.append({
            "patient_id": patient_id,
            "series_uid": uid,
            "label": label,
            "patch_path": patch_path,
            "mask_path": mask_path,
            "diameter_mm": diameter,
            "centroid_x": coord_w[0],
            "centroid_y": coord_w[1],
            "centroid_z": coord_w[2],
        })

        if (i + 1) % 100 == 0:
            logger.info("  %d / %d done", i + 1, total)

    df = pd.DataFrame(records)
    logger.info(
        "LUNA16 dataset: %d nodules | %d malignant | %d benign",
        len(df), (df.label == 1).sum(), (df.label == 0).sum(),
    )
    return df


def add_kfold_splits(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Patient-level stratified k-fold (reuses same logic as lidc_loader)."""
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


def load_and_split_luna16(
    subset_dir: str,
    candidates_csv: str,
    metadata_csv: str = METADATA_CSV,
    lidc_path: str = f"{DRIVE_BASE}/lidc_idri",
    output_dir: str = "/content/drive/MyDrive/lung_nodule_interim_luna16",
    processed_path: str = "/content/drive/MyDrive/lung_nodule_processed_luna16",
    n_folds: int = 5,
    seed: int = 42,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Full LUNA16 pipeline with caching. Returns labels DataFrame."""
    labels_csv = Path(processed_path) / "labels_luna16.csv"
    if labels_csv.exists() and not force_rebuild:
        logger.info("Loading cached LUNA16 labels from %s", labels_csv)
        return pd.read_csv(labels_csv)

    df = build_luna16_dataset(
        subset_dir=subset_dir,
        candidates_csv=candidates_csv,
        metadata_csv=metadata_csv,
        lidc_path=lidc_path,
        output_dir=output_dir,
    )
    df = add_kfold_splits(df, n_folds=n_folds, seed=seed)

    os.makedirs(processed_path, exist_ok=True)
    df.to_csv(labels_csv, index=False)
    logger.info("Saved LUNA16 labels to %s", labels_csv)
    return df
