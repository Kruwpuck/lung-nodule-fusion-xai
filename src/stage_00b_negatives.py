"""Stage 00b: LUNA16 hard-negative ("no-nodule") patches for arm D.

Mandatory gate: world(mm,LPS)->voxel transform is self-tested against
LUNA16 annotations.csv (matched to pylidc nodule centroids) BEFORE any
negative patch is generated. If the transform doesn't check out (z-axis
flip is the classic failure), negatives would be silently wrong — crops
of empty space or unrelated tissue that still "look" plausible. See
Paper/PLAN_MALIGNANCY_GRADING.md section 3.5.

Negatives are honestly labeled "no-nodule (hard-negative, LUNA16
detector-FP)" — these are CAD false positives (vessels/fissures/scars),
NOT a representative sample of normal parenchyma. See compass doc
section A/4 for why that distinction matters for the paper.
"""
import argparse
import logging
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LUNA_DIRS = [
    r"C:\Users\Adaptive Network\Documents\Lung Cancer\luna16 part 1",
    r"C:\Users\Adaptive Network\Documents\Lung Cancer\luna16 part 2",
]


def _find_luna_file(name: str) -> str:
    for d in LUNA_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found under {LUNA_DIRS}")


def _ensure_candidates_v2(work_dir: str) -> str:
    """Unzip candidates_V2.zip (72MB) if not already extracted. Nothing else is unzipped."""
    out_csv = os.path.join(work_dir, "candidates_V2.csv")
    if os.path.exists(out_csv):
        return out_csv
    zp = _find_luna_file("candidates_V2.zip")
    with zipfile.ZipFile(zp) as z:
        members = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise RuntimeError(f"No CSV inside {zp}")
        os.makedirs(work_dir, exist_ok=True)
        with z.open(members[0]) as src, open(out_csv, "wb") as dst:
            dst.write(src.read())
    return out_csv


def _world_to_voxel(coordX, coordY, coordZ, origin_xyz, spacing_xyz):
    """LUNA16 world coords (mm, LPS, x/y/z order) -> voxel (z, y, x)."""
    ox, oy, oz = origin_xyz
    psx, psy, psz = spacing_xyz
    vx = (coordX - ox) / psx
    vy = (coordY - oy) / psy
    vz = (coordZ - oz) / psz
    return vz, vy, vx


def _scan_origin_spacing(scan):
    """DICOM origin (x,y,z LPS) + voxel spacing (x,y,z) for a pylidc scan."""
    dcm = scan.load_all_dicom_images(verbose=False)  # z-sorted by pylidc
    ox, oy, oz = map(float, dcm[0].ImagePositionPatient)
    psx = psy = float(scan.pixel_spacing)
    psz = float(getattr(scan, "slice_spacing", None) or getattr(scan, "slice_thickness", 1.0))
    return (ox, oy, oz), (psx, psy, psz)


def gate_transform_selftest(min_hit_rate: float = 0.95) -> float:
    """Self-test the world->voxel transform against LUNA16 annotations.csv,
    which gives ground-truth (world_mm, diameter_mm) for real nodules already
    known to pylidc. For each annotation, transform to voxel, find the
    nearest pylidc nodule centroid on the same scan, measure distance in mm.
    Passes if >=min_hit_rate of matches land within their own diameter_mm.

    Returns the observed hit rate. Caller MUST stop and not generate any
    negatives if this is below min_hit_rate.
    """
    import pylidc as pl
    from pylidc.utils import consensus

    ann_csv = _find_luna_file("annotations.csv")
    ann = pd.read_csv(ann_csv)
    logger.info("annotations.csv: %d rows, %d unique series", len(ann), ann.seriesuid.nunique())

    scans_by_uid = {s.series_instance_uid: s for s in pl.query(pl.Scan).all()}

    hits, total = 0, 0
    for seriesuid, grp in ann.groupby("seriesuid"):
        scan = scans_by_uid.get(seriesuid)
        if scan is None:
            continue
        try:
            origin, spacing = _scan_origin_spacing(scan)
        except Exception as e:
            logger.warning("origin/spacing failed for %s: %s", seriesuid, e)
            continue

        # pylidc nodule centroids (voxel, Z Y X) on this scan
        try:
            nodule_groups = scan.cluster_annotations()
        except Exception:
            continue
        pylidc_centroids = []
        for anns_grp in nodule_groups:
            try:
                cmask, cbbox, _ = consensus(anns_grp, clevel=0.5, pad=0)  # pylidc 0.2.2: no tuple pad
            except Exception:
                continue
            if cmask.sum() == 0:
                continue
            # cmask/cbbox are pylidc order (Y, X, Z)
            nz = np.array(cmask.nonzero())
            cy = cbbox[0].start + nz[0].mean()
            cx = cbbox[1].start + nz[1].mean()
            cz = cbbox[2].start + nz[2].mean()
            pylidc_centroids.append((cz, cy, cx))
        if not pylidc_centroids:
            continue
        pylidc_centroids = np.array(pylidc_centroids)  # voxel (z,y,x)

        for _, row in grp.iterrows():
            vz, vy, vx = _world_to_voxel(
                row["coordX"], row["coordY"], row["coordZ"], origin, spacing
            )
            v = np.array([vz, vy, vx])
            # distance in mm: scale each axis by its own spacing (z,y,x order)
            spacing_zyx = np.array([spacing[2], spacing[1], spacing[0]])
            d_vox = pylidc_centroids - v
            d_mm = np.sqrt(((d_vox * spacing_zyx) ** 2).sum(axis=1))
            nearest_mm = d_mm.min()
            total += 1
            diam = float(row.get("diameter_mm", 0.0)) or 5.0
            if nearest_mm < diam:
                hits += 1

    hit_rate = hits / total if total else 0.0
    logger.info("Transform self-test: %d/%d matched within diameter_mm -> hit_rate=%.4f",
                hits, total, hit_rate)
    return hit_rate


def generate_negatives(
    labels_csv: str,
    work_dir: str,
    patches_out: str,
    window_mm=(64.0, 64.0, 16.0),
    target_spacing=(1.0, 1.0, 1.0),
    min_dist_from_nodule_mm: float = 10.0,
    ratio_to_nodules: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample LUNA16 class==0 candidates on scans already in labels_csv,
    crop with the identical pipeline used for real nodules, and return a
    DataFrame in the same schema as labels.csv (grade4=0, label=-1,
    median_rating=NaN) ready to be appended.
    """
    import pylidc as pl
    from src.data_loading.lidc_loader import crop_and_resample_point

    cand_csv = _ensure_candidates_v2(work_dir)
    cand = pd.read_csv(cand_csv)
    cand0 = cand[cand["class"] == 0].copy()
    logger.info("candidates_V2 class==0: %d rows", len(cand0))

    df = pd.read_csv(labels_csv)
    patient_scan = df.drop_duplicates("patient_id")[["patient_id", "fold"]]
    fold_by_patient = dict(zip(patient_scan.patient_id, patient_scan.fold))

    scans = pl.query(pl.Scan).all()
    scans_by_uid = {s.series_instance_uid: s for s in scans}
    # only series whose patient already has real nodules in our dataset
    usable_uids = {
        s.series_instance_uid: s for s in scans if s.patient_id in fold_by_patient
    }
    cand0 = cand0[cand0.seriesuid.isin(usable_uids)]
    logger.info("candidates usable (patient already in labels.csv): %d", len(cand0))

    # labels.csv stores voxel-local (patch) coords only, not world coords, so
    # the too-close-to-a-real-nodule filter below uses LUNA16 annotations.csv directly.
    ann_csv = _find_luna_file("annotations.csv")
    ann = pd.read_csv(ann_csv)

    rng = np.random.default_rng(seed)
    n_target = int(round(len(df) * ratio_to_nodules))

    records = []
    os.makedirs(patches_out, exist_ok=True)

    # shuffle candidates, then take per-patient until we hit n_target
    cand0 = cand0.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    seen_patients_quota: dict[str, int] = {}
    for _, row in cand0.iterrows():
        if len(records) >= n_target:
            break
        seriesuid = row["seriesuid"]
        scan = usable_uids[seriesuid]
        pid = scan.patient_id

        # drop candidates too close to any real annotated nodule on this series
        ann_this = ann[ann.seriesuid == seriesuid]
        if len(ann_this):
            dx = ann_this["coordX"].values - row["coordX"]
            dy = ann_this["coordY"].values - row["coordY"]
            dz = ann_this["coordZ"].values - row["coordZ"]
            dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            if dist.min() < min_dist_from_nodule_mm:
                continue

        try:
            origin, spacing = _scan_origin_spacing(scan)
            vz, vy, vx = _world_to_voxel(row["coordX"], row["coordY"], row["coordZ"], origin, spacing)
            vol_yxz = scan.to_volume(verbose=False)
            vol = vol_yxz.transpose(2, 0, 1)
            native_spacing = (spacing[2], spacing[1], spacing[0])  # (sz, sy, sx)
            center_zyx = (int(round(vz)), int(round(vy)), int(round(vx)))
            patch = crop_and_resample_point(vol, center_zyx, native_spacing, window_mm, target_spacing)
        except Exception as e:
            logger.warning("negative crop failed %s: %s", seriesuid, e)
            continue

        # synthetic 10mm-diameter spherical ROI at patch center for radiomics
        tz, ty, tx = patch.shape
        zz, yy, xx = np.ogrid[:tz, :ty, :tx]
        cz, cy, cx = tz // 2, ty // 2, tx // 2
        r_vox = 5.0 / target_spacing[0]  # 10mm diameter = 5mm radius, isotropic 1mm spacing
        mask = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= r_vox ** 2
        mask = mask.astype(np.uint8)

        idx = seen_patients_quota.get(pid, 0)
        seen_patients_quota[pid] = idx + 1
        nodule_dir = Path(patches_out) / pid / f"neg_{idx:03d}"
        nodule_dir.mkdir(parents=True, exist_ok=True)
        patch_path = str(nodule_dir / "patch.npy")
        mask_path = str(nodule_dir / "mask.npy")
        np.save(patch_path, patch.astype(np.float32))
        np.save(mask_path, mask)

        records.append({
            "patient_id": pid,
            "scan_id": scan.id,
            "nodule_idx": -1 - idx,  # negative sentinel range, won't collide with real nidx>=0
            "label": -1,
            "grade3": -1,
            "grade4": 0,
            "median_rating": np.nan,
            "n_annotations": 0,
            "rating_std": np.nan,
            "mask_path": mask_path,
            "patch_path": patch_path,
            "centroid_z": float(cz), "centroid_y": float(cy), "centroid_x": float(cx),
            "spacing_z": native_spacing[0], "spacing_y": native_spacing[1], "spacing_x": native_spacing[2],
            "fold": fold_by_patient[pid],
            "class_kind": "no_nodule_hard_negative",
        })

    logger.info("Generated %d negative patches (target was %d)", len(records), n_target)
    return pd.DataFrame(records)


def run(cfg: dict) -> None:
    labels_csv = os.path.join(cfg["paths"]["interim"], "labels.csv")
    work_dir = cfg.get("negatives", {}).get("work_dir", "./artifacts/luna_work")
    patches_out = cfg.get("negatives", {}).get("patches_out", "./artifacts/patches_neg")
    min_hit_rate = cfg.get("negatives", {}).get("min_transform_hit_rate", 0.95)

    hit_rate = gate_transform_selftest(min_hit_rate=min_hit_rate)
    print(f"[GATE] transform self-test hit_rate={hit_rate:.4f} (need >= {min_hit_rate})")
    if hit_rate < min_hit_rate:
        print("[STOP] transform self-test FAILED — not generating any negatives. "
              "Suspect z-axis flip; inspect _world_to_voxel / origin extraction before retrying.")
        raise SystemExit(1)

    neg_df = generate_negatives(
        labels_csv=labels_csv,
        work_dir=work_dir,
        patches_out=patches_out,
        ratio_to_nodules=cfg.get("negatives", {}).get("ratio_to_nodules", 1.0),
        min_dist_from_nodule_mm=cfg.get("negatives", {}).get("min_dist_from_nodule_mm", 10.0),
        seed=cfg.get("seed", 42),
    )

    df = pd.read_csv(labels_csv)
    if "class_kind" not in df.columns:
        df["class_kind"] = "nodule"
    combined = pd.concat([df, neg_df], ignore_index=True)
    combined.to_csv(labels_csv, index=False)
    print(f"[DONE] appended {len(neg_df)} negatives -> {labels_csv} ({len(combined)} total rows)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg)


if __name__ == "__main__":
    main()
