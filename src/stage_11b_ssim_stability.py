"""Stage 11b (rev3 Fase 2): robustness of the run03 CAM maps under physical perturbation.

Why this stage exists. Stage 11a asks whether a map reflects the decision. This one asks
whether the map is stable at all: if a 3 degree tilt or a trace of scanner noise moves it,
then a map that looks convincing is telling the reader less than it appears to. For
Track 2 that is a second axis of the silent-failure thesis -- alongside the finding that
the best explanation *site* moves with the training protocol, a map that moves under
perturbations a radiologist would not even notice is the same failure seen from another
side.

Perturbations follow Qin (2025): tilt of +/-3 degrees and additive Gaussian noise with
sigma 0.02. Inputs are min-max scaled from HU [-1000, 400] (`_load_patch_tensor`), so
sigma 0.02 is a 28 HU disturbance -- within the noise band of a real CT acquisition.

Two adaptations rev3 requires be stated explicitly rather than buried:

  1. **2.5D, not 3D.** Qin computes 3D SSIM over a volume. The input here is 2.5D -- three
     slices stacked as channels -- but `compute_gradcam` returns a single 2D map, so SSIM
     is computed on the 2D map and the dimensionality mismatch never arises. No per-slice
     averaging is needed because there are no per-slice maps.
  2. **Rotation is undone before comparison.** A map computed on a tilted input lives in
     the tilted frame. Comparing it directly against the unrotated map would measure the
     tilt, not the instability, and would report every model as unstable. So the map is
     rotated back by -theta first, and the comparison is restricted to the largest centred
     square that stays inside the image under rotation -- otherwise the empty corners
     rotation introduces would depress SSIM for a reason that has nothing to do with the
     explanation. The same crop is applied to the noise branch so the three numbers sit on
     one scale.

2.2 (cross-patient consistency) is computed in the same pass because every map it needs is
already in memory: mean pairwise SSIM between maps of different patients carrying the same
label, within a fold. rev3 marks it lower priority, so it is reported as a column rather
than headlined.

Outputs (nothing stage_10d or stage_11a wrote is touched):
    artifacts/results/run03/ssim_stability_run03.csv            one row per backbone per fold
    artifacts/results/run03/ssim_stability_run03_persample.csv  one row per sample

Run:
    .venv/Scripts/python.exe -m src.stage_11b_ssim_stability
    .venv/Scripts/python.exe -m src.stage_11b_ssim_stability --self-check
"""
from __future__ import annotations

import argparse
import itertools
import logging
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from skimage.metrics import structural_similarity
from torchvision.transforms.functional import InterpolationMode, rotate

from src.models.registry import _NAME_MAP
from src.stage_08b_run02_xai import _commit_sha, _load_patch_tensor
from src.stage_10d_xai_run03 import (
    BACKBONES,
    N_FOLDS,
    RUN_ID,
    SAMPLE_SEED,
    UNFREEZE_PCT,
    _display_ids,
    _load_model,
    _mtime,
    _resolve_layer,
    _samples,
)
from src.stage_11a_faithfulness import _cam_map
from src.utils.tracks import track_input_size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join("configs", "config.yaml")
OUT_DIR = os.path.join("artifacts", "results", "run03")
OUT_CSV = os.path.join(OUT_DIR, "ssim_stability_run03.csv")
PERSAMPLE_CSV = os.path.join(OUT_DIR, "ssim_stability_run03_persample.csv")

ROTATE_DEG = 3.0     # Qin (2025)
NOISE_SIGMA = 0.02   # Qin (2025); 0.02 * 1400 HU = 28 HU
# Cap on how many same-label pairs feed 2.2. All 60 maps of a fold give at most ~435 pairs
# per class, which is cheap, so the cap only guards against a fold that is far larger.
MAX_CONSISTENCY_PAIRS = 500


def _valid_crop(n: int, degrees: float) -> int:
    """Side of the largest centred square that stays inside an n x n image under rotation.

    A square rotated by theta needs (cos|theta| + sin|theta|) times its side to fit in its
    own bounding box, so the reciprocal gives the side that survives. At n=64 and 3 degrees
    this is 60: four rows and columns of rotation-induced emptiness are dropped rather than
    being scored as instability.
    """
    t = math.radians(abs(degrees))
    return max(1, int(n / (math.cos(t) + math.sin(t))))


def _centre_crop(arr: np.ndarray, side: int) -> np.ndarray:
    h, w = arr.shape[-2:]
    top, left = (h - side) // 2, (w - side) // 2
    return arr[..., top:top + side, left:left + side]


def _rotate_map(cam: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate a 2D map about its centre, bilinear, zero fill."""
    t = torch.from_numpy(cam.astype(np.float32))[None, None, ...]
    out = rotate(t, degrees, interpolation=InterpolationMode.BILINEAR, fill=[0.0])
    return out[0, 0].numpy()


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM between two maps already normalised to [0, 1]."""
    return float(structural_similarity(a, b, data_range=1.0))


def run_cell(backbone: str, fold: int, cfg: dict, device, sha: str, display: set[str]):
    """Return (summary row, per-sample frame) for one backbone at one fold."""
    internal = _NAME_MAP.get(backbone, backbone)
    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)
    input_size = track_input_size(cfg, backbone)

    samples = _samples(cfg, fold)
    model, ckpt = _load_model(cfg, backbone, fold, device)
    dummy = torch.zeros(1, n_slices, patch_xy, patch_xy, device=device)
    layer = _resolve_layer(model, internal, dummy)
    rng = np.random.default_rng(SAMPLE_SEED + fold)
    side = _valid_crop(patch_xy, ROTATE_DEG)

    per_sample, base_maps = [], []
    for i, row in samples.iterrows():
        img = _load_patch_tensor(row["patch_path"], n_slices, patch_xy).to(device)
        with torch.no_grad():
            logits = model(img)
            pred = int(logits.argmax(dim=1).item())
            base_prob = float(F.softmax(logits, dim=1)[0, pred].item())

        base = _cam_map(model, img, internal)
        base_c = _centre_crop(base, side)

        ssim_rot = {}
        for degrees in (ROTATE_DEG, -ROTATE_DEG):
            tilted = rotate(img, degrees, interpolation=InterpolationMode.BILINEAR, fill=[0.0])
            cam_t = _cam_map(model, tilted, internal)
            # Back into the original frame before comparing, then crop away the corners
            # rotation emptied.
            ssim_rot[degrees] = _ssim(base_c, _centre_crop(_rotate_map(cam_t, -degrees), side))

        noise = torch.from_numpy(
            rng.normal(0.0, NOISE_SIGMA, size=tuple(img.shape)).astype(np.float32)
        ).to(img.device)
        cam_n = _cam_map(model, img + noise, internal)
        ssim_noise = _ssim(base_c, _centre_crop(cam_n, side))

        base_maps.append((int(row["label"]), base_c))
        case = f"{row['patient_id']}#{int(row['nodule_idx'])}"
        per_sample.append({
            "run_id": RUN_ID, "commit_sha": sha, "backbone": backbone,
            "unfreeze_pct": UNFREEZE_PCT, "fold": fold, "sample_idx": int(i),
            "patient_id": row["patient_id"], "nodule_idx": int(row["nodule_idx"]),
            "in_display_set": int(case in display),
            "label": int(row["label"]), "pred": pred, "prob_pred": base_prob,
            "ssim_rot_pos": ssim_rot[ROTATE_DEG], "ssim_rot_neg": ssim_rot[-ROTATE_DEG],
            "ssim_rot_mean": (ssim_rot[ROTATE_DEG] + ssim_rot[-ROTATE_DEG]) / 2.0,
            "ssim_noise": ssim_noise,
            "cam_max": float(base.max()),
        })

    del model
    torch.cuda.empty_cache()

    # 2.2: mean pairwise SSIM between different patients carrying the same label.
    consistency = {}
    for lab in (0, 1):
        maps = [m for l, m in base_maps if l == lab]
        pairs = list(itertools.combinations(range(len(maps)), 2))[:MAX_CONSISTENCY_PAIRS]
        consistency[lab] = (float(np.mean([_ssim(maps[a], maps[b]) for a, b in pairs]))
                            if pairs else np.nan)

    ps = pd.DataFrame(per_sample)
    summary = {
        "run_id": RUN_ID, "commit_sha": sha, "backbone": backbone,
        "internal_name": internal, "unfreeze_pct": UNFREEZE_PCT, "fold": fold,
        "n": len(ps), "input_size": input_size, **layer,
        "checkpoint": ckpt.replace("\\", "/"), "checkpoint_mtime": _mtime(ckpt),
        "rotate_deg": ROTATE_DEG, "noise_sigma": NOISE_SIGMA,
        "compare_side_px": side, "patch_xy": patch_xy,
        "ssim_rot_pos": ps["ssim_rot_pos"].mean(),
        "ssim_rot_neg": ps["ssim_rot_neg"].mean(),
        "ssim_rot_mean": ps["ssim_rot_mean"].mean(),
        "ssim_rot_sd": ps["ssim_rot_mean"].std(),
        "ssim_noise": ps["ssim_noise"].mean(),
        "ssim_noise_sd": ps["ssim_noise"].std(),
        "ssim_consistency_benign": consistency[0],
        "ssim_consistency_malignant": consistency[1],
        "n_zero_cam": int((ps["cam_max"] <= 0).sum()),
        "n_display_set": int(ps["in_display_set"].sum()),
    }
    return summary, ps


def self_check(df: pd.DataFrame, persample: pd.DataFrame) -> None:
    """Asserts these numbers have to survive before they may enter the manuscript."""
    # 1. Same degenerate-site guard the other two stages carry.
    degenerate = df[(df["spatial_h"].astype(str) == "1") & (df["spatial_w"].astype(str) == "1")]
    assert degenerate.empty, (
        f"1x1 target layer: {degenerate[['backbone', 'fold']].to_dict('records')}")

    # 2. Provenance, as in stage_10d and stage_11a.
    missing = df[df["checkpoint_mtime"].fillna("") == ""]
    assert missing.empty, (
        f"checkpoint_mtime kosong: {missing[['backbone', 'fold']].to_dict('records')}")

    # 3. SSIM is bounded on [-1, 1] by definition. A value outside it means the maps were
    #    not normalised the way `data_range=1.0` assumes.
    for col in ("ssim_rot_pos", "ssim_rot_neg", "ssim_noise"):
        assert persample[col].between(-1.0, 1.0).all(), f"{col} keluar rentang [-1, 1]"

    # 4. The comparison window must actually exclude the rotation-emptied border, or the
    #    stability numbers are measuring geometry. 3 degrees on a 64 px patch leaves 60.
    assert (df["compare_side_px"] < df["patch_xy"]).all(), \
        "jendela banding nol dipangkas: sudut kosong akibat rotasi ikut terhitung"

    # 5. Complete grid.
    assert len(df) == len(BACKBONES) * N_FOLDS, f"{len(df)} sel, harus {len(BACKBONES) * N_FOLDS}"

    # 6. Not an assert: low stability is the Track 2 finding, not a failure of the stage.
    fragile = df[df["ssim_rot_mean"] < 0.5]
    if not fragile.empty:
        print("[SELF-CHECK] PERHATIAN: SSIM rotasi < 0,50 (peta rapuh terhadap kemiringan 3 "
              f"derajat) di {fragile[['backbone', 'fold']].to_dict('records')}")
    print("[SELF-CHECK] kelima assert lolos")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--self-check", action="store_true", help="jalankan assert pasca-hitung")
    p.add_argument("--backbone", default=None, help="batasi ke satu backbone")
    p.add_argument("--fold", type=int, default=None, help="batasi ke satu fold")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sha = _commit_sha()
    display = _display_ids()

    backbones = [args.backbone] if args.backbone else BACKBONES
    folds = [args.fold] if args.fold is not None else list(range(N_FOLDS))

    rows, per_sample = [], []
    for backbone in backbones:
        for fold in folds:
            summary, ps = run_cell(backbone, fold, cfg, device, sha, display)
            rows.append(summary)
            per_sample.append(ps)
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            pd.concat(per_sample, ignore_index=True).to_csv(PERSAMPLE_CSV, index=False)
            logger.info("[%s fold %d] SSIM rotasi=%.4f noise=%.4f konsistensi "
                        "jinak=%.4f ganas=%.4f",
                        backbone, fold, summary["ssim_rot_mean"], summary["ssim_noise"],
                        summary["ssim_consistency_benign"],
                        summary["ssim_consistency_malignant"])

    df = pd.DataFrame(rows)
    ps_all = pd.concat(per_sample, ignore_index=True)

    with pd.option_context("display.max_columns", None, "display.width", 250):
        print("\n=== stabilitas SSIM, checkpoint run03 uf100 ===")
        print(df[["backbone", "fold", "n", "ssim_rot_pos", "ssim_rot_neg", "ssim_rot_mean",
                  "ssim_rot_sd", "ssim_noise", "ssim_consistency_benign",
                  "ssim_consistency_malignant"]].to_string(index=False))
        print("\n=== rata-rata lintas fold ===")
        print(df.groupby("backbone")[["ssim_rot_mean", "ssim_noise",
                                      "ssim_consistency_benign",
                                      "ssim_consistency_malignant"]].mean().to_string())

    print(f"\n[WROTE] {OUT_CSV} ({len(df)} baris)")
    print(f"[WROTE] {PERSAMPLE_CSV} ({len(ps_all)} baris)")

    if args.self_check:
        self_check(df, ps_all)


if __name__ == "__main__":
    main()
