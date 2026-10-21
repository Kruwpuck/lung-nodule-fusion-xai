"""Stage 09h: what a model-free map scores on the same localisation metrics.

Every localisation number in Track 2 -- pointing accuracy, Dice, IoU, energy -- is reported
without a floor. That omission is not survivable here, because the patches are centred on
the nodule by construction: `_crop_and_resample_nodule` cuts a fixed 64x64x16 mm window
around the consensus-mask centroid, and `stage_07b_fixed_samples` verifies the centring
before a sample is allowed into the display set. A blob sitting at the middle of the image,
produced by nothing at all, therefore scores above chance by design.

Until that number is on the page, "the model localises the nodule" cannot be falsified.
This stage puts it there, using the same samples, the same mask slice and the same metric
functions as `stage_05_xai`, so the result drops straight into the same table.

Four model-free maps are scored, weakest claim first:

  uniform         - a constant map. Degenerate control: argmax falls on index (0,0), so
                    pointing accuracy collapses. Shows what "no information" looks like.
  random          - seeded uniform noise. The empirical chance level.
  center_point    - an isotropic Gaussian one pixel wide at the exact centre. The strongest
                    possible centre prior, and the pointing-game ceiling a centre bias can
                    reach.
  center_gaussian - an isotropic Gaussian with sigma = side/6, roughly the extent of a real
                    upsampled CAM blob. The honest analogue of a model's map.

`mask_area_frac_mean` is reported alongside, because it is the analytic chance level for
both pointing accuracy and energy: a map with no spatial preference hits the mask exactly
as often as the mask covers the patch.

No model is loaded and no checkpoint is read, so this is CPU-only and runs in seconds.
`checkpoint_mtime` is carried as "none" rather than dropped, so the column set still
matches every other result CSV.

Run:
    .venv/Scripts/python.exe -m src.stage_09h_center_baseline
    .venv/Scripts/python.exe -m src.stage_09h_center_baseline --self-check

Remember `git add -f`: artifacts/ is gitignored in full, so the output will not appear in
a plain `git status`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd
import yaml

from src.utils.io import cached
from src.xai.gradcam_utils import (
    dice_iou,
    dice_size_matched,
    energy_pointing_game,
    pointing_hit,
)

RUN_ID = "2026-10-14-center-baseline"
CONFIG_PATH = "configs/config.yaml"
OUT_DIR = "artifacts/results/center_baseline"
OUT_CSV = os.path.join(OUT_DIR, "center_baseline.csv")
OUT_PERSAMPLE = os.path.join(OUT_DIR, "center_baseline_persample.csv")

# Same cap and seed as stage_05_xai.py:15,55 -- the comparison is void if either drifts.
N_SAMPLES_METRICS = 60
SAMPLE_SEED = 42

COLUMNS = [
    "run_id", "commit_sha", "input_size", "checkpoint_mtime",
    "sample_set", "baseline", "n", "mask_area_frac_mean",
    "pointing_acc", "dice", "iou", "dice_size_matched", "energy_mean", "threshold_pct",
]


def _commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _gaussian_map(shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Isotropic Gaussian centred on the geometric centre of the map.

    The centre is (n - 1) / 2, not n // 2: on an even-sided patch the true middle falls
    between two pixels, and rounding it to one of them would shift the blob half a pixel
    off the centre the crop was built around.
    """
    h, w = shape
    yy = np.arange(h)[:, None] - (h - 1) / 2.0
    xx = np.arange(w)[None, :] - (w - 1) / 2.0
    return np.exp(-(yy ** 2 + xx ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


def _baseline_maps(shape: tuple[int, int], rng: np.random.Generator) -> dict[str, np.ndarray]:
    side = float(min(shape))
    return {
        "uniform": np.ones(shape, dtype=np.float32),
        "random": rng.random(shape).astype(np.float32),
        "center_point": _gaussian_map(shape, sigma=1.0),
        "center_gaussian": _gaussian_map(shape, sigma=side / 6.0),
    }


def _resolve_mask(mask_path: str, interim: str) -> str:
    """Rebase a labels.csv mask path onto the configured patch root.

    labels.csv stores `artifacts/patches/<patient>/<nodule>/mask.npy`, relative to whatever
    repo root wrote it. Reading those patches from another checkout -- or from a backup on
    a different drive -- makes the stored path wrong while `paths.interim` is right, so the
    last three components are rejoined onto it. The stored path still wins when it resolves,
    which keeps the normal in-repo run byte-identical.
    """
    if os.path.exists(mask_path):
        return mask_path
    parts = mask_path.replace("\\", "/").split("/")
    return os.path.join(interim, *parts[-3:])


def _mask2d(mask_path: str, interim: str) -> np.ndarray:
    """Central slice of the 3-D mask -- the same slice stage_05_xai.py:126-128 scores."""
    mask_full = np.load(_resolve_mask(mask_path, interim)).astype(np.float32)
    return mask_full[mask_full.shape[0] // 2]


def _sample_sets(cfg: dict) -> dict[str, pd.DataFrame]:
    """fold0_60 reproduces stage_05_xai's metric subset; fixed6 is the display set."""
    labels_path = os.path.join(cfg["paths"]["interim"], "labels.csv")
    df = pd.read_csv(labels_path)

    fold0 = df[(df["fold"] == 0) & (df["label"] != -1)].reset_index(drop=True)
    if len(fold0) > N_SAMPLES_METRICS:
        fold0 = fold0.sample(n=N_SAMPLES_METRICS, random_state=SAMPLE_SEED).reset_index(drop=True)
    sets = {"fold0_60": fold0}

    fixed_json = cfg.get("paths", {}).get("fixed_samples")
    if not fixed_json:
        fixed_json = os.path.join(os.path.dirname(cfg["paths"]["splits"]), os.pardir,
                                  "xai", "fixed_display_samples.json")
    fixed_json = os.path.normpath(fixed_json)
    if os.path.exists(fixed_json):
        with open(fixed_json, encoding="utf-8") as f:
            picked = json.load(f)["samples"]
        # stage_07b has written this as both a slot->sample mapping and a flat list; accept
        # either rather than pinning one and breaking when the other is regenerated.
        entries = picked.values() if isinstance(picked, dict) else picked
        keys = {(s["patient_id"], s["nodule_idx"]) for s in entries}
        rows = df[df.apply(lambda r: (r["patient_id"], r["nodule_idx"]) in keys, axis=1)]
        if len(rows):
            sets["fixed6"] = rows.reset_index(drop=True)
    else:
        print(f"[LEWAT] fixed6: {fixed_json} tidak ada")
    return sets


def run(cfg: dict) -> pd.DataFrame | None:
    if cached(OUT_CSV) and not cfg.get("force_rerun", False):
        print(f"[SKIP] {OUT_CSV} sudah ada")
        return None

    sha = _commit_sha()
    input_size = cfg.get("tracks", {}).get("track1", {}).get("input_size")
    rng = np.random.default_rng(SAMPLE_SEED)

    rows, per_sample = [], []
    for set_name, sdf in _sample_sets(cfg).items():
        acc: dict[str, dict[str, list]] = {}
        areas = []
        for _, row in sdf.iterrows():
            mask = _mask2d(row["mask_path"], cfg["paths"]["interim"])
            areas.append(float(mask.astype(bool).mean()))
            for bname, cam in _baseline_maps(mask.shape, rng).items():
                d, iou = dice_iou(cam, mask)
                rec = {
                    "pointing": float(pointing_hit(cam, mask)),
                    "dice": d,
                    "iou": iou,
                    "dice_sm": dice_size_matched(cam, mask),
                    "energy": energy_pointing_game(cam, mask),
                }
                bucket = acc.setdefault(bname, {k: [] for k in rec})
                for k, v in rec.items():
                    bucket[k].append(v)
                per_sample.append({
                    "run_id": RUN_ID, "commit_sha": sha, "sample_set": set_name,
                    "baseline": bname, "patient_id": row["patient_id"],
                    "nodule_idx": row["nodule_idx"], "label": row["label"],
                    "mask_area_frac": areas[-1], **rec,
                })

        for bname, m in acc.items():
            rows.append({
                "run_id": RUN_ID, "commit_sha": sha, "input_size": input_size,
                "checkpoint_mtime": "none",
                "sample_set": set_name, "baseline": bname, "n": len(sdf),
                "mask_area_frac_mean": float(np.mean(areas)),
                "pointing_acc": float(np.mean(m["pointing"])),
                "dice": float(np.mean(m["dice"])),
                "iou": float(np.mean(m["iou"])),
                "dice_size_matched": float(np.mean(m["dice_sm"])),
                "energy_mean": float(np.mean(m["energy"])),
                "threshold_pct": 0.80,
            })

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(OUT_CSV, index=False)
    pd.DataFrame(per_sample).to_csv(OUT_PERSAMPLE, index=False)

    show = ["sample_set", "baseline", "n", "mask_area_frac_mean",
            "pointing_acc", "dice", "dice_size_matched", "energy_mean"]
    print(df[show].to_string(index=False))
    print(f"\n[DITULIS] {OUT_CSV}  ({len(df)} baris, run_id={RUN_ID}, commit={sha})")
    print(f"[DITULIS] {OUT_PERSAMPLE}  ({len(per_sample)} baris)")
    print("  pembanding xai_metrics.csv: densenet121 0.717 | convnext_tiny 0.717 | "
          "densenet201 0.700 | googlenet 0.000")
    print(f"  jangan lupa: git add -f {OUT_CSV} {OUT_PERSAMPLE}")
    return df


def self_check() -> None:
    """Assert the baselines behave on masks whose answer is known by construction.

    Catches what would silently produce a flattering floor: a centre that is off by half a
    pixel, a blob that is not actually centred, or metrics wired to the wrong argument.
    """
    shape = (64, 64)
    rng = np.random.default_rng(0)
    maps = _baseline_maps(shape, rng)

    # A mask covering the middle: every centre-biased map must point inside it.
    centre_mask = np.zeros(shape, dtype=np.float32)
    centre_mask[28:36, 28:36] = 1.0
    for name in ("center_point", "center_gaussian"):
        assert pointing_hit(maps[name], centre_mask), f"{name} tidak menunjuk ke tengah"

    # A mask in the corner only: the centre prior must now miss. If it still hits, the blob
    # is not where this stage claims it is.
    corner_mask = np.zeros(shape, dtype=np.float32)
    corner_mask[:8, :8] = 1.0
    for name in ("center_point", "center_gaussian"):
        assert not pointing_hit(maps[name], corner_mask), f"{name} menunjuk ke sudut"

    # The Gaussian must be symmetric about the true centre, or it is biased toward one
    # quadrant and every number below inherits that bias.
    g = _gaussian_map(shape, sigma=8.0)
    assert np.allclose(g, g[::-1, :], atol=1e-6), "gaussian tidak simetris vertikal"
    assert np.allclose(g, g[:, ::-1], atol=1e-6), "gaussian tidak simetris horizontal"

    # energy_pointing_game subtracts the map minimum before summing, so a constant map is
    # annihilated and scores exactly 0 -- it measures energy relative to contrast, not raw
    # energy. Pinned here because the natural assumption (uniform scores the area fraction)
    # is wrong, and a reader comparing a model's energy against "chance" needs the right
    # floor: it comes from `random`, not from `uniform`.
    assert energy_pointing_game(maps["uniform"], centre_mask) == 0.0, \
        "peta konstan seharusnya nol energi setelah pengurangan minimum"

    frac = float(centre_mask.mean())
    e_rand = energy_pointing_game(maps["random"], centre_mask)
    assert abs(e_rand - frac) < 0.02, f"energi acak {e_rand} jauh dari fraksi area {frac}"

    print("[OK] self-check lolos")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--self-check", action="store_true", help="jalankan assert saja, tanpa menulis")
    args = p.parse_args()

    if args.self_check:
        self_check()
        return

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)


if __name__ == "__main__":
    main()
