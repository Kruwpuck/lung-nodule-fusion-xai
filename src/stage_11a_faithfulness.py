"""Stage 11a (rev3 Fase 1): faithfulness of the run03 CAM maps.

Why this stage exists. Every explainability number this project reports measures
*localisation* -- dice, IoU, pointing accuracy, energy pointing game -- and all four ask
the same question: does the map land on the nodule. None of them asks whether the map
reflects what the classifier actually used. `paper/track1/main.tex:305` states that gap
as an explicit open item. This stage closes it.

Two families, both on the fixed 60-nodule sample of each fold, both from the same run03
uf100 checkpoints stage_10d scored:

  * ROAD (Rong et al., ICML 2022) -- remove the most-relevant pixels first (MoRF) and the
    least-relevant first (LeRF) at nine percentiles, imputing rather than blanking so the
    perturbation itself does not become the signal. A faithful map gives MoRF well below
    LeRF. `road_combined` folds the pair into one number, higher is more faithful.
  * PGI / PGU (prediction gap on important / unimportant features) -- Gaussian noise on
    the top-k percent of the map versus the bottom-k, R repeats, mean absolute change in
    predicted probability. Needs no ground-truth mask, so it complements pointing
    accuracy, which does.

Nothing here implements a perturbation scheme of its own: `pytorch_grad_cam.metrics.road`
is already installed and is the reference implementation.

`fusion_late` is not measured separately, and that is a result rather than an omission.
Late fusion is `average_fusion(cnn_prob, rad_prob, weight_cnn=0.5)`
(`stage_03b_fusion.py:388`). A ROAD perturbation moves pixels; the radiomics branch is
computed from the unperturbed patch and is constant under it. So

    ROAD(fusion_late) = 0.5 * ROAD(cnn_only)

exactly, and the `*_fusion_late` columns carry that derivation rather than a second
measurement. What it means is worth stating plainly in the manuscript: averaging a
perturbation-invariant branch into the score halves the measured degradation, so late
fusion makes an unfaithful map *harder to detect by perturbation testing*, without
making it any more faithful.

Caveat that has to travel with these numbers (rev3 section 1.1): Tempel et al. report
that perturbation tests carry a built-in bias favouring spatial explanations over
feature-attribution ones. Track 1 compares spatial CAM against feature SHAP, so the
caveat belongs where the result is reported, not only in Limitations.

Outputs (nothing stage_10d wrote is touched):
    artifacts/results/run03/faithfulness_run03.csv            one row per backbone per fold
    artifacts/results/run03/faithfulness_run03_persample.csv  one row per sample

Run:
    .venv/Scripts/python.exe -m src.stage_11a_faithfulness
    .venv/Scripts/python.exe -m src.stage_11a_faithfulness --self-check
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from pytorch_grad_cam.metrics.road import (
    ROADLeastRelevantFirstAverage,
    ROADMostRelevantFirstAverage,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget

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
from src.utils.tracks import track_input_size
from src.xai.gradcam_utils import compute_gradcam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join("configs", "config.yaml")
OUT_DIR = os.path.join("artifacts", "results", "run03")
OUT_CSV = os.path.join(OUT_DIR, "faithfulness_run03.csv")
PERSAMPLE_CSV = os.path.join(OUT_DIR, "faithfulness_run03_persample.csv")

# ROAD's own defaults. Nine percentiles, both directions, averaged.
ROAD_PERCENTILES = [10, 20, 30, 40, 50, 60, 70, 80, 90]

# PGI/PGU. k is the fraction of the map treated as important; R is repeats, because a
# single noise draw is a sample of size one and would make the metric a coin flip.
PGI_K = 0.20
PGI_REPEATS = 10
# Inputs are min-max scaled from HU [-1000, 400] (`_load_patch_tensor`), so one unit of
# input equals 1400 HU. sigma 0.10 is therefore a 140 HU disturbance: large enough to
# matter on a CT patch, far short of erasing the region.
PGI_SIGMA = 0.10

# The weight late fusion gives the CNN branch, stage_03b_fusion.py:388. The radiomics
# branch is constant under pixel perturbation, so this is the exact factor by which any
# perturbation-based degradation of the CNN branch reaches the fused score.
LATE_FUSION_CNN_WEIGHT = 0.5


def _cam_map(model, img: torch.Tensor, internal: str) -> np.ndarray:
    """The canonical CAM for one sample, as float32 in [0, 1] at input resolution."""
    cam = compute_gradcam(model, img, backbone_name=internal).astype(np.float32)
    lo, hi = float(cam.min()), float(cam.max())
    return (cam - lo) / (hi - lo) if hi > lo else np.zeros_like(cam)


def _road_pair(model, img: torch.Tensor, cam: np.ndarray, pred: int) -> tuple[float, float]:
    """(MoRF, LeRF) averaged across the nine percentiles.

    Both are signed changes in the predicted-class probability after imputation, so both
    are normally negative and a faithful map is the one whose MoRF is the more negative.
    """
    targets = [ClassifierOutputSoftmaxTarget(pred)]
    cams = cam[None, ...]
    morf = ROADMostRelevantFirstAverage(percentiles=ROAD_PERCENTILES)
    lerf = ROADLeastRelevantFirstAverage(percentiles=ROAD_PERCENTILES)
    return (float(np.ravel(morf(img, cams, targets, model))[0]),
            float(np.ravel(lerf(img, cams, targets, model))[0]))


def _quantile_mask(cam: np.ndarray, k: float, most_important: bool) -> np.ndarray:
    """Boolean mask over the k fraction of pixels with the highest (or lowest) CAM value.

    Rank-based rather than value-based: a map whose mass sits in a few saturated pixels
    and a map that is nearly flat both hand back exactly k*H*W pixels, so PGI and PGU stay
    comparable across backbones instead of drifting with the map's dynamic range.
    """
    flat = cam.ravel()
    n = max(1, int(round(k * flat.size)))
    order = np.argsort(flat, kind="stable")
    picked = order[-n:] if most_important else order[:n]
    mask = np.zeros(flat.size, dtype=bool)
    mask[picked] = True
    return mask.reshape(cam.shape)


def _prediction_gap(model, img: torch.Tensor, mask: np.ndarray, pred: int,
                    base_prob: float, rng: np.random.Generator) -> float:
    """Mean |probability change| over PGI_REPEATS Gaussian draws inside `mask`."""
    m = torch.from_numpy(mask.astype(np.float32)).to(img.device)
    gaps = []
    for _ in range(PGI_REPEATS):
        noise = torch.from_numpy(
            rng.normal(0.0, PGI_SIGMA, size=tuple(img.shape)).astype(np.float32)
        ).to(img.device)
        with torch.no_grad():
            p = F.softmax(model(img + noise * m), dim=1)[0, pred].item()
        gaps.append(abs(p - base_prob))
    return float(np.mean(gaps))


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
    # Seeded per cell, not per process: re-running one backbone must reproduce that
    # backbone's numbers whether or not the other two ran first.
    rng = np.random.default_rng(SAMPLE_SEED + fold)

    per_sample = []
    for i, row in samples.iterrows():
        img = _load_patch_tensor(row["patch_path"], n_slices, patch_xy).to(device)
        with torch.no_grad():
            logits = model(img)
            pred = int(logits.argmax(dim=1).item())
            base_prob = float(F.softmax(logits, dim=1)[0, pred].item())

        cam = _cam_map(model, img, internal)
        morf, lerf = _road_pair(model, img, cam, pred)
        pgi = _prediction_gap(model, img, _quantile_mask(cam, PGI_K, True), pred, base_prob, rng)
        pgu = _prediction_gap(model, img, _quantile_mask(cam, PGI_K, False), pred, base_prob, rng)

        case = f"{row['patient_id']}#{int(row['nodule_idx'])}"
        per_sample.append({
            "run_id": RUN_ID, "commit_sha": sha, "backbone": backbone,
            "unfreeze_pct": UNFREEZE_PCT, "fold": fold, "sample_idx": int(i),
            "patient_id": row["patient_id"], "nodule_idx": int(row["nodule_idx"]),
            "in_display_set": int(case in display),
            "label": int(row["label"]), "pred": pred, "prob_pred": base_prob,
            "road_morf": morf, "road_lerf": lerf, "road_combined": (lerf - morf) / 2.0,
            "pgi": pgi, "pgu": pgu, "pg_ratio": pgi / pgu if pgu > 0 else np.nan,
            "cam_max": float(cam.max()),
        })

    del model
    torch.cuda.empty_cache()

    ps = pd.DataFrame(per_sample)
    summary = {
        "run_id": RUN_ID, "commit_sha": sha, "backbone": backbone,
        "internal_name": internal, "unfreeze_pct": UNFREEZE_PCT, "fold": fold,
        "n": len(ps), "input_size": input_size, **layer,
        "checkpoint": ckpt.replace("\\", "/"), "checkpoint_mtime": _mtime(ckpt),
        "road_percentiles": "|".join(str(p) for p in ROAD_PERCENTILES),
        "pgi_k": PGI_K, "pgi_sigma": PGI_SIGMA, "pgi_repeats": PGI_REPEATS,
        "road_morf": ps["road_morf"].mean(),
        "road_lerf": ps["road_lerf"].mean(),
        "road_combined": ps["road_combined"].mean(),
        # Derived, not measured -- see the module docstring. Kept as an explicit column so
        # the halving is visible in the table instead of being asserted in prose.
        "road_morf_fusion_late": ps["road_morf"].mean() * LATE_FUSION_CNN_WEIGHT,
        "road_combined_fusion_late": ps["road_combined"].mean() * LATE_FUSION_CNN_WEIGHT,
        "late_fusion_cnn_weight": LATE_FUSION_CNN_WEIGHT,
        "pgi": ps["pgi"].mean(), "pgu": ps["pgu"].mean(),
        "pg_ratio": ps["pgi"].mean() / ps["pgu"].mean() if ps["pgu"].mean() > 0 else np.nan,
        "n_faithful_ordering": int((ps["road_morf"] < ps["road_lerf"]).sum()),
        "n_zero_cam": int((ps["cam_max"] <= 0).sum()),
        "n_display_set": int(ps["in_display_set"].sum()),
    }
    return summary, ps


def self_check(df: pd.DataFrame, persample: pd.DataFrame) -> None:
    """Asserts these numbers have to survive before they may enter the manuscript."""
    # 1. Same degenerate-site guard stage_10d carries: a 1x1 map localises nothing, so a
    #    faithfulness score computed on one measures the pooling layer, not the map.
    degenerate = df[(df["spatial_h"].astype(str) == "1") & (df["spatial_w"].astype(str) == "1")]
    assert degenerate.empty, (
        f"1x1 target layer: {degenerate[['backbone', 'fold']].to_dict('records')}")

    # 2. Provenance. A row whose checkpoint mtime is missing cannot be checked for
    #    staleness later, which is exactly the densenet121 incident F-4 was built around.
    missing = df[df["checkpoint_mtime"].fillna("") == ""]
    assert missing.empty, (
        f"checkpoint_mtime kosong: {missing[['backbone', 'fold']].to_dict('records')}")

    # 3. PGI and PGU are mean absolute probability changes, so both are non-negative by
    #    construction. A negative value means an absolute value went missing somewhere --
    #    a coding error, not a finding.
    assert (persample["pgu"] >= 0).all() and (persample["pgi"] >= 0).all(), \
        "PGI/PGU negatif: gapnya bukan nilai mutlak"

    # 4. The cell grid is complete. A missing cell would otherwise average away silently.
    assert len(df) == len(BACKBONES) * N_FOLDS, f"{len(df)} sel, harus {len(BACKBONES) * N_FOLDS}"

    # 5. Not an assert: the faithfulness ordering itself is a finding either way. A cell
    #    where MoRF does not sit below LeRF says the map is uninformative about the
    #    decision -- which is the Track 2 thesis, not a bug -- so it is reported, loudly,
    #    rather than crashing the stage.
    unfaithful = df[df["road_morf"] >= df["road_lerf"]]
    if not unfaithful.empty:
        print("[SELF-CHECK] PERHATIAN: MoRF >= LeRF (peta nol informatif soal keputusan) di "
              f"{unfaithful[['backbone', 'fold']].to_dict('records')}")
    print("[SELF-CHECK] keempat assert lolos")


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
            # Written after every cell: a crash on cell 12 must not cost cells 1-11.
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            pd.concat(per_sample, ignore_index=True).to_csv(PERSAMPLE_CSV, index=False)
            logger.info("[%s fold %d] MoRF=%.4f LeRF=%.4f combined=%.4f PGI=%.4f PGU=%.4f "
                        "faithful %d/%d",
                        backbone, fold, summary["road_morf"], summary["road_lerf"],
                        summary["road_combined"], summary["pgi"], summary["pgu"],
                        summary["n_faithful_ordering"], summary["n"])

    df = pd.DataFrame(rows)
    ps_all = pd.concat(per_sample, ignore_index=True)

    with pd.option_context("display.max_columns", None, "display.width", 250):
        print("\n=== faithfulness, checkpoint run03 uf100 ===")
        print(df[["backbone", "fold", "n", "road_morf", "road_lerf", "road_combined",
                  "pgi", "pgu", "pg_ratio", "n_faithful_ordering",
                  "n_zero_cam"]].to_string(index=False))
        print("\n=== rata-rata lintas fold ===")
        print(df.groupby("backbone")[["road_morf", "road_lerf", "road_combined", "pgi",
                                      "pgu"]].mean().to_string())

    print(f"\n[WROTE] {OUT_CSV} ({len(df)} baris)")
    print(f"[WROTE] {PERSAMPLE_CSV} ({len(ps_all)} baris)")

    if args.self_check:
        self_check(df, ps_all)


if __name__ == "__main__":
    main()
