"""Stage 09d: recompute CAM localisation metrics for all twelve backbones under the
corrected target-layer rule, in one pass, one run_id, one commit.

Why this stage exists. Until now the CAM target layer was chosen by searching for a
feature map whose height fell in [7, 10]. Six of the twelve backbones never produce such
a height and fell through to a hand-written table; for GoogLeNet that table pointed at a
Dropout sitting after the global average pool, so the CAM was 1x1, normalised to
identically zero, and its published pointing accuracy and energy were exactly 0.0. The
band premise itself was measured and is false (artifacts/results/track2rev/
googlenet_layer_sweep.csv). `_last_spatial_target_layer` now applies the canonical rule
instead -- explain at the last layer that still has spatial extent -- to all twelve, and
this stage measures what that does to every backbone's numbers.

The sample set, patch loading, model construction and checkpoint loading are the ones
`src/stage_05_xai.py` used for the published numbers: fold 0, labelled nodules,
`sample(n=60, random_state=42)`. The metric primitives are imported from
`src.xai.gradcam_utils`, not reimplemented, so nothing can drift from the published
definitions. Only the target layer differs, which is the point of the comparison.

Outputs (artifacts/results/xai/xai_metrics.csv is left untouched as the historical trace):
    artifacts/results/track2rev/cam_12.csv            one row per backbone, with each
        metric's published value and the delta alongside the new value
    artifacts/results/track2rev/cam_12_persample.csv  per-sample metrics, which nothing
        in this repo persisted before, so dice and energy confidence intervals become
        computable for the first time

Run:
    .venv/Scripts/python.exe -m src.stage_09d_cam_12
    .venv/Scripts/python.exe -m src.stage_09d_cam_12 --self-check
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml

from src.models.registry import _NAME_MAP, build_model
from src.stage_08b_run02_xai import _load_patch_tensor  # exact copy of stage_05's loader
from src.stage_09a_target_layer_audit import _module_path, _observe
from src.xai.gradcam_utils import (
    _get_target_layer,
    _last_spatial_target_layer,
    compute_gradcam,
    dice_iou,
    dice_size_matched,
    energy_pointing_game,
    pointing_hit,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUN_ID = "2026-08-20-run04"
N_SAMPLES = 60          # stage_05_xai.N_SAMPLES_METRICS
SAMPLE_SEED = 42        # stage_05_xai line 55
THRESHOLD_PCT = 0.80

CONFIG_PATH = os.path.join("configs", "config.yaml")
OUT_DIR = os.path.join("artifacts", "results", "track2rev")
OUT_CSV = os.path.join(OUT_DIR, "cam_12.csv")
PERSAMPLE_CSV = os.path.join(OUT_DIR, "cam_12_persample.csv")
GUARD_CSV = os.path.join(OUT_DIR, "track1_guard.csv")
PUBLISHED_CSV = os.path.join("artifacts", "results", "xai", "xai_metrics.csv")

BACKBONES = [
    "mobilenetv3_small", "efficientnet_b0", "densenet121", "resnet50", "vgg16",
    "vit_base", "inceptionv3", "xception", "googlenet", "convnext_tiny",
    "inception_resnet_v2", "densenet201",
]
TRACK1_BACKBONES = ["densenet121", "densenet201", "convnext_tiny"]
METRICS = ["dice", "iou", "dice_size_matched", "pointing_acc", "energy_mean"]


def _commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _mtime(path: str) -> str:
    if not os.path.exists(path):
        return ""
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")


def _samples(cfg: dict) -> pd.DataFrame:
    """The 60 fold-0 nodules stage_05_xai measured on (lines 53-55), same seed."""
    df = pd.read_csv(os.path.join(cfg["paths"]["interim"], "labels.csv"))
    val_df = df[(df["fold"] == 0) & (df["label"] != -1)].reset_index(drop=True)
    if len(val_df) > N_SAMPLES:
        val_df = val_df.sample(n=N_SAMPLES, random_state=SAMPLE_SEED).reset_index(drop=True)
    return val_df


def _load_model(backbone: str, cfg: dict, device):
    """Same construction and checkpoint load as stage_05_xai lines 87-96."""
    best_pt = os.path.join(cfg["paths"]["checkpoints"], backbone, "fold0_best.pt")
    model = build_model(backbone, cfg, task="binary").to(device)
    try:
        state = torch.load(best_pt, weights_only=True, map_location="cpu")
    except TypeError:
        state = torch.load(best_pt, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model, best_pt


def _resolve_layer(model, internal: str, dummy):
    """Mirror compute_gradcam's dispatch and report what it landed on."""
    is_vit = "vit" in internal.lower()
    target = None if is_vit else _last_spatial_target_layer(model, dummy)
    path_taken = "auto" if target is not None else "fallback"
    if target is None:
        target = _get_target_layer(model, internal)
    _, h, w = _observe(model, dummy, target)
    return {
        "path_taken": path_taken,
        "module_class": type(target).__name__,
        "module_path": _module_path(model, target),
        "spatial_h": "" if h is None else h,
        "spatial_w": "" if w is None else w,
    }


def run_backbone(backbone: str, cfg: dict, samples: pd.DataFrame, device, sha: str):
    """Return (summary row, per-sample rows) for one backbone."""
    import torch.nn.functional as F

    internal = _NAME_MAP.get(backbone, backbone)
    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)

    model, ckpt = _load_model(backbone, cfg, device)
    dummy = torch.zeros(1, n_slices, patch_xy, patch_xy, device=device)
    layer = _resolve_layer(model, internal, dummy)

    per_sample = []
    for i, row in samples.iterrows():
        img = _load_patch_tensor(row["patch_path"], n_slices, patch_xy).to(device)
        with torch.no_grad():
            logits = model(img)
            prob = F.softmax(logits, dim=1)[0, 1].item()
            pred = int(logits.argmax(dim=1).item())

        cam = compute_gradcam(model, img, backbone_name=internal)
        mask_full = np.load(row["mask_path"]).astype(np.float32)
        mask2d = mask_full[mask_full.shape[0] // 2]

        d, iou = dice_iou(cam, mask2d, pct=THRESHOLD_PCT)
        per_sample.append({
            "run_id": RUN_ID, "commit_sha": sha, "backbone": backbone, "sample_idx": int(i),
            "patient_id": row["patient_id"], "nodule_idx": int(row["nodule_idx"]),
            "label": int(row["label"]), "pred": pred, "prob_malignant": prob,
            "dice": d, "iou": iou,
            "dice_size_matched": dice_size_matched(cam, mask2d),
            "pointing_hit": int(pointing_hit(cam, mask2d)),
            "energy": energy_pointing_game(cam, mask2d),
            "cam_max": float(cam.max()), "cam_min": float(cam.min()),
            "mask_px": int(mask2d.astype(bool).sum()),
        })

    del model
    torch.cuda.empty_cache()

    ps = pd.DataFrame(per_sample)
    summary = {
        "run_id": RUN_ID, "commit_sha": sha, "backbone": backbone, "internal_name": internal,
        "n": len(ps), **layer, "threshold_pct": THRESHOLD_PCT,
        "checkpoint_mtime": _mtime(ckpt), "published_mtime": _mtime(PUBLISHED_CSV),
        "dice": ps["dice"].mean(), "iou": ps["iou"].mean(),
        "dice_size_matched": ps["dice_size_matched"].mean(),
        "pointing_acc": ps["pointing_hit"].mean(), "energy_mean": ps["energy"].mean(),
        "n_zero_cam": int((ps["cam_max"] <= 0).sum()),
    }
    return summary, ps


def _with_published(rows: list[dict]) -> pd.DataFrame:
    """Attach each metric's published value from the historical CSV, plus the delta."""
    df = pd.DataFrame(rows)
    pub = pd.read_csv(PUBLISHED_CSV).set_index("backbone")
    for m in METRICS:
        df[f"{m}_published"] = df["backbone"].map(pub[m])
        df[f"{m}_delta"] = df[m] - df[f"{m}_published"]
    lead = ["run_id", "commit_sha", "backbone", "internal_name", "n", "path_taken",
            "module_class", "module_path", "spatial_h", "spatial_w"]
    metric_cols = [c for m in METRICS for c in (m, f"{m}_published", f"{m}_delta")]
    rest = [c for c in df.columns if c not in lead + metric_cols]
    return df[lead + metric_cols + rest]


def self_check(df: pd.DataFrame, persample: pd.DataFrame) -> None:
    """Three asserts the corrected resolver has to survive."""
    # 1. Nothing may explain a post-global-pool 1x1 map. ViT has no spatial map at all
    #    and is checked by its reshape transform instead, so it is exempt by construction.
    degenerate = df[(df["spatial_h"].astype(str) == "1") & (df["spatial_w"].astype(str) == "1")]
    assert degenerate.empty, f"1x1 target layer: {degenerate['backbone'].tolist()}"

    # 2. A CAM that min-max normalises to identically zero carries no localisation at all.
    #    vit_base is exempt and only vit_base: its path is untouched by this change (it has
    #    no 4-D intermediates, resolves through _get_target_layer and reproduces its
    #    published numbers to 1e-17), and it produced identically-zero maps on 53 of the 60
    #    samples before the change as well. That is a pre-existing property of explaining a
    #    LayerNorm through a reshape transform, reported as a finding, not repaired here.
    #    Every other backbone must produce a live map on every sample.
    dead = persample[(persample["cam_max"] <= 0) & (persample["backbone"] != "vit_base")]
    assert dead.empty, (f"{len(dead)} identically-zero CAMs: "
                        f"{dead[['backbone', 'sample_idx']].to_dict('records')[:10]}")

    # 3. The three backbones the published Track 1 manuscript rests on must still explain
    #    the same tensor. The guard CSV holds the before/after record; equality of its
    #    cam_sha1 columns is the actual proof, since a container and its last Sequential
    #    child emit the same tensor under different names (convnext_tiny resolves to
    #    features.0.7.2, the last child of the features.0.7 recorded before the change).
    guard = pd.read_csv(GUARD_CSV)
    before = guard[guard["tag"] == "before"].set_index("backbone")
    after = guard[guard["tag"] == "after"].set_index("backbone")
    for b in TRACK1_BACKBONES:
        now = df.loc[df["backbone"] == b, "module_path"].iloc[0]
        assert now == after.loc[b, "module_path"], (
            f"{b}: resolves to {now}, guard recorded {after.loc[b, 'module_path']}")
        assert before.loc[b, "cam_sha1"] == after.loc[b, "cam_sha1"], (
            f"{b}: CAM fingerprint moved, {before.loc[b, 'cam_sha1']} -> "
            f"{after.loc[b, 'cam_sha1']}")
    print("[SELF-CHECK] all three asserts passed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--self-check", action="store_true", help="run the post-hoc asserts")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = _samples(cfg)
    sha = _commit_sha()

    rows, per_sample = [], []
    for backbone in BACKBONES:
        summary, ps = run_backbone(backbone, cfg, samples, device, sha)
        rows.append(summary)
        per_sample.append(ps)
        # Written after every backbone: a crash on backbone 9 must not cost backbones 1-8.
        _with_published(rows).to_csv(OUT_CSV, index=False)
        pd.concat(per_sample, ignore_index=True).to_csv(PERSAMPLE_CSV, index=False)
        logger.info("[%s] %s %s (%sx%s) pointing_acc=%.4f dice=%.4f energy=%.4f",
                    backbone, summary["path_taken"], summary["module_path"],
                    summary["spatial_h"], summary["spatial_w"],
                    summary["pointing_acc"], summary["dice"], summary["energy_mean"])

    df = _with_published(rows)
    ps_all = pd.concat(per_sample, ignore_index=True)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df[["backbone", "module_class", "module_path", "spatial_h", "spatial_w",
                  "pointing_acc", "pointing_acc_published", "pointing_acc_delta",
                  "energy_mean", "energy_mean_delta", "dice", "dice_delta"]].to_string(index=False))
    print(f"\n[WROTE] {OUT_CSV} ({len(df)} rows)")
    print(f"[WROTE] {PERSAMPLE_CSV} ({len(ps_all)} rows)")

    if args.self_check:
        self_check(df, ps_all)


if __name__ == "__main__":
    main()
