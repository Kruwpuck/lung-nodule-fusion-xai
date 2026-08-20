"""Stage 09a: audit which layer the Grad-CAM machinery actually explains, per backbone.

Stage 05 never names a target layer explicitly. For every non-ViT backbone it first asks
`_auto_target_layer` for the deepest 4-D module whose feature-map height falls inside the
[7, 10] band, and only when that search comes back empty does it fall back to the
hand-written substring table in `_get_target_layer`. Those two paths can land on very
different modules, and a module that sits after global pooling produces a 1x1 map, which
makes every localisation metric degenerate. This stage records, for each backbone, which
path won, which module was chosen, how large that module's output is, and every distinct
4-D feature-map height observed during a single forward pass. The last column is what
tells us whether the [7, 10] band was reachable at all.

Weights are deliberately not loaded. Layer resolution depends only on the architecture and
on the input size the model resizes to, never on the parameter values, so `pretrained=False`
and no checkpoint load. The checkpoints are up to 650 MB each and loading twelve of them
would cost minutes of I/O for exactly zero change in the answer.

Run:
    .venv/Scripts/python.exe -m src.stage_09a_target_layer_audit
"""
from __future__ import annotations

import os
import subprocess

import pandas as pd
import torch
import yaml

from src.models.backbones import BackboneClassifier
from src.models.registry import _NAME_MAP
from src.utils.tracks import track_input_size
from src.xai.gradcam_utils import _auto_target_layer, _get_target_layer

RUN_ID = "2026-08-20-run04"
BAND_LO, BAND_HI = 7, 10  # the band `_auto_target_layer` searches, recorded so the CSV self-describes

CONFIG_PATH = os.path.join("configs", "config.yaml")
OUT_DIR = os.path.join("artifacts", "results", "track2rev")
OUT_CSV = os.path.join(OUT_DIR, "target_layer_audit.csv")

BACKBONES = [
    "mobilenetv3_small",
    "efficientnet_b0",
    "densenet121",
    "resnet50",
    "vgg16",
    "vit_base",
    "inceptionv3",
    "xception",
    "googlenet",
    "convnext_tiny",
    "inception_resnet_v2",
    "densenet201",
]

COLUMNS = [
    "run_id", "commit_sha", "backbone", "internal_name", "input_size", "path_taken",
    "module_class", "module_path", "spatial_h", "spatial_w", "heights_seen",
    "band_lo", "band_hi", "error",
]


def _commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _observe(model, sample_input, target_module):
    """Run one forward pass and report the 4-D heights seen plus the target's output size.

    Returns a tuple of (sorted distinct heights, target height, target width). The target
    height and width are None when the chosen module never emitted a 4-D tensor, which is
    itself a meaningful finding.
    """
    heights: set[int] = set()
    target_shape: list = [None, None]

    def hook(mod, inp, out):
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            heights.add(int(out.shape[2]))
            if mod is target_module:
                target_shape[0], target_shape[1] = int(out.shape[2]), int(out.shape[3])

    handles = [m.register_forward_hook(hook) for m in model.modules()]
    try:
        with torch.no_grad():
            model(sample_input)
    finally:
        for h in handles:
            h.remove()
    return sorted(heights), target_shape[0], target_shape[1]


def _module_path(model, target_module) -> str:
    """Dotted path of the module inside the model, matched by object identity."""
    for name, mod in model.named_modules():
        if mod is target_module:
            return name or "<root>"
    return "<unnamed>"


def audit_backbone(name: str, cfg: dict, commit_sha: str) -> dict:
    row = {c: "" for c in COLUMNS}
    row.update(
        run_id=RUN_ID, commit_sha=commit_sha, backbone=name,
        band_lo=BAND_LO, band_hi=BAND_HI,
    )

    internal = _NAME_MAP.get(name, name)
    row["internal_name"] = internal
    try:
        # Same construction path as src/models/registry.build_model, except pretrained=False.
        n_slices = cfg.get("data", {}).get("n_slices", 3)
        patch_xy = cfg.get("data", {}).get("patch_xy", 64)
        model = BackboneClassifier(
            internal, n_input_channels=n_slices, n_classes=2, pretrained=False,
            input_size=track_input_size(cfg, name),
        )
        model.eval()
        # The size the model actually operates at, read off the model rather than a table.
        row["input_size"] = model._resize_to if model._resize_to is not None else patch_xy

        # Stage 05 hands compute_gradcam a raw 64x64 patch; the model resizes internally.
        dummy = torch.zeros(1, n_slices, patch_xy, patch_xy)

        # Mirror the dispatch in compute_gradcam: ViT skips the auto search entirely.
        is_vit = "vit" in internal.lower()
        target = None if is_vit else _auto_target_layer(model, dummy, BAND_LO, BAND_HI)
        row["path_taken"] = "auto" if target is not None else "fallback"
        if target is None:
            target = _get_target_layer(model, internal)

        heights, h, w = _observe(model, dummy, target)
        row["module_class"] = type(target).__name__
        row["module_path"] = _module_path(model, target)
        row["spatial_h"] = "" if h is None else h
        row["spatial_w"] = "" if w is None else w
        row["heights_seen"] = "|".join(str(x) for x in heights)
    except Exception as e:  # one broken backbone must not take the other eleven down
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def main() -> None:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    commit_sha = _commit_sha()
    rows = [audit_backbone(name, cfg, commit_sha) for name in BACKBONES]

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(OUT_CSV, index=False)

    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df.drop(columns=["run_id", "commit_sha", "band_lo", "band_hi"]).to_string(index=False))
    print(f"\n[WROTE] {OUT_CSV}  ({len(df)} rows, run_id={RUN_ID}, commit={commit_sha})")


if __name__ == "__main__":
    main()
