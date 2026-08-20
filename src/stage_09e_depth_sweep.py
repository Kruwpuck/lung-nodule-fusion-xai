"""Stage 09e: how CAM localisation quality varies with explanation depth, all twelve backbones.

Two heuristics have now been measured on these models and both fail. The `[7, 10]` height
band silently returns nothing on backbones that never produce such a map, pushing the choice
into a hand-written table that pointed GoogLeNet at a Dropout after the global average pool
(a 1x1 map, identically zero after normalisation, published as pointing accuracy 0.0). The
canonical rule that replaced it -- explain at the deepest layer that still has spatial extent
-- degrades broadly instead of silently: at a 64 px input a stride-32 backbone's last spatial
layer is 2x2. Comparing the two picks within this one run, the canonical rule scores worse on
pointing accuracy than the band at six of the twelve backbones, better at one (GoogLeNet, the
one where the band fell through to a 1x1 layer), and ties at five. A larger figure of nine
appears elsewhere in this session's history; that one counts backbones whose numbers fell
against the *published* values, which is a different comparison, because the published set also
carries a stale densenet121 row whose checkpoint was retrained after its metrics were written.

Two heuristics give two points per backbone, and two points cannot tell a monotone
relationship from a peaked one from no relationship at all. The shape of that relationship is
the Track 2 claim, so this stage measures the whole curve: a ladder of explanation sites down
the depth of each network, the same 60 fold-0 samples and the same metric primitives at every
site, with the two heuristics' picks marked as points on the curve rather than reported apart
from it.

Choosing the ladder. Every module that emits a 4-D tensor during one forward pass is a
candidate; the ladder takes the last such module at each distinct output resolution, which is
the natural granularity because sites at the same resolution differ only in how much of the
block has run. Three eligibility rules are carried over from `_last_spatial_target_layer` in
`src/xai/gradcam_utils.py`, for the reasons documented there: a module that fires more than
once per pass is skipped (pytorch-grad-cam would hand the CAM its first execution, not the one
whose shape we recorded), a module enclosed by a block that has already collapsed to 1x1 is
skipped (it is one parallel branch of an Inception-style block, and explaining it hides the
evidence flowing through its siblings -- that produced 23 identically-zero maps on
inception_resnet_v2), and a winning `nn.Sequential` is replaced by its last child so the site
names the operation that made the map. The collapsed 1x1 sites themselves are kept as
candidates: a curve that falls to zero because the map has no spatial extent left is the
evidence that the site rather than the architecture is responsible.

ViT has no 4-D intermediates at all, so its depth axis is the transformer stack instead:
`features.encoder.layers.encoder_layer_{i}.ln_1` for every i, read back into a grid by the
existing `vit_reshape_transform`.

Depth is recorded on several axes rather than collapsed into one invented score, because
depth means different things in different networks: the module's execution index normalised
over the forward pass, the output spatial size, and the dotted module path.

Outputs (artifacts/results/track2rev/):
    depth_sweep_12.csv            one row per (backbone, site)
    depth_sweep_12_persample.csv  per-sample values, so confidence intervals are computable
    fig_t2_2_depth_sweep.png      small multiples, pointing accuracy against depth

Run:
    .venv/Scripts/python.exe -m src.stage_09e_depth_sweep
    .venv/Scripts/python.exe -m src.stage_09e_depth_sweep --self-check
    .venv/Scripts/python.exe -m src.stage_09e_depth_sweep --list-sites   # ladders only, no CAMs
"""
from __future__ import annotations

import argparse
import itertools
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from src.models.registry import _NAME_MAP
# Sample selection, patch loading and checkpoint loading are stage 05's, reached through
# stage 09d so the headline table and this sweep cannot drift apart.
from src.stage_09d_cam_12 import (
    _commit_sha,
    _load_model,
    _load_patch_tensor,
    _samples,
)
from src.xai.gradcam_utils import (
    _auto_target_layer,
    _get_target_layer,
    _last_spatial_target_layer,
    dice_iou,
    dice_size_matched,
    energy_pointing_game,
    pointing_hit,
    vit_reshape_transform,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUN_ID = "2026-08-20-run04"
METHOD = "layercam"     # the repo default, for comparability with everything published
THRESHOLD_PCT = 0.80    # stage_05_xai's dice_iou threshold

CONFIG_PATH = os.path.join("configs", "config.yaml")
OUT_DIR = os.path.join("artifacts", "results", "track2rev")
OUT_CSV = os.path.join(OUT_DIR, "depth_sweep_12.csv")
PERSAMPLE_CSV = os.path.join(OUT_DIR, "depth_sweep_12_persample.csv")
FIG_PNG = os.path.join(OUT_DIR, "fig_t2_2_depth_sweep.png")
CAM12_CSV = os.path.join(OUT_DIR, "cam_12.csv")

BACKBONES = [
    "mobilenetv3_small", "efficientnet_b0", "densenet121", "resnet50", "vgg16",
    "vit_base", "inceptionv3", "xception", "googlenet", "convnext_tiny",
    "inception_resnet_v2", "densenet201",
]

SUMMARY_COLS = [
    "run_id", "commit_sha", "backbone", "internal_name", "site_rank",
    "module_class", "module_path", "spatial_h", "spatial_w",
    "exec_index", "n_module_calls", "exec_index_norm", "n",
    "dice", "iou", "dice_size_matched", "pointing_acc", "energy_mean",
    "n_zero_cam", "is_band_pick", "is_canonical_pick", "cam_method", "seconds",
]


# --- forward-pass bookkeeping ----------------------------------------------------------


def _forward_trace(model: nn.Module, x: torch.Tensor):
    """One forward pass, recording every module execution in order.

    Returns `(fired, first_index, n_calls, total_calls)` where `fired` is the list of
    `(exec_index, module, h, w)` for executions that produced a 4-D tensor, `first_index`
    maps module id to the execution index of its first call, `n_calls` maps module id to
    how many times it ran, and `total_calls` is the length of the whole pass. The execution
    index is what makes depth comparable across architectures once normalised by
    `total_calls`, since layer counts differ by an order of magnitude between them.
    """
    fired: list[tuple[int, nn.Module, int, int]] = []
    first_index: dict[int, int] = {}
    n_calls: dict[int, int] = {}
    counter = itertools.count()

    def hook(mod, inp, out):
        idx = next(counter)
        first_index.setdefault(id(mod), idx)
        n_calls[id(mod)] = n_calls.get(id(mod), 0) + 1
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            fired.append((idx, mod, int(out.shape[2]), int(out.shape[3])))

    handles = [m.register_forward_hook(hook) for m in model.modules()]
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    return fired, first_index, n_calls, next(counter)


def _candidate_sites(model: nn.Module, x: torch.Tensor) -> list[dict]:
    """The ladder of explanation sites: the last eligible module at each output resolution.

    Eligibility follows `_last_spatial_target_layer`: one execution per pass, not enclosed by
    a block that has already collapsed to 1x1, and a `nn.Sequential` winner replaced by its
    last child. Unlike that function, 1x1 sites are kept rather than discarded, and one site
    per resolution is returned rather than only the deepest.
    """
    fired, first_index, n_calls, total = _forward_trace(model, x)
    paths = {id(m): name for name, m in model.named_modules()}
    collapsed = [paths[id(m)] for _, m, h, w in fired if h == 1 and w == 1 and paths.get(id(m))]

    def inside_collapsed(path: str) -> bool:
        return any(path.startswith(c + ".") for c in collapsed)

    # `h == w` drops channels-last 4-D tensors, which are not feature maps. ConvNeXt blocks
    # permute to (N, H, W, C) around their LayerNorm and Linear layers, so those modules emit
    # 4-D outputs shaped like 3x3072 that a shape test alone would read as a 3x3072 "map".
    # Every genuine map here is square because the input patches are square, so squareness
    # separates the two without needing to know which architectures permute.
    eligible = [
        (idx, m, h, w) for idx, m, h, w in fired
        if h == w and n_calls[id(m)] == 1 and paths.get(id(m))
        and not inside_collapsed(paths[id(m)])
    ]
    shapes = {id(m): (h, w) for _, m, h, w in eligible}

    # Last eligible module at each distinct resolution, in execution order.
    by_res: dict[tuple[int, int], tuple[int, nn.Module]] = {}
    for idx, m, h, w in eligible:
        by_res[(h, w)] = (idx, m)

    sites = []
    for (h, w), (idx, m) in by_res.items():
        while isinstance(m, nn.Sequential) and len(m) and shapes.get(id(m[-1])) == (h, w):
            m = m[-1]
        sites.append(_site(model, m, first_index.get(id(m), idx), total, h, w, paths))
    sites.sort(key=lambda s: s["exec_index"])
    return sites


def _site(model, module, exec_index, total, h, w, paths=None) -> dict:
    """One row's worth of identity and depth bookkeeping for a candidate site."""
    if paths is None:
        paths = {id(m): name for name, m in model.named_modules()}
    return {
        "module": module,
        "module_class": type(module).__name__,
        "module_path": paths.get(id(module), "<unnamed>"),
        "spatial_h": h,
        "spatial_w": w,
        "exec_index": exec_index,
        "n_module_calls": total,
        "exec_index_norm": round(exec_index / max(total - 1, 1), 4),
    }


def _vit_sites(model: nn.Module, x: torch.Tensor) -> list[dict]:
    """ViT's depth axis: the `ln_1` of every transformer encoder layer.

    ViT emits no 4-D intermediate, so there is no resolution ladder to walk. The encoder
    stack is the depth axis instead, and every site is read back into a grid by the same
    `vit_reshape_transform` the published path uses, so its token grid size is constant
    down the stack and the spatial columns record that grid rather than a feature map.
    """
    _, first_index, _, total = _forward_trace(model, x)
    layers = model.features.encoder.layers
    sites = []
    for name, layer in layers.named_children():
        ln = layer.ln_1
        with torch.no_grad():
            tokens = None

            def grab(mod, inp, out):
                nonlocal tokens
                tokens = out

            hnd = ln.register_forward_hook(grab)
            try:
                model(x)
            finally:
                hnd.remove()
        grid = int(round((tokens.shape[1] - 1) ** 0.5))
        sites.append(_site(model, ln, first_index.get(id(ln), 0), total, grid, grid))
    sites.sort(key=lambda s: s["exec_index"])
    return sites


# --- CAM and metrics --------------------------------------------------------------------


def _cams_at(model, module, samples, reshape=None) -> list[np.ndarray]:
    """Layer-CAM of the predicted class at one explicitly named site, one map per sample."""
    from pytorch_grad_cam import LayerCAM

    with LayerCAM(model=model, target_layers=[module], reshape_transform=reshape) as cam:
        return [cam(input_tensor=s["img"], targets=None)[0] for s in samples]


def _score(cams, samples, backbone, site, sha, internal) -> tuple[dict, list[dict]]:
    """Aggregate and per-sample metrics, using the published primitives unmodified."""
    rows = []
    for s, cam in zip(samples, cams):
        d, iou = dice_iou(cam, s["mask"], pct=THRESHOLD_PCT)
        rows.append({
            "run_id": RUN_ID, "commit_sha": sha, "backbone": backbone,
            "module_path": site["module_path"], "module_class": site["module_class"],
            "spatial_h": site["spatial_h"], "spatial_w": site["spatial_w"],
            "exec_index_norm": site["exec_index_norm"],
            "sample_idx": s["sample_idx"], "patient_id": s["patient_id"],
            "nodule_idx": s["nodule_idx"], "label": s["label"], "pred": s["pred"],
            "prob_malignant": s["prob"],
            "dice": d, "iou": iou,
            "dice_size_matched": dice_size_matched(cam, s["mask"]),
            "pointing_hit": int(pointing_hit(cam, s["mask"])),
            "energy": energy_pointing_game(cam, s["mask"]),
            "cam_max": float(cam.max()), "cam_min": float(cam.min()),
            "mask_px": s["mask_px"],
        })
    ps = pd.DataFrame(rows)
    summary = {
        "run_id": RUN_ID, "commit_sha": sha, "backbone": backbone, "internal_name": internal,
        "module_class": site["module_class"], "module_path": site["module_path"],
        "spatial_h": site["spatial_h"], "spatial_w": site["spatial_w"],
        "exec_index": site["exec_index"], "n_module_calls": site["n_module_calls"],
        "exec_index_norm": site["exec_index_norm"], "n": len(ps),
        "dice": ps["dice"].mean(), "iou": ps["iou"].mean(),
        "dice_size_matched": ps["dice_size_matched"].mean(),
        "pointing_acc": ps["pointing_hit"].mean(), "energy_mean": ps["energy"].mean(),
        "n_zero_cam": int((ps["cam_max"] <= 0).sum()),
        "cam_method": METHOD,
    }
    return summary, rows


# --- per-backbone driver ----------------------------------------------------------------


def _sample_bundle(cfg: dict, samples_df: pd.DataFrame, model, device) -> list[dict]:
    """Patches, masks and this model's prediction for the 60 stage-05 fold-0 samples."""
    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)
    out = []
    for i, row in samples_df.iterrows():
        img = _load_patch_tensor(row["patch_path"], n_slices, patch_xy).to(device)
        with torch.no_grad():
            logits = model(img)
            prob = F.softmax(logits, dim=1)[0, 1].item()
            pred = int(logits.argmax(dim=1).item())
        mask_full = np.load(row["mask_path"]).astype(np.float32)
        mask2d = mask_full[mask_full.shape[0] // 2]
        out.append({
            "img": img, "mask": mask2d, "sample_idx": int(i),
            "patient_id": row["patient_id"], "nodule_idx": int(row["nodule_idx"]),
            "label": int(row["label"]), "pred": pred, "prob": prob,
            "mask_px": int(mask2d.astype(bool).sum()),
        })
    return out


def _ladder(model, internal: str, dummy) -> tuple[list[dict], nn.Module, nn.Module]:
    """Candidate sites plus the two reference modules, resolved through the shared code.

    The canonical pick comes from `_last_spatial_target_layer` and the band pick from
    `_auto_target_layer`, both called exactly as `compute_gradcam` calls them (before and
    after commit f81ba0d respectively), so the marked points are the real heuristics rather
    than a re-description of them. Either pick is appended to the ladder if the resolution
    ladder did not already contain it.
    """
    is_vit = "vit" in internal.lower()
    sites = _vit_sites(model, dummy) if is_vit else _candidate_sites(model, dummy)

    canonical = None if is_vit else _last_spatial_target_layer(model, dummy)
    if canonical is None:
        canonical = _get_target_layer(model, internal)
    band = None if is_vit else _auto_target_layer(model, dummy)
    if band is None:
        band = _get_target_layer(model, internal)

    known = {id(s["module"]) for s in sites}
    for extra in (canonical, band):
        if id(extra) not in known:
            fired, first_index, _, total = _forward_trace(model, dummy)
            hw = next(((h, w) for _, m, h, w in fired if m is extra), (0, 0))
            sites.append(_site(model, extra, first_index.get(id(extra), total), total, *hw))
            known.add(id(extra))
    sites.sort(key=lambda s: s["exec_index"])
    for rank, s in enumerate(sites):
        s["site_rank"] = rank
        s["is_canonical_pick"] = int(s["module"] is canonical)
        s["is_band_pick"] = int(s["module"] is band)
    return sites


def sweep_backbone(backbone: str, cfg: dict, samples_df, device, sha: str):
    internal = _NAME_MAP.get(backbone, backbone)
    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)

    model, _ = _load_model(backbone, cfg, device)
    dummy = torch.zeros(1, n_slices, patch_xy, patch_xy, device=device)
    sites = _ladder(model, internal, dummy)
    reshape = vit_reshape_transform if "vit" in internal.lower() else None

    samples = _sample_bundle(cfg, samples_df, model, device)
    summaries, per_sample = [], []
    for site in sites:
        t0 = time.perf_counter()
        cams = _cams_at(model, site["module"], samples, reshape)
        summary, rows = _score(cams, samples, backbone, site, sha, internal)
        summary.update(site_rank=site["site_rank"],
                       is_band_pick=site["is_band_pick"],
                       is_canonical_pick=site["is_canonical_pick"],
                       seconds=round(time.perf_counter() - t0, 2))
        summaries.append(summary)
        per_sample.extend(rows)
        logger.info("  [%s] %-38s %2dx%-2d d=%.3f point=%.4f energy=%.4f zero=%2d %s%s",
                    backbone, summary["module_path"], summary["spatial_h"],
                    summary["spatial_w"], summary["exec_index_norm"],
                    summary["pointing_acc"], summary["energy_mean"], summary["n_zero_cam"],
                    "BAND " if site["is_band_pick"] else "",
                    "CANON" if site["is_canonical_pick"] else "")

    del model
    torch.cuda.empty_cache()
    return summaries, per_sample


# --- figure ------------------------------------------------------------------------------


def make_figure(df: pd.DataFrame, path: str, chance: float | None = None) -> None:
    """Small multiples: pointing accuracy against explanation depth, one panel per backbone.

    The x axis is the ladder position rather than the normalised execution index, because
    the execution index bunches the last three or four sites of most backbones into the
    final few percent of the pass and makes them unreadable at print size. The ordering is
    the same either way -- both axes are execution order -- and the exact index of every
    point is in `exec_index_norm` in the CSV. Each tick names the site's feature-map size,
    which is the second depth axis and the one that explains the shape of the curve.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [b for b in BACKBONES if b in set(df["backbone"])]
    ncol = 4
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14.0, 3.2 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, backbone in zip(axes, order):
        d = df[df["backbone"] == backbone].sort_values("exec_index_norm").reset_index(drop=True)
        x = np.arange(len(d))
        if chance is not None:
            ax.axhline(chance, color="#888888", linestyle="--", linewidth=0.9, zorder=1)
        ax.plot(x, d["pointing_acc"], "-o", color="#3b6ea5",
                markersize=5, linewidth=1.6, zorder=2)
        sel = d["is_band_pick"] == 1
        ax.scatter(x[sel], d.loc[sel, "pointing_acc"], s=150, marker="s",
                   facecolors="none", edgecolors="#e08214", linewidths=2.2, zorder=3)
        sel = d["is_canonical_pick"] == 1
        ax.scatter(x[sel], d.loc[sel, "pointing_acc"], s=180, marker="*",
                   color="#d7191c", zorder=4)
        ax.set_title(backbone, fontsize=11)
        ax.set_ylim(-0.06, 1.0)
        ax.set_xlim(-0.5, len(d) - 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}²" for h in d["spatial_h"]], fontsize=7.5, rotation=45)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(axis="y", labelsize=9)
    for ax in axes[len(order):]:
        ax.axis("off")
    for i, ax in enumerate(axes[:len(order)]):
        if i % ncol == 0:
            ax.set_ylabel("pointing accuracy\n(fraction of 60 samples)", fontsize=9)
        if i >= len(order) - ncol:
            ax.set_xlabel("explanation site, shallow to deep\n(tick = feature-map size in pixels)",
                          fontsize=9)

    handles = [
        plt.Line2D([], [], color="#3b6ea5", marker="o", markersize=5,
                   label=f"candidate explanation site ({METHOD}, 60 fold-0 samples)"),
        plt.Line2D([], [], color="#e08214", marker="s", markerfacecolor="none",
                   markersize=10, linestyle="none", markeredgewidth=2.2,
                   label="[7, 10] height-band pick (the superseded selector)"),
        plt.Line2D([], [], color="#d7191c", marker="*", markersize=14, linestyle="none",
                   label="canonical last-spatial-layer pick (reported configuration)"),
    ]
    if chance is not None:
        handles.append(plt.Line2D([], [], color="#888888", linestyle="--", linewidth=0.9,
                                  label=f"chance = {chance:.3f} (mean lesion area fraction)"))
    # The legend sits under the panels, in reserved space, so it cannot cover any data.
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9.5)
    fig.suptitle("CAM localisation quality against explanation depth, twelve backbones",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.085, 1, 0.965))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --- self-check ---------------------------------------------------------------------------


def _chance(per_sample: pd.DataFrame, patch_xy: int) -> float:
    """Chance pointing rate: the mean lesion area fraction over the 60 samples.

    A CAM whose maximum lands uniformly at random hits the lesion this often, so it is the
    line a curve has to clear to mean anything.
    """
    one = per_sample.drop_duplicates("sample_idx")
    return float((one["mask_px"] / float(patch_xy * patch_xy)).mean())


def self_check(df: pd.DataFrame) -> None:
    """Three asserts: a curve exists, the canonical pick is marked, and it reproduces cam_12."""
    # 1. Three points is the minimum that can distinguish a peak from a monotone trend.
    for backbone, d in df.groupby("backbone"):
        assert len(d) >= 3, f"{backbone}: only {len(d)} candidate sites, that is not a curve"

    # 2. Exactly one canonical pick per backbone, and it is the layer the headline table used.
    cam12 = pd.read_csv(CAM12_CSV).set_index("backbone")
    for backbone, d in df.groupby("backbone"):
        picks = d[d["is_canonical_pick"] == 1]
        assert len(picks) == 1, f"{backbone}: {len(picks)} rows flagged is_canonical_pick"
        got = picks.iloc[0]["module_path"]
        want = cam12.loc[backbone, "module_path"]
        assert got == want, f"{backbone}: canonical pick {got}, cam_12 recorded {want}"

    # 3. The metrics at that pick must equal the headline table's. This is what proves the
    #    sweep and the published row come out of one code path rather than two similar ones.
    for backbone, d in df.groupby("backbone"):
        pick = d[d["is_canonical_pick"] == 1].iloc[0]
        for metric in ("dice", "iou", "dice_size_matched", "pointing_acc", "energy_mean"):
            a, b = float(pick[metric]), float(cam12.loc[backbone, metric])
            assert abs(a - b) < 1e-9, f"{backbone}.{metric}: sweep {a!r} vs cam_12 {b!r}"
    print("[SELF-CHECK] all three asserts passed")


# --- driver --------------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--backbones", nargs="*", default=BACKBONES,
                   help="subset to sweep; the output CSVs then hold only that subset")
    p.add_argument("--list-sites", action="store_true",
                   help="print each backbone's ladder and exit, computing no CAMs")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--figure-only", action="store_true",
                   help="redraw the figure from the existing CSV")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sha = _commit_sha()

    patch_xy = cfg["data"].get("patch_xy", 64)
    if args.figure_only:
        df = pd.read_csv(OUT_CSV)
        make_figure(df, FIG_PNG, _chance(pd.read_csv(PERSAMPLE_CSV), patch_xy))
        print(f"[WROTE] {FIG_PNG}")
        if args.self_check:
            self_check(df)
        return

    if args.list_sites:
        n_slices = cfg["data"].get("n_slices", 3)
        patch_xy = cfg["data"].get("patch_xy", 64)
        for backbone in args.backbones:
            internal = _NAME_MAP.get(backbone, backbone)
            model, _ = _load_model(backbone, cfg, device)
            dummy = torch.zeros(1, n_slices, patch_xy, patch_xy, device=device)
            for s in _ladder(model, internal, dummy):
                print(f"{backbone:<20} {s['site_rank']:>2} {s['module_path']:<44} "
                      f"{s['spatial_h']}x{s['spatial_w']:<4} norm={s['exec_index_norm']:.3f} "
                      f"{'BAND ' if s['is_band_pick'] else '':<5}"
                      f"{'CANON' if s['is_canonical_pick'] else ''}")
            del model
            torch.cuda.empty_cache()
        return

    samples_df = _samples(cfg)
    rows: list[dict] = []
    per_sample: list[dict] = []
    for backbone in args.backbones:
        t0 = time.perf_counter()
        summaries, ps = sweep_backbone(backbone, cfg, samples_df, device, sha)
        rows.extend(summaries)
        per_sample.extend(ps)
        # Written after every backbone: a crash on backbone 9 must not cost backbones 1-8.
        pd.DataFrame(rows)[SUMMARY_COLS].to_csv(OUT_CSV, index=False)
        pd.DataFrame(per_sample).to_csv(PERSAMPLE_CSV, index=False)
        logger.info("[%s] %d sites in %.1fs", backbone, len(summaries), time.perf_counter() - t0)

    df = pd.DataFrame(rows)[SUMMARY_COLS]
    make_figure(df, FIG_PNG, _chance(pd.DataFrame(per_sample), patch_xy))
    print(f"\n[WROTE] {OUT_CSV} ({len(df)} rows)")
    print(f"[WROTE] {PERSAMPLE_CSV} ({len(per_sample)} rows)")
    print(f"[WROTE] {FIG_PNG}")

    if args.self_check:
        self_check(df)


if __name__ == "__main__":
    main()
