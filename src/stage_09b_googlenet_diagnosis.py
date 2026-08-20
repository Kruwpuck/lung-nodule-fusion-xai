"""Stage 09b: why GoogLeNet's Grad-CAM scores are exactly zero.

Stage 09a established that for GoogLeNet the `[7, 10]` auto-search finds nothing (the
heights seen at a 96x96 input are 1, 3, 6, 12, 24, 48), so `compute_gradcam` falls back
to the hand-written table, which returns `model.features[-2]`. Because `build_2d_backbone`
keeps `avgpool` and `dropout` inside `features`, `features[-2]` is the Dropout that sits
*after* the global average pool, and its output is 1x1. The published `xai_metrics.csv`
row for googlenet is `pointing_acc = 0.0` and `energy_mean = 0.0` exactly, the only
backbone whose energy is exactly zero.

This stage measures rather than fixes. It never touches `src/xai/gradcam_utils.py`; the
CAM classes are called directly with an explicitly chosen target layer, so the shared
resolver keeps behaving exactly as it does for the published numbers.

Four checks plus a control:

    1. Layer sweep. The same 60 fold-0 samples, the same metric primitives, at six
       candidate GoogLeNet layers. The discriminating column is `frac_zero_cam`: the
       fraction of samples whose CAM comes back identically zero.
    2. CAM-method cross-check at the best non-degenerate layer, with wall-clock cost.
    3. Adebayo-style sanity checks: cascading weight randomisation, plus a class-flip
       sensitivity probe. A map that survives randomisation is an edge detector, not an
       explanation.
    4. The `features` module chain dumped verbatim, so the position of avgpool, dropout
       and Flatten is data rather than a claim in prose.

    Control: densenet121, which resolves to a 3x3 BatchNorm2d and scores pointing_acc
    0.7167 in the published table. Without it, "the map is non-zero at layer X" is
    unfalsifiable.

Run:
    .venv/Scripts/python.exe -m src.stage_09b_googlenet_diagnosis
    .venv/Scripts/python.exe -m src.stage_09b_googlenet_diagnosis --self-check
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from src.models.backbones import BackboneClassifier
from src.models.registry import _NAME_MAP
from src.utils.tracks import track_input_size
from src.xai.gradcam_utils import (
    dice_iou,
    dice_size_matched,
    energy_pointing_game,
    pointing_hit,
)

RUN_ID = "2026-08-20-run04"
N_SAMPLES = 60          # identical to stage 05's N_SAMPLES_METRICS, so the numbers compare
RANDOM_STATE = 42       # identical to stage 05's sample(random_state=...)

CONFIG_PATH = os.path.join("configs", "config.yaml")
OUT_DIR = os.path.join("artifacts", "results", "track2rev")
SWEEP_CSV = os.path.join(OUT_DIR, "googlenet_layer_sweep.csv")
METHODS_CSV = os.path.join(OUT_DIR, "googlenet_cam_methods.csv")
SANITY_CSV = os.path.join(OUT_DIR, "googlenet_sanity.csv")
HEAD_TXT = os.path.join(OUT_DIR, "googlenet_head_structure.txt")
FIG_PNG = os.path.join(OUT_DIR, "fig_t2_1_googlenet_diagnosis.png")

# Candidate target layers, given as dotted paths inside BackboneClassifier. The GoogLeNet
# list is read off the real module chain: features is
# Sequential(conv1, maxpool1, conv2, conv3, maxpool2, inception3a, inception3b, maxpool3,
#            inception4a..4e, maxpool4, inception5a, inception5b, avgpool, dropout, Flatten)
# so index 17 is the Dropout the resolver picks and index 16 is the AdaptiveAvgPool2d.
GOOGLENET_CANDIDATES = [
    ("features.17", "dropout (resolved by _get_target_layer)"),
    ("features.16", "adaptive avgpool"),
    ("features.15", "inception5b"),
    ("features.13", "maxpool4"),
    ("features.12", "inception4e"),
    ("features.3", "conv3"),
]
# Control. densenet121's features is Sequential(torchvision densenet.features, pool, Flatten),
# so the resolved BatchNorm2d lives at features.0.norm5.
DENSENET_CANDIDATES = [
    ("features.0.norm5", "norm5 (resolved by _get_target_layer)"),
    ("features.0.denseblock4", "denseblock4"),
    ("features.0.denseblock3", "denseblock3"),
]

CAM_METHODS = ["layercam", "hirescam", "gradcam", "scorecam"]

# Cumulative cascading-randomisation stages, top of the network first. Modules with no
# parameters (the pools) are skipped: re-initialising them is a no-op.
CASCADE_STAGES = [
    "classifier", "features.15", "features.14", "features.12",
    "features.10", "features.8", "features.5", "features.0",
]

VERDICT_CRITERION = (
    "artifact iff (C1) frac_zero_cam == 1.0 at the resolved layer features.17, AND "
    "(C2) some candidate layer has frac_zero_cam == 0.0 and pointing_acc above the "
    "per-patch chance rate plus two standard errors, where chance is the mean nodule-mask "
    "area fraction over the same 60 samples, AND (C3) mean Spearman between the CAM at "
    "that layer and the fully weight-randomised CAM is below 0.5. not_artifact if C1 "
    "fails (the resolved layer already produces a usable map). inconclusive otherwise: "
    "C1 without C2 means no layer recovers localisation; C1 and C2 without C3 means the "
    "recovered map does not depend on the trained weights and is an edge detector."
)


def _commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


# --- data, copied from stage 05 so the sample set is byte-for-byte the published one ----

HU_MIN, HU_MAX = -1000.0, 400.0


def _load_patch_tensor(patch_path: str, n_slices: int, patch_xy: int) -> torch.Tensor:
    vol = np.load(patch_path).astype(np.float32)
    vol = np.clip(vol, HU_MIN, HU_MAX)
    vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)
    Z, H, W = vol.shape
    cz = Z // 2
    half = n_slices // 2
    cy, cx = H // 2, W // 2
    y0, y1 = max(0, cy - patch_xy // 2), min(H, cy + patch_xy // 2)
    x0, x1 = max(0, cx - patch_xy // 2), min(W, cx + patch_xy // 2)
    slices = []
    for offset in range(-half, half + 1):
        z = max(0, min(Z - 1, cz + offset))
        sl = np.zeros((patch_xy, patch_xy), dtype=np.float32)
        crop = vol[z, y0:y1, x0:x1]
        sl[: crop.shape[0], : crop.shape[1]] = crop
        slices.append(sl)
    arr = np.stack(slices, axis=0)
    return torch.from_numpy(arr).unsqueeze(0)


def load_samples(cfg: dict, device: torch.device) -> list[dict]:
    """The stage-05 fold-0 evaluation set: fold 0, label != -1, 60 rows, seed 42."""
    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)
    df = pd.read_csv(os.path.join(cfg["paths"]["interim"], "labels.csv"))
    val_df = df[(df["fold"] == 0) & (df["label"] != -1)].reset_index(drop=True)
    if len(val_df) > N_SAMPLES:
        val_df = val_df.sample(n=N_SAMPLES, random_state=RANDOM_STATE).reset_index(drop=True)

    samples = []
    for _, row in val_df.iterrows():
        img = _load_patch_tensor(row["patch_path"], n_slices, patch_xy).to(device)
        mask_full = np.load(row["mask_path"]).astype(np.float32)
        mask2d = mask_full[mask_full.shape[0] // 2]
        samples.append({"img": img, "mask": mask2d, "label": int(row["label"])})
    return samples


def build_loaded_model(backbone: str, cfg: dict, device: torch.device) -> BackboneClassifier:
    """Same construction and checkpoint load as stage 05."""
    internal = _NAME_MAP.get(backbone, backbone)
    model = BackboneClassifier(
        internal, n_input_channels=cfg["data"].get("n_slices", 3), n_classes=2,
        pretrained=True, input_size=track_input_size(cfg, backbone),
    ).to(device)
    ckpt = os.path.join(cfg["paths"]["checkpoints"], backbone, "fold0_best.pt")
    try:
        state = torch.load(ckpt, weights_only=True, map_location="cpu")
    except TypeError:
        state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model


# --- CAM at an explicitly named layer --------------------------------------------------


def _cam_class(method: str):
    from pytorch_grad_cam import GradCAM, HiResCAM, LayerCAM, ScoreCAM
    return {"gradcam": GradCAM, "hirescam": HiResCAM,
            "layercam": LayerCAM, "scorecam": ScoreCAM}[method]


def _module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    mods = dict(model.named_modules())
    if path not in mods:
        raise KeyError(f"{path!r} not in model; have e.g. {list(mods)[:5]}")
    return mods[path]


def cams_at_layer(model, layer_path, samples, method="layercam", target_class=None):
    """CAMs for every sample at one explicitly named layer. Returns a list of (H, W) arrays.

    The CAM object is built once and reused across samples, which is what the library's
    context manager is for; per-sample construction would give identical maps at higher cost.
    """
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    layer = _module(model, layer_path)
    targets = None if target_class is None else [ClassifierOutputTarget(target_class)]
    out = []
    with _cam_class(method)(model=model, target_layers=[layer]) as cam:
        for s in samples:
            out.append(cam(input_tensor=s["img"], targets=targets)[0])
    return out


def spatial_size(model, layer_path, sample_img) -> str:
    """Output spatial size of one layer, observed rather than assumed."""
    got: list = []

    def hook(mod, inp, out):
        if isinstance(out, torch.Tensor) and out.dim() == 4:
            got.append((int(out.shape[2]), int(out.shape[3])))

    h = _module(model, layer_path).register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(sample_img)
    finally:
        h.remove()
    return f"{got[0][0]}x{got[0][1]}" if got else "not-4d"


def score_cams(cams, samples) -> dict:
    """Aggregate the four published metric primitives plus the degeneracy fraction."""
    d, i, dsm, hit, en, zero = [], [], [], [], [], []
    for cam, s in zip(cams, samples):
        dd, ii = dice_iou(cam, s["mask"])
        d.append(dd)
        i.append(ii)
        dsm.append(dice_size_matched(cam, s["mask"]))
        hit.append(pointing_hit(cam, s["mask"]))
        en.append(energy_pointing_game(cam, s["mask"]))
        zero.append(float(cam.max()) == 0.0)
    return {
        "n": len(cams),
        "dice": float(np.mean(d)), "iou": float(np.mean(i)),
        "dice_size_matched": float(np.mean(dsm)),
        "pointing_acc": float(np.mean(hit)),
        "energy_mean": float(np.mean(en)),
        "frac_zero_cam": float(np.mean(zero)),
    }


# --- checks ----------------------------------------------------------------------------


def check1_sweep(models, samples, chance, sha) -> pd.DataFrame:
    rows, cams_cache = [], {}
    for backbone, candidates in (("googlenet", GOOGLENET_CANDIDATES),
                                 ("densenet121", DENSENET_CANDIDATES)):
        model = models[backbone]
        for path, note in candidates:
            t0 = time.perf_counter()
            cams = cams_at_layer(model, path, samples)
            m = score_cams(cams, samples)
            cams_cache[(backbone, path)] = cams
            rows.append({
                "run_id": RUN_ID, "commit_sha": sha, "backbone": backbone,
                "layer_path": path, "layer_note": note,
                "layer_class": type(_module(model, path)).__name__,
                "spatial_size": spatial_size(model, path, samples[0]["img"]),
                **m,
                "chance_pointing": chance,
                "usable": "no" if m["frac_zero_cam"] > 0 else "yes",
                "seconds": round(time.perf_counter() - t0, 2),
            })
            print(f"  [{backbone}] {path:<24} {rows[-1]['spatial_size']:>7}  "
                  f"point={m['pointing_acc']:.4f} energy={m['energy_mean']:.4f} "
                  f"zero={m['frac_zero_cam']:.2f}")
    return pd.DataFrame(rows), cams_cache


def check2_methods(model, layer_path, samples, sha) -> pd.DataFrame:
    rows = []
    for method in CAM_METHODS:
        t0 = time.perf_counter()
        cams = cams_at_layer(model, layer_path, samples, method=method)
        secs = time.perf_counter() - t0
        rows.append({
            "run_id": RUN_ID, "commit_sha": sha, "backbone": "googlenet",
            "layer_path": layer_path, "method": method,
            **score_cams(cams, samples),
            "seconds_total": round(secs, 2),
            "seconds_per_sample": round(secs / len(samples), 4),
        })
        print(f"  {method:<10} point={rows[-1]['pointing_acc']:.4f} "
              f"energy={rows[-1]['energy_mean']:.4f} {secs:.1f}s")
    return pd.DataFrame(rows)


def _spearman_mean(a_list, b_list) -> tuple[float, float]:
    """Mean and median per-sample Spearman rank correlation; constant maps give nan."""
    from scipy.stats import spearmanr

    vals = []
    for a, b in zip(a_list, b_list):
        r = spearmanr(a.ravel(), b.ravel()).statistic
        vals.append(r)
    arr = np.array(vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(np.median(finite))


def check3_sanity(cfg, backbone, layer_path, samples, reference_cams, device, sha):
    """Cascading weight randomisation plus a class-flip sensitivity probe."""
    internal = _NAME_MAP.get(backbone, backbone)
    trained = build_loaded_model(backbone, cfg, device)
    # A second, never-trained model supplies the random weights, so "re-initialised" means
    # the architecture's own init scheme rather than an ad-hoc noise injection.
    rand = BackboneClassifier(
        internal, n_input_channels=cfg["data"].get("n_slices", 3), n_classes=2,
        pretrained=False, input_size=track_input_size(cfg, backbone),
    ).to(device).eval()
    rand_state = rand.state_dict()

    rows, done = [], []
    model = copy.deepcopy(trained)
    for stage in CASCADE_STAGES:
        prefix = stage + "."
        swap = {k: v for k, v in rand_state.items() if k == stage or k.startswith(prefix)}
        if not swap:
            continue
        state = model.state_dict()
        state.update(swap)
        model.load_state_dict(state)
        model.eval()
        done.append(stage)
        cams = cams_at_layer(model, layer_path, samples)
        mean_r, med_r = _spearman_mean(reference_cams, cams)
        rows.append({
            "run_id": RUN_ID, "commit_sha": sha, "backbone": backbone,
            "layer_path": layer_path, "check": "cascading_randomisation",
            "stage": stage, "randomised_so_far": "|".join(done),
            "n": len(samples), "spearman_mean": mean_r, "spearman_median": med_r,
            "frac_zero_cam": score_cams(cams, samples)["frac_zero_cam"],
            "verdict": "", "criterion": "",
        })
        print(f"  cascade +{stage:<14} spearman_mean={mean_r:.4f}")
    cascade_final = rows[-1]["spearman_mean"] if rows else float("nan")

    # Class-flip probe. Adebayo's label-randomisation test retrains on permuted labels,
    # which is not affordable here; this is the cheap sibling — explain the other class
    # instead of the predicted one on the trained model. It answers a narrower question
    # (does the map depend on which logit is explained) and is labelled as such.
    flipped = []
    for s in samples:
        with torch.no_grad():
            pred = int(trained(s["img"]).argmax(dim=1).item())
        flipped.append(cams_at_layer(trained, layer_path, [s], target_class=1 - pred)[0])
    mean_r, med_r = _spearman_mean(reference_cams, flipped)
    rows.append({
        "run_id": RUN_ID, "commit_sha": sha, "backbone": backbone,
        "layer_path": layer_path, "check": "class_flip_sensitivity",
        "stage": "explain_other_class", "randomised_so_far": "",
        "n": len(samples), "spearman_mean": mean_r, "spearman_median": med_r,
        "frac_zero_cam": score_cams(flipped, samples)["frac_zero_cam"],
        "verdict": "", "criterion": "",
    })
    print(f"  class-flip           spearman_mean={mean_r:.4f}")
    return pd.DataFrame(rows), cascade_final


def check4_head_structure(model, sample_img, path: str) -> None:
    """Dump the features chain with observed output sizes, one line per child."""
    seen: dict[int, str] = {}
    handles = []
    for idx, child in enumerate(model.features):
        def hook(mod, inp, out, idx=idx):
            if isinstance(out, torch.Tensor):
                seen[idx] = "x".join(str(d) for d in tuple(out.shape))
            else:
                seen[idx] = type(out).__name__
        handles.append(child.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(sample_img)
    finally:
        for h in handles:
            h.remove()

    lines = [
        f"run_id={RUN_ID}",
        "GoogLeNet BackboneClassifier.features, one line per direct child.",
        f"Input handed to the model: {tuple(sample_img.shape)}; the model resizes internally "
        f"to {model._resize_to}x{model._resize_to}.",
        "",
        f"{'idx':>4}  {'class':<22}  output shape",
    ]
    for idx, child in enumerate(model.features):
        lines.append(f"{idx:>4}  {type(child).__name__:<22}  {seen.get(idx, '-')}")
    lines += [
        "",
        "features[-1] is Flatten, features[-2] is the Dropout, features[-3] is the "
        "AdaptiveAvgPool2d. `_get_target_layer`'s googlenet branch returns features[-2], "
        "i.e. the Dropout, whose output is 1x1 because it sits after the pool.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def make_figure(cams_bad, cams_good, samples, good_layer, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Show the samples with the largest lesions: the pointing game is least ambiguous there.
    order = sorted(range(len(samples)), key=lambda i: samples[i]["mask"].sum(), reverse=True)[:3]
    fig, axes = plt.subplots(len(order), 3, figsize=(9.5, 3.2 * len(order)))
    for r, i in enumerate(order):
        s = samples[i]
        ct = s["img"][0].detach().cpu().numpy()
        ct = ct[ct.shape[0] // 2]
        for c, (data, title) in enumerate([
            (None, f"CT + mask (label={s['label']})"),
            (cams_bad[i], "features.17 Dropout (1x1)\nresolved today"),
            (cams_good[i], f"{good_layer}\ncorrected"),
        ]):
            ax = axes[r, c]
            ax.imshow(ct, cmap="gray")
            if data is not None:
                ax.imshow(data, alpha=0.5, cmap="jet", vmin=0.0, vmax=1.0)
                ax.set_title(f"{title}\nmax={data.max():.3f}", fontsize=8)
            else:
                ax.set_title(title, fontsize=8)
            ax.contour(s["mask"], colors="lime", linewidths=1.0)
            ax.axis("off")
    fig.suptitle("GoogLeNet Layer-CAM: degenerate vs corrected target layer", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


# --- driver ----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--self-check", action="store_true",
                    help="run the diagnosis, then assert the three claims it must support")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sha = _commit_sha()
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_samples(cfg, device)
    chance = float(np.mean([s["mask"].sum() / s["mask"].size for s in samples]))
    print(f"[setup] {len(samples)} fold-0 samples, device={device}, "
          f"chance pointing rate={chance:.4f}")

    models = {b: build_loaded_model(b, cfg, device) for b in ("googlenet", "densenet121")}

    print("\n[check 1] target-layer sweep")
    sweep, cams_cache = check1_sweep(models, samples, chance, sha)
    sweep.to_csv(SWEEP_CSV, index=False)

    # Best non-degenerate GoogLeNet layer by pointing accuracy, the metric the published
    # table reports as 0.0.
    g = sweep[(sweep.backbone == "googlenet") & (sweep.frac_zero_cam == 0.0)]
    best_layer = g.sort_values("pointing_acc", ascending=False).iloc[0]["layer_path"]
    print(f"  -> best non-degenerate googlenet layer: {best_layer}")

    print("\n[check 2] CAM methods at", best_layer)
    methods = check2_methods(models["googlenet"], best_layer, samples, sha)
    methods.to_csv(METHODS_CSV, index=False)

    print("\n[check 3] sanity checks")
    reference = cams_cache[("googlenet", best_layer)]
    sanity, cascade_final = check3_sanity(
        cfg, "googlenet", best_layer, samples, reference, device, sha
    )

    # Verdict, computed from the numbers rather than asserted in prose.
    bad = sweep[(sweep.backbone == "googlenet") & (sweep.layer_path == "features.17")].iloc[0]
    c1 = bool(bad["frac_zero_cam"] == 1.0)
    se = float(np.sqrt(chance * (1 - chance) / len(samples)))
    c2_rows = g[g.pointing_acc > chance + 2 * se]
    c2 = bool(len(c2_rows) > 0)
    c3 = bool(np.isfinite(cascade_final) and cascade_final < 0.5)
    verdict = "artifact" if (c1 and c2 and c3) else ("not_artifact" if not c1 else "inconclusive")
    print(f"\n[verdict] C1={c1} C2={c2} C3={c3} -> {verdict}")

    sanity = pd.concat([sanity, pd.DataFrame([{
        "run_id": RUN_ID, "commit_sha": sha, "backbone": "googlenet",
        "layer_path": best_layer, "check": "VERDICT",
        "stage": f"C1_resolved_layer_all_zero={c1}|C2_layer_recovers_localisation={c2}|"
                 f"C3_randomisation_changes_cam={c3}",
        "randomised_so_far": "", "n": len(samples),
        "spearman_mean": cascade_final, "spearman_median": float("nan"),
        "frac_zero_cam": float(bad["frac_zero_cam"]),
        "verdict": verdict, "criterion": VERDICT_CRITERION,
    }])], ignore_index=True)
    sanity.to_csv(SANITY_CSV, index=False)

    print("\n[check 4] head structure")
    check4_head_structure(models["googlenet"], samples[0]["img"], HEAD_TXT)

    make_figure(cams_cache[("googlenet", "features.17")],
                cams_cache[("googlenet", best_layer)], samples, best_layer, FIG_PNG)

    for p in (SWEEP_CSV, METHODS_CSV, SANITY_CSV, HEAD_TXT, FIG_PNG):
        print(f"[WROTE] {p}")

    if args.self_check:
        assert bad["frac_zero_cam"] == 1.0, (
            f"expected the resolved Dropout layer to be degenerate on every sample, "
            f"got frac_zero_cam={bad['frac_zero_cam']}"
        )
        good = sweep[(sweep.backbone == "googlenet") & (sweep.layer_path == best_layer)].iloc[0]
        assert good["frac_zero_cam"] == 0.0 and good["energy_mean"] > 0.0, (
            f"expected a live CAM at {best_layer}, got frac_zero_cam="
            f"{good['frac_zero_cam']} energy_mean={good['energy_mean']}"
        )
        assert np.isfinite(cascade_final) and cascade_final < 0.9, (
            f"fully randomised weights left the CAM at spearman={cascade_final}; "
            f"that map is not explaining the model"
        )
        print("[self-check] all 3 asserts passed")


if __name__ == "__main__":
    main()
