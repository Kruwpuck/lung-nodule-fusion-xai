"""Stage 10d (F-4): re-derive the CAM localisation metrics from the run03 checkpoints.

Why this stage exists. F-1 retrained the three Track 1 backbones under nested cross
validation, so every metric derived from the published checkpoints is, for those three,
a metric of a model that the honest protocol does not produce. The AUC side of that was
settled in F-2 and F-3. This is the explainability side: it asks whether the localisation
numbers the manuscript reports move when the checkpoint underneath them is replaced.

What is held fixed, so that only the checkpoint differs:

  * the sample set -- `artifacts/xai/fixed_display_samples.json` is read but never
    rewritten, and the metric samples are drawn exactly as `stage_05_xai` drew them,
    `sample(n=60, random_state=42)` over the labelled nodules of the scored fold;
  * the patch loader -- `_load_patch_tensor`, imported rather than copied;
  * the metric primitives -- `src.xai.gradcam_utils`, imported rather than copied;
  * the target-layer rule -- `_last_spatial_target_layer`, the corrected resolver that
    `stage_09d_cam_12.py` established.

The published run scored fold 0 only, so only the fold-0 rows carry `_published` and
`_delta` columns. The other four folds are reported because the checkpoints exist and a
single fold is a thin basis for a claim about localisation; they are new measurements
with nothing to be compared against, and are labelled as such rather than silently
averaged into the fold-0 comparison.

The unfreeze cell is `uf100`, the same cell F-3 used as `honest_nested_cv`. Its bound is
the one `f2_sensitivity.md` section 6 states: a lower bound on the honest CNN branch, not
an estimate of it, because four protocol differences move together.

Outputs (nothing under artifacts/results/xai/ is touched):
    artifacts/results/run03/xai_metrics_run03.csv        one row per backbone per fold
    artifacts/results/run03/xai_run03_persample.csv      one row per sample

Run:
    .venv/Scripts/python.exe -m src.stage_10d_xai_run03
    .venv/Scripts/python.exe -m src.stage_10d_xai_run03 --self-check
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml

from src.models.registry import _NAME_MAP, build_model
from src.stage_08b_run02_xai import _commit_sha, _load_patch_tensor
from src.stage_09a_target_layer_audit import _module_path, _observe
from src.utils.tracks import track_input_size
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

RUN_ID = "2026-08-22-run03"
UNFREEZE_PCT = 100      # the cell F-3 uses as honest_nested_cv
N_SAMPLES = 60          # stage_05_xai.N_SAMPLES_METRICS
SAMPLE_SEED = 42        # stage_05_xai line 55
THRESHOLD_PCT = 0.80
N_FOLDS = 5
PUBLISHED_FOLD = 0      # the only fold the published numbers were measured on

CONFIG_PATH = os.path.join("configs", "config.yaml")
OUT_DIR = os.path.join("artifacts", "results", "run03")
OUT_CSV = os.path.join(OUT_DIR, "xai_metrics_run03.csv")
PERSAMPLE_CSV = os.path.join(OUT_DIR, "xai_run03_persample.csv")
SWEEP_CSV = os.path.join(OUT_DIR, "xai_depth_sweep_run03.csv")
SWEEP_PERSAMPLE_CSV = os.path.join(OUT_DIR, "xai_depth_sweep_run03_persample.csv")
EFFICIENCY_CSV = os.path.join(OUT_DIR, "efficiency_run03.csv")
PUBLISHED_CSV = os.path.join("artifacts", "results", "xai", "xai_metrics.csv")
FIXED_SAMPLES_JSON = os.path.join("artifacts", "xai", "fixed_display_samples.json")

BACKBONES = ["convnext_tiny", "densenet201", "densenet121"]
METRICS = ["dice", "iou", "dice_size_matched", "pointing_acc", "energy_mean"]


def _mtime(path: str) -> str:
    if not os.path.exists(path):
        return ""
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")


def _display_ids() -> set[str]:
    """The fixed display set, read only, so a row can record whether a sample is in it.

    stage_07b wrote this file and nothing here may rewrite it: the panels in both
    manuscripts show these nodules, and a regenerated set would silently change which
    nodules the figures and the numbers refer to.
    """
    if not os.path.exists(FIXED_SAMPLES_JSON):
        return set()
    with open(FIXED_SAMPLES_JSON, encoding="utf-8") as f:
        payload = json.load(f)
    entries = payload.get("samples", payload) if isinstance(payload, dict) else payload
    ids = set()
    for e in entries if isinstance(entries, list) else []:
        if isinstance(e, dict) and "patient_id" in e and "nodule_idx" in e:
            ids.add(f"{e['patient_id']}#{int(e['nodule_idx'])}")
    return ids


def _samples(cfg: dict, fold: int) -> pd.DataFrame:
    """The 60 held-out nodules of one fold, drawn exactly as stage_05_xai drew fold 0."""
    df = pd.read_csv(os.path.join(cfg["paths"]["interim"], "labels.csv"))
    val_df = df[(df["fold"] == fold) & (df["label"] != -1)].reset_index(drop=True)
    if len(val_df) > N_SAMPLES:
        val_df = val_df.sample(n=N_SAMPLES, random_state=SAMPLE_SEED).reset_index(drop=True)
    return val_df


def _checkpoint(cfg: dict, backbone: str, fold: int) -> str:
    cell = f"{backbone}_uf{UNFREEZE_PCT}"
    return os.path.join(cfg["paths"]["checkpoints"], "run03", cell, f"fold{fold}_best.pt")


def _load_model(cfg: dict, backbone: str, fold: int, device):
    best_pt = _checkpoint(cfg, backbone, fold)
    if not os.path.exists(best_pt):
        raise FileNotFoundError(
            f"{best_pt} tidak ada. Jalankan dulu: python -m src.stage_10b_finetune "
            f"--backbone {backbone} --fold {fold} --unfreeze {UNFREEZE_PCT}")
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
    target = _last_spatial_target_layer(model, dummy)
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


def run_cell(backbone: str, fold: int, cfg: dict, device, sha: str, display: set[str]):
    """Return (summary row, per-sample frame) for one backbone at one fold."""
    import torch.nn.functional as F

    internal = _NAME_MAP.get(backbone, backbone)
    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)
    input_size = track_input_size(cfg, backbone)

    samples = _samples(cfg, fold)
    model, ckpt = _load_model(cfg, backbone, fold, device)
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
        case = f"{row['patient_id']}#{int(row['nodule_idx'])}"
        per_sample.append({
            "run_id": RUN_ID, "commit_sha": sha, "backbone": backbone,
            "unfreeze_pct": UNFREEZE_PCT, "fold": fold, "sample_idx": int(i),
            "patient_id": row["patient_id"], "nodule_idx": int(row["nodule_idx"]),
            "in_display_set": int(case in display),
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
        "run_id": RUN_ID, "commit_sha": sha, "backbone": backbone,
        "internal_name": internal, "unfreeze_pct": UNFREEZE_PCT, "fold": fold,
        "n": len(ps), "input_size": input_size, **layer,
        "threshold_pct": THRESHOLD_PCT,
        "checkpoint": ckpt.replace("\\", "/"), "checkpoint_mtime": _mtime(ckpt),
        "published_mtime": _mtime(PUBLISHED_CSV),
        "dice": ps["dice"].mean(), "iou": ps["iou"].mean(),
        "dice_size_matched": ps["dice_size_matched"].mean(),
        "pointing_acc": ps["pointing_hit"].mean(), "energy_mean": ps["energy"].mean(),
        "n_zero_cam": int((ps["cam_max"] <= 0).sum()),
        "n_display_set": int(ps["in_display_set"].sum()),
    }
    return summary, ps


def measure_efficiency(cfg: dict, sha: str) -> pd.DataFrame:
    """Re-measure cost for the three backbones and compare against the published table.

    Params and FLOPs are properties of the architecture and the input size, and F-1
    changed neither, so they are a control: any movement in them means the model being
    measured is not the model that was published. Latency is the quantity F-4 asks about,
    because it is a property of the machine as well, and the machine has been in use
    since `efficiency_7.csv` was written.

    The measurement is `stage_09f_efficiency_7._measure_one`, imported unchanged, so the
    two tables are produced by identical code and their difference is not a difference in
    how they were measured.
    """
    from src.stage_09f_efficiency_7 import (BATCH_SIZE, N_FORWARDS, N_REPEATS, OUT_CSV
                                            as PUB_EFF_CSV, _measure_one)

    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)
    input_res = (n_slices, patch_xy, patch_xy)
    gpu = torch.cuda.is_available()
    pub = pd.read_csv(PUB_EFF_CSV).set_index("model")

    cols = ["params_M", "gflops", "latency_cpu_ms_median", "latency_gpu_ms_median"]
    rows = []
    for backbone in BACKBONES:
        m = _measure_one(backbone, cfg, input_res, gpu)
        row = {"run_id": RUN_ID, "commit_sha": sha, "model": backbone,
               "input_size": track_input_size(cfg, backbone), **m,
               "input_res": f"{n_slices}x{patch_xy}x{patch_xy}", "batch_size": BATCH_SIZE,
               "n_repeats": N_REPEATS, "n_forwards_per_repeat": N_FORWARDS,
               "gpu_device": torch.cuda.get_device_name(0) if gpu else "",
               "torch_version": torch.__version__}
        for c in cols:
            published = float(pub.loc[backbone, c]) if backbone in pub.index else float("nan")
            row[f"{c}_published"] = published
            row[f"{c}_delta"] = m[c] - published
        rows.append(row)
        logger.info("[%s] params %.3fM (delta %+.3f)  gflops %.4f (delta %+.4f)  "
                    "gpu %.3f ms (delta %+.3f)", backbone, m["params_M"],
                    row["params_M_delta"], m["gflops"], row["gflops_delta"],
                    m["latency_gpu_ms_median"], row["latency_gpu_ms_median_delta"])
    df = pd.DataFrame(rows)

    # The control, asserted rather than eyeballed: architecture cost may not move.
    assert (df["params_M_delta"].abs() < 1e-9).all(), \
        f"params bergerak, arsitekturnya ikut berubah: {df[['model', 'params_M_delta']]}"
    assert (df["gflops_delta"].abs() < 1e-9).all(), \
        f"FLOPs bergerak, arsitekturnya ikut berubah: {df[['model', 'gflops_delta']]}"
    return df


def sweep_cell(backbone: str, fold: int, cfg: dict, device, sha: str):
    """Layer-CAM at every candidate site of one run03 model, not only the resolved one.

    This is the exclusion the pre-registered criteria in the Track 2 manuscript demand
    before an extraordinary localisation number is reported as a property of the model:
    if some other site in the same frozen network recovers pointing accuracy above
    chance, the collapse belongs to the explanation site, and if none does, it belongs
    to the network. The site ladder, the CAM call and the metric aggregation are
    `stage_09e_depth_sweep`'s, imported rather than reimplemented, so the run03 curve is
    directly comparable with the published curve in `track2rev/depth_curve_summary.csv`.
    """
    from src.stage_09e_depth_sweep import (_cams_at, _ladder, _sample_bundle,
                                           _score as _sweep_score)
    from src.xai.gradcam_utils import vit_reshape_transform

    internal = _NAME_MAP.get(backbone, backbone)
    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)

    model, ckpt = _load_model(cfg, backbone, fold, device)
    dummy = torch.zeros(1, n_slices, patch_xy, patch_xy, device=device)
    sites = _ladder(model, internal, dummy)
    reshape = vit_reshape_transform if "vit" in internal.lower() else None
    samples = _sample_bundle(cfg, _samples(cfg, fold), model, device)

    summaries, per_sample = [], []
    for site in sites:
        cams = _cams_at(model, site["module"], samples, reshape)
        summary, rows = _sweep_score(cams, samples, backbone, site, sha, internal)
        extra = {"run_id": RUN_ID, "fold": fold, "unfreeze_pct": UNFREEZE_PCT,
                 "site_rank": site["site_rank"],
                 "is_canonical_pick": site["is_canonical_pick"],
                 "checkpoint_mtime": _mtime(ckpt), "input_size": track_input_size(cfg, backbone)}
        summaries.append({**summary, **extra})
        per_sample.extend({**r, **extra} for r in rows)
        logger.info("  [%s fold %d] %-38s %sx%-3s pointing_acc=%.4f n_zero_cam=%d%s",
                    backbone, fold, site["module_path"], site["spatial_h"],
                    site["spatial_w"], summary["pointing_acc"], summary["n_zero_cam"],
                    "  <- canonical" if site["is_canonical_pick"] else "")

    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(summaries), pd.DataFrame(per_sample)


def _with_published(rows: list[dict]) -> pd.DataFrame:
    """Attach the published value and delta -- on fold 0 only.

    The published run measured fold 0 and nothing else, so a `_published` column on
    folds 1 to 4 would be comparing two different sets of nodules under one name.
    Those cells are left blank rather than filled with the fold-0 number.
    """
    df = pd.DataFrame(rows)
    pub = pd.read_csv(PUBLISHED_CSV).set_index("backbone")
    is_pub_fold = df["fold"] == PUBLISHED_FOLD
    for m in METRICS:
        published = df["backbone"].map(pub[m]).where(is_pub_fold)
        df[f"{m}_published"] = published
        df[f"{m}_delta"] = (df[m] - published).where(is_pub_fold)
    lead = ["run_id", "commit_sha", "backbone", "internal_name", "unfreeze_pct", "fold",
            "n", "input_size", "path_taken", "module_class", "module_path",
            "spatial_h", "spatial_w"]
    metric_cols = [c for m in METRICS for c in (m, f"{m}_published", f"{m}_delta")]
    rest = [c for c in df.columns if c not in lead + metric_cols]
    return df[lead + metric_cols + rest]


def self_check(df: pd.DataFrame, persample: pd.DataFrame) -> None:
    """Four asserts the run03 explainability numbers have to survive."""
    # 1. A post-global-pool 1x1 map normalises to a constant and localises nothing,
    #    which is the failure mode stage_09d found in GoogLeNet. None of these three
    #    may resolve to one.
    degenerate = df[(df["spatial_h"].astype(str) == "1") & (df["spatial_w"].astype(str) == "1")]
    assert degenerate.empty, (
        f"1x1 target layer: {degenerate[['backbone', 'fold']].to_dict('records')}")

    # 2. An identically-zero CAM still yields a finite Dice, so it passes unnoticed unless
    #    it is asserted on. The assertable condition is the one the Track 2 manuscript
    #    pre-registered as criterion (i) of its artifact verdict: a site that emits nothing
    #    on *every* sample explains nothing and its metrics are meaningless. A cell that
    #    emits a live map on most samples and a zero on a few is a different object -- the
    #    site works and some inputs drive it flat -- and it is reported through `n_zero_cam`
    #    on every row rather than crashing the stage. densenet121 fold 4 is exactly that
    #    case, 4 of 60, and it is stated in f4_downstream.md instead of being smoothed away.
    per_cell = persample.groupby(["backbone", "fold"])["cam_max"]
    fully_dead = per_cell.max()[lambda s: s <= 0]
    assert fully_dead.empty, (
        f"situs mati total (peta nol di seluruh 60 sampel): {fully_dead.index.tolist()}")
    counted = (persample.assign(z=persample["cam_max"] <= 0)
               .groupby(["backbone", "fold"])["z"].sum())
    # Matched on the (backbone, fold) key, not positionally: groupby sorts its keys and
    # BACKBONES does not, so a positional comparison would silently pair densenet201's
    # summary row with densenet121's per-sample count.
    recounted = df.set_index(["backbone", "fold"]).index.map(counted)
    assert (df["n_zero_cam"].to_numpy() == recounted.to_numpy()).all(), \
        "n_zero_cam tidak cocok dengan hitungan per sampel"

    # 3. Provenance is the point of F-4: a row whose checkpoint mtime is missing cannot
    #    be checked for staleness later, which is exactly the densenet121 incident.
    missing = df[df["checkpoint_mtime"].fillna("") == ""]
    assert missing.empty, (
        f"checkpoint_mtime kosong: {missing[['backbone', 'fold']].to_dict('records')}")

    # 4. The fold-0 rows are the ones that carry the comparison against the published
    #    numbers, and all three backbones must have it.
    pub_rows = df[df["fold"] == PUBLISHED_FOLD]
    assert len(pub_rows) == len(BACKBONES), f"fold {PUBLISHED_FOLD}: {len(pub_rows)} baris"
    assert pub_rows[[f"{m}_delta" for m in METRICS]].notna().all().all(), \
        "baris fold 0 kehilangan delta terhadap angka terbit"
    assert df[df["fold"] != PUBLISHED_FOLD][[f"{m}_delta" for m in METRICS]].isna().all().all(), \
        "fold selain 0 tidak boleh membawa delta -- angka terbit cuma ada untuk fold 0"
    print("[SELF-CHECK] keempat assert lolos")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=CONFIG_PATH)
    p.add_argument("--self-check", action="store_true", help="jalankan assert pasca-hitung")
    p.add_argument("--efficiency", action="store_true",
                   help="ukur ulang params/FLOPs/latensi dan bandingkan dengan tabel terbit")
    p.add_argument("--depth-sweep", action="store_true",
                   help="sweep seluruh situs penjelasan, bukan cuma yang teresolusi")
    p.add_argument("--fold", type=int, default=PUBLISHED_FOLD,
                   help="fold untuk --depth-sweep")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sha = _commit_sha()
    display = _display_ids()

    if args.efficiency:
        df = measure_efficiency(cfg, sha)
        df.to_csv(EFFICIENCY_CSV, index=False)
        with pd.option_context("display.max_columns", None, "display.width", 250):
            print("\n=== biaya, run03 lawan efficiency_7.csv terbit ===")
            print(df[["model", "params_M", "params_M_delta", "gflops", "gflops_delta",
                      "latency_gpu_ms_median", "latency_gpu_ms_median_published",
                      "latency_gpu_ms_median_delta", "latency_cpu_ms_median",
                      "latency_cpu_ms_median_delta"]].to_string(index=False))
        print(f"\n[WROTE] {EFFICIENCY_CSV} ({len(df)} baris)")
        return

    if args.depth_sweep:
        summaries, per_sample = [], []
        for backbone in BACKBONES:
            s, ps = sweep_cell(backbone, args.fold, cfg, device, sha)
            summaries.append(s)
            per_sample.append(ps)
            pd.concat(summaries, ignore_index=True).to_csv(SWEEP_CSV, index=False)
            pd.concat(per_sample, ignore_index=True).to_csv(SWEEP_PERSAMPLE_CSV, index=False)
        df = pd.concat(summaries, ignore_index=True)
        ps_all = pd.concat(per_sample, ignore_index=True)
        chance = float((ps_all.drop_duplicates(["backbone", "sample_idx"])["mask_px"]
                        / float(cfg["data"].get("patch_xy", 64) ** 2)).mean())
        with pd.option_context("display.max_columns", None, "display.width", 250):
            print(f"\n=== sweep kedalaman, checkpoint run03, fold {args.fold} "
                  f"(garis kebetulan {chance:.4f}) ===")
            print(df[["backbone", "site_rank", "module_class", "module_path", "spatial_h",
                      "pointing_acc", "dice", "energy_mean", "n_zero_cam",
                      "is_canonical_pick"]].to_string(index=False))
            best = df.loc[df.groupby("backbone")["pointing_acc"].idxmax()]
            print("\n=== situs terbaik per backbone ===")
            print(best[["backbone", "module_path", "spatial_h", "pointing_acc",
                        "is_canonical_pick"]].to_string(index=False))
        print(f"\n[WROTE] {SWEEP_CSV} ({len(df)} baris)")
        print(f"[WROTE] {SWEEP_PERSAMPLE_CSV} ({len(ps_all)} baris)")
        return

    logger.info("himpunan tampilan tetap: %d nodul dibaca dari %s (read-only)",
                len(display), FIXED_SAMPLES_JSON)

    rows, per_sample = [], []
    for backbone in BACKBONES:
        for fold in range(N_FOLDS):
            summary, ps = run_cell(backbone, fold, cfg, device, sha, display)
            rows.append(summary)
            per_sample.append(ps)
            # Written after every cell: a crash on cell 12 must not cost cells 1-11.
            _with_published(rows).to_csv(OUT_CSV, index=False)
            pd.concat(per_sample, ignore_index=True).to_csv(PERSAMPLE_CSV, index=False)
            logger.info("[%s fold %d] %s (%sx%s) pointing_acc=%.4f dice=%.4f energy=%.4f "
                        "n_zero_cam=%d",
                        backbone, fold, summary["module_path"], summary["spatial_h"],
                        summary["spatial_w"], summary["pointing_acc"], summary["dice"],
                        summary["energy_mean"], summary["n_zero_cam"])

    df = _with_published(rows)
    ps_all = pd.concat(per_sample, ignore_index=True)

    with pd.option_context("display.max_columns", None, "display.width", 250):
        print("\n=== fold 0: run03 lawan angka terbit ===")
        print(df[df["fold"] == PUBLISHED_FOLD][
            ["backbone", "module_path", "spatial_h", "pointing_acc", "pointing_acc_published",
             "pointing_acc_delta", "dice", "dice_delta", "energy_mean", "energy_mean_delta",
             "n_zero_cam"]].to_string(index=False))
        print("\n=== kelima fold, checkpoint run03 (nol pembanding terbit di luar fold 0) ===")
        print(df[["backbone", "fold", "n", "pointing_acc", "dice", "iou",
                  "dice_size_matched", "energy_mean", "n_zero_cam",
                  "checkpoint_mtime"]].to_string(index=False))

    print(f"\n[WROTE] {OUT_CSV} ({len(df)} baris)")
    print(f"[WROTE] {PERSAMPLE_CSV} ({len(ps_all)} baris)")

    if args.self_check:
        self_check(df, ps_all)


if __name__ == "__main__":
    main()
