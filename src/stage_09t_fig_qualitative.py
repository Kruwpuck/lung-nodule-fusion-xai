"""Stage 09t: qualitative Layer-CAM grid at the band, canonical and best explanation sites (Track 2 manuscript).

POST HOC, DESCRIPTIVE, ADDITIVE. Writes only under artifacts/results/track2rev/qualitative/. Checkpoints, patches and masks
are opened read-only (torch.load / np.load); nothing is saved to artifacts/checkpoints or artifacts/patches.

Needs the run04 fold-0 checkpoints and the 60-nodule patch/mask files, which are not on every machine.
`--dry-run` needs only the CSVs: it prints the checkpoint paths and mtimes it would read, the selected nodules and the
output paths, and computes nothing. This script has NOT been run end to end on the machine where it was written.

Backbones (a priori, not by panel appearance): ConvNeXt-Tiny, DenseNet121, DenseNet201 (the three nested-CV retrained
backbones, the only ones with faithfulness numbers in the manuscript) and GoogLeNet (site-degenerate case: the band rule
resolves it to a 1x1 Dropout that emits an all-zero map).

Nodules (fixed rule that never looks at a CAM; every row is the same nodule in every panel):
  S-small : malignant, predicted malignant by all 12 backbones, smallest mask
  S-large : same, largest mask
  S-miss  : malignant and predicted benign by all selected backbones; among those, lowest mean malignant probability

Sites: band, canonical and best-in-candidate-set modules from depth_curve_summary.csv (the same sites as the manuscript's
tables). Where two kinds are the same module the two maps are identical by construction. Maps are Layer-CAM of the
predicted class, per-map scaled to [0, 1] and drawn at alpha 0.5 over the middle slice with a fixed 0-1 colorbar; the
ground-truth consensus mask contour is lime.

Inputs (read only): artifacts/results/track2rev/{depth_sweep_12_persample,depth_curve_summary}.csv, configs/config.yaml,
                    <paths.checkpoints>/<backbone>/fold0_best.pt, patches and masks of the 60 nodules.
Outputs: artifacts/results/track2rev/qualitative/{t2_qualitative_selection.csv, fig_t2_qualitative_print.png, .pdf}
         (stage with `git add -f`)

Run:  .venv/Scripts/python.exe -m src.stage_09t_fig_qualitative --dry-run
      .venv/Scripts/python.exe -m src.stage_09t_fig_qualitative [--force]
      .venv/Scripts/python.exe -m src.stage_09t_fig_qualitative --self-check
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess

import numpy as np
import pandas as pd

DIR = os.path.join("artifacts", "results", "track2rev")
OUT_DIR = os.path.join(DIR, "qualitative")
OUT_CSV = os.path.join(OUT_DIR, "t2_qualitative_selection.csv")
OUT_PNG = os.path.join(OUT_DIR, "fig_t2_qualitative_print.png")
OUT_PDF = os.path.join(OUT_DIR, "fig_t2_qualitative_print.pdf")
CONFIG_PATH = os.path.join("configs", "config.yaml")
BACKBONES = ["convnext_tiny", "densenet121", "densenet201", "googlenet"]
NAME = {"densenet121": "DenseNet121", "googlenet": "GoogLeNet", "convnext_tiny": "ConvNeXt-Tiny", "densenet201": "DenseNet201"}
KINDS = [("band_site", "band"), ("canonical_site", "canonical"), ("best_site", "best")]
# A site outside the candidate set that the manuscript reports for one backbone (GoogLeNet, Fig. 1: 0.767 at features.13).
EXTRA = {"googlenet": "features.13"}


def cached() -> bool:
    return all(os.path.exists(p) for p in (OUT_CSV, OUT_PNG, OUT_PDF))


def select_nodules(backbones: list[str]) -> pd.DataFrame:
    p = pd.read_csv(os.path.join(DIR, "depth_sweep_12_persample.csv"))
    q = p.drop_duplicates(["backbone", "sample_idx"])
    g = q.groupby("sample_idx").agg(label=("label", "first"), n_pred_malignant=("pred", "sum"), mask_px=("mask_px", "first"),
                                    patient_id=("patient_id", "first"), nodule_idx=("nodule_idx", "first"),
                                    n_backbones=("backbone", "nunique"), run_id=("run_id", "first"), commit_sha=("commit_sha", "first"))
    assert (g["n_backbones"] == 12).all()
    ok = g[(g["label"] == 1) & (g["n_pred_malignant"] == 12)].sort_values("mask_px")
    sub = q[q["backbone"].isin(backbones)]
    sub_g = sub.groupby("sample_idx").agg(n_sel_pred_malignant=("pred", "sum"), mean_prob=("prob_malignant", "mean"))
    miss = g.join(sub_g)
    miss = miss[(miss["label"] == 1) & (miss["n_sel_pred_malignant"] == 0)].sort_values("mean_prob")
    assert len(ok) >= 2 and len(miss) >= 1, "selection rule found no nodule"
    sel = pd.concat([ok.iloc[[0]], ok.iloc[[-1]], miss.iloc[[0]]])
    sel["rule"] = ["S-small: smallest mask among all-correct malignant", "S-large: largest mask among all-correct malignant",
                   "S-miss: malignant missed by all selected backbones, lowest mean probability"]
    return sel.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=CONFIG_PATH)
    ap.add_argument("--backbones", nargs="*", default=BACKBONES)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="recompute even if the outputs exist")
    ap.add_argument("--self-check", action="store_true", help="validate existing outputs and exit")
    args = ap.parse_args()

    sel = select_nodules(args.backbones)
    sites = pd.read_csv(os.path.join(DIR, "depth_curve_summary.csv")).set_index("backbone")
    cfg_ckpt = None
    if os.path.exists(args.config):
        try:
            import yaml
            cfg_ckpt = yaml.safe_load(open(args.config, encoding="utf-8"))["paths"]["checkpoints"]
        except ImportError:  # dry-run on a machine without PyYAML: read the one line we need
            import re
            m = re.search(r"^\s*checkpoints:\s*[\"']?([^\"'\s#]+)", open(args.config, encoding="utf-8").read(), re.M)
            cfg_ckpt = m.group(1) if m else None

    if args.self_check:
        assert cached(), "outputs missing"
        out = pd.read_csv(OUT_CSV)
        need = {"run_id", "commit_sha", "input_size", "checkpoint_mtime", "gpu_device", "torch_version", "backbone", "sample_idx",
                "site_kind", "module_path", "prob_malignant", "cam_max", "patch_xy", "source_run_id", "source_commit_sha", "script_sha256"}
        assert need <= set(out.columns), need - set(out.columns)
        per = {b: len(KINDS) + (b in EXTRA) for b in args.backbones}
        assert len(out) == sum(per.values()) * len(sel), len(out)
        assert all(len(g) == per[b] for (b, _), g in out.groupby(["backbone", "sample_idx"]))
        print("[SELF-CHECK OK]", len(out), "rows")
        return

    print("selected nodules:")
    print(sel[["sample_idx", "patient_id", "nodule_idx", "mask_px", "rule"]].to_string(index=False))
    print("sites and checkpoints:")
    for b in args.backbones:
        r = sites.loc[b]
        ck = os.path.join(cfg_ckpt, b, "fold0_best.pt") if cfg_ckpt else "<config not found>"
        mt = (pd.Timestamp(os.path.getmtime(ck), unit="s").isoformat() if os.path.exists(ck) else "MISSING on this machine")
        print(f"  {b:15s} band {r.band_site} | canonical {r.canonical_site} | best {r.best_site}"
              + (f" | outside set {EXTRA[b]}" if b in EXTRA else "") + "\n"
              f"  {'':15s} checkpoint {ck} (mtime {mt})")
    print("outputs:", OUT_CSV, OUT_PNG, OUT_PDF)
    if args.dry_run:
        return
    if cached() and not args.force:
        print("[CACHED] outputs exist; use --force to recompute")
        return
    os.makedirs(OUT_DIR, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    import yaml
    from pytorch_grad_cam import LayerCAM

    from src.stage_09d_cam_12 import _load_model, _load_patch_tensor, _samples
    from src.xai.gradcam_utils import vit_reshape_transform

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    n_slices, patch_xy = cfg["data"].get("n_slices", 3), cfg["data"].get("patch_xy", 64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    samples = _samples(cfg)
    rows = [samples.loc[int(i)] for i in sel["sample_idx"]]
    # sample_idx in the stage-09e CSVs is the row index of _samples(cfg); fail loudly if this machine's labels.csv disagrees
    for row, (_, want) in zip(rows, sel.iterrows()):
        assert (row["patient_id"], int(row["nodule_idx"])) == (want["patient_id"], int(want["nodule_idx"])), \
            f"sample_idx {want['sample_idx']}: expected {want['patient_id']} nodule {want['nodule_idx']}, got {row['patient_id']} nodule {row['nodule_idx']}"
        assert int(row["label"]) == int(want["label"]) == 1
    imgs = [_load_patch_tensor(r["patch_path"], n_slices, patch_xy).to(device) for r in rows]
    masks = []
    for r in rows:
        m = np.load(r["mask_path"]).astype(np.float32)
        masks.append(m[m.shape[0] // 2])
    cts = [img[0, img.shape[1] // 2].cpu().numpy() for img in imgs]

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
                         "font.size": 7, "axes.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    nb, nn_ = len(args.backbones), len(rows)
    nk = len(KINDS) + (1 if any(b in EXTRA for b in args.backbones) else 0)
    fig, axes = plt.subplots(nb + 2, 1 + nn_ * nk, figsize=(7.16, 0.72 * (nb + 1) + 0.9),
                             gridspec_kw={"height_ratios": [0.55] + [1] * nb + [0.12], "width_ratios": [1.9] + [1] * (nn_ * nk)})
    tags = ["small malignant", "large malignant", "missed malignant"]
    for k in range(nn_):
        for j in range(nk):
            ax = axes[0, 1 + k * nk + j]
            ax.imshow(cts[k], cmap="gray")
            ax.contour(masks[k], colors="lime", linewidths=0.7)
            ax.set_title(KINDS[j][1] if j < len(KINDS) else "outside set", fontsize=6.5, pad=1.5)
        axes[0, 1 + k * nk + 1].text(0.5, 1.45, tags[k], transform=axes[0, 1 + k * nk + 1].transAxes, ha="center", fontsize=7)
    axes[0, 0].text(0.95, 0.5, "CT, mask", ha="right", va="center", fontsize=7)
    records, last_im = [], None
    for i, b in enumerate(args.backbones, start=1):
        model, ckpt = _load_model(b, cfg, device)
        mtime = pd.Timestamp(os.path.getmtime(ckpt), unit="s").isoformat()
        reshape = vit_reshape_transform if "vit" in b else None
        mods = dict(model.named_modules())
        axes[i, 0].text(0.95, 0.5, NAME[b], ha="right", va="center", fontsize=7)
        for k, img in enumerate(imgs):
            with torch.no_grad():
                prob = F.softmax(model(img), dim=1)[0, 1].item()
            site_list = [(sites.loc[b, col], kind) for col, kind in KINDS] + ([(EXTRA[b], "outside_candidate_set")] if b in EXTRA else [])
            if len(site_list) < nk:
                axes[i, 1 + k * nk + nk - 1].set_visible(False)
            for j, (path, kind) in enumerate(site_list):
                with LayerCAM(model=model, target_layers=[mods[path]], reshape_transform=reshape) as cam:
                    m = cam(input_tensor=img, targets=None)[0]
                ax = axes[i, 1 + k * nk + j]
                ax.imshow(cts[k], cmap="gray")
                last_im = ax.imshow(m, alpha=0.5, cmap="jet", vmin=0, vmax=1)
                ax.contour(masks[k], colors="lime", linewidths=0.5)
                records.append({"backbone": b, "sample_idx": int(sel["sample_idx"].iloc[k]), "site_kind": kind, "module_path": path,
                                "prob_malignant": prob, "cam_max": float(m.max()), "checkpoint": ckpt, "checkpoint_mtime": mtime})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    cax = fig.add_axes([0.40, 0.035, 0.30, 0.018])
    fig.colorbar(last_im, cax=cax, orientation="horizontal").set_label("Layer-CAM (per-map scaled)", fontsize=6.5, labelpad=1)
    cax.tick_params(labelsize=6, length=2)
    for ax in axes.ravel():
        if ax is cax:
            continue
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[-1, 0].set_visible(False)
    for ax in axes[-1, :]:
        ax.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.9, bottom=0.1, wspace=0.03, hspace=0.04)
    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    out = pd.DataFrame(records)
    # Last commit (on any fetched ref) that touched this script, not the checked-out HEAD: the script may be run from another
    # branch's working tree. `git log -1 -- file` alone follows HEAD and returns nothing there, hence --all.
    try:
        here = os.path.abspath(__file__)
        sha = subprocess.check_output(["git", "log", "--all", "-1", "--format=%h", "--", here], text=True,
                                      cwd=os.path.dirname(here)).strip() or "unknown"
    except Exception:  # noqa: BLE001
        sha = "unknown"
    t1 = cfg.get("tracks", {}).get("track1", {})
    out["run_id"] = "2026-10-qualitative-posthoc"
    out["commit_sha"] = sha
    out["source_run_id"] = sel["run_id"].iloc[0]
    out["source_commit_sha"] = sel["commit_sha"].iloc[0]
    out["script_sha256"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    # model input: the track-1 backbones upsample the 64 px crop to tracks.track1.input_size inside forward
    out["input_size"] = out["backbone"].map(lambda b: t1.get("input_size", patch_xy) if b in set(t1.get("backbones", [])) else patch_xy)
    out["patch_xy"] = patch_xy
    out["gpu_device"] = gpu
    out["torch_version"] = torch.__version__
    out["note"] = "post_hoc_descriptive; nodules chosen by fixed rule, not by map quality"
    out.to_csv(OUT_CSV, index=False)
    print("git add -f", OUT_CSV, OUT_PNG, OUT_PDF, "src/stage_09t_fig_qualitative.py")


if __name__ == "__main__":
    main()
