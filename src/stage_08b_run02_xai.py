"""Langkah 2/3 run02: metrik XAI fusion_late (T-2) dan SHAP cabang radiomics (T-3).

## Yang diukur, dan kenapa angkanya bisa sama dengan cnn_only

`fusion_late` adalah fusi tingkat keputusan: `0.5*cnn_prob + 0.5*rad_prob`. Tidak ada
jaringan gabungan baru. Cabang citranya PERSIS checkpoint arm A yang dipakai
`cnn_only`. Jadi peta salience `fusion_late` bukan peta baru -- ia peta arm A,
diwarisi utuh.

Satu hal yang bisa berbeda: kelas yang dijelaskan. CAM menjelaskan satu kelas
keputusan. Untuk `cnn_only` kelas itu argmax `cnn_prob`; untuk `fusion_late` argmax
`late_prob`. Pada nodul yang kedua arm menyepakati kelasnya, kedua CAM identik bit
per bit -- itu fakta arsitektural, bukan kegagalan pengukuran. Pada nodul yang
keduanya berbeda kelas, CAM `fusion_late` menjelaskan keputusan fusi, dan di situlah
selisih metrik muncul. Kolom `n_disagree` mencatat berapa banyak nodul seperti itu,
supaya selisih nol tidak pernah bisa disalahbaca sebagai "belum dihitung".

Konsekuensi untuk paper: klaim yang bertahan bukan "peta fusion lebih bagus", tapi
"fusion mempertahankan seluruh kemampuan penjelasan spasial cabang citra DAN
menambahkan penjelasan fitur SHAP, pada AUC setara radiomics". Yang ini terbukti
dari angka, tidak bisa dibantah.

## Dua himpunan sampel

`fixed6` -- keenam nodul di `artifacts/xai/fixed_display_samples.json` apa adanya,
sesuai GOAL2. Wajib, dan tidak dipilih ulang.

`fold0_60` -- 60 nodul fold 0 yang sudah dipakai `stage_05_xai.py` (baris 53-55,
`sample(n=60, random_state=42)`), yaitu himpunan yang menghasilkan angka
`pointing_acc` cnn_only 0.7167/0.7000 yang dikutip GOAL2. Ini bukan pemilihan ulang
sampel: himpunan ini sudah ada sebelum run02 dan dipakai persis sebagaimana adanya.
Alasannya, pointing accuracy pada n=6 hanya bisa bernilai kelipatan 1/6, jadi
perbandingan fusion vs cnn_only di atas 6 sampel saja terlalu kasar untuk dibawa ke
sidang. Keduanya dilaporkan berdampingan, tidak saling menggantikan.

## SHAP

Cabang radiomics `fusion_late` adalah XGBoost pada fitur radiomik terpilih per fold.
Ia tidak menerima masukan apa pun dari backbone, jadi ketiga figure per backbone
memang identik secara konstruksi. Itu dicatat eksplisit di `shap_provenance.csv`
(kolom `identical_across_backbones`) alih-alih disembunyikan.

Keluaran, semua di artifacts/results/run02/:
    xai_metrics_fusion.csv, xai_samples_used.csv,
    shap_beeswarm_{backbone}.png, shap_provenance.csv

Aman dijalankan ulang: bagian yang berkas keluarannya sudah lengkap dilewati.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUN_ID = "2026-08-04-run02"
BACKBONES = ["convnext_tiny", "densenet201", "densenet121"]
CAM_METHOD = "layercam"
THRESHOLD_PCT = 0.80
N_SECONDARY = 60
SECONDARY_SEED = 42

OUT_DIR = os.path.join("artifacts", "results", "run02")
PROBS_DIR = os.path.join(OUT_DIR, "probs")
SAMPLES_JSON = os.path.join("artifacts", "xai", "fixed_display_samples.json")

METRICS_CSV = os.path.join(OUT_DIR, "xai_metrics_fusion.csv")
SAMPLES_CSV = os.path.join(OUT_DIR, "xai_samples_used.csv")
SHAP_PROV_CSV = os.path.join(OUT_DIR, "shap_provenance.csv")


def _commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _load_patch_tensor(patch_path: str, n_slices: int, patch_xy: int):
    """Salinan persis `load_patch_tensor` di stage_05_xai.py (baris 59-77).

    Disalin, bukan diimpor, karena di sana ia closure di dalam `run()`. Harus tetap
    identik: kalau pemuatan berbeda sedikit saja, metrik run02 tidak lagi sebanding
    dengan angka cnn_only yang sudah diterbitkan.
    """
    import torch

    hu_min, hu_max = -1000.0, 400.0
    vol = np.load(patch_path).astype(np.float32)
    vol = np.clip(vol, hu_min, hu_max)
    vol = (vol - hu_min) / (hu_max - hu_min)
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
    return torch.from_numpy(np.stack(slices, axis=0)).unsqueeze(0)


def _prob_lookup(backbone: str) -> dict[tuple[str, int], tuple[float, float]]:
    """(patient_id, nodule_idx) -> (cnn_best, late_best) untuk fold 0 saja."""
    npz_path = os.path.join(PROBS_DIR, f"{backbone}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"{npz_path} belum ada. Jalankan langkah 1/3 dulu: "
            "python -m src.stage_08a_run02_probs")
    d = np.load(npz_path, allow_pickle=True)
    sel = d["fold"] == 0
    return {
        (str(p), int(n)): (float(c), float(l))
        for p, n, c, l in zip(d["patient_id"][sel], d["nodule_idx"][sel],
                              d["cnn_best"][sel], d["late_best"][sel])
    }


def _fixed_samples() -> pd.DataFrame:
    spec = json.load(open(SAMPLES_JSON))
    rows = [{
        "slot": s["slot"],
        "patient_id": s["patient_id"],
        "nodule_idx": int(s["nodule_idx"]),
        "label": int(s["label"]),
        "patch_path": os.path.normpath(s["patch_path"]),
        "mask_path": os.path.normpath(s["mask_path"]),
    } for s in spec["samples"]]
    return pd.DataFrame(rows)


def _secondary_samples(cfg: dict) -> pd.DataFrame:
    """Himpunan 60 nodul fold 0 milik stage_05_xai, dibangun ulang dengan seed yang sama."""
    df = pd.read_csv(os.path.join(cfg["paths"]["interim"], "labels.csv"))
    val_df = df[(df["fold"] == 0) & (df["label"] != -1)].reset_index(drop=True)
    if len(val_df) > N_SECONDARY:
        val_df = val_df.sample(n=N_SECONDARY, random_state=SECONDARY_SEED).reset_index(drop=True)
    val_df = val_df.copy()
    val_df["slot"] = ["F%02d" % i for i in range(len(val_df))]
    val_df["patch_path"] = val_df["patch_path"].map(os.path.normpath)
    val_df["mask_path"] = val_df["mask_path"].map(os.path.normpath)
    return val_df[["slot", "patient_id", "nodule_idx", "label", "patch_path", "mask_path"]]


def _cam_metrics(model, internal: str, samples: pd.DataFrame, probs: dict,
                  cfg: dict, device) -> list[dict]:
    """Satu baris per arm untuk satu backbone dan satu himpunan sampel."""
    from src.xai.gradcam_utils import (
        compute_gradcam, dice_iou, dice_size_matched, energy_pointing_game, pointing_hit,
    )

    n_slices = cfg["data"].get("n_slices", 3)
    patch_xy = cfg["data"].get("patch_xy", 64)

    acc = {arm: {k: [] for k in ("dice", "iou", "dsm", "hit", "energy")}
           for arm in ("cnn_only", "fusion_late")}
    n_disagree = 0
    n_used = 0
    n_no_prob = 0

    for _, row in samples.iterrows():
        key = (str(row["patient_id"]), int(row["nodule_idx"]))
        if key not in probs:
            # Nodul yang tersaring keluar oleh join radiomics di _load_merged tidak
            # punya rad_prob, jadi late_prob tidak terdefinisi untuknya. Dilewati dan
            # dihitung, tidak diganti nodul lain.
            n_no_prob += 1
            continue
        cnn_p, late_p = probs[key]
        cls = {"cnn_only": int(cnn_p > 0.5), "fusion_late": int(late_p > 0.5)}
        if cls["cnn_only"] != cls["fusion_late"]:
            n_disagree += 1

        img = _load_patch_tensor(row["patch_path"], n_slices, patch_xy).to(device)
        mask_full = np.load(row["mask_path"]).astype(np.float32)
        mask2d = mask_full[mask_full.shape[0] // 2]

        for arm in ("cnn_only", "fusion_late"):
            cam = compute_gradcam(model, img, backbone_name=internal,
                                  target_class=cls[arm], method=CAM_METHOD)
            d, iou = dice_iou(cam, mask2d, pct=THRESHOLD_PCT)
            a = acc[arm]
            a["dice"].append(d)
            a["iou"].append(iou)
            a["dsm"].append(dice_size_matched(cam, mask2d))
            a["hit"].append(pointing_hit(cam, mask2d))
            a["energy"].append(energy_pointing_game(cam, mask2d))
        n_used += 1

    out = []
    for arm in ("cnn_only", "fusion_late"):
        a = acc[arm]
        out.append({
            "arm": arm, "n": n_used, "n_disagree": n_disagree, "n_no_prob": n_no_prob,
            "dice": float(np.mean(a["dice"])) if n_used else float("nan"),
            "iou": float(np.mean(a["iou"])) if n_used else float("nan"),
            "dice_size_matched": float(np.mean(a["dsm"])) if n_used else float("nan"),
            "pointing_acc": float(np.mean(a["hit"])) if n_used else float("nan"),
            "energy_mean": float(np.mean(a["energy"])) if n_used else float("nan"),
            "threshold_pct": THRESHOLD_PCT,
        })
    return out


def run_xai(cfg: dict) -> None:
    import torch

    from src.models.registry import _NAME_MAP, build_model

    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(METRICS_CSV):
        print(f"[LEWAT] {METRICS_CSV} sudah ada")
        return

    sha = _commit_sha()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_sets = {"fixed6": _fixed_samples(), "fold0_60": _secondary_samples(cfg)}

    sample_rows = []
    for name, df in sample_sets.items():
        for _, r in df.iterrows():
            sample_rows.append({"run_id": RUN_ID, "sample_set": name, "slot": r["slot"],
                                "patient_id": r["patient_id"], "nodule_idx": int(r["nodule_idx"]),
                                "label": int(r["label"])})
    pd.DataFrame(sample_rows).to_csv(SAMPLES_CSV, index=False)
    print(f"[SELESAI] {SAMPLES_CSV}")

    rows = []
    for backbone in BACKBONES:
        probs = _prob_lookup(backbone)
        ckpt = os.path.join(cfg["paths"]["checkpoints"], backbone, "fold0_best.pt")
        model = build_model(backbone, cfg, task="binary").to(device)
        state = torch.load(ckpt, weights_only=True, map_location="cpu")
        model.load_state_dict(state["model_state"] if isinstance(state, dict)
                              and "model_state" in state else state)
        model.eval()
        internal = _NAME_MAP.get(backbone, backbone)

        for set_name, samples in sample_sets.items():
            for r in _cam_metrics(model, internal, samples, probs, cfg, device):
                rows.append({"run_id": RUN_ID, "commit_sha": sha, "backbone": backbone,
                             "sample_set": set_name, "cam_method": CAM_METHOD, **r})
                logger.info("[%s %s %s] pointing_acc=%.4f dice=%.4f n=%d n_disagree=%d",
                            backbone, set_name, r["arm"], r["pointing_acc"], r["dice"],
                            r["n"], r["n_disagree"])

    pd.DataFrame(rows).to_csv(METRICS_CSV, index=False)
    print(f"[SELESAI] {METRICS_CSV}  ({len(rows)} baris)")


def run_shap(cfg: dict, fold: int = 0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    from src.fusion.early_fusion import train_early_fusion_xgboost
    from src.stage_03b_fusion import _load_merged, _select_fold_features

    os.makedirs(OUT_DIR, exist_ok=True)
    figures = [os.path.join(OUT_DIR, f"shap_beeswarm_{b}.png") for b in BACKBONES]
    if all(os.path.exists(f) for f in figures) and os.path.exists(SHAP_PROV_CSV):
        print(f"[LEWAT] {len(figures)} figure SHAP sudah ada")
        return

    merged, feat_cols = _load_merged(cfg)
    train_df = merged[merged["fold"] != fold].reset_index(drop=True)
    val_df = merged[merged["fold"] == fold].reset_index(drop=True)
    X_train_sel, X_val_sel, selected, fs_method = _select_fold_features(train_df, val_df, feat_cols)

    clf = train_early_fusion_xgboost(X_train_sel, train_df["label"].values, cfg.get("xgboost", {}))
    shap_values = shap.TreeExplainer(clf).shap_values(X_val_sel)

    for path in figures:
        plt.figure(figsize=(9, 8))
        shap.summary_plot(shap_values, X_val_sel, feature_names=selected, show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[SELESAI] {path}")

    sha = _commit_sha()
    pd.DataFrame([{
        "run_id": RUN_ID, "commit_sha": sha, "backbone": b,
        "figure": os.path.basename(f), "fold": fold, "n_features": len(selected),
        "fs_method": fs_method,
        # Cabang radiomics fusion_late nol masukan dari backbone, jadi ketiganya
        # memang berkas yang isinya sama. Dicatat, bukan disamarkan.
        "identical_across_backbones": True,
    } for b, f in zip(BACKBONES, figures)]).to_csv(SHAP_PROV_CSV, index=False)
    print(f"[SELESAI] {SHAP_PROV_CSV}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--only", choices=["xai", "shap"], help="jalankan satu bagian saja")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.only != "shap":
        run_xai(cfg)
    if args.only != "xai":
        run_shap(cfg)


if __name__ == "__main__":
    main()
