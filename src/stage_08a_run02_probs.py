"""Langkah 1/3 run02: hitung sekali, simpan, probabilitas per arm untuk fusion_late.

Kenapa tahap ini ada. `stage_03b_fusion` menghitung `cnn_prob`, `rad_prob` dan
`late_prob` di memori lalu membuangnya; yang tersisa di disk hanya rerata metrik
per fold dan satu tabel DeLong melawan arm tunggal terbaik. Semua tugas run02
(T-0, T-1, T-2, T-4) butuh probabilitas mentahnya, jadi tahap ini menghitung ulang
bagian yang murah -- inferensi CNN dan XGBoost radiomics, tanpa melatih FusionNet
apa pun -- lalu menyimpannya.

Resep per fold identik dengan `_run_backbone_arms`: subset nodul yang sama, seleksi
fitur per fold yang sama, XGBoost yang sama, `average_fusion(..., weight_cnn=0.5)`
yang sama. Yang ditambahkan cuma satu: varian `cnn_last`.

Kenapa `cnn_last` ada. `stage_03_train` memilih `fold{f}_best.pt` berdasarkan AUC
pada fold yang sama yang kemudian dipakai melaporkan hasil (`val_df = df[df.fold ==
fold]`, baris 184, lalu `is_best` pada baris 268 dan `save_ckpt(best_pt, ...)` pada
baris 294). Jadi `cnn_only` dan komponen CNN `fusion_late` sama-sama menikmati
seleksi checkpoint di atas fold penilaiannya sendiri, sementara `radiomics_only`
tidak. `fold{f}_last.pt` adalah checkpoint akhir latihan yang tidak dipilih
berdasarkan apa pun, jadi ia batas bawah yang jujur: kalau keunggulan `fusion_late`
bertahan dengan `cnn_last`, keunggulan itu bukan artefak seleksi checkpoint.

Ini BUKAN nested CV penuh. Nested CV sejati perlu inner-val terpisah, artinya
melatih ulang 3 backbone x 5 fold. Yang di sini analisis sensitivitas tanpa latihan
ulang, dan wajib dilaporkan sebagai itu.

Keluaran, satu berkas per backbone:

    artifacts/results/run02/probs/{backbone}.npz

berisi array sejajar sepanjang seluruh 5 fold yang digabung:
    y_true, fold, patient_id, nodule_idx, cnn_best, cnn_last, rad, late_best, late_last

Aman dijalankan ulang: backbone yang berkasnya sudah ada dilewati.
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dikunci oleh handoff/GOAL2.md. Agent dilarang mengubah daftar ini.
BACKBONES = ["convnext_tiny", "densenet201", "densenet121"]
OUT_DIR = os.path.join("artifacts", "results", "run02", "probs")


def run(cfg: dict) -> None:
    import torch

    from src.fusion.early_fusion import train_early_fusion_xgboost
    from src.fusion.late_fusion import average_fusion
    from src.models.registry import _NAME_MAP
    from src.stage_03b_fusion import _cnn_only_preds, _load_merged, _select_fold_features

    os.makedirs(OUT_DIR, exist_ok=True)
    merged, feat_cols = _load_merged(cfg)
    n_folds = cfg["data"].get("n_folds", 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in BACKBONES:
        out = os.path.join(OUT_DIR, f"{model_name}.npz")
        if os.path.exists(out):
            print(f"[LEWAT] {out} sudah ada")
            continue

        internal = _NAME_MAP.get(model_name, model_name)
        cols: dict[str, list] = {k: [] for k in (
            "y_true", "fold", "patient_id", "nodule_idx",
            "cnn_best", "cnn_last", "rad",
        )}

        for fold in range(n_folds):
            train_df = merged[merged["fold"] != fold].reset_index(drop=True)
            val_df = merged[merged["fold"] == fold].reset_index(drop=True)

            X_train_sel, X_val_sel, selected, fs_method = _select_fold_features(
                train_df, val_df, feat_cols)
            clf = train_early_fusion_xgboost(
                X_train_sel, train_df["label"].values, cfg.get("xgboost", {}))

            cols["rad"].append(clf.predict_proba(X_val_sel)[:, 1])
            cols["cnn_best"].append(
                _cnn_only_preds(model_name, internal, cfg, fold, val_df, device, "best"))
            cols["cnn_last"].append(
                _cnn_only_preds(model_name, internal, cfg, fold, val_df, device, "last"))
            cols["y_true"].append(val_df["label"].values)
            cols["fold"].append(np.full(len(val_df), fold))
            cols["patient_id"].append(val_df["patient_id"].values.astype(str))
            cols["nodule_idx"].append(val_df["nodule_idx"].values)

            logger.info("[%s fold %d] n_val=%d n_feat=%d fs_method=%s",
                        model_name, fold, len(val_df), len(selected), fs_method)

        arr = {k: np.concatenate(v) for k, v in cols.items()}
        arr["late_best"] = average_fusion(arr["cnn_best"], arr["rad"], weight_cnn=0.5)
        arr["late_last"] = average_fusion(arr["cnn_last"], arr["rad"], weight_cnn=0.5)
        np.savez(out, **arr)
        print(f"[SELESAI] {out}  ({len(arr['y_true'])} baris)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    run(yaml.safe_load(open(args.config)))


if __name__ == "__main__":
    main()
