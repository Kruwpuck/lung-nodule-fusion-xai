"""Langkah 3/3 run02: analisis (T-0, T-1, T-4) lalu periksa keenam gate GOAL2.

Dua mode:

    python -m src.stage_08c_run02_gates            # bangun tabel lalu periksa
    python -m src.stage_08c_run02_gates --check    # periksa saja, nol tulisan

`--check` keluar dengan kode 0 hanya kalau keenam gate lolos. Tempel keluarannya apa
adanya ke ledger; jangan diringkas jadi "sudah lolos".

Semua AUC di sini AUC gabungan lintas 5 fold yang saling lepas, resep yang sama
dengan `_run_backbone_arms` di stage_03b (satu vektor probabilitas per arm, satu
`roc_auc_score`, satu DeLong berpasangan). Bukan rerata AUC per fold, jadi angkanya
tidak akan persis sama dengan kolom `auc` di `ablation_summary.csv` -- itu memang dua
besaran berbeda, dan yang dipakai DeLong harus yang gabungan.

T-0 memberi dua kolom untuk setiap arm yang punya cabang CNN: `_best` memakai
checkpoint yang dipilih di atas fold penilaiannya sendiri (semua angka lama), `_last`
memakai checkpoint akhir latihan yang tidak dipilih apa pun. Kalau `delta_late`
negatif besar, keunggulan `fusion_late` memang bergantung pada seleksi checkpoint dan
klaim paper wajib menyesuaikan -- itu hasil yang dilaporkan apa adanya, bukan
kegagalan run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

RUN_ID = "2026-08-04-run02"
BACKBONES = ["convnext_tiny", "densenet201", "densenet121"]
ALPHA = 0.05

OUT_DIR = os.path.join("artifacts", "results", "run02")
PROBS_DIR = os.path.join(OUT_DIR, "probs")
SAMPLES_JSON = os.path.join("artifacts", "xai", "fixed_display_samples.json")

T0_CSV = os.path.join(OUT_DIR, "t0_checkpoint_sensitivity.csv")
DELONG_CSV = os.path.join(OUT_DIR, "delong_run02.csv")
XAI_METRICS_CSV = os.path.join(OUT_DIR, "xai_metrics_fusion.csv")
XAI_SAMPLES_CSV = os.path.join(OUT_DIR, "xai_samples_used.csv")
XAI_DELTA_CSV = os.path.join(OUT_DIR, "xai_fusion_vs_cnn.csv")
SHAP_PROV_CSV = os.path.join(OUT_DIR, "shap_provenance.csv")
TABLE_CSV = os.path.join(OUT_DIR, "combined_advantage_table.csv")
TABLE_MD = os.path.join(OUT_DIR, "combined_advantage_table.md")

XAI_METRICS = ["dice", "iou", "pointing_acc", "energy_mean"]


def _commit_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _read(path: str):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _probs(backbone: str):
    path = os.path.join(PROBS_DIR, f"{backbone}.npz")
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None


# --- pembangunan tabel ------------------------------------------------------


def build() -> None:
    from sklearn.metrics import roc_auc_score

    from src.evaluation.statistical_tests import delong_test

    sha = _commit_sha()
    t0_rows, dl_rows = [], []

    for backbone in BACKBONES:
        d = _probs(backbone)
        if d is None:
            print(f"[TUNGGU] {os.path.join(PROBS_DIR, backbone + '.npz')} belum ada")
            continue

        y = d["y_true"]
        auc = {k: float(roc_auc_score(y, d[k]))
               for k in ("cnn_best", "cnn_last", "rad", "late_best", "late_last")}

        _, p_lr_best, _ = delong_test(y, d["late_best"], d["rad"])
        _, p_lr_last, _ = delong_test(y, d["late_last"], d["rad"])
        t0_rows.append({
            "run_id": RUN_ID, "commit_sha": sha, "backbone": backbone,
            "auc_cnn_best": auc["cnn_best"], "auc_cnn_last": auc["cnn_last"],
            "auc_rad": auc["rad"],
            "auc_late_best": auc["late_best"], "auc_late_last": auc["late_last"],
            "delta_late": auc["late_last"] - auc["late_best"],
            "p_late_vs_rad_best": p_lr_best, "p_late_vs_rad_last": p_lr_last,
            "late_beats_rad_best": bool(auc["late_best"] > auc["rad"]),
            "late_beats_rad_last": bool(auc["late_last"] > auc["rad"]),
        })

        # T-1 dan pendampingnya. Varian `last` ikut diuji supaya setiap klaim
        # signifikansi punya pasangan tanpa seleksi checkpoint.
        for kind, cnn_key, late_key in (("best", "cnn_best", "late_best"),
                                         ("last", "cnn_last", "late_last")):
            for comparison, a_arm, a_key, b_arm, b_key in (
                ("fusion_late_vs_cnn_only", "fusion_late", late_key, "cnn_only", cnn_key),
                ("fusion_late_vs_radiomics_only", "fusion_late", late_key, "radiomics_only", "rad"),
            ):
                _, p, _ = delong_test(y, d[a_key], d[b_key])
                dl_rows.append({
                    "run_id": RUN_ID, "commit_sha": sha, "backbone": backbone,
                    "ckpt_kind": kind, "comparison": comparison,
                    "arm_a": a_arm, "auc_a": auc[a_key],
                    "arm_b": b_arm, "auc_b": auc[b_key],
                    "delong_p": float(p),
                    "significant": bool(p < ALPHA),
                    "a_better": bool(auc[a_key] > auc[b_key]),
                })

    os.makedirs(OUT_DIR, exist_ok=True)
    if t0_rows:
        pd.DataFrame(t0_rows).to_csv(T0_CSV, index=False)
        print(f"[SELESAI] {T0_CSV}  ({len(t0_rows)} baris)")
    if dl_rows:
        pd.DataFrame(dl_rows).to_csv(DELONG_CSV, index=False)
        print(f"[SELESAI] {DELONG_CSV}  ({len(dl_rows)} baris)")

    _build_xai_delta()
    _build_combined_table()


def _build_xai_delta() -> None:
    xai = _read(XAI_METRICS_CSV)
    if xai is None:
        print(f"[TUNGGU] {XAI_METRICS_CSV} belum ada")
        return

    rows = []
    for (backbone, sset), grp in xai.groupby(["backbone", "sample_set"]):
        cnn = grp[grp["arm"] == "cnn_only"]
        fus = grp[grp["arm"] == "fusion_late"]
        if cnn.empty or fus.empty:
            continue
        c, f = cnn.iloc[0], fus.iloc[0]
        for metric in XAI_METRICS:
            rows.append({
                "run_id": RUN_ID, "backbone": backbone, "sample_set": sset,
                "n": int(f["n"]), "n_disagree": int(f["n_disagree"]),
                "metric": metric,
                "cnn_only": float(c[metric]), "fusion_late": float(f[metric]),
                "delta": float(f[metric]) - float(c[metric]),
            })
    if rows:
        pd.DataFrame(rows).to_csv(XAI_DELTA_CSV, index=False)
        print(f"[SELESAI] {XAI_DELTA_CSV}  ({len(rows)} baris)")


def _fmt_range(vals) -> str:
    vals = [v for v in vals if v == v]
    if not vals:
        return "-"
    return f"{np.mean(vals):.4f} ({min(vals):.4f}-{max(vals):.4f})"


def _build_combined_table() -> None:
    t0 = _read(T0_CSV)
    dl = _read(DELONG_CSV)
    xai = _read(XAI_METRICS_CSV)
    if t0 is None or dl is None or xai is None:
        print("[TUNGGU] tabel gabungan butuh T0, DeLong dan metrik XAI lengkap")
        return

    dl_best = dl[dl["ckpt_kind"] == "best"]
    vs_cnn = dl_best[dl_best["comparison"] == "fusion_late_vs_cnn_only"]
    vs_rad = dl_best[dl_best["comparison"] == "fusion_late_vs_radiomics_only"]

    # Pointing accuracy dilaporkan dari himpunan 60 nodul; n=6 terlalu kasar untuk
    # jadi angka tunggal di tabel paper, dan nilainya tetap tercetak di gate G-2.
    pa = xai[xai["sample_set"] == "fold0_60"]
    pa_cnn = pa[pa["arm"] == "cnn_only"]["pointing_acc"].tolist()
    pa_fus = pa[pa["arm"] == "fusion_late"]["pointing_acc"].tolist()

    rows = [
        {"criterion": "AUC gabungan 5 fold (rerata 3 backbone)",
         "cnn_only": _fmt_range(t0["auc_cnn_best"]),
         "radiomics_only": _fmt_range(t0["auc_rad"]),
         "fusion_late": _fmt_range(t0["auc_late_best"])},
        {"criterion": "AUC dengan checkpoint tanpa seleksi (T-0)",
         "cnn_only": _fmt_range(t0["auc_cnn_last"]),
         "radiomics_only": _fmt_range(t0["auc_rad"]) + " (tidak terpengaruh)",
         "fusion_late": _fmt_range(t0["auc_late_last"])},
        {"criterion": "DeLong vs fusion_late (p, 3 backbone)",
         "cnn_only": _fmt_range(vs_cnn["delong_p"]) +
                      f"; signifikan {int(vs_cnn['significant'].sum())}/3",
         "radiomics_only": _fmt_range(vs_rad["delong_p"]) +
                            f"; signifikan {int(vs_rad['significant'].sum())}/3",
         "fusion_late": "-"},
        {"criterion": "Peta salience spasial (Grad-CAM/Layer-CAM)",
         "cnn_only": "ada", "radiomics_only": "mustahil secara struktural",
         "fusion_late": "ada (diwarisi dari cabang citra)"},
        {"criterion": "Pointing accuracy (60 nodul fold 0, rerata 3 backbone)",
         "cnn_only": _fmt_range(pa_cnn),
         "radiomics_only": "tidak terdefinisi",
         "fusion_late": _fmt_range(pa_fus)},
        {"criterion": "SHAP fitur radiomik",
         "cnn_only": "tidak ada", "radiomics_only": "ada", "fusion_late": "ada"},
        {"criterion": "Penjelasan spasial DAN fitur sekaligus",
         "cnn_only": "tidak", "radiomics_only": "tidak", "fusion_late": "ya"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_CSV, index=False)
    with open(TABLE_MD, "w", encoding="utf-8") as fh:
        fh.write(f"# Tabel keunggulan gabungan ({RUN_ID}, commit {_commit_sha()})\n\n")
        fh.write(df.to_markdown(index=False))
        fh.write("\n")
    print(f"[SELESAI] {TABLE_CSV}")
    print(f"[SELESAI] {TABLE_MD}")


# --- gate -------------------------------------------------------------------


def _gate(ok: bool, label: str, detail: str) -> bool:
    print(f"  {'LULUS' if ok else 'GAGAL'}  {label}: {detail}")
    return bool(ok)


def check() -> bool:
    results = []

    # G-0 verifikasi T-0 tuntas, dilaporkan ke arah mana pun hasilnya.
    t0 = _read(T0_CSV)
    if t0 is None:
        results.append(_gate(False, "G-0 verifikasi T-0", f"{T0_CSV} tidak ada"))
    else:
        need = ["auc_cnn_best", "auc_cnn_last", "auc_rad", "auc_late_best",
                "auc_late_last", "delta_late", "p_late_vs_rad_best", "p_late_vs_rad_last"]
        miss = [c for c in need if c not in t0.columns]
        nan = int(t0[need].isna().sum().sum()) if not miss else -1
        results.append(_gate(
            len(t0) == len(BACKBONES) and not miss and nan == 0,
            "G-0 verifikasi T-0",
            f"{len(t0)} backbone, diharapkan {len(BACKBONES)}; kolom hilang {miss}; sel kosong {nan}"))
        for _, r in t0.iterrows():
            print(f"         {r['backbone']}: late_best {r['auc_late_best']:.4f} vs rad "
                  f"{r['auc_rad']:.4f} (p {r['p_late_vs_rad_best']:.4f}) | "
                  f"late_last {r['auc_late_last']:.4f} (p {r['p_late_vs_rad_last']:.4f}) | "
                  f"delta {r['delta_late']:+.4f}")

    # G-1 DeLong fusion_late vs cnn_only, 3 backbone.
    dl = _read(DELONG_CSV)
    if dl is None:
        results.append(_gate(False, "G-1 DeLong fusion vs cnn_only", f"{DELONG_CSV} tidak ada"))
    else:
        sel = dl[(dl["comparison"] == "fusion_late_vs_cnn_only") & (dl["ckpt_kind"] == "best")]
        sig = sel[sel["significant"] & sel["a_better"]]
        results.append(_gate(
            len(sel) == len(BACKBONES) and int(sel["delong_p"].isna().sum()) == 0,
            "G-1 DeLong fusion vs cnn_only",
            f"{len(sel)} uji, diharapkan {len(BACKBONES)}; "
            f"fusion signifikan lebih baik {len(sig)}/{len(sel)}; "
            f"p = {[round(float(p), 4) for p in sel['delong_p']]}"))

    # G-2 metrik XAI fusion lengkap, di atas sampel yang identik dengan fixed_display_samples.
    xai = _read(XAI_METRICS_CSV)
    used = _read(XAI_SAMPLES_CSV)
    if xai is None or used is None:
        results.append(_gate(False, "G-2 metrik XAI fusion",
                              f"{XAI_METRICS_CSV} atau {XAI_SAMPLES_CSV} tidak ada"))
    else:
        fus = xai[(xai["arm"] == "fusion_late") & (xai["sample_set"] == "fixed6")]
        nan = int(fus[XAI_METRICS].isna().sum().sum()) if not fus.empty else -1
        want = {(s["patient_id"], int(s["nodule_idx"]))
                for s in json.load(open(SAMPLES_JSON))["samples"]}
        got = {(r["patient_id"], int(r["nodule_idx"]))
               for _, r in used[used["sample_set"] == "fixed6"].iterrows()}
        results.append(_gate(
            len(fus) == len(BACKBONES) and nan == 0 and got == want,
            "G-2 metrik XAI fusion",
            f"{len(fus)} backbone x {len(XAI_METRICS)} metrik, sel kosong {nan}; "
            f"sampel cocok dengan fixed_display_samples.json: {got == want}"))
        for pid, nidx in sorted(want):
            print(f"         sampel tetap: {pid} nodule_idx={nidx}")

    # G-3 tabel selisih XAI fusion vs cnn_only per backbone.
    delta = _read(XAI_DELTA_CSV)
    if delta is None:
        results.append(_gate(False, "G-3 selisih XAI", f"{XAI_DELTA_CSV} tidak ada"))
    else:
        expected = len(BACKBONES) * 2 * len(XAI_METRICS)  # 2 himpunan sampel
        results.append(_gate(
            len(delta) == expected and int(delta["delta"].isna().sum()) == 0,
            "G-3 selisih XAI",
            f"{len(delta)} baris, diharapkan {expected}; "
            f"nodul beda kelas keputusan {sorted(set(delta['n_disagree']))}"))
        pa = delta[(delta["metric"] == "pointing_acc") & (delta["sample_set"] == "fold0_60")]
        for _, r in pa.iterrows():
            print(f"         {r['backbone']} pointing_acc cnn {r['cnn_only']:.4f} -> "
                  f"fusion {r['fusion_late']:.4f} (selisih {r['delta']:+.4f})")

    # G-4 SHAP beeswarm 3 figure dengan provenance.
    prov = _read(SHAP_PROV_CSV)
    figs = [os.path.join(OUT_DIR, f"shap_beeswarm_{b}.png") for b in BACKBONES]
    exist = [f for f in figs if os.path.exists(f) and os.path.getsize(f) > 0]
    ok_prov = prov is not None and {"run_id", "commit_sha"} <= set(prov.columns) \
        and int(prov[["run_id", "commit_sha"]].isna().sum().sum()) == 0 \
        and len(prov) == len(BACKBONES)
    results.append(_gate(
        len(exist) == len(BACKBONES) and ok_prov, "G-4 SHAP beeswarm",
        f"{len(exist)}/{len(BACKBONES)} figure ada, provenance lengkap: {bool(ok_prov)}"))

    # G-5 tabel keunggulan gabungan, nol sel kosong.
    tbl = _read(TABLE_CSV)
    if tbl is None:
        results.append(_gate(False, "G-5 tabel gabungan", f"{TABLE_CSV} tidak ada"))
    else:
        blank = int(tbl.isna().sum().sum() + (tbl.astype(str) == "").sum().sum())
        results.append(_gate(
            len(tbl) >= 6 and blank == 0, "G-5 tabel gabungan",
            f"{len(tbl)} baris kriteria, sel kosong {blank}"))

    ok = all(results)
    print()
    print("Keenam gate lulus." if ok else f"{sum(results)}/{len(results)} gate lulus.")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="periksa gate saja, jangan menulis apa pun")
    args = p.parse_args()
    if not args.check:
        build()
        print()
    sys.exit(0 if check() else 1)


if __name__ == "__main__":
    main()
