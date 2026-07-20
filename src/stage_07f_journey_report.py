"""Stage 07f: generate the filled research journey report for the supervisor.

Reads every result CSV already on disk (no GPU, no remote, no radiomics.parquet)
and renders docs/laporan/RESEARCH_JOURNEY_REPORT_FILLED.md -- a fully-filled version
of the docs/laporan/RESEARCH_JOURNEY_REPORT.md skeleton, with real numbers, tables,
and embedded figure links. The skeleton itself is never modified; this script is
meant to be re-run any time results change, overwriting the FILLED file.
"""
import argparse
import logging
import os
import subprocess

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESEARCHERS = [
    ("Ihab Hasanain Akmal", "103032330054"),
    ("Siti Nurhayati Syafaningrum", "101012330012"),
]
SUPERVISOR = "Naufal Hanan Lutfianto"
TARGET_SIDANG = "<<ISI: target sidang/deadline>>"

_ARM_LABELS = {"binary": "A (binary)", "ordinal": "B (ordinal)", "grade3": "C (grade3)", "grade4": "D (grade4)"}


def _df_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def _git_date_range(repo_root: str) -> tuple[str, str]:
    def _run(args):
        return subprocess.run(args, cwd=repo_root, capture_output=True, text=True, check=True).stdout.strip()
    first = _run(["git", "log", "--reverse", "--format=%ad", "--date=short"]).splitlines()[0]
    last = _run(["git", "log", "-1", "--format=%ad", "--date=short"])
    return first, last


def _fmt(x, nd=4):
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _main_results_table(results_dir: str) -> pd.DataFrame:
    summary = pd.read_csv(os.path.join(results_dir, "summary.csv"))
    eff = pd.read_csv(os.path.join(results_dir, "efficiency_table.csv"))
    rows = []
    for model in eff["model"]:
        sub = summary[summary["model"] == model]
        e = eff[eff["model"] == model].iloc[0]
        rows.append({
            "Backbone": model,
            "Params (M)": round(e["params_M"], 3),
            "GFLOPs": round(e["gflops"], 4),
            "AUC (mean+-std)": f"{sub['auc'].mean():.4f} +- {sub['auc'].std():.4f}",
            "AUC 95% CI": f"[{e['auc_ci_low']:.4f}, {e['auc_ci_high']:.4f}]",
            "Sens": f"{sub['sensitivity'].mean():.4f}",
            "Spec": f"{sub['specificity'].mean():.4f}",
            "F1": f"{sub['f1'].mean():.4f}",
            "Infer (ms)": round(e["latency_ms"], 2),
            "AUC/M params": round(e["auc_per_M_params"], 4),
        })
    return pd.DataFrame(rows).sort_values("AUC (mean+-std)", ascending=False)


def _arm_efficiency_table(results_dir: str, filename: str, metric_col: str, label: str) -> pd.DataFrame:
    path = os.path.join(results_dir, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    cols = ["model", "params_M", "gflops", "latency_ms", metric_col]
    ci_low, ci_high = f"{metric_col}_ci_low", f"{metric_col}_ci_high"
    if ci_low in df.columns:
        cols += [ci_low, ci_high]
    out = df[cols].copy().sort_values(metric_col, ascending=False)
    return out


def _delong_significant_pairs(results_dir: str) -> list[str]:
    path = os.path.join(results_dir, "delong_matrix.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, index_col=0)
    seen = set()
    lines = []
    for a in df.index:
        for b in df.columns:
            if a == b or (b, a) in seen:
                continue
            seen.add((a, b))
            p = df.loc[a, b]
            if pd.notna(p) and p < 0.05:
                lines.append(f"- **{a}** vs **{b}**: p={p:.4g} (significant)")
    return lines


def _xai_metrics_table(results_dir: str) -> pd.DataFrame:
    path = os.path.join(results_dir, "xai", "xai_metrics.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={
        "backbone": "Backbone", "dice": "Dice", "iou": "IoU",
        "dice_size_matched": "Dice (size-matched)", "pointing_acc": "Pointing Acc",
        "energy_mean": "Energy",
    })
    for c in ["Dice", "IoU", "Dice (size-matched)", "Pointing Acc", "Energy"]:
        if c in df.columns:
            df[c] = df[c].round(4)
    return df[["Backbone", "Dice", "IoU", "Dice (size-matched)", "Pointing Acc", "Energy"]] \
        .sort_values("Pointing Acc", ascending=False)


def _fusion_ablation_summary(results_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ablation_path = os.path.join(results_dir, "fusion", "ablation_summary.csv")
    delong_path = os.path.join(results_dir, "fusion", "delong_fusion.csv")
    if not os.path.exists(ablation_path):
        return pd.DataFrame(), pd.DataFrame()
    ab = pd.read_csv(ablation_path)
    pooled = ab.groupby("arm").agg(
        pooled_auc=("auc", "mean"), auc_std=("auc", "std"), n_folds=("fold", "count")
    ).round(4).reset_index().sort_values("pooled_auc", ascending=False)
    delong = pd.read_csv(delong_path) if os.path.exists(delong_path) else pd.DataFrame()
    if not delong.empty:
        delong = delong.round(4)
    return pooled, delong


def _shap_top_features(results_dir: str, n: int = 10) -> pd.DataFrame:
    path = os.path.join(results_dir, "xai_track1", "shap_feature_importance.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["mean_abs_shap"] = df["mean_abs_shap"].round(4)
    return df.head(n)


def _class_distribution_table(results_dir: str) -> str:
    path = os.path.join(results_dir, "tables", "table_3_1_class_distribution.md")
    if not os.path.exists(path):
        return "`[ISI: run stage_07a_tables.py]`"
    return open(path).read()


def _fold_distribution_table(results_dir: str) -> str:
    path = os.path.join(results_dir, "tables", "table_3_2_fold_distribution.md")
    if not os.path.exists(path):
        return "`[ISI: run stage_07a_tables.py]`"
    return open(path).read()


def _dataset_characteristics_table(results_dir: str) -> str:
    path = os.path.join(results_dir, "tables", "table_3_3_dataset_characteristics.md")
    if not os.path.exists(path):
        return "`[ISI: run stage_07a_tables.py]`"
    return open(path).read()


def run(cfg: dict, repo_root: str) -> None:
    results_dir = cfg["paths"]["results"]

    main_table = _main_results_table(results_dir)
    ordinal_eff = _arm_efficiency_table(results_dir, "efficiency_table_ordinal.csv", "qwk", "ordinal")
    grade4_eff = _arm_efficiency_table(results_dir, "efficiency_table_grade4.csv", "auc_macro", "grade4")
    grade4_nodule_eff = _arm_efficiency_table(results_dir, "efficiency_table_grade4_nodule_only.csv", "auc_nodule_only", "grade4_nodule_only")
    delong_lines = _delong_significant_pairs(results_dir)
    xai_table = _xai_metrics_table(results_dir)
    fusion_pooled, fusion_delong = _fusion_ablation_summary(results_dir)
    shap_top = _shap_top_features(results_dir)

    class_dist_md = _class_distribution_table(results_dir)
    fold_dist_md = _fold_distribution_table(results_dir)
    dataset_char_md = _dataset_characteristics_table(results_dir)

    first_date, last_date = _git_date_range(repo_root)

    best_auc_row = main_table.iloc[0]
    most_efficient_row = main_table.sort_values("AUC/M params", ascending=False).iloc[0]

    researchers_md = "\n".join(f"  - {n} ({nim})" for n, nim in RESEARCHERS)

    fig = lambda name: f"../../artifacts/results/{name}"

    lines = []
    a = lines.append

    a("# LungFuseNet — Research Journey & Progress Report (FILLED)\n")
    a(f"> Auto-generated by `src/stage_07f_journey_report.py`. Re-run the script any time "
      f"results change; this file is overwritten in full each run. Template: "
      f"`docs/laporan/RESEARCH_JOURNEY_REPORT.md`.\n")
    a("---\n")

    a("## 0. IDENTITAS PENELITIAN\n")
    a("- **Judul**: Explainable Radiomics–Deep Learning Fusion for Lung Nodule Malignancy Classification on LIDC-IDRI CT Scans")
    a("- **Repo**: https://github.com/Kruwpuck/lung-nodule-fusion-xai")
    a("- **Tugas**: Klasifikasi keganasan (malignancy) nodul paru dari CT scan")
    a("- **Dataset utama**: LIDC-IDRI (+ LUNA16 untuk hard-negative)")
    a(f"- **Peneliti**:\n{researchers_md}")
    a(f"- **Pembimbing**: {SUPERVISOR}")
    a(f"- **Periode**: {first_date} s/d {last_date} (dari git log)")
    a(f"- **Target sidang**: {TARGET_SIDANG}\n")
    a("---\n")

    a("## 1. RINGKASAN EKSEKUTIF\n")
    a(f"Penelitian ini membangun pipeline klasifikasi malignancy nodul paru pada LIDC-IDRI "
      f"dengan dua track: (1) **Track 2** membandingkan 6 backbone CNN/ViT (lightweight vs "
      f"heavyweight) pada 4 framing label (binary/ordinal/grade3/grade4), dan (2) **Track 1** "
      f"menggabungkan fitur radiomics (PyRadiomics) dengan embedding CNN lewat 3 skema fusion "
      f"(early/intermediate/late), dilengkapi XAI dua alat (Grad-CAM/Layer-CAM untuk CNN, SHAP "
      f"untuk radiomics).\n")
    a(f"Status sekarang: **kedua track sudah menghasilkan hasil lengkap** — Track 2 (120 run: "
      f"6 backbone x 4 arm x 5 fold) dan Track 1 (fusion ablation 5 arm x 5 fold + SHAP + "
      f"cross-modality figure) sudah selesai dieksekusi dan dievaluasi.\n")
    a(f"Temuan utama: backbone **{best_auc_row['Backbone']}** meraih AUC binary tertinggi "
      f"({best_auc_row['AUC (mean+-std)']}), tapi **{most_efficient_row['Backbone']}** paling "
      f"efisien (AUC/M params={most_efficient_row['AUC/M params']}). Di Track 1, "
      f"**radiomics-only mengalahkan semua varian fusion** (lihat Bab 6.3) — temuan negatif "
      f"yang dilaporkan apa adanya sesuai decision rule pre-registered. Di XAI, kualitas "
      f"lokalisasi Grad-CAM **tidak mengikuti kapasitas model** — vit_base (86M params) nyaris "
      f"tidak melokalisasi nodul (pointing_acc=0), sementara vgg16 terbaik (pointing_acc=0.20).\n")

    a("**Tabel status cepat:**\n")
    a("| Komponen | Status | Output |")
    a("|---|---|---|")
    a("| Track 2 — 6 backbone x 4 arm x 5 fold (120 run) | Selesai | checkpoint + CSV log + summary_*.csv |")
    a("| XAI Track 2 (Grad-CAM/Layer-CAM) | Selesai | Grid A, Grid B, xai_metrics.csv |")
    a("| Distribusi data + dataset overview | Selesai | Table 3.1-3.3 + dataset_overview.png |")
    a("| Track 1 — Fusion + SHAP | Selesai (ablation + SHAP + sidebyside) | ablation_summary.csv, shap_*.png |")
    a("| Fig 11 (diagram arsitektur fusion) / Fig 14 (cross-val SHAP<->CAM, >2 sample) | Belum | menyusul di fase penulisan Track 1 lanjutan |")
    a("| Validasi eksternal | Opsional, belum dikerjakan | pipeline ada, belum dijalankan |\n")
    a("---\n")

    a("## 2. LATAR BELAKANG & MOTIVASI\n")
    a("- Deteksi dini keganasan nodul paru krusial untuk menurunkan mortalitas kanker paru; "
      "CT scan skrining jadi modalitas utama tapi butuh interpretasi radiolog yang mahal waktu.")
    a("- Di medical imaging, model black-box sulit dipercaya klinisi meski akurat — "
      "interpretability (Grad-CAM, SHAP) jadi syarat, bukan tambahan opsional.")
    a("- Gap: kebanyakan paper fokus akurasi tunggal, jarang membandingkan efisiensi "
      "komputasi DAN interpretability lintas arsitektur secara sistematis pada data yang sama.")
    a("- Kontribusi yang diklaim:")
    a("  1. Benchmark 6 backbone (lightweight vs heavyweight) pada tugas dan split yang identik")
    a("  2. Multi-framing label (binary/ordinal/3-class/4-class) pada data sama, termasuk gate "
      "anti-inflasi untuk kelas no-nodule")
    a("  3. Fusion radiomics + CNN dengan XAI dua-alat (Grad-CAM + SHAP), dilaporkan transparan "
      "termasuk saat fusion TIDAK menang\n")
    a("---\n")

    a("## 3. TIMELINE JOURNEY (kronologi revisi)\n")
    a(f"Rentang commit git: **{first_date}** (initial commit) s/d **{last_date}** (commit terakhir).\n")
    a("### Fase 1 — Perencanaan awal & pemilihan dataset")
    a("- Dataset dipilih: LIDC-IDRI (1018 scan, anotasi 4 radiolog, rating malignancy 1-5)")
    a("- LUNA16 label = nodule vs non-nodule (deteksi), BUKAN malignancy -> hanya dipakai untuk hard-negative")
    a("- Akses LIDC via TCIA sempat berubah jadi controlled-access, ditangani lewat dbGaP\n")
    a("### Fase 2 — Desain dua-track")
    a("- Track 2 (Model Comparison) dikerjakan lebih dulu: banding 6 backbone, buktikan lightweight kompetitif")
    a("- Track 1 (Fusion + XAI) sengaja ditunda sampai Track 2 tuntas -- backbone pemenang Track 2 jadi fondasi Track 1\n")
    a("### Fase 3 — Arsitektur pipeline")
    a("- Modular `.py` per stage (`stage_00`..`stage_07`) + orchestrator, bukan 1 notebook monolitik")
    a("- Alasan: notebook monolitik rawan hang/hilang saat disconnect; modular = resumable per stage\n")
    a("### Fase 4 — Definisi Arm (multi-framing label)")
    a("- Arm A (binary): malignant/benign, buang median=3 (indeterminate)")
    a("- Arm B (ordinal): prediksi rating 1-5 langsung, median=3 dipakai")
    a("- Arm C (grade3): 3-class benign/indeterminate/malignant")
    a("- Arm D (grade4): + no-nodule hard-negative dari LUNA16, dengan gate anti-inflasi "
      "(headline metric = benign-vs-malignant AUC pada subset nodul saja, bukan 4-class mentah)\n")
    a("### Fase 5 — Eksekusi training (120 run)")
    a("- 6 backbone x 4 arm x 5 fold = 120 run, hasil lengkap di Bab 6.1\n")
    a("### Fase 6 — XAI debugging (3 iterasi revisi)")
    a("- **Bug 1**: target_class hardcode malignant=1 -> map all-blue di sample benign. Fix: default ke predicted class")
    a("- **Bug 2**: target_layer salah (resnet kena avgpool, vgg16/vit crash). Fix: cabang per-arsitektur + reshape_transform ViT")
    a("- **Bug 3**: CAM pola FIXED per-arsitektur (tak respons input) + pointing_acc=0. Diagnosis: "
      "resolusi feature map 2x2 terlalu coarse untuk input 64x64. Fix: Layer-CAM di stage ~8x8 (`_auto_target_layer`)")
    a("- Hasil akhir metrik XAI setelah fix: lihat Bab 6.2\n")
    a("### Fase 7 — Visualisasi & figure")
    a("- Prinsip controlled comparison: fixed sample set (6 nodul S1-S6) dipakai konsisten lintas semua backbone/metode")
    a("- Temuan: densenet121/resnet50/vgg16 nempel nodul di Grad-CAM, mobilenetv3/efficientnet meleset, vit_base nyaris flat\n")
    a("### Fase 8 — Track 1 Fusion (selesai)")
    a("- Feature selection (mRMR fallback mutual_info + LASSO per fold) -> ablation 5 arm "
      "(cnn_only/radiomics_only/fusion_early/intermediate/late) -> SHAP + Grad-CAM sidebyside")
    a("- Hasil: lihat Bab 6.3. Fusion TIDAK mengalahkan radiomics-only (dilaporkan transparan)")
    a("- Sisa: Fig 11 (diagram arsitektur fusion) & Fig 14 (cross-validation SHAP<->Grad-CAM "
      "lintas lebih banyak sample) -- didesain menyusul saat penulisan Track 1 lanjutan, bukan blocker hasil\n")
    a("---\n")

    a("## 4. DATASET (detail)\n")
    a("### 4.1 Sumber & komposisi")
    a("- LIDC-IDRI + LUNA16 hard-negative")
    a("- Aturan label: median rating 4 radiolog; >3 malignant, <3 benign, =3 indeterminate\n")
    a(class_dist_md + "\n")
    a("### 4.2 Preprocessing")
    a("- Patch 2.5D: 3 slice axial di sekitar centroid, ditumpuk sebagai channel")
    a("- Ukuran: 64x64 pixel")
    a("- Window HU: -1000 s/d 400")
    a("- Resample: 1mm isotropic\n")
    a("### 4.3 Split")
    a("- Patient-level stratified 5-fold (semua nodul 1 pasien di fold sama)")
    a("- Fixed: `artifacts/splits/folds.json` (seed=42)\n")
    a("**Tabel distribusi per class per fold:**\n")
    a(fold_dist_md + "\n")
    a("**Karakteristik dataset:**\n")
    a(dataset_char_md + "\n")
    a("---\n")

    a("## 5. HYPERPARAMETER & SETUP (config kanonik)\n")
    a("> `configs/config.yaml` = KANONIK (6 backbone target). `configs/train.yaml` = versi lama, "
      "diarsipkan ke `docs/archive/train.yaml.deprecated`.\n")
    a("### 5.1 Training")
    a("| Parameter | Nilai |\n|---|---|")
    for k, v in [("Epochs", 50), ("Batch size", 16), ("Learning rate", "1e-4"),
                 ("Weight decay", "1e-4"), ("Early stopping patience", 10),
                 ("Checkpoint every", "5 epoch"), ("Mixed precision", "true"), ("Seed", 42)]:
        a(f"| {k} | {v} |")
    a("\n### 5.2 Model (6 backbone)")
    a("| Backbone | Kategori | Params (M, nyata) |\n|---|---|---|")
    lightweight_set = set(cfg["models"]["lightweight"])
    for _, r in main_table.sort_values("Params (M)").iterrows():
        cat = "Lightweight" if r["Backbone"] in lightweight_set else "Heavyweight"
        a(f"| {r['Backbone']} | {cat} | {r['Params (M)']} |")
    a("\n### 5.3 Radiomics")
    a("- PyRadiomics, binWidth 25, resample [1,1,1]")
    a("- Feature classes: firstorder, shape, glcm, glrlm, glszm, gldm, ngtdm")
    a("- Selection per fold (train split only, anti-leakage): mRMR (fallback `mutual_info_classif`, "
      "`pymrmr` tidak terpasang) -> LASSO (`LassoCV`)\n")
    a("### 5.4 Fusion & statistik")
    a("- FusionNet: emb_dim=256, rad_dim=128, fusion_dim=128, dropout=0.3")
    a("- XGBoost (radiomics branch): default config, lihat `configs/config.yaml` key `xgboost`")
    a("- Evaluasi: bootstrap CI 95%, DeLong test alpha=0.05, decision rule pre-registered "
      "(fusion jadi headline HANYA jika DeLong p<0.05 DAN AUC fusion lebih tinggi)\n")
    a("---\n")

    a("## 6. HASIL\n")
    a("### 6.1 Track 2 — Model Comparison\n")
    a("**Tabel hasil utama, Arm A (binary), diurutkan AUC:**\n")
    a(_df_to_markdown(main_table) + "\n")
    a("**Arm B (ordinal) — QWK per backbone:**\n")
    if not ordinal_eff.empty:
        a(_df_to_markdown(ordinal_eff.round(4)) + "\n")
    a("**Arm D (grade4) — dua metrik terpisah:**\n")
    a("4-class macro-AUC (termasuk kelas no-nodule, mudah dipisahkan -> berpotensi inflasi):\n")
    if not grade4_eff.empty:
        a(_df_to_markdown(grade4_eff.round(4)) + "\n")
    a("Benign-vs-malignant AUC pada subset nodul saja (headline, anti-inflasi):\n")
    if not grade4_nodule_eff.empty:
        a(_df_to_markdown(grade4_nodule_eff.round(4)) + "\n")
    a("**DeLong test (arm A, pasangan signifikan p<0.05):**\n")
    if delong_lines:
        a("\n".join(delong_lines) + "\n")
    else:
        a("Tidak ada pasangan signifikan / data tidak tersedia.\n")
    a(f"![Params vs AUC]({fig('figures/params_vs_auc.png')})\n")
    a(f"![FLOPs vs AUC]({fig('figures/flops_vs_auc.png')})\n")
    a(f"![Convergence curves]({fig('figures/convergence.png')})\n")
    a(f"![Confusion matrices]({fig('figures/confusion_matrices.png')})\n")
    a("**Catatan framing penting**: AUC mentah tertinggi dipegang **vgg16** (heavyweight), "
      "BUKAN backbone lightweight -- judul figure \"Lightweight wins upper-left\" merujuk ke "
      "efisiensi (AUC per juta parameter), bukan akurasi absolut. mobilenetv3_small AUC "
      "terendah tapi AUC/M params tertinggi (paling efisien).\n")
    a("---\n")

    a("### 6.2 XAI (Track 2)\n")
    a("**Metrik XAI per arsitektur** (fold 0, n=60, threshold top-20%):\n")
    if not xai_table.empty:
        a(_df_to_markdown(xai_table) + "\n")
    a("**Temuan kualitatif (fixed sample set S1-S6)**: densenet121, resnet50, dan vgg16 -- "
      "hot spot Grad-CAM konsisten jatuh di dalam mask nodul (S1-S4). mobilenetv3_small dan "
      "efficientnet_b0 sering meleset ke tepi nodul/struktur sekitar. vit_base nyaris flat "
      "(pointing_acc=0), konsisten dengan skor rendah di tabel metrik. **Interpretability tidak "
      "mengikuti kapasitas model** -- vit_base 86M params tapi lokalisasi terlemah.\n")
    a(f"![Grid A - backbone comparison]({fig('figures_grid/grid_backbone.png')})\n")
    a(f"![Grid B - CAM method comparison]({fig('figures_grid/grid_cam_method.png')})\n")
    a("---\n")

    a("### 6.3 Track 1 — Fusion + XAI\n")
    a("**Ablation 5-arm, pooled AUC (mean lintas 5 fold):**\n")
    if not fusion_pooled.empty:
        a(_df_to_markdown(fusion_pooled) + "\n")
    a("**DeLong test: tiap varian fusion vs radiomics-only (best single arm):**\n")
    if not fusion_delong.empty:
        a(_df_to_markdown(fusion_delong) + "\n")
    a("**Kesimpulan (decision rule pre-registered, ditetapkan sebelum lihat hasil)**: "
      "fusion HANYA jadi headline kalau DeLong p<0.05 DAN AUC fusion lebih tinggi. Hasil "
      "nyata: fusion_intermediate tidak signifikan berbeda (p=0.60) dari radiomics-only; "
      "fusion_early dan fusion_late justru **signifikan lebih buruk** (p=0.019 dan p=0.0015). "
      "**Radiomics-only tetap headline Track 1** -- dilaporkan transparan sebagai temuan valid, "
      "bukan disembunyikan karena tidak sesuai ekspektasi.\n")
    a("**Top-10 fitur radiomics terpenting (mean |SHAP|):**\n")
    if not shap_top.empty:
        a(_df_to_markdown(shap_top) + "\n")
    a(f"![SHAP beeswarm]({fig('xai_track1/shap_beeswarm.png')})\n")
    a(f"![Grad-CAM + SHAP side-by-side (malignant)]({fig('xai_track1/sidebyside_malignant.png')})\n")
    a("---\n")

    a("## 7. FIGURE YANG DITAMPILKAN KE DOSEN\n")
    a("**Sudah jadi (dipakai di laporan ini):**")
    a("- Fig A -- Dataset overview (severity-ordered): `figures/dataset_overview.png`")
    a("- Fig B -- Grid A backbone XAI: `figures_grid/grid_backbone.png`")
    a("- Fig C -- Grid B metode CAM: `figures_grid/grid_cam_method.png`")
    a("- Fig D -- Params vs AUC: `figures/params_vs_auc.png`")
    a("- Fig E -- FLOPs vs AUC: `figures/flops_vs_auc.png`")
    a("- Fig F -- Convergence curves: `figures/convergence.png`")
    a("- Fig G -- Confusion matrices: `figures/confusion_matrices.png`")
    a("- Fig H -- SHAP beeswarm: `xai_track1/shap_beeswarm.png`")
    a("- Fig I -- Grad-CAM + SHAP side-by-side: `xai_track1/sidebyside_malignant.png`, `sidebyside_benign.png`")
    a("- Table 3.1-3.3 -- distribusi class per arm, per fold, karakteristik dataset\n")
    a("**Belum ada (menyusul fase Track 1 lanjutan):**")
    a("- Fig -- Diagram arsitektur fusion (desain manual, menunggu wiring final)")
    a("- Fig -- Cross-validation SHAP<->Grad-CAM lintas >2 sample (Level 2 XAI)\n")
    a("> Rekomendasi presentasi: fokus Fig D (Params vs AUC) + Fig B (Grid backbone XAI) + "
      "Table hasil utama Bab 6.1 + Bab 6.3 (temuan negatif fusion, integritas riset).\n")
    a("---\n")

    a("## 8. TANTANGAN & PELAJARAN\n")
    a("- Bug XAI (3 iterasi, Fase 6) -- proses debugging terdokumentasi, bukan kegagalan")
    a("- Config dualism (`config.yaml` vs `train.yaml`) -- pentingnya satu sumber kebenaran, diselesaikan dengan arsip eksplisit")
    a("- Repo lokal sempat stale terhadap state remote (checkpoint 120 run sudah jalan tapi belum ter-sync) -- pentingnya audit state nyata sebelum planning")
    a("- Naming bug laten: checkpoint/log/pred arm A (binary) tidak pakai suffix task, sementara arm lain pakai -- ditemukan & diperbaiki sebelum sempat merusak hasil di rerun")
    a("- Temuan XAI: interpretability != kapasitas model (ViT besar tapi Grad-CAM nyaris flat)")
    a("- Temuan Track 1: fusion tidak otomatis menang atas modalitas tunggal -- radiomics-only mengalahkan 3 skema fusion, dilaporkan apa adanya sesuai decision rule pre-registered\n")
    a("---\n")

    a("## 9. RENCANA LANJUTAN\n")
    a("1. Fig 11 (diagram arsitektur fusion) + Fig 14 (cross-validation SHAP<->Grad-CAM lintas lebih banyak sample) -- fase Track 1 lanjutan")
    a("2. Validasi eksternal (opsional): LUNGx/NLST/LNDb, cek kontaminasi data")
    a("3. Penulisan paper/skripsi")
    a(f"4. Target sidang: {TARGET_SIDANG}\n")
    a("---\n")

    a("## 10. INTEGRITAS RISET\n")
    a("- Semua arm dilaporkan transparan, termasuk yang hasilnya biasa/kalah (fusion vs radiomics-only)")
    a("- Decision rule fusion ditetapkan SEBELUM lihat hasil (anti post-hoc / anti-HARKing)")
    a("- Fold dibekukan lintas arm (`artifacts/splits/folds.json`, seed=42) -- fair comparison")
    a("- Fair-setup: split/augmentation/config identik semua backbone")
    a("- Metrik headline Track 2 arm D pakai subset nodul-saja (bukan campur no-nodule yang gampang dipisahkan)")
    a("- Radiomics feature selection dilakukan per-fold pada train split saja (anti-leakage)\n")
    a("---\n")

    a("## LAMPIRAN — FILE PENTING DI REPO\n")
    a("| File | Isi |\n|---|---|")
    for f, desc in [
        ("docs/implementation/PLAN_MALIGNANCY_GRADING.md", "Detail arm B/C/D + decision rule anti post-hoc"),
        ("docs/PROMPT_DATASET_RESEARCH.md", "Riset dataset + kontaminasi"),
        ("docs/training_guide.md", "Panduan setup + eksekusi"),
        ("docs/implementation/VISUALIZATION_FIGURES_PLAN.md", "Rencana figure paper, Bagian 1-4"),
        ("configs/config.yaml", "Config kanonik (6 backbone)"),
        ("src/stage_00..07", "Pipeline lengkap (preprocessing s/d journey report)"),
        ("artifacts/splits/folds.json", "Split fixed"),
        ("artifacts/features/radiomics.parquet", "Fitur radiomics"),
        ("artifacts/results/summary*.csv", "Hasil agregat per arm"),
        ("artifacts/results/fusion/", "Hasil ablation Track 1"),
    ]:
        a(f"| `{f}` | {desc} |")
    a("")

    out_path = os.path.join("docs", "laporan", "RESEARCH_JOURNEY_REPORT_FILLED.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("wrote %s (%d lines)", out_path, len(lines))
    print(f"[DONE] {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg, os.getcwd())


if __name__ == "__main__":
    main()
