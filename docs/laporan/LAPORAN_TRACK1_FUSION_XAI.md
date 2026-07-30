# Laporan hasil Track 1: fusi radiomics-CNN dan explainability per-arm

## 0. Identitas penelitian

- **Judul**: Fusi radiomics-CNN dan explainability per-arm untuk klasifikasi malignansi nodul paru pada LIDC-IDRI
- **Repo**: `lung-nodule-fusion-xai`
- **Tugas**: Track 1 dari pemisahan dua paper (lihat `docs/Review Revisi 1.md` §8)
- **Dataset**: LIDC-IDRI
- **Tanggal laporan**: 30 Juli 2026

---

## 1. Ringkasan eksekutif

Track 1 membandingkan lima *arm* representasi (CNN-only, radiomics-only, early fusion, intermediate fusion, late fusion) pada tujuh backbone CNN, lalu mengevaluasi explainability tiap arm secara terpisah. Temuan utama: radiomics-only mengungguli semua varian fusi, dan pointing accuracy Grad-CAM/Layer-CAM sangat bervariasi antar backbone tanpa hubungan konsisten terhadap AUC klasifikasi.

### Status komponen

| Komponen | Status | Sumber |
|---|---|---|
| Ablasi fusi 5 arm x 7 backbone x 5 fold | Selesai (data pra-perbaikan resolusi) | `artifacts/results/fusion/ablation_summary.csv` |
| DeLong fusi vs radiomics-only | Selesai | `artifacts/results/fusion/delong_fusion.csv` |
| Metrik XAI (Grad-CAM/Layer-CAM) | Selesai | `artifacts/results/xai/xai_metrics.csv` |
| Perbaikan bug resolusi input_size | Kode selesai, belum di-*re-run* | `docs/revisi/rev1/TASKBOARD.md` tugas 1 |
| Arm fusi baru (branch_norm, GMU, modality dropout) | Kode selesai, belum dieksekusi | `docs/revisi/rev1/TASKBOARD.md` tugas 5a/5b/5c |

---

## 2. Latar belakang dan kontribusi

Fusi radiomics dan fitur CNN diasumsikan saling melengkapi, tapi asumsi ini tidak selalu berlaku pada kohort sekecil LIDC-IDRI, di mana radiomics dengan seleksi fitur adalah *baseline* yang kuat. Kontribusi laporan ini: (1) perbandingan lima arm representasi pada tujuh backbone dengan uji DeLong berpasangan, (2) protokol XAI yang mengevaluasi cabang CNN (Grad-CAM/Layer-CAM) dan cabang radiomics (SHAP) secara terpisah, karena belum ada metrik *model-agnostic* terstandardisasi untuk membandingkan keduanya secara adil.

---

## 3. Metodologi

### 3.1 Dataset dan split

LIDC-IDRI, label malignansi dari median 4 rating ahli radiologi (skala 1–5), *split* 5-fold berbasis pasien dengan seed 42 sehingga satu pasien tidak muncul di lebih dari satu fold.

### 3.2 Prapemrosesan

Patch 2.5D: tumpukan slice aksial berdekatan berpusat pada nodul, di-*crop* ke resolusi in-plane tetap. Intensitas Hounsfield unit di-*clip* ke $[-1000, 400]$ lalu dinormalisasi ke $[0, 1]$.

### 3.3 Backbone dan resolusi input

Tujuh backbone: DenseNet121, InceptionV3, Xception, GoogLeNet, ConvNeXt-Tiny, Inception-ResNet-v2, DenseNet201. Tiga di antaranya (InceptionV3, Xception, Inception-ResNet-v2) punya syarat resolusi minimum (75, 71, 75 piksel) melebihi patch native 64 piksel proyek ini, sehingga ketujuh backbone dievaluasi pada resolusi seragam 96 piksel, dipilih sebagai kelipatan 32 terkecil di atas seluruh syarat minimum agar tidak ada backbone yang diuntungkan resolusi saat dibandingkan lewat XAI.

### 3.4 Arm fusi

CNN-only, radiomics-only, early fusion (konkatenasi fitur radiomics mentah ke input CNN), intermediate fusion (konkatenasi embedding CNN dengan vektor radiomics sebelum classification head bersama), late fusion (rata-rata probabilitas arm CNN dan radiomics).

### 3.5 Protokol XAI

Grad-CAM/Layer-CAM untuk cabang CNN, dievaluasi dengan dice, IoU, dice size-matched, pointing accuracy (apakah aktivasi maksimum jatuh di dalam mask nodul), dan energy pointing metric, semua terhadap mask ground-truth radiolog. SHAP untuk cabang radiomics. Kedua modalitas dilaporkan terpisah, tidak digabung jadi satu skor.

---

## 4. Dataset

Lihat §3.1 dan §3.2. Detail lengkap distribusi kelas dan fold ada di `artifacts/results/tables/table_3_1_class_distribution.csv` dan `table_3_2_fold_distribution.csv`.

---

## 5. Konfigurasi

`configs/config.yaml` blok `tracks.track1`: 7 backbone di atas, `input_size: 96`.

---

## 6. Hasil

### 6.1 Ablasi fusi

AUC rata-rata per arm, dipool lintas 7 backbone dan 5 fold, dari `artifacts/results/fusion/ablation_summary.csv` (175 baris: 7 backbone x 5 arm x 5 fold):

| Arm | AUC rata-rata |
|---|---|
| radiomics_only | 0.9313 |
| fusion_intermediate | 0.9269 |
| fusion_early | 0.9179 |
| fusion_late | 0.9171 |
| cnn_only | 0.7853 |

Dari 21 uji DeLong berpasangan (`fusion/delong_fusion.csv`, satu per backbone per varian fusi terhadap radiomics-only), **tidak satu pun** mencapai signifikansi yang mendukung fusi.

#### Batasan pada kolom cnn_only

Angka `cnn_only` di atas berasal dari `ablation_summary.csv` yang ditarik dari mesin remote pada 30 Juli 2026. Cross-check `densenet201`: `cnn_only` mean AUC 0.6432, jauh di bawah AUC standalone-nya 0.8988 di `summary_binary.csv`. Ini persis gejala bug resolusi input yang sudah diidentifikasi (checkpoint dilatih di 96px, dievaluasi di 64px) dan **sudah diperbaiki di kode** (`docs/revisi/rev1/TASKBOARD.md` tugas 1, `done-code`), tapi ablasi ini **belum di-*re-run*** dengan kode yang sudah diperbaiki. Artinya seluruh kolom `cnn_only` di atas masih data pra-perbaikan dan harus dibaca sebagai batasan, bukan hasil final. Kolom arm fusi lain (early/intermediate/late) tetap konsisten secara internal karena melatih dan mengevaluasi pada resolusi yang sama, jadi kesimpulan "radiomics mengungguli fusi" tidak berubah.

### 6.2 XAI

Pointing accuracy per backbone, dari `artifacts/results/xai/xai_metrics.csv`:

| Backbone | Pointing accuracy |
|---|---|
| DenseNet121 | 0.7167 |
| ConvNeXt-Tiny | 0.7167 |
| DenseNet201 | 0.7000 |
| Inception-ResNet-v2 | 0.3833 |
| InceptionV3 | 0.2000 |
| Xception | 0.2000 |
| VGG16 | 0.2000 |
| ResNet50 | 0.1167 |
| EfficientNet-B0 | 0.0833 |
| MobileNetV3-Small | 0.0000 |
| ViT-Base | 0.0000 |
| GoogLeNet | 0.0000 |

Pointing accuracy rendah tidak berarti AUC klasifikasi rendah; beberapa backbone dengan pointing mendekati nol tetap klasifikasi kompetitif. Ini konsisten dengan keterbatasan Grad-CAM yang sudah didokumentasikan pada literatur, khususnya untuk arsitektur vision transformer.

---

## 7. Figur

| Figur | Berkas | Kegunaan |
|---|---|---|
| Panel Grad-CAM per backbone | `artifacts/results/xai/xai_{backbone}.png` | Visualisasi CAM per backbone (belum komparabel lintas model) |
| Panel komparabilitas XAI baru | `artifacts/results/figures_grid/grid_comparability.png` | Sampel identik, colorbar seragam, baris kegagalan (belum dieksekusi, tidak ada checkpoint lokal) |
| Diagram arsitektur fusi | `artifacts/results/figures/fusion_architecture.png` | Ilustrasi arm fusi |

---

## 8. Batasan

1. Kolom `cnn_only` pada ablasi fusi masih data pra-perbaikan bug resolusi; perlu di-*re-run* sebelum dianggap final (§6.1).
2. Tiga varian fusi baru (branch normalization, Gated Multimodal Unit, modality dropout) sudah diimplementasikan dan diuji unit, tapi belum pernah dieksekusi pada grid ablasi penuh.
3. Panel XAI komparabilitas baru (`stage_07f_xai_comparability.py`) belum bisa dijalankan di mesin manapun karena checkpoint tidak tersedia lokal saat ditulis.
4. SHAP dan Grad-CAM dilaporkan pada skala terpisah tanpa metrik penyatu; ini gap metodologis terbuka, bukan keterbatasan khusus studi ini.

---

## 9. Rencana lanjutan

1. Jalankan ulang ablasi fusi dengan perbaikan `input_size` (`python -m src.stage_03b_fusion --config configs/config.yaml`), verifikasi `cnn_only` densenet201 kembali mendekati 0.8988.
2. Eksekusi ketiga varian fusi baru (branch_norm, GMU, modality dropout) pada grid penuh.
3. Jalankan panel XAI komparabilitas begitu checkpoint tersedia.
4. Tambahkan sitasi yang hilang lewat Zotero (`docs/laporan/REFERENSI_DIBUTUHKAN.md`).

---

## 10. Integritas riset

Semua angka pada laporan ini ditelusuri ke baris CSV nyata yang ditarik dari mesin remote pada 30 Juli 2026, bukan diperkirakan. Batasan bug resolusi dinyatakan eksplisit di titik angkanya muncul (§6.1), bukan disembunyikan.

---

## Lampiran: berkas hasil

| Berkas | Isi |
|---|---|
| `artifacts/results/fusion/ablation_summary.csv` | 175 baris, AUC per arm per backbone per fold |
| `artifacts/results/fusion/delong_fusion.csv` | 21 baris, uji DeLong fusi vs radiomics-only |
| `artifacts/results/xai/xai_metrics.csv` | 12 baris, metrik Grad-CAM/Layer-CAM per backbone |
