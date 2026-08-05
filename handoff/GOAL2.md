# GOAL run02 (revisi) — Membangun Kasus Keunggulan Fusion
 
## Klaim utama paper
 
> **Late fusion mencapai performa klasifikasi yang secara statistik setara dengan radiomics-only, sekaligus menyediakan penjelasan spasial yang secara struktural tidak dapat dihasilkan radiomics. Pada kriteria gabungan performa + explainability, fusion adalah pilihan yang lebih unggul.**
 
Ini bukan klaim kompromi — ini klaim menang pada sumbu yang benar. Radiomics unggul hanya pada satu metrik tunggal (AUC), dan itu pun tidak signifikan. Fusion unggul pada kriteria yang relevan secara klinis: model yang dapat dijelaskan secara spasial.
 
**Angka pendukung yang sudah ada:**
 
| Arm | AUC | Grad-CAM | Pointing accuracy |
|---|---|---|---|
| `fusion_late` | **0.9333** | ✅ ada | terukur |
| `radiomics_only` | 0.9318 | ❌ **mustahil** | tidak terdefinisi |
| `cnn_only` | 0.9018 | ✅ ada | 0.70–0.72 |
 
DeLong `fusion_late` vs `radiomics_only`: **tidak berbeda signifikan** (p 0.204–0.902). Fusion nominal lebih tinggi.
 
---
 
## Slot isian
 
```
PROJECT_ROOT     = <path repo lung-nodule-fusion-xai>
GOAL             = Kuantifikasi keunggulan gabungan fusion_late (AUC setara radiomics, unggul XAI, unggul atas cnn_only)
DONE_CRITERIA    = 6 gate di bawah, semua wajib lolos dengan bukti tercetak
GPU_HOST         = PC remote ini
MAX_ITER         = 15
MAX_SUBAGENT     = 5
RUN_ID           = 2026-08-04-run02
RESERVED_ENV     = artifacts/results/_baseline_pre_rev2/, artifacts/splits/folds.json, .venv
```
 
---
 
## Tugas
 
### T-0. VERIFIKASI KRITIS — apakah keunggulan fusion_late nyata? (kerjakan PERTAMA)
 
**Ini menentukan apakah klaim utama bertahan.**
 
Audit sebelumnya mencatat `fusion_late` = rerata `cnn_prob` (bocor) + `rad_prob` (bersih), dan dinilai "separuh bocor". Nested CV **tidak menyentuhnya** karena perbaikan hanya berlaku di `_train_fusion_fold`, sementara `fusion_late` memakai probabilitas dari checkpoint Arm A yang dipilih di luar stage_03b.
 
Artinya: checkpoint Arm A dipilih berdasarkan AUC fold luar — keuntungan seleksi yang tidak dimiliki `radiomics_only`.
 
**Yang harus dijawab:** apakah keunggulan 0.9333 vs 0.9318 bertahan kalau komponen CNN-nya memakai seleksi checkpoint yang bersih (inner-val)?
 
- Kalau **bertahan** → klaim utama kokoh, lanjut T-1 sampai T-4.
- Kalau **hilang** → laporkan apa adanya, klaim menyesuaikan jadi "fusion setara/sedikit di bawah radiomics dalam AUC, tetap unggul dalam explainability". Argumen XAI tidak terpengaruh sama sekali.
Lebih baik ketahuan sekarang daripada saat sidang.
 
### T-1. DeLong fusion_late vs cnn_only
 
Pakai preds yang ada. 3 backbone. Menetapkan bahwa fusion mengungguli citra-sendirian secara statistik, bukan cuma rerata.
 
### T-2. XAI model fusion (GPU)
 
Layer-CAM untuk `fusion_late` pada 3 backbone, memakai **`fixed_display_samples.json` yang sama**. Hitung pointing accuracy, IoU, Dice, energy-based pointing game.
 
Bandingkan dengan `cnn_only` (DenseNet121 0.7167, ConvNeXt-Tiny 0.7167, DenseNet201 0.7000).
 
### T-3. SHAP cabang radiomics pada fusion
 
Beeswarm global, 3 backbone. Menunjukkan fusion memberi **dua jenis penjelasan sekaligus** — spasial dan fitur.
 
### T-4. Tabel keunggulan gabungan
 
Satu tabel ringkas untuk paper, membandingkan ketiga arm pada semua sumbu:
 
| Kriteria | cnn_only | radiomics_only | fusion_late |
|---|---|---|---|
| AUC | | | |
| DeLong vs fusion_late | | | — |
| Grad-CAM tersedia | ✅ | ❌ | ✅ |
| Pointing accuracy | | tidak terdefinisi | |
| SHAP fitur | ❌ | ✅ | ✅ |
| Penjelasan spasial + fitur | ❌ | ❌ | ✅ |
 
Baris terakhir adalah inti klaim: **hanya fusion yang memberi keduanya.**
 
---
 
## Kriteria selesai (6 gate)
 
**G-0. Verifikasi T-0 tuntas.** Hasilnya dilaporkan apa pun arahnya, dengan angka sebelum-sesudah.
 
**G-1. DeLong fusion_late vs cnn_only lengkap** — 3 backbone, cetak p-value dan berapa yang signifikan.
 
**G-2. Metrik XAI fusion lengkap** — `xai_metrics_fusion.csv`, 3 backbone × 4 metrik. Cetak daftar sample ID sebagai bukti sample identik dengan `fixed_display_samples.json`.
 
**G-3. Tabel perbandingan XAI fusion vs cnn_only tersaji** dengan selisih per backbone.
 
**G-4. SHAP beeswarm terhasilkan** — 3 figure, dengan provenance (run_id, commit sha).
 
**G-5. Tabel keunggulan gabungan (T-4) terisi lengkap**, nol sel kosong.
 
---
 
## Batasan
 
- Sample XAI **wajib** dari `fixed_display_samples.json`, tidak boleh dipilih ulang
- Backbone tetap: `convnext_tiny`, `densenet201`, `densenet121`
- Dilarang menyentuh `_baseline_pre_rev2/`, `folds.json`, `.venv`
- Dilarang mengubah ambang atau definisi metrik
- Dilarang `--force`, `reset --hard`, `rm -rf`, `git push`
## Larangan khusus
 
- **Dilarang mencari konfigurasi fusi baru.** Run ini mengukur dan memverifikasi, bukan mencari. Arm yang diuji sudah ditetapkan: `fusion_late`.
- Dilarang mengubah `fixed_display_samples.json`
- Dilarang menyimpulkan keunggulan XAI tanpa angka pendukung
- **Dilarang menyembunyikan hasil T-0 kalau tidak menguntungkan**
---
 
## Catatan framing untuk paper
 
Klaim ini kuat justru karena **tidak overclaim**. Menulis "fusion mengalahkan semua" ketika tabel menunjukkan radiomics nominal dekat akan langsung dibantah penguji. Menulis "fusion setara dalam AUC dan satu-satunya yang menyediakan penjelasan spasial" tidak bisa dibantah — karena keduanya terbukti dari data.
 
Posisi terkuat: **radiomics unggul pada satu metrik tunggal yang selisihnya tidak signifikan; fusion unggul pada kriteria gabungan yang relevan untuk penggunaan klinis.**