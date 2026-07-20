# LungFuseNet — Project Audit Checklist

Dokumen untuk mengecek **apa yang sudah terlaksana vs belum**, artifact/figure apa yang diperlukan, dan data apa yang harus siap. Dipakai berkala untuk audit status repo sebelum menulis paper/skripsi.

Cara pakai: centang `[x]` kalau sudah beres + tulis path artifact-nya. Biarkan `[ ]` kalau belum. Kolom **Status** diisi: ✅ selesai / 🟡 sebagian / ❌ belum / ⏸️ ditunda sengaja.

---

## RINGKASAN STATUS CEPAT

| Komponen | Status | Catatan |
|---|---|---|
| Track 2 — Arm A (binary) | 🟡 | Training ada, reporting perlu dicek |
| Track 2 — Arm B (ordinal) | 🟡 | Baru training, reporting belum |
| Track 2 — Arm C (grade3 3-class) | ❌ | Diputuskan dikerjakan, belum jalan |
| Track 2 — Arm D (grade4 + no-nodule) | 🟡 | Baru training, reporting belum |
| Track 2 — Efficiency benchmark | 🟡 | Modul ada, tabel final perlu dicek |
| XAI Track 2 — Grad-CAM fix | 🟡 | Bug struktural sedang dibereskan |
| Track 1 — Fusion | ❌ | Modul ada, belum di-wire/training |
| Track 1 — Ablation (3 arm) | ❌ | Belum jalan |
| Track 1 — XAI (Grad-CAM + SHAP) | ❌ | Belum jalan |
| Validasi eksternal (patologi) | ❌ | Belum, opsional |

> Update kolom ini tiap kali audit. Angka detail ada di bagian per-track di bawah.

---

## BAGIAN 0 — FONDASI DATA (prasyarat semua track)

### Data yang harus ada di disk
- [ ] LIDC-IDRI lengkap (1018 scan) — path: `__________`
- [ ] LUNA16 (annotations.csv, candidates_V2, subset0–9) — path: `__________`
- [ ] Verifikasi 888/888 seriesuid LUNA16 ↔ LIDC match — bukti: `__________`
- [ ] Patch hasil preprocessing (cache) — path: `artifacts/patches/`
- [ ] Consensus mask dari pylidc (untuk XAI + radiomics) — path: `__________`

### Label & split
- [ ] Label binary (Arm A): median>3 malignant, median<3 benign, median==3 dibuang (956 nodul) — file: `__________`
- [ ] Label ordinal (Arm B): median_rating 1–5 — file: `__________`
- [ ] Label grade3 (Arm C): benign/indeterminate/malignant — kolom: `grade3`
- [ ] Label grade4 (Arm D): no-nodule/benign/indeterminate/malignant — file: `__________`
- [ ] Patient-level stratified 5-fold split (FIXED, dipakai SEMUA arm) — file: `artifacts/splits/folds.json`
- [ ] **GATE: split identik lintas arm** — verifikasi tidak ada 1 pasien pun pindah fold antar arm

### Sanity check data (WAJIB sebelum XAI)
- [ ] Centering check: mask centroid ≤4px dari center patch — script + log: `__________`
- [ ] Buang patch dengan mask kosong / centroid melenceng — jumlah dibuang: `____`
- [ ] Cek axis-order (z/y/x pylidc vs SimpleITK) konsisten
- [ ] Cek slice tengah 2.5D = z-index centroid di volume yang di-crop

**Artifact fondasi yang diperlukan:**
- [ ] Tabel distribusi kelas per arm (berapa benign/malignant/indeterminate/no-nodule)
- [ ] Diagram flow preprocessing (DICOM → patch 2.5D 64×64×3)
- [ ] Tabel karakteristik dataset (jumlah pasien, nodul, slice thickness)

---

## BAGIAN 1 — TRACK 2: MODEL COMPARISON (Lightweight vs Heavyweight)

### 1.1 Kesiapan model (6 backbone)
Cek tiap backbone sudah terimplementasi + emb_dim benar:
- [ ] mobilenetv3_small (~2.5M) — emb_dim 576
- [ ] efficientnet_b0 (~5.3M)
- [ ] densenet121 (~8.0M)
- [ ] resnet50 (~25.6M)
- [ ] vgg16 (~138M) — emb_dim 512
- [ ] vit_base (~86M) — emb_dim 768

### 1.2 Fair setup (WAJIB dijaga)
- [ ] **SATU config aktif** — bukan dua (`config.yaml` vs `train.yaml`). Config aktif: `__________`
- [ ] Split sama semua model
- [ ] Augmentation sama semua model
- [ ] Pretraining sama (ImageNet, atau semua from-scratch — pilih satu)
- [ ] Optimizer, LR, scheduler, epoch sama
- [ ] Seed fixed (idealnya multi-seed 3–5 run untuk mean±std)

### 1.3 Status training per Arm × Backbone
Isi AUC (val, mean 5-fold) tiap sel. Kosong = belum training.

| Backbone | Arm A (binary) | Arm B (ordinal) | Arm C (grade3) | Arm D (grade4) |
|---|---|---|---|---|
| mobilenetv3_small | `____` | `____` | `____` | `____` |
| efficientnet_b0 | `____` | `____` | `____` | `____` |
| densenet121 | `____` | `____` | `____` | `____` |
| resnet50 | `____` | `____` | `____` | `____` |
| vgg16 | `____` | `____` | `____` | `____` |
| vit_base | `____` | `____` | `____` | `____` |

- [ ] Semua sel Arm A terisi
- [ ] Semua sel Arm B terisi
- [ ] Semua sel Arm C terisi (JIKA dikerjakan)
- [ ] Semua sel Arm D terisi

### 1.4 Metrik & CSV (harus tersimpan, tidak perlu training ulang)
- [ ] `artifacts/logs/{model}_{fold}.csv` — convergence per epoch (loss, acc, auc)
- [ ] `artifacts/logs/summary.csv` — 1 baris per model×fold (params, flops, auc, sens, spec, f1, infer_ms)
- [ ] Efficiency: params, FLOPs, inference time, memory per backbone — file: `__________`

### 1.5 Metrik khusus per Arm
- [ ] Arm A: AUC, acc, sensitivity, specificity, F1, balanced acc
- [ ] Arm B (ordinal): **QWK**, **MAE**, one-off accuracy (dalam-1-grade), macro-AUC
- [ ] Arm C (grade3): confusion matrix 3-kelas, macro-F1, per-class AUC
- [ ] Arm D (grade4): **DUA metrik terpisah** —
  - [ ] (1) metrik 4-kelas penuh
  - [ ] (2) benign-vs-malignant pada **subset nodul saja** (exclude no-nodule)
  - [ ] **GATE anti-inflasi**: headline perbandingan model pakai metrik (2), BUKAN (1)

### 1.6 Uji statistik
- [ ] DeLong test antar backbone (paired AUC)
- [ ] 95% CI via bootstrap
- [ ] Koreksi multiple-comparison (Bonferroni/Holm) ATAU deklarasi perbandingan utama = 1

**Figure/tabel Track 2 yang diperlukan untuk paper:**
- [ ] **Tabel utama**: model | params | FLOPs | AUC | sens | spec | F1 | infer_ms (per arm)
- [ ] **Scatter plot "Params vs AUC"** (sumbu-x log params) — bukti lightweight vs heavyweight
- [ ] **Scatter plot "FLOPs vs AUC"** — efisiensi komputasi
- [ ] **Grafik convergence** (train/val loss per epoch) — dari CSV
- [ ] **Confusion matrix** (Arm A minimal, Arm C/D kalau ada)
- [ ] **Tabel akurasi per juta parameter** — angka tunggal efisiensi
- [ ] **Tabel perbandingan semua arm** (A/B/C/D) — transparan, termasuk yang kalah
- [ ] Ordinal (Arm B): plot prediksi vs true rating (scatter/heatmap)

---

## BAGIAN 2 — XAI TRACK 2 (Grad-CAM per arsitektur)

### 2.1 Bug fix (cek sudah beres)
- [ ] target_class = predicted class (bukan hardcode malignant=1)
- [ ] target_layer benar per arsitektur (bukan kena avgpool)
- [ ] Cabang vgg16 (`features[-1]`) ada
- [ ] Cabang vit (`blocks[-1].norm1` + reshape_transform buang CLS) ada
- [ ] **ViT reshape grid benar**: 4×4 (16 token) untuk input 64×64, BUKAN hardcode 14×14

### 2.2 Diagnostik struktural (WAJIB sebelum percaya angka)
- [ ] D0: CAM berubah saat input berubah (bukan pola FIXED per-arsitektur)
- [ ] D1: raw activation bervariasi antar input (hook di modul benar)
- [ ] D2: gradient magnitude di target layer non-zero
- [ ] Cek ukuran feature map di target layer (curigaan: 2×2 terlalu coarse)

### 2.3 Fix resolusi
- [ ] Retarget ke stage 8×8 (output-stride 8) + Layer-CAM
- [ ] Patch tetap 64×64 (TIDAK diperbesar — jaga faithfulness)
- [ ] Re-run metrik: pointing_acc naik dari 0 ke angka wajar?

### 2.4 Metrik XAI kuantitatif (per arsitektur)
- [ ] IoU (thresholded CAM vs mask)
- [ ] Dice
- [ ] Pointing game (argmax CAM di dalam mask) — **metrik utama**
- [ ] Energy-based pointing game (rename dari `cam_in_nodule_fraction`, sitir Score-CAM)
- [ ] Catat threshold yang dipakai (mis. top-20%)
- [ ] **Catat di paper**: nodul <10% area → IoU punya ceiling rendah (~0.3), wajar

### 2.5 (Opsional, fase lanjutan) Banding metode CAM
- [ ] Grad-CAM vs Grad-CAM++ vs HiResCAM vs Layer-CAM
- [ ] +1 gradient-free (Ablation-CAM / Score-CAM) sebagai bukti saturasi
- [ ] Pilih terbaik by energy-in-mask + pointing_acc

**Figure XAI Track 2 yang diperlukan:**
- [ ] **Overlay Grad-CAM** (true-positive malignant, high-confidence) — merah di nodul + overlay mask
- [ ] Grid 6 backbone × beberapa sample — bukti per-arsitektur
- [ ] **Tabel metrik XAI per arsitektur** (IoU/Dice/pointing/energy)
- [ ] Panel failure case (jujur, dipisah)
- [ ] (Jika banding metode) tabel Grad-CAM vs metode lain

---

## BAGIAN 3 — TRACK 1: FUSION + XAI (belum dikerjakan, ditunda sengaja)

### 3.1 Radiomics pipeline
- [ ] ROI/mask siap (Route A: pylidc consensus / Route B: auto-seg)
- [ ] PyRadiomics extraction (IBSI-compliant YAML) — setting: binWidth 25, resample [1,1,1]
- [ ] Feature classes: firstorder, shape, glcm, glrlm, glszm, gldm, ngtdm
- [ ] Filter: wavelet + LoG
- [ ] Feature selection: ICC→mRMR→LASSO
- [ ] Fitur radiomics tersimpan (cache) — path: `artifacts/features/radiomics.csv`

### 3.2 Ablation 3 arm (INTI PEMBUKTIAN Track 1)
- [ ] **Arm 1 — CNN-only**: gambar saja
- [ ] **Arm 2 — Radiomics-only**: XGBoost/LightGBM pada fitur tabular
- [ ] **Arm 3 — Fusion**: gambar + radiomics
  - [ ] Early fusion (concat → XGBoost)
  - [ ] Intermediate/joint fusion (default) — CNN embedding + radiomic vector → dense
  - [ ] Late fusion (ensemble probabilitas)
- [ ] **GATE fair**: fold/split/preprocessing IDENTIK ketiga arm
- [ ] **Decision rule** (tetapkan SEBELUM lihat hasil): fusion headline HANYA jika signifikan (DeLong) atas modality terbaik
- [ ] Catat: kadang radiomics-only > CNN-only — itu temuan sah, bukan gagal

### 3.3 XAI Track 1 — DUA level
**Level 1 — per branch (wajib):**
- [ ] Grad-CAM/Layer-CAM untuk branch CNN (DI MANA model melihat)
- [ ] SHAP untuk branch radiomics (FITUR mana penting)
  - [ ] TreeSHAP pada classifier tabular
  - [ ] Beeswarm/summary plot (global)
  - [ ] Waterfall/force plot (local, per-nodul)

**Level 2 — cross-validation antar branch (nilai jual tinggi):**
- [ ] Cek fitur SHAP-tinggi (texture/shape) berkorelasi spasial dengan region Grad-CAM
- [ ] Deteksi spurious correlation (Grad-CAM di luar nodul + SHAP fitur non-robust)

**Level 3 — (opsional ambisius):**
- [ ] Grad-CAM fusion vs Grad-CAM CNN-only — apakah radiomics "memandu" perhatian CNN?

**Figure Track 1 yang diperlukan:**
- [ ] **Tabel ablation**: CNN-only vs radiomics-only vs fusion (3 tipe) — AUC + DeLong p-value
- [ ] SHAP beeswarm (fitur radiomics terpenting)
- [ ] Side-by-side Grad-CAM (gambar) + SHAP (radiomics) untuk nodul sama
- [ ] Diagram arsitektur fusion (CNN branch + radiomic branch → dense)
- [ ] (Level 2) figure cross-validation spasial SHAP↔Grad-CAM

---

## BAGIAN 4 — VALIDASI EKSTERNAL (opsional, memperkuat)

- [ ] Pilih kohort patologi-confirmed (LUNGx / NLST / LNDb)
- [ ] **CEK KONTAMINASI**: pastikan tidak ada overlap pasien dengan LIDC
  - [ ] LUNGx: cek overlap (sama-sama arsip U. Chicago) — via DICOM header + image hash
  - [ ] Verifikasi via PatientID / SeriesInstanceUID
- [ ] Report AUC drop internal→eksternal (estimasi generalisasi jujur)
- [ ] Catat: label LIDC = opini radiolog, bukan patologi (kelemahan utama)

---

## BAGIAN 5 — INTEGRITAS RISET (untuk paper/skripsi)

- [ ] **Laporkan SEMUA arm** (A/B/C/D), termasuk yang hasilnya biasa/gagal
- [ ] Decision rule ditetapkan SEBELUM lihat hasil (anti post-hoc / HARKing)
- [ ] Statement transparansi & multiple-comparison di metodologi
- [ ] Reproducibility: versi library, config YAML, seed, versi LIDC dicatat
- [ ] Sensitivity analysis aturan agregasi (median vs mean vs ≥3-of-4)
- [ ] Deklarasi: metrik headline Track 2 pakai nodul saja (bukan campur no-nodule)

---

## BAGIAN 6 — HYGIENE REPO

- [ ] Satu config aktif (hapus/arsipkan dualisme `config.yaml`/`train.yaml`)
- [ ] File log root (`_orchestrator_err.log`, `_test_run.log`) masuk `.gitignore`
- [ ] README update (LIDC bukan lagi "124GB download bebas" — controlled access)
- [ ] `artifacts/` di-gitignore (jangan commit checkpoint besar)
- [ ] Orchestrator jalan di lokal + Colab (2 versi)
- [ ] Error PermissionError Windows (file lock) — sudah diatasi?

---

## PRIORITAS EKSEKUSI (urutan disarankan)

1. **Fondasi + hygiene** — pastikan satu config, split fixed lintas arm, sanity check data.
2. **Tutup Track 2 yang sudah jalan** — reporting Arm B & D (figure + metrik), jaga gate anti-inflasi Arm D.
3. **Fix XAI Track 2** — rule out bug struktural → Layer-CAM 8×8 → metrik. (Banding metode = fase lanjutan.)
4. **Arm C (grade3)** — reuse infra, lengkapi spektrum grading.
5. **Baru buka Track 1** — radiomics pipeline → ablation 3 arm → XAI 2 level. (Potongan terbesar, sengaja terakhir.)
6. **Validasi eksternal** — kalau waktu memungkinkan, perkuat klaim.

> Prinsip: tutup dulu yang sudah jalan sampai menghasilkan output, baru buka pekerjaan besar baru. Jangan buka banyak front sekaligus.
