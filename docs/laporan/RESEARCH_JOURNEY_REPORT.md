# LungFuseNet — Research Journey & Progress Report

> **Kerangka dokumen untuk dosen.** Struktur ini berisi SEMUA yang harus ada dalam laporan journey penelitian. Bagian `[ISI: ...]` = placeholder untuk data yang ada di local/remote (belum di GitHub). Isi angka nyata dari repo local sebelum kirim ke dosen.

---

## 0. IDENTITAS PENELITIAN

- **Judul**: Explainable Radiomics–Deep Learning Fusion for Lung Nodule Malignancy Classification on LIDC-IDRI CT Scans
- **Repo**: https://github.com/Kruwpuck/lung-nodule-fusion-xai
- **Tugas**: Klasifikasi keganasan (malignancy) nodul paru dari CT scan
- **Dataset utama**: LIDC-IDRI (+ LUNA16 untuk hard-negative)
- **Peneliti**: `[ISI: nama, NIM]`
- **Pembimbing**: `[ISI]`
- **Periode**: `[ISI: tanggal mulai – sekarang]`

---

## 1. RINGKASAN EKSEKUTIF (untuk dosen, 1 halaman)

`[ISI singkat: 3-4 paragraf]`
- Apa masalah나ya (klasifikasi malignancy nodul + interpretability)
- Pendekatan (2 track: model comparison + fusion XAI)
- Status sekarang (Track 2 + XAI selesai, Track 1 fusion sedang dikerjakan)
- Temuan utama sejauh ini (angka AUC terbaik, temuan XAI antar-arsitektur)

**Tabel status cepat:**

| Komponen | Status | Output |
|---|---|---|
| Track 2 — 6 backbone × 4 arm × 5 fold (120 run) | ✅ Selesai | checkpoint + CSV log |
| XAI Track 2 (Grad-CAM/Layer-CAM) | ✅ Selesai | Grid A, Grid B, metrik |
| Distribusi data + dataset overview | ✅ Selesai | tabel + figure |
| Track 1 — Fusion + SHAP | 🟡 Belum wired | kode ada, belum training |
| Validasi eksternal | ⏸️ Opsional | pipeline ada |

---

## 2. LATAR BELAKANG & MOTIVASI

`[ISI:]`
- Kenapa klasifikasi malignancy nodul paru penting (deteksi dini kanker paru)
- Kenapa interpretability penting di medical imaging (trust klinisi, bukan black-box)
- Gap: kebanyakan paper fokus akurasi, kurang bandingkan efisiensi model + interpretability antar-arsitektur
- Kontribusi yang diklaim:
  1. Benchmark 6 backbone (lightweight vs heavyweight) di tugas yang sama
  2. Multi-framing label (binary/ordinal/3-class/4-class) pada data sama
  3. Fusion radiomics + CNN dengan XAI dua-alat (Grad-CAM + SHAP)

---

## 3. TIMELINE JOURNEY (kronologi revisi)

Bagian inti untuk dosen — tunjukkan evolusi pemikiran. Tiap fase: apa yang dilakukan, apa yang berubah, kenapa.

### Fase 1 — Perencanaan awal & pemilihan dataset
- **Rencana awal**: `[ISI: rencana pertama, mungkin lebih sederhana]`
- **Keputusan dataset**: LIDC-IDRI dipilih (1018 scan, anotasi 4 radiolog, rating malignancy 1–5)
- **Temuan penting**: LUNA16 label = nodule vs non-nodule (deteksi), BUKAN malignancy → hanya dipakai untuk hard-negative
- **Temuan akses**: LIDC via TCIA berubah jadi controlled-access (Juli 2025) — butuh dbGaP
- **Revisi**: `[ISI: apa yang berubah dari rencana awal]`

### Fase 2 — Desain dua-track
- **Track 2** (Model Comparison): banding 6 backbone, buktikan lightweight bisa kompetitif
- **Track 1** (Fusion + XAI): radiomics + CNN, sengaja DITUNDA sampai Track 2 tuntas
- **Alasan urutan**: tutup dulu yang bisa dieksekusi, backbone pemenang Track 2 jadi fondasi Track 1

### Fase 3 — Arsitektur pipeline
- **Keputusan**: modular `.py` per stage (stage_00 → stage_06) + orchestrator, BUKAN 1 notebook monolitik
- **Alasan**: notebook monolitik hang/hilang saat disconnect; modular = resumable
- **Dua versi**: local (`run_local.sh`) + Colab (`run_colab.ipynb`)

### Fase 4 — Definisi Arm (multi-framing label)
Data source sama, target label beda:
- **Arm A** (binary): malignant/benign, buang median=3 (956 nodul indeterminate)
- **Arm B** (ordinal): prediksi rating 1–5 langsung, median=3 DIPAKAI
- **Arm C** (grade3): 3-class benign/indeterminate/malignant
- **Arm D** (grade4): + no-nodule hard-negative dari LUNA16
- **Revisi penting**: Arm C awalnya placeholder tak jelas → didefinisikan ulang jadi grade3 3-class

### Fase 5 — Eksekusi training (120 run)
- 6 backbone × 4 arm × 5 fold = 120 run
- **Hasil**: `[ISI: AUC per backbone per arm dari summary.csv]`

### Fase 6 — XAI debugging (3 iterasi revisi)
Bagian dengan revisi paling banyak — tunjukkan proses debugging ke dosen:
- **Bug 1**: target_class hardcode malignant=1 → map all-blue di sample benign. **Fix**: default ke predicted class
- **Bug 2**: target_layer salah (resnet kena avgpool, vgg16/vit crash). **Fix**: cabang per-arsitektur + reshape_transform ViT
- **Bug 3**: CAM pola FIXED per-arsitektur (tak respons input) + pointing_acc=0. **Diagnosis**: bug struktural + resolusi 2×2 terlalu coarse untuk input 64×64. **Fix**: Layer-CAM @ stage 8×8
- **Hasil akhir**: `[ISI: metrik XAI setelah fix — pointing_acc, IoU, dll]`

### Fase 7 — Visualisasi & figure
- Prinsip controlled comparison: fixed sample set (6 nodul S1–S6) lintas semua backbone/metode
- **Temuan**: `[ISI: densenet/resnet/vgg nempel nodul, mobilenet/efficientnet meleset, vit flat]`

### Fase 8 (sekarang) — Track 1 Fusion
- Status: kode fusion + radiomics ada, belum di-wire ke training
- **Next**: feature selection → ablation 3 arm (CNN-only/radiomics-only/fusion) → XAI 2 level

---

## 4. DATASET (detail)

### 4.1 Sumber & komposisi
- LIDC-IDRI: `[ISI: jumlah scan, nodul total, pasien]`
- Aturan label: median rating 4 radiolog; >3 malignant, <3 benign, =3 indeterminate
- `[ISI: jumlah benign/malignant/indeterminate nyata dari data]`
- No-nodule: LUNA16 candidates class-0 (hard-negative)

### 4.2 Preprocessing
- Patch 2.5D: 3 slice axial di sekitar centroid, ditumpuk sebagai channel
- Ukuran: 64×64 pixel
- Window HU: −1000 s/d 400
- Resample: 1mm isotropic
- Verifikasi: 888/888 seriesuid LUNA16 ↔ LIDC match

### 4.3 Split
- Patient-level stratified 5-fold (semua nodul 1 pasien di fold sama)
- Fixed: `artifacts/splits/folds.json` (seed=42)
- **Tabel distribusi per class per fold**: `[ISI dari Table 3.2]`

---

## 5. HYPERPARAMETER & SETUP (config kanonik)

> **Catatan**: repo punya 2 config. `config.yaml` = KANONIK (6 backbone target). `train.yaml` = versi lama (9 backbone, deprecated). Pastikan lapor yang benar.

### 5.1 Training (dari config.yaml)
| Parameter | Nilai |
|---|---|
| Epochs | 50 |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Early stopping patience | 10 |
| Checkpoint every | 5 epoch |
| Mixed precision | true |
| Scheduler | cosine |
| Seed | 42 |

### 5.2 Augmentation
Horizontal/vertical flip (0.5), rotation (±15°), brightness jitter (0.1), zoom (0.9–1.1)

### 5.3 Model (6 backbone target)
| Backbone | Kategori | Params (approx) |
|---|---|---|
| mobilenetv3_small | Lightweight | ~2.5M |
| efficientnet_b0 | Lightweight | ~5.3M |
| densenet121 | Lightweight | ~8.0M |
| resnet50 | Heavyweight | ~25.6M |
| vgg16 | Heavyweight | ~138M |
| vit_base | Heavyweight | ~86M |

`[ISI: params/FLOPs NYATA dari summary.csv — angka di atas approx]`

### 5.4 Radiomics (config radiomics_params.yaml)
- PyRadiomics, binWidth 25, resample [1,1,1]
- Feature classes: firstorder, shape, glcm, glrlm, glszm, gldm, ngtdm
- Selection: ICC → mRMR → LASSO

### 5.5 Fusion & statistik
- FusionNet: embedding_dim 256, radiomic_hidden 128, dropout 0.3
- XGBoost: n_estimators 400, max_depth 4, lr 0.05
- Evaluasi: bootstrap 2000 iter, 95% CI, DeLong α=0.05

---

## 6. HASIL (isi dari local)

### 6.1 Track 2 — Model Comparison
`[ISI dari summary.csv]`

**Tabel hasil utama** (per arm):
| Backbone | Params | FLOPs | AUC | Sens | Spec | F1 | Infer(ms) |
|---|---|---|---|---|---|---|---|
| mobilenetv3_small | | | | | | | |
| efficientnet_b0 | | | | | | | |
| densenet121 | | | | | | | |
| resnet50 | | | | | | | |
| vgg16 | | | | | | | |
| vit_base | | | | | | | |

**Perbandingan antar-arm** (A/B/C/D): `[ISI]`
**Metrik ordinal Arm B**: QWK, MAE, one-off acc `[ISI]`
**Arm D — dua metrik terpisah**: (1) 4-kelas, (2) benign-vs-malignant nodul-saja `[ISI]`
**DeLong test**: `[ISI signifikansi antar-backbone]`

### 6.2 XAI
`[ISI]`
**Metrik XAI per arsitektur** (IoU/Dice/pointing/energy):
| Backbone | IoU | Dice | Pointing | Energy |
|---|---|---|---|---|
| ... | | | | |

**Temuan kualitatif**: `[ISI: mana yang nempel nodul, mana yang meleset/flat]`

### 6.3 Track 1 — Fusion
`[BELUM: sedang dikerjakan]`

---

## 7. FIGURE YANG MAU DITAMPILKAN KE DOSEN

Prioritas figure untuk laporan (yang sudah jadi ✅, yang belum ⏳):

**Sudah jadi (ada di local):**
- ✅ **Fig A — Dataset overview**: 12 patch per class (benign/malignant/indeterminate/no-nodule) `[file: dataset_overview.png]`
- ✅ **Fig B — Grid A (backbone XAI)**: 6 backbone × 6 sample, Layer-CAM, fixed samples `[file: grid_backbone.png]`
- ✅ **Fig C — Grid B (metode CAM)**: 5 metode × 6 sample di densenet121 `[file: grid_cam_method.png]`
- ✅ **Table 3.1–3.3** — distribusi class per arm, per fold, karakteristik dataset

**Perlu digenerate dari summary.csv (Track 2 closeout):**
- ⏳ **Fig D — Params vs AUC scatter** (bukti lightweight vs heavyweight) ← FIGURE KUNCI
- ⏳ **Fig E — FLOPs vs AUC scatter**
- ⏳ **Fig F — Convergence curves** (dari CSV log)
- ⏳ **Fig G — Confusion matrix** per arm
- ⏳ **Table hasil utama** (per backbone) ← TABLE KUNCI

**Belum (Track 1, nanti):**
- ⏳ Fig — Diagram arsitektur fusion
- ⏳ Fig — SHAP beeswarm
- ⏳ Fig — Side-by-side Grad-CAM + SHAP
- ⏳ Table ablation fusion

> **Rekomendasi untuk presentasi ke dosen**: fokus Fig D (Params vs AUC) + Fig B (Grid backbone XAI) + Table hasil utama. Tiga ini paling kuat menunjukkan kontribusi.

---

## 8. TANTANGAN & PELAJARAN (untuk diskusi dengan dosen)

`[ISI:]`
- Bug XAI (3 iterasi) — proses debugging, bukan kegagalan
- Config dualism — pentingnya satu sumber kebenaran
- Repo stale local vs remote — pentingnya sync
- Temuan: interpretability ≠ kapasitas model (ViT besar tapi CAM flat)
- `[ISI: tantangan lain yang kamu alami]`

---

## 9. RENCANA LANJUTAN

1. Track 2 closeout: generate summary.csv + figure kunci (Fig D/E/F/G)
2. Track 1 fusion: feature selection → ablation 3 arm → XAI 2 level
3. Validasi eksternal (opsional): LUNGx/NLST/LNDb, cek kontaminasi
4. Penulisan paper/skripsi
5. `[ISI: target sidang/deadline]`

---

## 10. INTEGRITAS RISET (poin untuk meyakinkan dosen)

- Semua arm dilaporkan transparan (termasuk yang hasilnya biasa/kalah)
- Decision rule ditetapkan SEBELUM lihat hasil (anti post-hoc)
- Fold dibekukan lintas arm (fair comparison)
- Fair-setup: split/augmentation/config sama semua backbone
- Metrik headline Track 2 pakai nodul-saja (bukan campur no-nodule yang gampang)

---

## LAMPIRAN — FILE PENTING DI REPO

| File | Isi |
|---|---|
| `docs/PLAN_MALIGNANCY_GRADING.md` | Detail arm B/C/D + decision rule anti post-hoc |
| `docs/PROMPT_DATASET_RESEARCH.md` | Riset dataset + kontaminasi |
| `docs/training_guide.md` | Panduan setup + eksekusi |
| `TRAINING_PIPELINE_PLAN.md` | Arsitektur pipeline |
| `configs/config.yaml` | Config kanonik (6 backbone) |
| `src/stage_00..06` | Pipeline lengkap |
| `artifacts/splits/folds.json` | Split fixed (local) |
| `artifacts/features/radiomics.parquet` | Fitur radiomics (local) |
| `summary.csv` | Hasil agregat (perlu digenerate) |

---

### CATATAN PENGISIAN (hapus sebelum kirim ke dosen)
Yang harus diisi dari local/remote sebelum kirim:
1. Semua `[ISI: ...]` — angka nyata dari summary.csv, folds.json, metrik XAI
2. Embed 3 figure yang sudah jada (dataset_overview, grid_backbone, grid_cam_method)
3. Generate + embed Fig D (Params vs AUC) — paling penting
4. Isi timeline dengan tanggal nyata
5. Cek angka params/FLOPs dari efficiency benchmark (jangan pakai approx)
