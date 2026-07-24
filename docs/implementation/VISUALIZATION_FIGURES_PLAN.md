# LungFuseNet — Visualization & Paper Figures Plan

Planning untuk (1) **konsistensi sample XAI** lintas backbone/metode, (2) **sample representatif per class**, (3) **tabel distribusi data**, dan (4) **rekomendasi figure lengkap untuk paper**. Ini prinsip publikasi + fair-comparison, bukan riset baru.

---

## PRINSIP INTI: CONTROLLED COMPARISON

Aturan tunggal yang mengikat semua visualisasi XAI:

> **Sample yang sama, divisualisasikan lintas semua kondisi yang dibandingkan.**

Kalau backbone A pakai nodul #1 dan backbone B pakai nodul #2, perbedaan heatmap bisa berasal dari **nodulnya**, bukan dari **backbone/metodenya**. Untuk klaim "XAI X lebih bagus dari Y", variabel nodul HARUS dikontrol (dibuat identik). Ini analog dengan fair-setup di Track 2 (split/augmentation sama) — di XAI, yang disamakan adalah **himpunan sample yang ditampilkan**.

Konsekuensi konkret:
- Pilih **satu himpunan sample tetap** (mis. 5–8 nodul), simpan ID-nya (`patient_id + nodule_id + fold`).
- SEMUA figure XAI (6 backbone × N metode CAM) pakai himpunan ID yang sama itu.
- Simpan daftar ID ini sebagai artifact: `artifacts/xai/fixed_display_samples.json` — reproducible, tidak berubah antar-run.

---

## BAGIAN 1 — PEMILIHAN SAMPLE XAI (fixed set)

### 1.1 Kriteria pemilihan
Sample yang ditampilkan di figure XAI harus:
- [ ] **True-positive malignant** (GT=malignant ∧ pred=malignant) untuk figure "model melihat lesi"
- [ ] **High-confidence** (predicted prob tinggi) — CAM lebih well-formed
- [ ] **Lolos centering check** (mask centroid ≤4px dari center, mask tidak kosong)
- [ ] **Bervariasi ukuran/lokasi nodul** — biar tidak semua kasus gampang
- [ ] Konsisten di fold yang sama (biar checkpoint yang dipakai jelas)

### 1.2 Struktur himpunan sample
Rekomendasi: pilih sample yang mencakup beberapa skenario, tapi **himpunannya tetap** untuk semua backbone/metode.

| Sample slot | Kriteria | Fungsi di figure |
|---|---|---|
| S1–S3 | TP malignant, high-conf, ukuran nodul beda | Figure utama "model melihat lesi" |
| S4 | TP malignant, nodul kecil (<5% area) | Uji batas resolusi CAM |
| S5 | True benign (opsional, panel terpisah) | Kontras: model TIDAK menyala di non-malignant |
| S6 | Failure case (FP/FN) | Panel kejujuran (dipisah, dilabeli) |

- [ ] Simpan ID fixed set → `artifacts/xai/fixed_display_samples.json`
- [ ] Verifikasi semua ID lolos centering check SEBELUM dipakai

### 1.3 Matriks visualisasi (yang harus dihasilkan)
Dengan fixed sample set, hasilkan grid:

**Grid A — perbandingan BACKBONE (metode CAM sama, mis. Layer-CAM):**
```
             S1    S2    S3    S4
mobilenetv3  [cam] [cam] [cam] [cam]
efficientnet [cam] [cam] [cam] [cam]
densenet121  [cam] [cam] [cam] [cam]
resnet50     [cam] [cam] [cam] [cam]
vgg16        [cam] [cam] [cam] [cam]
vit_base     [cam] [cam] [cam] [cam]
```
→ Baca per kolom: nodul sama, backbone beda. Bisa bandingkan mana backbone yang CAM-nya paling nempel di lesi.

**Grid B — perbandingan METODE CAM (backbone terbaik, mis. densenet121):**
```
              S1    S2    S3    S4
Grad-CAM      [cam] [cam] [cam] [cam]
Grad-CAM++    [cam] [cam] [cam] [cam]
HiResCAM      [cam] [cam] [cam] [cam]
Layer-CAM     [cam] [cam] [cam] [cam]
Ablation-CAM  [cam] [cam] [cam] [cam]
```
→ Baca per kolom: nodul sama, metode beda. Bandingkan mana metode CAM terbaik.

- [ ] Setiap sel: overlay CAM + **outline mask konsensus** (biar pembaca lihat apakah merah jatuh di dalam mask)
- [ ] Colorbar konsisten (skala sama semua sel — jangan per-image auto-scale yang bikin misleading)
- [ ] Baris paling atas: gambar CT asli + mask (referensi)

---

## BAGIAN 2 — SAMPLE REPRESENTATIF PER CLASS

Pembaca perlu lihat "class ini kelihatan seperti apa". Untuk tiap arm, tampilkan contoh gambar.

### 2.1 Untuk Arm A (binary) & Arm C/D (multi-class)
- [ ] **Benign**: 2–3 contoh nodul (CT patch + mask outline)
- [ ] **Malignant**: 2–3 contoh
- [ ] **Indeterminate** (Arm C/D): 2–3 contoh — tunjukkan kenapa ambigu
- [ ] **No-nodule** (Arm D): 2–3 contoh parenchyma normal / hard-negative

### 2.2 Format
```
Benign:        [patch] [patch] [patch]
Malignant:     [patch] [patch] [patch]
Indeterminate: [patch] [patch] [patch]
No-nodule:     [patch] [patch] [patch]
```
- [ ] Tiap patch: slice tengah 2.5D, window HU konsisten (-1000..400)
- [ ] Overlay mask outline (kecuali no-nodule)
- [ ] Caption: malignancy rating asli (mis. "median=1", "median=5", "median=3")
- [ ] Pilih contoh yang jelas/tipikal, bukan outlier

> Ini figure "dataset overview" — biasanya diletakkan di awal section Data/Method.

---

## BAGIAN 3 — TABEL DISTRIBUSI DATA

### 3.1 Distribusi per class (WAJIB)
- [ ] Tabel jumlah sample per class, per arm:

| Class | Arm A | Arm B | Arm C | Arm D |
|---|---|---|---|---|
| Benign (median<3) | `____` | — | `____` | `____` |
| Malignant (median>3) | `____` | — | `____` | `____` |
| Indeterminate (median=3) | dibuang | included | `____` | `____` |
| No-nodule | — | — | — | `____` |
| Rating 1 / 2 / 3 / 4 / 5 | — | `_/_/_/_/_` | — | — |
| **Total** | `____` | `____` | `____` | `____` |

### 3.2 Distribusi per fold (untuk verifikasi stratifikasi)
- [ ] Tabel: jumlah tiap class per fold (0–4) — buktikan stratified split seimbang
- [ ] Cek: rasio class konsisten lintas fold (bukti stratifikasi jalan)

### 3.3 Karakteristik dataset (deskriptif)
- [ ] Jumlah pasien, jumlah scan, jumlah nodul unik
- [ ] Slice thickness (min/max/median)
- [ ] Ukuran nodul (diameter median, range)
- [ ] Aturan agregasi label (median 4 radiolog) + aturan exclude (median=3 untuk Arm A)
- [ ] Sumber no-nodule (LUNA16 candidates class-0 / LIDC non-nodule)

---

## BAGIAN 4 — REKOMENDASI FIGURE LENGKAP UNTUK PAPER

Urutan sesuai alur paper standar (medical imaging DL).

### Section: Dataset & Method
- [ ] **Fig 1 — Dataset overview**: contoh patch per class (Bagian 2) + tabel distribusi (Bagian 3.1)
- [ ] **Fig 2 — Preprocessing pipeline**: diagram DICOM → crop 2.5D → resample 1mm → patch 64×64×3
- [ ] **Fig 3 — Arsitektur**: diagram 6 backbone (atau diagram fusion untuk Track 1)
- [ ] **Table 1 — Karakteristik dataset** (Bagian 3.3)
- [ ] **Table 2 — Distribusi per fold** (Bagian 3.2)

### Section: Results — Track 2 (Model Comparison)
- [ ] **Fig 4 — Params vs AUC scatter** (bukti lightweight vs heavyweight) ← figure kunci
- [ ] **Fig 5 — FLOPs vs AUC scatter** (efisiensi komputasi)
- [ ] **Fig 6 — Convergence curves** (train/val loss per epoch, dari CSV)
- [ ] **Fig 7 — Confusion matrix** (per arm, atau backbone terbaik)
- [ ] **Table 3 — Hasil utama per backbone** (params, FLOPs, AUC, sens, spec, F1, infer_ms) ← table kunci
- [ ] **Table 4 — Perbandingan semua arm** (A/B/C/D, transparan termasuk yang kalah)
- [ ] **Table 5 — Metrik ordinal Arm B** (QWK, MAE, one-off acc)
- [ ] **Table 6 — DeLong test** (signifikansi antar backbone)

### Section: Results — XAI
- [ ] **Fig 8 — Grid backbone (fixed samples)** (Grid A, Bagian 1.3) ← controlled comparison
- [ ] **Fig 9 — Grid metode CAM (fixed samples)** (Grid B, Bagian 1.3)
- [ ] **Fig 10 — Failure cases** (panel kejujuran, dipisah)
- [ ] **Table 7 — Metrik XAI per arsitektur** (IoU/Dice/pointing/energy) + catatan ceiling IoU

### Section: Results — Track 1 (Fusion, kalau sudah)
- [ ] **Fig 11 — Diagram arsitektur fusion** (CNN branch + radiomic branch)
- [ ] **Fig 12 — SHAP beeswarm** (fitur radiomics terpenting)
- [ ] **Fig 13 — Side-by-side Grad-CAM + SHAP** (nodul sama, dua penjelasan)
- [ ] **Fig 14 — Cross-validation SHAP↔Grad-CAM** (Level 2, nilai jual tinggi)
- [ ] **Table 8 — Ablation** (CNN-only vs radiomics-only vs fusion + DeLong p-value) ← table kunci Track 1

### Supplementary (opsional)
- [ ] Distribusi confidence prediksi
- [ ] Calibration curve
- [ ] Per-fold breakdown lengkap
- [ ] Grid XAI untuk semua sample (bukan cuma yang di main figure)

---

## PRIORITAS FIGURE (mana yang paling penting)

**Tier 1 — wajib ada (inti klaim paper):**
- Fig 4 (Params vs AUC) — bukti utama Track 2
- Table 3 (hasil per backbone) — angka utama
- Fig 8 (grid backbone XAI, fixed samples) — bukti interpretability
- Table 8 (ablation fusion) — bukti Track 1 (kalau ada)

**Tier 2 — memperkuat:**
- Fig 1 (dataset overview), Table 1–2 (distribusi)
- Fig 5 (FLOPs vs AUC), Table 6 (DeLong)
- Table 7 (metrik XAI)

**Tier 3 — pelengkap:**
- Convergence, confusion matrix, failure cases, supplementary

---

## CHECKLIST IMPLEMENTASI VISUALISASI

- [ ] Buat `artifacts/xai/fixed_display_samples.json` (himpunan sample tetap)
- [ ] Verifikasi semua sample lolos centering check
- [ ] Script generate Grid A (backbone, fixed samples, metode sama)
- [ ] Script generate Grid B (metode CAM, fixed samples, backbone terbaik)
- [ ] Colorbar/scale konsisten (bukan per-image auto-scale untuk perbandingan)
- [ ] Overlay mask outline di semua CAM figure
- [ ] Script sample per class (Bagian 2)
- [ ] Script tabel distribusi (Bagian 3, dari folds.json + label)
- [ ] Semua figure pakai style/font konsisten (kalau untuk paper)

---

## CATATAN PENTING

1. **Fixed sample set = kunci fair XAI comparison.** Tanpa ini, klaim "backbone/metode X lebih bagus" tidak sah. Simpan ID-nya sebagai artifact, jangan pilih ulang tiap run.
2. **Colorbar konsisten.** pytorch-grad-cam normalize per-image (min-max) — untuk perbandingan visual antar-sel, ini bisa misleading (map lemah di-stretch jadi kelihatan kuat). Pertimbangkan skala global atau catat di caption.
3. **Mask outline wajib di figure XAI.** Tanpa outline mask, pembaca tidak bisa menilai apakah merah benar-benar jatuh di lesi.
4. **Sample per class ≠ sample XAI.** Yang per-class untuk "dataset overview" (boleh nodul apa saja yang tipikal). Yang XAI harus TP malignant high-conf + fixed lintas backbone.
5. **Distribusi per fold membuktikan stratifikasi** — reviewer sering minta ini untuk memastikan tidak ada fold yang pincang.
