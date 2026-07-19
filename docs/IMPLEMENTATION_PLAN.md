# LungFuseNet — Implementation Plan (Post-Audit)

Plan ini disusun berdasarkan **audit status nyata** (bukan asumsi checklist lama). Temuan utama: banyak komponen yang dikira "sudah jalan" ternyata **ada kodenya tapi belum di-wire** (fusion, radiomics) atau **belum ada sama sekali** (training script Arm A/B/C/D). Ada juga **mismatch backbone kritis** yang harus diselesaikan sebelum apapun.

Prinsip plan: **selesaikan blocker dulu, jangan buka front baru di atas fondasi yang belum jelas.**

---

## FASE 0 — INVESTIGASI BLOCKER (WAJIB, sebelum nulis kode apapun)

Dua temuan memblok semua kerjaan lain. Selesaikan ini dulu.

### 0.1 Resolusi mismatch backbone (paling kritis)
**Masalah:** `backbones.py` isinya `mobilenet_v3_large, efficientnet_b0, resnet50, densenet121, convnext_tiny, resnet3d_10`. Tapi checklist + output XAI kemarin nyebut `mobilenetv3_small, vgg16, vit_base` — yang TIDAK ADA di `backbones.py`.

**Langkah investigasi (urut):**
- [ ] Cari file `src/models/registry.py` — audit bilang tidak ketemu, tapi diagnostic script Grad-CAM kemarin memakainya (`_NAME_MAP`). Konfirmasi: ada atau tidak?
- [ ] Kalau `registry.py` ADA: baca `_NAME_MAP`. Kemungkinan ini yang dipakai XAI pipeline, bukan `backbones.py`. Berarti ada DUA sumber definisi model.
- [ ] Kalau `registry.py` TIDAK ADA: dari mana output XAI vgg16/vit_base berasal? Cek apakah ada notebook / script lain yang mendefinisikan model (grep `vit_base`, `vgg16` di seluruh repo).
- [ ] Cek: apakah checkpoint untuk vgg16/vit_base/mobilenetv3_small benar-benar ADA di disk? (`ls artifacts/checkpoints/`). Kalau XAI menghasilkan output untuk backbone ini, checkpoint-nya harus ada.

**Decision point setelah investigasi:**
- **Skenario A** — XAI pakai `registry.py` dengan set backbone yang benar (6 target), `backbones.py` outdated → **update/hapus `backbones.py`, jadikan `registry.py` satu-satunya sumber.**
- **Skenario B** — checkpoint vgg16/vit_base tidak ada, output XAI itu dari run lama/berbeda → **backbone target harus di-training dari nol** (masuk Fase 2).
- **Skenario C** — dua-duanya dipakai di tempat berbeda → **konsolidasi ke satu registry**, ini sumber bug fair-setup.

> **GATE: jangan lanjut ke Fase 1+ sebelum tahu skenario mana yang benar.** Semua training bergantung pada set backbone yang pasti.

### 0.2 Konfirmasi keberadaan training script
**Masalah:** Tidak ada stage/training script untuk Arm A/B/C/D. Yang ada cuma `stage_05_xai.py`. Tapi XAI butuh model terlatih — berarti training terjadi di suatu tempat.

**Langkah:**
- [ ] Grep seluruh repo untuk `.fit(`, `train_loop`, `backward()`, `optimizer.step()` — di luar `stage_05`.
- [ ] Cek apakah ada notebook (`.ipynb`) yang melakukan training (checklist lama mungkin ditulis basis notebook yang belum diaudit).
- [ ] Cek `artifacts/checkpoints/` — kalau ada `best.pt`/`last.pt`, training PERNAH jalan (di suatu tempat). Kalau kosong, belum pernah.

**Decision point:**
- Kalau training ada di notebook → **port ke `stage_03_train.py` yang proper** (resumable, CSV logging) supaya reproducible.
- Kalau training belum pernah jalan → **tulis `stage_03_train.py` dari nol** (masuk Fase 2).

---

## FASE 1 — FONDASI YANG BISA DIVERIFIKASI

Setelah blocker jelas, bangun fondasi yang selama ini diasumsikan ada tapi tidak.

### 1.1 Split sebagai artifact tersimpan (bukan on-the-fly)
**Masalah:** Split dihitung ulang tiap panggil `add_kfold_splits(seed=42)`. Gate "split identik lintas arm" TIDAK bisa diverifikasi kalau split tidak dibekukan ke file.

- [ ] Tulis `stage_02_split.py` yang generate split SEKALI → simpan `artifacts/splits/folds.json`
- [ ] Format: mapping `patient_id → fold_id` (patient-level, bukan nodule-level)
- [ ] Semua training (semua arm) BACA dari file ini, bukan hitung ulang
- [ ] Tambah assertion: kalau `folds.json` sudah ada, jangan regenerate (biar tidak berubah diam-diam)
- [ ] **GATE**: verifikasi determinisme — hash folds.json, pastikan sama tiap load

### 1.2 Config wiring (config ada tapi belum tersambung)
**Masalah:** `train.yaml` ada tapi grep menunjukkan tidak ada `.py` yang import/argparse config ini. Config belum benar-benar dipakai.

- [ ] Wire `train.yaml` ke training script (argparse `--config`)
- [ ] Pastikan hyperparameter (LR, epoch, augmentation) benar-benar dibaca dari config, bukan hardcode di script
- [ ] **GATE fair-setup**: satu config → semua backbone. Verifikasi tidak ada hyperparameter hardcode yang bypass config.

### 1.3 Sanity check data (untuk XAI yang valid)
- [ ] Centering check: mask centroid ≤4px dari center patch
- [ ] Buang patch mask kosong / centroid melenceng, catat jumlahnya
- [ ] Simpan daftar patch valid → dipakai konsisten semua arm

### 1.4 Hygiene (cepat, sekalian)
- [ ] `.gitignore`: tambah `artifacts/`, `results/`, `*.log`, checkpoint besar
- [ ] Bersihkan file log yang terlanjur ke-commit

---

## FASE 2 — TRACK 2 TRAINING (inti yang ternyata belum ada)

Ini bagian terbesar yang audit ungkap: **training script Arm A/B/C/D belum ada.** Tulis dari nol (atau port dari notebook).

### 2.1 Training engine (`stage_03_train.py`)
- [ ] Argumen: `--backbone`, `--arm` (A/B/C/D), `--fold`, `--config`
- [ ] Baca split dari `folds.json` (Fase 1.1)
- [ ] Resumable: `maybe_resume` dari `last.pt`, checkpoint tiap N epoch
- [ ] CSV logging per epoch: `artifacts/logs/{backbone}_{arm}_{fold}.csv` (loss, acc, auc)
- [ ] `best.pt` (AUC terbaik) + `last.pt` (resume)
- [ ] Head fleksibel per arm:
  - [ ] Arm A: binary (BCE / CE 2-kelas)
  - [ ] Arm B: ordinal (SmoothL1 pada rating 1–5, ATAU ordinal regression)
  - [ ] Arm C: 3-class CE (benign/indeterminate/malignant, kolom grade3)
  - [ ] Arm D: 4-class CE (+ no-nodule)

### 2.2 Backbone final (dari resolusi Fase 0.1)
Pastikan 6 backbone target siap DENGAN emb_dim benar:
- [ ] mobilenetv3_small (~2.5M, emb 576)
- [ ] efficientnet_b0 (~5.3M)
- [ ] densenet121 (~8.0M)
- [ ] resnet50 (~25.6M)
- [ ] vgg16 (~138M, emb 512)
- [ ] vit_base (~86M, emb 768)

> Kalau Fase 0.1 memutuskan set backbone berbeda (mis. tetap pakai yang di `backbones.py`), UPDATE checklist ini agar konsisten. Yang penting: **satu set backbone final, dipakai konsisten Track 2 + XAI + Track 1.**

### 2.3 Eksekusi training
Matriks: 6 backbone × 4 arm × 5 fold = 120 run. Prioritaskan:
- [ ] Arm A dulu (binary, paling established) — 6×5 = 30 run
- [ ] Arm B (ordinal) — 30 run
- [ ] Arm D (grade4) — 30 run
- [ ] Arm C (grade3) — 30 run
- [ ] Orchestrator (lokal `run.sh` + Colab notebook) yang skip run selesai (resumable)

### 2.4 Reporting (`stage_06_report.py`)
- [ ] Baca semua CSV → `summary.csv` (params, flops, auc, sens, spec, f1, infer_ms)
- [ ] Metrik per arm (lihat detail di checklist audit: QWK/MAE untuk B, confusion matrix untuk C, **dua metrik terpisah untuk D**)
- [ ] **GATE anti-inflasi Arm D**: headline pakai benign-vs-malignant nodul-saja, BUKAN 4-kelas campur
- [ ] DeLong + bootstrap CI (kodenya SUDAH ada di `statistical_tests.py` — tinggal panggil)

---

## FASE 3 — XAI TRACK 2 (sudah sebagian, rapikan)

Grad-CAM fix kemarin sudah jalan. Tapi output nyebut backbone yang mungkin tidak match (tergantung resolusi Fase 0.1).

- [ ] **Setelah backbone final pasti**: re-run XAI untuk 6 backbone yang benar
- [ ] Pastikan cabang vgg16 (`features[-1]`) + vit (`blocks[-1].norm1`, reshape 4×4 bukan 14×14) benar
- [ ] Diagnostik struktural D0–D2 (CAM berubah saat input berubah?)
- [ ] Fix resolusi: Layer-CAM @ 8×8, patch tetap 64×64
- [ ] Metrik: IoU/Dice/pointing_acc/energy — pointing_acc sebagai utama
- [ ] Catat ceiling IoU rendah (nodul <10% area) di paper

---

## FASE 4 — TRACK 1 FUSION (kodenya ada, belum di-wire)

Fusion + radiomics logic SUDAH ada (bukan stub), tapi **tidak ada stage yang import** — belum tersambung ke training.

### 4.1 Wire radiomics (kode ada, belum pernah jalan)
- [ ] Jalankan `extraction.py` → hasilkan `artifacts/features/radiomics.csv` (belum ada cache)
- [ ] Jalankan `feature_selection.py` (ICC→mRMR→LASSO)
- [ ] Verifikasi fitur tersimpan + reproducible

### 4.2 Wire fusion ke training (`stage_03b_fusion.py`)
- [ ] Import `early/intermediate/late_fusion.py` (yang sudah ada) ke stage training
- [ ] Ablation 3 arm: CNN-only / radiomics-only / fusion
- [ ] **GATE fair**: split/preprocessing identik ketiga arm (baca `folds.json` yang sama)
- [ ] Decision rule ditetapkan SEBELUM lihat hasil (DeLong)

### 4.3 XAI Track 1 (2 level)
- [ ] Level 1: Grad-CAM (branch CNN) + SHAP (branch radiomics)
- [ ] Level 2: cross-validation spasial SHAP↔Grad-CAM

---

## URUTAN EKSEKUSI & DEPENDENCY

```
FASE 0 (investigasi blocker)  ← WAJIB PERTAMA, tidak bisa di-skip
   │
   ├─ 0.1 resolusi backbone mismatch ──┐
   └─ 0.2 konfirmasi training script ──┤
                                        ▼
FASE 1 (fondasi verifiable)   ← split fixed, config wiring, sanity check
   │
   ▼
FASE 2 (Track 2 training)     ← bagian terbesar, training Arm A/B/C/D dari nol
   │
   ├──────────────┐
   ▼              ▼
FASE 3 (XAI)   (bisa paralel setelah ada checkpoint)
   │
   ▼
FASE 4 (Track 1 fusion)       ← terakhir, potongan terbesar, sengaja ditunda
```

**Aturan:** jangan mulai fase berikutnya sebelum GATE fase sebelumnya lolos. Khususnya:
- **Fase 0 harus selesai** sebelum Fase 2 (set backbone harus pasti sebelum training).
- **Split fixed (1.1) harus ada** sebelum training apapun (kalau tidak, fair-setup tidak bisa diverifikasi).

---

## LANGKAH PERTAMA KONKRET

Jawab pertanyaan di akhir audit: **ya, cek `src/models/registry.py` / `_NAME_MAP` dulu.** Ini menentukan skenario A/B/C di Fase 0.1, dan semua training bergantung padanya.

Setelah itu jelas, langkah kedua: konfirmasi apakah training script benar-benar belum ada atau tersembunyi di notebook (Fase 0.2). Dua investigasi ini menentukan seberapa besar Fase 2 sebenarnya — apakah "port dari notebook" (lebih ringan) atau "tulis dari nol" (lebih berat).

Jangan tulis training script atau perbaiki backbone sampai dua investigasi ini selesai — supaya tidak membangun di atas asumsi yang salah lagi (seperti checklist sebelumnya).
