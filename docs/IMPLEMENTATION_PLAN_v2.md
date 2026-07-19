# LungFuseNet — Implementation Plan (v2, basis kondisi REMOTE nyata)

> **Catatan revisi:** Plan v1 (`IMPLEMENTATION_PLAN.md` lama) disusun dari audit repo **local yang stale**. Fase 0/1/2 di plan itu (investigasi backbone mismatch, tulis split artifact, tulis training script dari nol) **sudah tidak relevan** — semua sudah kelar di remote. Plan ini ditulis ulang basis kondisi remote yang sebenarnya.

---

## KONDISI NYATA (remote, terverifikasi)

### ✅ Sudah beres
- `artifacts/splits/folds.json` — split fixed sudah ada (bukan on-the-fly lagi)
- **Training lengkap**: 6 backbone × 4 arm (binary/ordinal/grade3/grade4) × 5 fold = **120 run**, checkpoint + CSV log lengkap semua
- `artifacts/features/radiomics.parquet` — radiomics extraction sudah pernah jalan
- Full pipeline `stage_00` → `stage_06` ada
- DeLong + bootstrap CI + consensus mask — sudah implement (dari audit sebelumnya)

### ❌ Belum / kurang (SISA KERJAAN NYATA)
1. `artifacts/logs/summary.csv` — belum ada (stage_06_report belum digenerate / output beda nama)
2. **Config dualism nyata**: `config.yaml` DAN `train.yaml` dua-duanya ada di remote — perlu cek mana yang aktif dipakai training
3. **Track 1 fusion belum wired** — `src.fusion` belum di-import di `stage_03_train.py` maupun `stage_04_evaluate.py` (terkonfirmasi, sesuai dugaan awal)

> Track 2 pada dasarnya SELESAI. Sisa kerjaan = reporting + hygiene + Track 1. Jauh lebih kecil dari plan v1.

---

## SISA KERJAAN (4 item, urut by dependency)

### LANGKAH 1 — Sync local dari remote (paling awal, wajib)
**Kenapa pertama:** semua investigasi/eksekusi berikutnya harus di atas kode remote yang benar. Kalau tidak sync, kita ulangi kesalahan "plan dari repo stale" untuk ketiga kalinya.

- [ ] Pull semua dari remote: `registry.py`, `stage_00`–`stage_06`, semua checkpoint + log
- [ ] Verifikasi `artifacts/` ter-sync: `folds.json`, `radiomics.parquet`, checkpoint 120 run, CSV log
- [ ] Konfirmasi struktur local == remote (tidak ada file lokal usang yang menimpa)

> **GATE:** jangan lakukan langkah 2–4 sebelum local benar-benar mirror remote.

### LANGKAH 2 — Resolve config dualism (sebelum percaya angka reporting)
**Kenapa sebelum reporting:** kalau 120 run dilatih pakai config tertentu, summary.csv harus mencerminkan config ITU. Kalau ada dua config dan kita salah rujuk, tabel final bisa salah/inkonsisten.

- [ ] Cek `stage_03_train.py`: config mana yang di-import/argparse (`config.yaml` atau `train.yaml`?)
- [ ] Grep `config.yaml` vs `train.yaml` di seluruh `src/` — mana yang benar-benar dibaca
- [ ] Tentukan **satu config kanonik** = yang dipakai 120 run
- [ ] Yang tidak dipakai: arsipkan/hapus/tandai deprecated (jangan biarkan ambigu)
- [ ] Dokumentasikan di README: config aktif yang mana
- [ ] **GATE fair-setup**: verifikasi 120 run memang pakai hyperparameter dari config kanonik yang sama (bukan campur dua config antar-run)

### LANGKAH 3 — Generate summary.csv + tabel final (Track 2 closeout)
**Kenapa sekarang:** training sudah selesai, tinggal agregasi. Ini menutup Track 2 jadi output yang bisa masuk paper.

- [ ] Jalankan `stage_06_report.py` — cek kenapa `summary.csv` belum ada:
  - [ ] Error saat run? → debug
  - [ ] Output beda nama? → cek nama file aktual di `artifacts/`, rename/sesuaikan
  - [ ] Belum pernah dipanggil? → jalankan
- [ ] Verifikasi `summary.csv` berisi: backbone | arm | params | flops | auc | sens | spec | f1 | infer_ms (per fold + agregat)
- [ ] Metrik per arm sesuai (dari checklist audit):
  - [ ] Arm A: AUC, acc, sens, spec, F1, balanced acc
  - [ ] Arm B (ordinal): **QWK, MAE, one-off accuracy**, macro-AUC
  - [ ] Arm C (grade3): confusion matrix 3-kelas, macro-F1, per-class AUC
  - [ ] Arm D (grade4): **DUA metrik terpisah** —
    - [ ] (1) 4-kelas penuh
    - [ ] (2) benign-vs-malignant nodul-saja (exclude no-nodule)
    - [ ] **GATE anti-inflasi**: headline pakai metrik (2)
- [ ] Panggil DeLong + bootstrap CI (kode sudah ada) untuk perbandingan backbone
- [ ] **Figure/tabel Track 2 yang dihasilkan:**
  - [ ] Tabel utama per arm (model × metrik)
  - [ ] Scatter "Params vs AUC" (bukti lightweight vs heavyweight)
  - [ ] Scatter "FLOPs vs AUC"
  - [ ] Grafik convergence (dari CSV log)
  - [ ] Confusion matrix
  - [ ] Tabel perbandingan semua arm (transparan, termasuk yang kalah)

> Setelah langkah 3: **Track 2 tuntas** — punya semua tabel/figure untuk paper.

### LANGKAH 4 — Track 1 Fusion (satu-satunya kerjaan besar tersisa)
**Kenapa terakhir:** ini potongan terbesar yang beneran belum jalan, dan sengaja ditunda sampai Track 2 tuntas (konsisten dengan keputusan sebelumnya: tutup dulu yang jalan, baru buka front baru). Radiomics extraction sudah ada (`radiomics.parquet`), fusion logic sudah ada, tinggal WIRE + training + XAI.

#### 4.1 Feature selection radiomics (kalau belum)
- [ ] Cek: `radiomics.parquet` sudah lewat ICC→mRMR→LASSO atau masih raw ~1000+ fitur?
- [ ] Kalau raw: jalankan `feature_selection.py` → simpan fitur terpilih
- [ ] Standardize (fit scaler di train fold saja)

#### 4.2 Wire fusion ke training (`stage_03b_fusion.py` atau extend stage_03)
- [ ] Import `early/intermediate/late_fusion.py` ke stage training (SEKARANG belum di-import)
- [ ] Baca `folds.json` yang SAMA dengan Track 2 (fair-setup lintas track)
- [ ] **Ablation 3 arm:**
  - [ ] CNN-only (sudah ada dari Track 2 — reuse checkpoint Arm A)
  - [ ] Radiomics-only (XGBoost/LightGBM pada fitur terpilih)
  - [ ] Fusion: early / intermediate (default) / late
- [ ] **GATE fair**: split/preprocessing identik ketiga arm
- [ ] **Decision rule** (tetapkan SEBELUM lihat hasil): fusion headline HANYA jika signifikan (DeLong) atas modality terbaik
- [ ] Catat: kalau radiomics-only > CNN-only, itu temuan sah

#### 4.3 XAI Track 1 (2 level)
- [ ] Level 1: Grad-CAM/Layer-CAM (branch CNN) + SHAP TreeSHAP (branch radiomics)
  - [ ] SHAP beeswarm (global) + waterfall (local)
- [ ] Level 2: cross-validation spasial — fitur SHAP-tinggi ↔ region Grad-CAM
- [ ] (Opsional) Grad-CAM fusion vs CNN-only: apakah radiomics memandu perhatian CNN?

#### Figure Track 1 yang diperlukan:
- [ ] Tabel ablation: CNN-only vs radiomics-only vs fusion (3 tipe) + DeLong p-value
- [ ] SHAP beeswarm
- [ ] Side-by-side Grad-CAM + SHAP untuk nodul sama
- [ ] Diagram arsitektur fusion
- [ ] (Level 2) figure cross-validation SHAP↔Grad-CAM

---

## URUTAN & DEPENDENCY

```
LANGKAH 1 (sync local)           ← wajib pertama, cegah stale ketiga kali
   │
   ▼
LANGKAH 2 (resolve config)       ← sebelum reporting (angka harus rujuk config benar)
   │
   ▼
LANGKAH 3 (summary.csv + tabel)  ← Track 2 closeout, tinggal agregasi
   │
   ▼
LANGKAH 4 (Track 1 fusion)       ← satu-satunya kerjaan besar tersisa
   ├─ 4.1 feature selection
   ├─ 4.2 wire + ablation 3 arm
   └─ 4.3 XAI 2 level
```

**Item yang sudah TIDAK perlu** (beda dari plan v1):
- ~~Investigasi backbone mismatch~~ → sudah jelas di remote
- ~~Tulis split artifact~~ → `folds.json` sudah ada
- ~~Tulis training script dari nol~~ → 120 run sudah selesai
- ~~Sanity check + training Arm A/B/C/D~~ → sudah kelar

---

## LANGKAH KONKRET BERIKUTNYA

Rekomendasi: **sync local (langkah 1) → cek config (langkah 2) → jalankan stage_06_report (langkah 3)** dalam satu sesi. Tiga ini cepat dan menutup Track 2. Track 1 (langkah 4) dikerjakan setelahnya sebagai fase tersendiri karena bebannya besar.

Kalau mau paling efisien: mulai dari langkah 1+2 barengan (sync sambil cek config mana yang di-import), lalu langsung langkah 3. Track 1 baru dibuka setelah summary.csv + tabel Track 2 benar-benar jadi.
