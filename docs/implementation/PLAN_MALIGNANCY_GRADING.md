# Plan — Malignancy Grading + Kelas No-Nodule

Status: **proposal**. Belum dieksekusi.
Tanggal: 2026-07-14
Konteks: lanjutan dari `refactored-finding-dusk.md` (fix axis-order preprocessing, sudah selesai).

---

## 0. Ringkasan Temuan (semua diukur, bukan asumsi)

Dijalankan di remote 100.98.9.120 hari ini:

| Fakta | Angka | Sumber |
|---|---|---|
| Scan LIDC di pylidc DB | 1018 | `pl.query(pl.Scan)` |
| Folder DICOM di disk | **1010** | `os.listdir(lidc_idri)` — **8 scan hilang** |
| Total nodul (min_ann=1) | 2651 | `cluster_annotations()` |
| **Nodul `median==3` DIBUANG** | **956 (36%)** | `_median_label(exclude_score=3)` |
| Lolos filter label | 1695 | 2651 − 956 |
| Ada di `labels.csv` | **1391** | **304 hilang di consensus/crop — belum diaudit** |
| Scan slice ≤2.5 mm | 897 / 1018 | `scan.slice_thickness` |
| LUNA16 lokal | `annotations.csv`, `candidates.csv`, `candidates_V2.zip`, subset0–9 (subset2 hilang) | `Lung Cancer\luna16 part {1,2}` |
| Match `seriesuid` LUNA16 → pylidc | **888 / 888 (100%)** | go/no-go check |
| Match `annotations.csv` → pylidc | **601 / 601 (100%)** | go/no-go check |
| Hard negative tersedia | **549.714** (median 580/scan) | `candidates.csv class==0` |

Distribusi `median_rating` penuh (2651 nodul):

| median | 1.0 | 1.5 | 2.0 | 2.5 | **3.0** | 3.5 | 4.0 | 4.5 | 5.0 |
|---|---|---|---|---|---|---|---|---|---|
| n | 324 | 36 | 515 | 275 | **956** | 168 | 192 | 70 | 115 |
| skema sekarang | 0 | 0 | 0 | 0 | **DIBUANG** | 1 | 1 | 1 | 1 |

**Kelas "indeterminate" (956) lebih besar dari kelas malignant (454) yang dipakai sekarang.** Ini bukan kasus pinggiran — ini sepertiga dataset yang dibuang.

---

## 1. Jawaban Pertanyaan Akademis (jujur, termasuk yang tidak enak)

### 1a. "Boleh gak nambah kelas non-cancer?"

**Boleh, tapi tiga syarat wajib. Kalau salah satu dilanggar, reviewer akan bantai.**

**Syarat 1 — Namanya salah.** "Non-cancer" itu keliru: nodul **benign JUGA non-cancer**. Kelas yang kamu maksud sebenarnya **"no-nodule" / "normal parenchyma"**. Di paper harus ditulis begitu. Kalau ditulis "non-cancer" sejajar dengan "benign", definisi kelasnya tumpang tindih dan itu cacat fatal.

**Syarat 2 — Ini task yang berbeda.** "Ada nodul gak?" = **deteksi**. "Nodul ini ganas gak?" = **karakterisasi**. Menggabungkan keduanya jadi satu softmax bikin kelas normal menang gampang (parenkim kosong vs nodul apapun ≈ terpisah sempurna), sehingga **macro-AUC/accuracy naik palsu** tanpa ada kemajuan nyata. RESEARCH_PLAN kamu sendiri sudah menyebut ini (baris 41: LUNA16 = deteksi, bukan malignancy).

**Syarat 3 — Negatifnya harus HARD negative.** Patch paru acak = terlalu gampang = metrik menggelembung. Yang sah: kandidat yang **algoritma deteksi kira nodul tapi ternyata bukan** (pembuluh darah, fissure, jaringan parut). Itu persis isi `candidates.csv class==0` LUNA16 — 549.714 lokasi, dan **100% scan-nya sudah ada di DICOM lokalmu**.

**Aturan pelaporan anti-inflasi (wajib):** selalu laporkan **DUA** set angka —
1. metrik 4-kelas (dengan no-nodule), dan
2. metrik benign-vs-malignant **pada subset nodul saja**.
Headline Track 2 (lightweight vs heavyweight) tetap pakai (2), supaya perbandingannya tidak terkontaminasi kelas gampang.

**Verdict:** kerjakan, tapi posisikan sebagai **eksperimen sekunder ("detection-aware extension")**, bukan endpoint utama. Endpoint utama tetap karakterisasi malignancy.

### 1b. "Boleh gak malignancy-nya digrading, jangan biner?"

**Boleh, dan ini justru upgrade akademis — bukan sekadar boleh.** Alasannya:

1. **Kamu buang 956 nodul (36%) tanpa alasan ilmiah.** Skema exclude-3 itu warisan konvensi, bukan hukum. Membuang sepertiga data adalah kelemahan yang bisa diserang.
2. **Kelompok indeterminate justru yang paling penting secara klinis.** Nodul median-3 adalah yang bikin dokter bingung — antara follow-up CT vs biopsi. Model yang cuma bisa bilang "jelas jinak" / "jelas ganas" tidak menolong di kasus yang justru butuh pertolongan.
3. **Biner membuang informasi urutan.** LIDC memberi skala 1–5. Memaksakan softmax 5-kelas juga salah (memperlakukan "salah 1 tingkat" sama beratnya dengan "salah 4 tingkat"). Yang benar: **ordinal**.

**Peringatan jujur:** "indeterminate" adalah **ketidakpastian anotator**, bukan keadaan biologis. Sebagian reviewer keberatan pada framing "3 kelas penyakit". Solusinya adalah framing **ordinal** (memprediksi skor malignancy 1–5), bukan "3 kelas penyakit" — secara metodologis lebih kuat dan tidak bisa diserang dari sisi itu.

---

## 2. Desain Terpilih

### Inti: SATU model ordinal, TIGA endpoint pelaporan

Head model mengeluarkan **1 skalar** = prediksi `median_rating` ∈ [1, 5]. Dilatih di **semua** nodul (termasuk median==3). Lalu:

| Endpoint | Cara turunkan | Gunanya |
|---|---|---|
| **Biner (AUC)** | Pada subset `median≠3`, skor = ŷ, label = `median>3` | **Langsung sebanding dengan angka 0.83–0.91 yang sudah ada** + sebanding dengan literatur |
| **Grading 3-kelas** | Bin ŷ: `<2.75`=benign, `2.75–3.25`=indeterminate, `>3.25`=malignant | Ini yang kamu mau. Confusion matrix + macro-F1 + recall per kelas |
| **Kualitas ordinal** | MAE, Quadratic Weighted Kappa, akurasi ±1 tingkat | Metrik yang benar untuk target berurut |

**Keunggulannya: satu training sweep, semua endpoint.** Threshold bisa diubah setelah training tanpa latih ulang. Dan endpoint biner-nya **apple-to-apple** dengan hasil sekarang, jadi klaim "ordinal tidak merusak apapun" bisa dibuktikan, bukan diklaim.

### Tiga arm eksperimen (bertahap, ada gate)

| Arm | Head / Loss | Data | Cost | Status |
|---|---|---|---|---|
| **A — baseline** | binary BCE | 1391 (exclude-3) | **0** | ✅ SUDAH ADA |
| **B — ordinal** | 1 skalar, SmoothL1 pada `median_rating` | ~2300 (semua) | 30 run ≈ 2,5 jam | BARU |
| **C — multi-task** | 2 head: BCE (masked, hanya median≠3) + ordinal (semua) | ~2300 | 30 run ≈ 2,5 jam | BARU, opsional |

**Decision rule (tetapkan SEKARANG, sebelum lihat hasil — biar tidak post-hoc):**
> Ordinal (B) menggantikan biner (A) sebagai headline **hanya jika** AUC biner turunan dari B tidak turun signifikan dari A (DeLong, α=0.05). Kalau turun signifikan, A tetap headline dan grading dilaporkan sebagai analisis tambahan. Kalau B setara, B menang telak — AUC sama, tapi pakai 36% lebih banyak data DAN memberi grading.

Ini meniru gaya decision rule yang sudah ada di `RESEARCH_PLAN.md` baris 173 (untuk fusion). Konsisten.

Arm C dijalankan **hanya kalau B lolos gate**. Jangan bakar 2,5 jam sebelum tahu B masuk akal.

### Arm D — no-nodule (terpisah, sekunder)

4-kelas: `no-nodule / benign / indeterminate / malignant`. Dijalankan **setelah** B selesai. Negatif dari LUNA16 `candidates_V2` class==0, disubsample ~1:1 dengan jumlah nodul (~2300 negatif), **hanya dari pasien yang sudah ada di split kita**, dan **ditaruh di fold pasien yang sama** (kalau tidak → leakage).

---

## 3. Perubahan Kode

### 3.1 `src/data_loading/lidc_loader.py` — jangan buang median==3

```python
def _malignancy_targets(anns: list) -> dict:
    """Semua target sekaligus. Tidak ada nodul yang dibuang."""
    ratings = [a.malignancy for a in anns]
    med = float(np.median(ratings))
    return {
        "median_rating": med,                                  # ordinal target, 1.0-5.0
        "label": int(med > 3) if med != 3 else -1,             # biner; -1 = indeterminate (di-mask di loss)
        "grade3": 0 if med < 3 else (1 if med == 3 else 2),    # 0=benign 1=indeterminate 2=malignant
        "n_annotations": len(anns),
        "rating_std": float(np.std(ratings)),                  # BARU: disagreement antar radiolog
    }
```

- Ganti `exclude_score` jadi parameter `include_indeterminate: bool = True`.
- **`label = -1` untuk median==3** (bukan dibuang). Loss biner me-mask `-1`. Ini yang bikin arm C (multi-task) mungkin.
- Tambah `rating_std` — ukuran ketidaksepakatan radiolog. Berguna buat analisis: apakah model paling salah persis di nodul yang radiolognya juga tidak sepakat? Itu paragraf diskusi gratis.

**`skip_existing=True`** — 1391 patch lama sudah benar (logika crop tidak berubah), jangan diekstrak ulang. Cuma ekstrak ~950 nodul baru. Hemat ~50 menit.

### 3.2 Fold assignment — WAJIB dibekukan

**Ini paling gampang bikin hasil tidak sebanding, dan paling gampang kelewat.**

`add_kfold_splits` melakukan `StratifiedKFold` atas `patient_label` (modus label pasien). Menambah 956 nodul mengubah modus itu → sebagian pasien **pindah fold** → AUC baru tidak bisa dibandingkan dengan AUC lama, dan kamu tidak akan sadar.

```python
def add_kfold_splits(df, n_folds=5, seed=42, freeze_from: str | None = None):
    """Patient-level stratified k-fold. freeze_from: labels.csv lama — fold pasien lama DIPERTAHANKAN."""
    frozen = {}
    if freeze_from and os.path.exists(freeze_from):
        old = pd.read_csv(freeze_from)
        frozen = dict(old.drop_duplicates("patient_id")[["patient_id", "fold"]].values)

    known = df[df.patient_id.isin(frozen)].copy()
    known["fold"] = known.patient_id.map(frozen)

    new = df[~df.patient_id.isin(frozen)].copy()
    if len(new):
        # stratifikasi pasien baru pakai grade3 modus (bukan label biner — pasien baru bisa all-indeterminate)
        pdf = new.groupby("patient_id").agg(pl=("grade3", lambda x: int(x.mode()[0]))).reset_index()
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        pdf["fold"] = -1
        for k, (_, vi) in enumerate(skf.split(pdf.patient_id, pdf.pl)):
            pdf.loc[vi, "fold"] = k
        new = new.merge(pdf[["patient_id", "fold"]], on="patient_id", how="left")
    return pd.concat([known, new], ignore_index=True)
```

Verifikasi wajib: `set(fold pasien lama di labels_baru) == set(fold pasien lama di labels_lama)`. Kalau ada satu saja yang geser, **berhenti**.

### 3.3 `src/models/` — head ordinal

Semua 6 backbone sekarang berakhir di `Linear(feat, 2)`. Tambah pilihan:

```python
# task="binary"  -> Linear(feat, 2), CrossEntropy
# task="ordinal" -> Linear(feat, 1), SmoothL1 pada median_rating
# task="multi"   -> Linear(feat, 2) + Linear(feat, 1), loss = CE(masked) + lambda * SmoothL1
# task="grade4"  -> Linear(feat, 4), CrossEntropy (arm D)
```

Backbone **tidak disentuh** — hanya head + loss. Ini yang menjaga perbandingan Track 2 (params/FLOPs) tetap valid: yang dibandingkan tetap arsitektur yang sama persis.

Catatan: params bertambah/berkurang beberapa ribu saja (head kecil). Sebutkan di tabel efficiency bahwa params dilaporkan untuk konfigurasi head yang sama.

### 3.4 `src/evaluation/metrics.py` — metrik ordinal

Tambah:
```python
def ordinal_metrics(y_rating_true, y_rating_pred):
    """MAE, QWK (pada skor dibulatkan 1-5), akurasi dalam ±1 tingkat."""
    from sklearn.metrics import cohen_kappa_score
    mae = np.abs(y_rating_true - y_rating_pred).mean()
    t = np.clip(np.round(y_rating_true), 1, 5).astype(int)
    p = np.clip(np.round(y_rating_pred), 1, 5).astype(int)
    return {
        "mae": float(mae),
        "qwk": float(cohen_kappa_score(t, p, weights="quadratic")),
        "acc_within_1": float((np.abs(t - p) <= 1).mean()),
    }

def derive_binary(y_rating_true, y_rating_pred):
    """AUC biner pada subset median != 3 — SEBANDING dengan hasil arm A."""
    m = y_rating_true != 3.0
    return compute_metrics(( y_rating_true[m] > 3 ).astype(int), y_rating_pred[m])

def derive_grade3(y_rating_true, y_rating_pred, lo=2.75, hi=3.25):
    to3 = lambda v: np.where(v < lo, 0, np.where(v > hi, 2, 1))
    return to3(y_rating_true), to3(y_rating_pred)
```

### 3.5 `src/stage_00b_negatives.py` — BARU (arm D)

1. Unzip `candidates_V2.zip` (72 MB CSV) — **satu-satunya hal yang perlu di-unzip. Subset .zip 59 GB TIDAK disentuh.**
2. World (mm, LPS) → voxel, pakai origin dari DICOM:
   ```python
   dsl   = scan.load_all_dicom_images(verbose=False)   # sudah urut z oleh pylidc
   ox, oy, oz = map(float, dsl[0].ImagePositionPatient)
   ps    = float(scan.pixel_spacing)
   sz    = float(scan.slice_spacing or scan.slice_thickness)
   vx = (coordX - ox) / ps
   vy = (coordY - oy) / ps
   vz = (coordZ - oz) / sz
   ```
3. **GATE WAJIB — self-test transform.** Pakai `annotations.csv` (1186 nodul asli, 601 scan, semua match). Transform ke voxel, cari centroid nodul pylidc terdekat di scan yang sama, ukur jarak dalam mm.
   - **Lolos jika ≥95% jarak < `diameter_mm`.**
   - **Kalau gagal → STOP.** Jangan generate negatif apapun. Kemungkinan besar sumbu z terbalik — coba `vz = (oz_max - coordZ) / sz`, tes ulang.
   - Ini tes gratis dan menentukan. **Jangan dilewati.** Transform yang salah = 2300 patch sampah yang kelihatan meyakinkan.
4. Sampling negatif:
   - hanya `class==0`, hanya `seriesuid` yang pasiennya sudah ada di `labels.csv`
   - buang kandidat yang **< 10 mm dari nodul asli manapun** (kalau tidak, "negatif"-mu sebenarnya di dalam nodul → label noise)
   - ambil ~1 per pasien sampai total ≈ jumlah nodul (rasio 1:1)
   - `fold` = fold pasiennya (bekukan, jangan re-split)
5. Crop window **64×64×16 mm, resample 1 mm** — **fungsi yang sama persis** (`_crop_and_resample_nodule` versi tanpa mask). Kalau beda pipeline, model bisa membedakan kelas dari artefak preprocessing, bukan dari anatomi. Ini jebakan halus.
6. **Mask untuk radiomics:** negatif tidak punya lesi → mask kosong → PyRadiomics gagal. Pakai **bola diameter 10 mm di titik tengah** sebagai ROI. **Catat ini sebagai deviasi metodologis di paper** — jangan disembunyikan.
7. Simpan ke `artifacts/patches_neg/`, append ke labels.csv dengan `grade4=0`, `median_rating=NaN`, `label=-1`.

### 3.6 `src/stage_06_report.py` — bagian baru

- `rating_scatter.png` — ŷ vs median_rating asli, warnai titik dengan `rating_std` (nodul yang radiolognya tak sepakat harusnya paling berantakan — kalau iya, itu bukti model belajar ketidakpastian yang benar)
- `grade3_confusion.png` — grid 6 model
- `ordinal_table.csv` — MAE / QWK / acc±1 per model
- `binary_comparison.csv` — **AUC arm A vs arm B berdampingan + p-value DeLong.** Ini tabel yang menentukan decision rule di §2.

---

## 4. Urutan Eksekusi

Semua remote via paramiko dari scratchpad (pola yang sudah dipakai).

| # | Langkah | Waktu | Gate |
|---|---|---|---|
| 1 | **Audit 304 nodul hilang** (1695 → 1391). Log alasan drop per nodul. Cek 8 scan yang ada di DB tapi tidak ada di disk. | 15 mnt | Kalau > 5% hilang karena bug (bukan data rusak), perbaiki dulu |
| 2 | Backup `labels.csv` → `labels_binary_v1.csv` (referensi fold + hasil arm A) | 1 mnt | — |
| 3 | Terapkan §3.1 + §3.2. Jalankan stage_00 `skip_existing=True` | ~30 mnt | Patch baru shape `(16,64,64)`, mask non-kosong |
| 4 | **GATE FOLD:** verifikasi tiap pasien lama tetap di fold yang sama | 1 mnt | **Geser 1 pasien pun → STOP** |
| 5 | stage_01 radiomics untuk ~950 nodul baru saja (incremental) | ~25 mnt | — |
| 6 | Terapkan §3.3 + §3.4. Train **arm B** (6 model × 5 fold) | ~2,5 jam | — |
| 7 | Eval + `binary_comparison.csv` + DeLong vs arm A | 10 mnt | **DECISION RULE §2** |
| 8 | Kalau B lolos → train **arm C** (multi-task) | ~2,5 jam | — |
| 9 | §3.5 + **GATE TRANSFORM** (self-test `annotations.csv`) | 30 mnt | **< 95% hit → STOP, jangan generate negatif** |
| 10 | Generate ~2300 patch negatif + train **arm D** (4-kelas) | ~3 jam | — |
| 11 | §3.6 report + commit | 20 mnt | — |

Total ≈ 10 jam kalau semua arm jalan. Bisa dipotong: **berhenti di langkah 7** dan sudah dapat grading malignancy yang diminta. Arm C dan D adalah bonus.

**Rekomendasi staging:** kerjakan 1–7 dulu. Itu menjawab "aku pengen ada malignancy-nya" secara penuh, ~4 jam, risiko rendah. Arm D (no-nodule) belakangan — dia yang paling berisiko secara akademis dan paling mahal.

---

## 5. Risiko (dan cara matinya)

| Risiko | Kenapa bahaya | Mitigasi |
|---|---|---|
| **Fold bergeser** | AUC baru vs lama jadi tidak sebanding, dan **tidak ada error** — kamu cuma dapat angka yang salah dibanding | Gate langkah 4. Bekukan fold map. |
| **Transform world→voxel salah** (z terbalik) | Menghasilkan 2300 patch "negatif" yang sebenarnya acak/kosong. **Kelihatan meyakinkan, tetap salah.** | Gate langkah 9 pakai `annotations.csv` sebagai ground truth. Gratis dan menentukan. |
| **Negatif terlalu gampang** | Macro-AUC menggelembung, reviewer bantai | Wajib pakai `candidates_V2` class==0 (detector FP), bukan patch acak. Buang yang <10mm dari nodul asli. |
| **Kelas normal mengontaminasi headline Track 2** | Klaim "lightweight menang" jadi tidak sah | Selalu laporkan benign-vs-malignant **pada subset nodul saja**, terpisah |
| **Pipeline preprocessing negatif ≠ positif** | Model membedakan kelas dari artefak, bukan anatomi | Pakai fungsi crop/resample yang sama persis |
| **AUC ordinal turun** | Panik, mundur ke biner | **Sudah diantisipasi** oleh decision rule §2. Turun signifikan = biner tetap headline. Itu hasil yang sah, bukan kegagalan. |
| **`load_and_split` return cache basi** | stage_00 tidak jalan sama sekali, diam-diam | Hapus `labels.csv` / `force_rebuild=True`. Sudah pernah kena. |

---

## 6. Apa yang TIDAK dilakukan

- **Tidak** unzip subset0–9 LUNA16 (59 GB). Tidak dibutuhkan — kita pakai DICOM lokal + CSV koordinat saja.
- **Tidak** memakai gambar `.mhd` LUNA16 sama sekali (resampling-nya beda dari pipeline kita → akan jadi confound).
- **Tidak** menyentuh Track 1 (Fusion + XAI). Masih belum diimplementasi, rencana terpisah.
- **Tidak** re-crop 1391 patch lama. Logika crop tidak berubah.
- **Tidak** menamai kelas "non-cancer". Namanya **"no-nodule"**.
