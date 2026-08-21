# Plan Implementasi — Hasil Audit Korpus (Prompt 1 & 3)

Garis besar implementasi dari dua audit NotebookLM. Disusun per fase dengan gate, konsisten dengan pola run01/run02.

---

## RINGKASAN: APA YANG LAYAK DIIMPLEMENTASI

| Dari | Item | Sifat | Biaya | Prioritas |
|---|---|---|---|---|
| P1 | LASSO pada vektor gabungan sebelum XGBoost | kode | rendah | sedang |
| P1 | Pola ukuran-dataset vs gain fusi | tulisan | nol | **tinggi** |
| P1 | Positioning: 1 dari 4 paper menguji signifikansi | tulisan | nol | **tinggi** |
| P3 | Perturbation-based AUC degradation (ROAD/deletion-insertion) | kode+GPU | sedang | **tinggi** |
| P3 | PGI & PGU | kode | rendah | sedang |
| P3 | SSIM Stability (rotasi ±3°, noise σ=0,02) | kode+GPU | sedang | sedang |
| P3 | SSIM Consistency lintas-pasien | kode+GPU | sedang | rendah |
| P3 | Framing kegagalan sanity check sebagai kontribusi | tulisan | nol | **tertinggi** |
| P3 | Protokol reader study | tulisan | nol | rendah |

> **Prasyarat mutlak:** angka Friedman/Nemenyi/DeLong per-arm tanpa CSV harus diselesaikan dulu (regenerasi atau keluarkan dari naskah). Itu blocker submit, bukan peningkatan. Jangan tambah eksperimen sebelum beres.

---

## FASE 0 — PENULISAN (nol kode, kerjakan duluan)

Nilai tertinggi, biaya nol. Tiga temuan dari audit yang langsung memperkuat kedua naskah.

### 0.1 Kegagalan sanity check sebagai kontribusi orisinal ⭐

**Temuan korpus:** nol dari ~29 paper menjalankan sanity check (Adebayo/Kindermans) pada model mereka sendiri. Sebagian merujuknya di tinjauan pustaka, tapi tidak ada yang mempraktikkannya.

**Yang kamu punya:** korelasi 0,71–0,85 setelah classifier + tiga blok teratas diacak; korelasi balik-kelas 0,7712.

**Klaim yang bisa ditulis:**
> Peta CAM pada tugas ini terbukti *activation-driven*, bukan *decision-driven* — dan sepanjang korpus yang kami periksa, tidak ada studi klasifikasi nodul paru yang memverifikasi hal ini pada modelnya sendiri.

- [ ] Masuk ke Limitations **kedua** naskah (Track 1 dan Track 2)
- [ ] Track 2: bisa naik dari Limitations ke Results, karena sejalan dengan tesis situs-penjelasan
- [ ] **Wajib dihedge**: "dalam korpus N paper yang kami periksa", bukan klaim absolut
- [ ] Sitir Adebayo et al. (2018) dan Kindermans et al. (2017)

### 0.2 Pola ukuran-dataset vs gain fusi

**Temuan korpus:** hubungan terbalik konsisten. n≈200–300 → gain +2,5% s.d. +4%. n≥1000 → gain +0,59% atau mendekati nol.

**Datamu:** n=1391, selisih `fusion_late` vs `radiomics_only` = +0,0015 AUC.

**Nilainya:** hasilmu **persis sesuai pola terdokumentasi**, lengkap dengan mekanisme penjelasnya (pada n besar, CNN sudah cukup data untuk mempelajari pola yang diekstrak radiomics → *diminishing returns*).

- [ ] Masuk Discussion Track 1 sebagai preseden bahwa hasil bukan anomali
- [ ] Sertakan Neha & Shukla (2026): gain fusi di banyak studi retrospektif adalah "artefak metodologis" akibat kebocoran sebelum CV; konkatenasi naif "memperkuat efek noise dan korelasi"
- [ ] Hubungkan ke temuanmu sendiri: kamu **menemukan dan memperbaiki** kebocoran semacam itu (early stopping leak → nested CV), dan hasilnya turun −0,0210. Konsisten dengan diagnosis mereka.

### 0.3 Positioning kekakuan metodologi

**Temuan korpus:** dari 4 paper dengan gain fusi, hanya **1** yang menguji signifikansi — itu pun *paired t-test* pada Dice untuk segmentasi, bukan klasifikasi.

**Kontras:** kamu menguji 21 pasang dengan DeLong berpasangan, plus nested CV, plus split patient-level dibekukan.

- [ ] Masuk Discussion/positioning kedua naskah
- [ ] Bingkai sebagai: gain kecil yang kami laporkan diuji, sementara gain besar di literatur sering tidak
- [ ] **Hedge**: berlaku untuk korpus yang diperiksa, bukan seluruh literatur

### 0.4 Standar evaluasi XAI korpus (bahan positioning)

**Temuan:** standar minimum korpus = visualisasi kualitatif 2–4 contoh berhasil, nol pengujian kuantitatif. Hanya <15% melakukan lokalisasi kuantitatif (IoU/Dice/pointing game). Neha & Shukla: <10% studi nodul paru mengintegrasikan XAI sama sekali.

- [ ] Pakai untuk membenarkan kedalaman evaluasi XAI-mu sebagai kontribusi, bukan pelengkap

### 0.5 Protokol reader study (sudah diputuskan sebelumnya)

Korpus memberi template konkret — Katar et al. (2026): 1 radiolog toraks bersertifikat (>10 tahun), 77 kasus test set, skala Likert 5-poin, 5 dimensi (anatomical accuracy, pathological completeness, terminology correctness, linguistic fluency, clinical usefulness), analisis mean±SD.

- [ ] Adaptasi ke konteks XAI (menilai peta, bukan laporan teks)
- [ ] Simpan di `docs/`, nyatakan future work di manuskrip
- [ ] Nol pengumpulan data manusia di fase ini

---

## FASE 1 — FAITHFULNESS (prioritas kode tertinggi)

Menjawab celah terbesar: metrikmu semua mengukur **lokalisasi**, belum ada yang mengukur **faithfulness**.

### 1.1 Perturbation-based AUC degradation

**Justifikasi korpus:** Tempel et al. (2025) memakai kurva degradasi performa berbasis perturbasi. ROAD/deletion-insertion tidak disebut namanya di korpus, tapi **prinsip matematisnya identik** — jadi implementasi ROAD sah dan bisa disitir lewat Tempel (prinsip) + paper ROAD asli (Rong et al., ICML 2022).

- [ ] Implementasi deletion/insertion curve pada 3 backbone utama
- [ ] Atau ROAD via `pytorch-grad-cam` (library sudah terpasang)
- [ ] Bandingkan `fusion_late` vs `cnn_only`
- [ ] Sudah tertulis di `handoff/GOAL3.md` — sekarang punya justifikasi korpus

**⚠️ Caveat wajib dicatat:** Tempel et al. melaporkan uji perturbasi punya **bias metodologis bawaan** yang condong mengunggulkan penjelasan spasial (Grad-CAM) dibanding SHAP. Ini **langsung relevan** karena klaim Track 1-mu membandingkan CAM spasial vs SHAP fitur. Nyatakan caveat ini saat melaporkan hasil — jangan sampai keunggulan CAM terbaca sebagai superioritas sejati padahal artefak desain perturbasi.

### 1.2 PGI & PGU

**Justifikasi:** Tempel et al. (2025). Menguji faithfulness **tanpa** butuh mask ground truth — pelengkap yang baik untuk pointing accuracy yang butuh mask.

- [ ] Implementasi pada set sample tetap yang sama
- [ ] Murah, tidak butuh anotasi tambahan

**Gate Fase 1:**
- [ ] Kurva degradasi tersaji untuk 3 backbone, `fusion_late` vs `cnn_only`
- [ ] PGI/PGU terhitung pada `fixed_display_samples.json`
- [ ] Caveat bias perturbasi tercatat di naskah

---

## FASE 2 — ROBUSTNESS (opsional, nilai sedang)

### 2.1 SSIM Stability

**Justifikasi:** Qin (2025) — mengukur kekokohan peta terhadap perturbasi fisik: kemiringan ±3° dan noise Gaussian σ=0,02.

**⚠️ Adaptasi teknis wajib:** Qin memakai **3D SSIM**; inputmu **2,5D** (3 slice ditumpuk sebagai channel). Tidak bisa disalin langsung. Opsi: SSIM 2D per-slice lalu dirata-rata, atau SSIM pada peta CAM 2D (bukan volume). Nyatakan adaptasinya eksplisit di metode.

- [ ] Tentukan skema adaptasi 2,5D dulu, dokumentasikan
- [ ] Terapkan perturbasi pada input, hitung SSIM antar peta CAM sebelum/sesudah
- [ ] 3 backbone utama

**Nilainya untuk Track 2:** kalau peta tidak stabil terhadap perturbasi kecil, itu **bukti tambahan** untuk tesis kegagalan senyap — peta yang tampak meyakinkan tapi rapuh.

### 2.2 SSIM Consistency lintas-pasien

**Justifikasi:** Qin (2025) — kemiripan peta antar pasien dengan label sama.

- [ ] Prioritas lebih rendah; kerjakan hanya jika Fase 1 dan 2.1 selesai dan waktu cukup
- [ ] Adaptasi 2,5D sama seperti 2.1

---

## FASE 3 — FUSI (upaya terakhir, pre-registered)

### 3.1 LASSO pada vektor gabungan sebelum XGBoost

**Justifikasi:** Kurra et al. (2026) — regularisasi L1 pada representasi gabungan berdimensi tinggi sebelum XGBoost, untuk menekan redundansi.

**Kenapa ini satu-satunya mekanisme yang layak dari enam yang ditemukan:**
- AutoGluon stacking → pada dasarnya late fusion lebih rumit; kamu sudah punya late fusion yang menang. AutoML juga membuat "mekanisme mana yang menolong" tidak terjawab — buruk untuk paper explainability.
- Learnable linear projection → tumpang tindih dengan `branch_norm` + FusionNet
- Two-stage SVM FP filter → tugas berbeda (deteksi), tidak berlaku
- Transformer tokenization → beban rekayasa terbesar, dan jalur CAM ViT-mu sudah rusak
- SGHF-Net → butuh prior patologi/biopsi; LIDC tidak punya (label = opini radiolog)

**Kenapa LASSO-on-joint layak:**
- Langsung menyasar mode kegagalan yang sudah didiagnosis (ketimpangan dimensi 256 vs ~24)
- Nol dependensi baru (sklearn sudah ada)
- `fusion_early` adalah arm terburuk (0,9126) — paling mungkin membaik

**⚠️ Aturan pre-registration (tetapkan SEBELUM jalan):**
- [ ] Satu mekanisme, satu kali jalan
- [ ] Dilaporkan apa pun hasilnya
- [ ] **Tidak ada pencarian lanjutan** kalau gagal
- [ ] Literatur sendiri memprediksi kemungkinan besar tidak menolong pada n=1391 — catat prediksi ini sebelum eksperimen

**Gate:** kalau `fusion_early` + LASSO tidak mengalahkan `radiomics_only` dengan DeLong p<0,05, pencarian fusi **ditutup permanen** dan hasilnya dilaporkan.

---

## VERIFIKASI WAJIB (sebelum apa pun masuk naskah)

NotebookLM grounded pada korpus, tapi tetap bisa salah baca. Cek langsung ke PDF untuk:

- [ ] **Lin et al. (2024) memakai LUNA16 untuk "benign vs malignant"** — kamu sudah menemukan sejak awal proyek bahwa label LUNA16 adalah *nodule vs non-nodule* (deteksi), bukan malignansi. Pastikan mereka melakukan linkage ke rating LIDC, atau ini salah baca. **Penting** karena angka ini dipakai untuk pola ukuran-dataset.
- [ ] Angka gain tiap paper di tabel Prompt 1
- [ ] Klaim "nol paper menjalankan sanity check" — ini klaim kuat, verifikasi minimal pada paper yang paling mungkin melakukannya (Tempel, Qin)
- [ ] Detail reader study Katar et al. (jumlah pembaca, kasus, dimensi)

**Batas klaim:** semua temuan "tidak ada paper yang…" hanya berlaku untuk korpusmu (~29 paper), bukan literatur secara umum. Tulis dengan hedge eksplisit.

---

## URUTAN EKSEKUSI

```
PRASYARAT: selesaikan angka statistik tanpa CSV (blocker submit)
   │
   ▼
FASE 0 (penulisan)          ← nol kode, nilai tertinggi, kerjakan duluan
   │
   ▼
FASE 1 (faithfulness)       ← celah terbesar; ROAD/deletion-insertion + PGI/PGU
   │
   ▼
FASE 2 (robustness)         ← opsional; SSIM stability lebih dulu dari consistency
   │
   ▼
FASE 3 (LASSO fusi)         ← upaya terakhir, pre-registered, gate penutup
```

**Alasan Fase 0 duluan:** nilai tertinggi, biaya nol, dan tiga temuannya langsung memperkuat naskah tanpa menunggu GPU. Kalau waktu habis di tengah jalan, Fase 0 saja sudah menaikkan kualitas kedua paper.

**Alasan Fase 1 sebelum 2 dan 3:** faithfulness adalah celah yang paling sering diserang reviewer (§Q6(c) brief sebelumnya), dan metrikmu saat ini semua mengukur lokalisasi. Menutup celah itu lebih berharga daripada menambah metrik robustness atau mengejar gain fusi yang literatur sendiri prediksi tidak akan datang.