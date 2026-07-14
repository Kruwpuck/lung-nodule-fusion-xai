# Prompt untuk Riset Dataset di Claude Chat

> Copy-paste blok di bawah ini ke Claude chat (nyalakan **web search**).
> Prompt ini sudah dimuati konteks nyata dari pipeline-mu, jadi jawabannya tidak akan generik.

---

Saya sedang mengerjakan riset klasifikasi malignancy nodul paru dari CT scan (skripsi/paper). Saya butuh bantuan **mencari dan mengevaluasi dataset**, bukan menulis kode. Tolong gunakan web search dan **sertakan sumber untuk setiap klaim angka**.

## Kondisi saya sekarang

**Data yang SUDAH saya punya di disk lokal:**
- **LIDC-IDRI lengkap**: 1018 scan (1010 folder DICOM), diakses lewat `pylidc`. Anotasi 4 radiolog dengan rating malignancy 1–5 dan kontur per-nodul.
- **LUNA16 lengkap**: `annotations.csv` (1186 nodul), `candidates.csv` (551.065 kandidat, 549.714 di antaranya kelas negatif), `candidates_V2.zip`, subset0–9 (kecuali subset2). Sudah saya verifikasi: **888/888 `seriesuid` LUNA16 cocok 100% dengan scan LIDC lokal saya**, jadi saya bisa crop patch dari DICOM asli pakai koordinat LUNA16 tanpa download apapun lagi.

**Preprocessing saya:** tiap nodul di-crop jadi window fisik **64×64×16 mm**, di-resample ke **1×1×1 mm isotropic**, diisi −1000 HU di luar batas scan. Output tetap `(16, 64, 64)` voxel. Split **patient-level stratified 5-fold**.

**Task sekarang:** biner benign vs malignant. Label = median rating 4 radiolog; median > 3 → malignant, median < 3 → benign, **median == 3 dibuang (956 nodul, 36% dari 2651)**.

**Yang mau saya ubah:**
1. Menambahkan **grading malignancy** (ordinal 1–5 atau 3-kelas benign/indeterminate/malignant) supaya 956 nodul indeterminate tidak terbuang.
2. Menambahkan kelas **"no-nodule" / normal parenchyma** (bukan "non-cancer" — saya paham benign juga non-cancer).

## Yang saya minta

### A. Sumber kelas "no-nodule / normal parenchyma"

Saya sudah tahu saya bisa pakai LUNA16 `candidates_V2` kelas 0 sebagai **hard negative** (false positive dari detektor: pembuluh darah, fissure, jaringan parut). Tolong **kritik pendekatan itu** dan carikan alternatif/pelengkap:

- Adakah dataset lain yang menyediakan patch paru normal terverifikasi (bukan sekadar "tidak ada anotasi")?
- Apakah **LIDC XML `<nonNodule>` markings** (radiolog secara eksplisit menandai "ini BUKAN nodul") bisa diakses? Di mana download-nya, berapa besar, berapa banyak markingnya? Saya tidak punya file XML-nya — DICOM saya image-only.
- Adakah literatur yang **memperingatkan** soal menggabungkan deteksi (ada nodul?) dan karakterisasi (nodul ini ganas?) dalam satu classifier? Saya curiga ini menggelembungkan metrik. Cari argumen dari **kedua sisi**.

### B. Validasi eksternal dengan label patologi

Kelemahan terbesar riset saya: **label LIDC adalah opini radiolog, bukan patologi.** Carikan kohort dengan **malignancy terkonfirmasi patologi/biopsi/follow-up** untuk validasi eksternal:

- SPIE-AAPM-LungX, NSCLC-Radiomics, LungCT-Diagnosis, LNDb, NLST, DLCS/Duke, dan apapun yang lebih baru (2024–2026)
- Untuk masing-masing: **ukuran, jenis label, ketebalan slice, rute akses (publik / DUA / dbGaP / butuh supervisor), lisensi, ukuran download**

### C. ⚠️ Cek kontaminasi (PENTING — jangan dilewati)

**Banyak dataset CT paru diturunkan dari LIDC-IDRI.** Kalau "validasi eksternal" saya ternyata berisi pasien yang sama dengan data training saya, validasinya **tidak sah** dan itu cacat fatal.

Untuk **setiap** dataset yang kamu rekomendasikan, jawab eksplisit:
- Apakah ini turunan/subset/re-annotation dari LIDC-IDRI atau LUNA16?
- Adakah irisan pasien dengan LIDC-IDRI?
- Kalau overlap, bagaimana cara mendeteksinya (PatientID? SeriesInstanceUID? hash gambar?)

### D. Justifikasi akademis untuk grading

Cari **paper nyata (dengan sitasi)** yang:
- memakai skema **ordinal / multi-class** pada rating malignancy LIDC, bukan biner exclude-3 — apa metrik yang mereka pakai (MAE? Quadratic Weighted Kappa? macro-AUC?), dan berapa hasilnya
- **mengkritik** skema buang-median-3, ATAU membelanya
- pernah menambahkan kelas normal/no-nodule ke classifier malignancy — apa yang terjadi pada metriknya?

## Format jawaban

1. **Tabel perbandingan dataset**: nama | ukuran | jenis label | patologi-confirmed? | turunan LIDC? | rute akses | ukuran download | rekomendasi (pakai / lewati / kenapa)
2. **Rekomendasi peringkat** untuk kelas no-nodule, dengan alasan
3. **Rekomendasi peringkat** untuk validasi eksternal patologi
4. **Verdict akademis**: apakah menambah kelas no-nodule ke classifier malignancy itu ide bagus atau buruk — beri argumen **kedua sisi**, lalu pilih satu dan pertahankan
5. **Sitasi paper** untuk skema grading ordinal
6. **Peringatan/jebakan** yang belum saya sebut

Jangan mengarang angka. Kalau tidak yakin, bilang tidak yakin dan sebutkan apa yang perlu diverifikasi.
