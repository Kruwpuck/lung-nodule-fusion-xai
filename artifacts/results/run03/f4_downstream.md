# F-4 — metrik hilir diturunkan ulang dari checkpoint run03

`run_id` 2026-08-22-run03 · kode `src/stage_10d_xai_run03.py` ·
sumber `run03/xai_metrics_run03.csv` (15 baris), `xai_run03_persample.csv` (900 baris),
`xai_depth_sweep_run03.csv` (18 baris), `efficiency_run03.csv` (3 baris)

Checkpoint berubah, jadi segala yang diturunkan darinya basi. F-2 dan F-3 sudah
menyelesaikan sisi AUC. Ini sisi penjelasan: apakah angka lokalisasi yang dilaporkan
naskah bergerak ketika checkpoint di bawahnya diganti.

Yang dibekukan supaya **hanya** checkpoint yang berbeda: himpunan sampel
(`sample(n=60, random_state=42)`, resep `stage_05_xai`), pemuat patch
(`_load_patch_tensor`, diimpor bukan disalin), primitif metrik (`src.xai.gradcam_utils`,
diimpor bukan disalin), dan aturan situs penjelasan (`_last_spatial_target_layer`, resolver
terkoreksi dari `stage_09d_cam_12.py`). `artifacts/xai/fixed_display_samples.json` dibaca,
nol ditulis.

Sel yang dipakai `uf100` — sel yang sama yang F-3 sebut `honest_nested_cv`. Batasnya tetap
`f2_sensitivity.md` §6: batas bawah cabang CNN jujur, bukan estimasinya.

---

## 1. Prasyarat: checkpoint benar termuat, dibuktikan bukan diasumsikan

Angka lokalisasi yang ambruk punya dua sebab yang tidak bisa dibedakan dari metriknya
sendiri: modelnya memang berbeda, atau `state_dict`-nya gagal muat diam-diam. Yang kedua
**tidak melempar error** — `load_state_dict` yang cocok bentuknya akan menerima bobot apa
pun. Jadi model yang dimuat `stage_10d` di-skor ulang dan dibandingkan dengan
`outer_auc_full` yang `stage_10b` catat saat menyimpan checkpoint itu.

| backbone | fold | AUC sekarang | AUC tercatat | selisih |
|---|---|---|---|---|
| convnext_tiny | 0 | 0,8788 | 0,8788 | +0,0000 |
| convnext_tiny | 4 | 0,8178 | 0,8178 | −0,0000 |
| densenet201 | 0 | 0,8655 | 0,8655 | −0,0000 |
| densenet201 | 4 | 0,8249 | 0,8249 | +0,0000 |
| densenet121 | 0 | 0,8965 | 0,8965 | +0,0000 |
| densenet121 | 4 | 0,8589 | 0,8589 | +0,0000 |

Enam dari enam bit-identik. **Model-modelnya benar dan bekerja** — AUC 0,82 sampai 0,90.
Apa pun yang terjadi pada peta CAM, ia tidak terjadi pada klasifikasinya.

---

## 2. Fold 0: run03 lawan angka terbit, situs yang sama

| backbone | situs | ukuran | pointing terbit | pointing run03 | Δ | dice Δ | energy Δ | peta nol |
|---|---|---|---|---|---|---|---|---|
| convnext_tiny | `features.0.7.2` | 3×3 | 0,7167 | 0,5000 | −0,2167 | −0,0176 | −0,0126 | 0 |
| densenet121 | `features.0.norm5` | 3×3 | 0,7167 | 0,2500 | −0,4667 | −0,0555 | −0,0103 | 0 |
| densenet201 | `features.0.norm5` | 3×3 | 0,7000 | **0,0000** | **−0,7000** | −0,0967 | −0,0240 | 0 |

Ketiga baris fold 0 memakai situs yang identik dengan tabel terbit, jadi selisihnya bukan
selisih situs.

## 3. Kelima fold

| backbone | pointing (rerata ± sd) | dice | energy | peta nol |
|---|---|---|---|---|
| convnext_tiny | 0,5000 ± 0,0808 | 0,1003 | 0,0373 | 0 |
| densenet121 | 0,1033 ± 0,0916 | 0,0227 | 0,0141 | 4 dari 300 |
| densenet201 | **0,0000 ± 0,0000** | 0,0010 | 0,0063 | 0 |

densenet201 nol di kelima fold, bukan satu fold sial. densenet121 turun ke 0,10 dengan
sebaran hampir selebar nilainya sendiri. convnext_tiny turun tapi bertahan.

Empat peta identik nol muncul pada densenet121 fold 4 (4 dari 60). Itu **bukan** situs
yang mati — 56 sampel lain menghasilkan peta hidup di situs yang sama. Dicatat lewat kolom
`n_zero_cam` di tiap baris, bukan dirata-ratakan diam-diam, karena peta nol tetap
menghasilkan Dice terhingga dan lolos tanpa terlihat kalau tidak dihitung.

---

## 4. Eksklusi wajib: apakah ini artefak situs penjelasan

Naskah Track 2 mempra-registrasi aturannya: pointing accuracy tepat nol adalah hasil luar
biasa dan **diperlakukan sebagai dugaan artefak implementasi sampai dieksklusi**. Verdik
`artifact` menuntut ketiga kriteria sekaligus. Sweep seluruh situs kandidat, fold 0, garis
kebetulan 0,0189:

| backbone | 48×48 | 24×24 | 12×12 | 6×6 | 3×3 (kanonik) | 1×1 |
|---|---|---|---|---|---|---|
| convnext_tiny | — | 0,4167 | 0,2000 | 0,0500 | **0,5000** | 0,0000 |
| densenet121 | **0,4167** | 0,2500 | 0,0500 | 0,1000 | 0,2500 | 0,0000 |
| densenet201 | **0,3500** | 0,1500 | 0,1000 | 0,1000 | 0,0000 | 0,0000 |

Putusan terhadap ketiga kriteria:

1. **Situs teresolusi mengeluarkan peta nol di seluruh sampel** — **tidak terpenuhi.**
   densenet201 punya `n_zero_cam` 0 di kelima fold. Petanya hidup, ia cuma mendarat di
   tempat yang salah.
2. **Ada situs kandidat yang memulihkan pointing di atas kebetulan** — **terpenuhi.**
   `features.0.relu0` memberi 0,3500 pada densenet201 dan 0,4167 pada densenet121, jauh di
   atas 0,0189 bahkan setelah dikurangi dua galat baku (≈0,12 pada n=60).
3. Korelasi terhadap model teracak — tidak dijalankan, dan tidak perlu: kriteria 1 sudah
   gagal.

**Verdiknya bukan artefak.** Ini pergerakan nyata pada model yang nyata.

### Yang sebenarnya bergerak: bentuk kurvanya, bukan tingginya

Dibandingkan `track2rev/depth_sweep_12.csv` situs per situs:

| backbone | situs terdangkal | situs kanonik terdalam |
|---|---|---|
| convnext_tiny | +0,0500 | −0,2167 |
| densenet121 | **+0,1500** | −0,3333 |
| densenet201 | **+0,1167** | **−0,7000** |

Di bawah checkpoint terbit, kurva kedalaman memuncak di situs spasial **terdalam** pada
ketiganya. Di bawah checkpoint nested-CV, puncak itu rata atau terbalik pada kedua
DenseNet sementara situs terdangkal justru **naik**. Bobot lokalisasinya berpindah dangkal,
bukan lenyap.

Naskah Track 2 berargumen bahwa yang mengatur lokalisasi CAM adalah lapisan tempat peta
diambil, bukan keluarga arsitektur maupun jumlah parameter. Hasil ini menambahkan sumbu
kedua ke argumen yang sama: **situs terbaik juga bukan properti arsitektur, ia bergerak
bersama protokol pelatihan yang menghasilkan bobotnya.** Tabel CAM lintas-arsitektur yang
memakai satu situs tetap karena situs itu terbaik pada satu checkpoint sedang melaporkan
checkpoint tersebut, bukan jaringannya.

Satu kaitan mekanistik tersedia tapi **tidak diklaim di sini**: kedua DenseNet menerima
pembekuan BatchNorm di F-1 dan situs kanoniknya memang modul `BatchNorm2d`
(`features.0.norm5`), sementara convnext_tiny memuat nol modul BatchNorm dan bertahan.
Pola itu konsisten, dan tetap dua backbone lawan satu — persis batas yang
`f1_summary.md` §4 sudah nyatakan: ketiganya bukan tiga replikasi perlakuan yang sama.
Memisahkannya butuh sel `bn_eval: false`, yang tidak dipra-registrasi dan tidak dijalankan.

---

## 5. Biaya: kontrol lolos, latensi terkonfounding

| backbone | params_M | Δ | GFLOPs | Δ | GPU ms terbit | GPU ms sekarang | rasio |
|---|---|---|---|---|---|---|---|
| convnext_tiny | 27,820 | **0,0** | 1,6486 | **0,0** | 2,518 | 7,091 | 2,82× |
| densenet201 | 18,097 | **0,0** | 1,6132 | **0,0** | 11,324 | 32,413 | 2,86× |
| densenet121 | 6,956 | **0,0** | 1,0649 | **0,0** | 6,718 | 20,644 | 3,07× |

Params dan FLOPs bergerak nol, di-assert bukan dilihat sekilas. Itu kontrol yang benar:
F-1 nol mengubah arsitektur maupun ukuran masukan, jadi pergerakan apa pun di kolom itu
berarti model yang diukur bukan model yang diterbitkan.

**Latensinya tidak dilaporkan sebagai angka run03.** Ia naik 2,8 sampai 3,1 kali di GPU
**dan** 2,0 sampai 2,8 kali di CPU secara seragam. Perlambatan multiplikatif yang muncul di
kedua perangkat sekaligus adalah tanda beban mesin, bukan properti model — dan mesinnya
memang jauh lebih sibuk saat pengukuran ini daripada saat `efficiency_7.csv` ditulis.
Latensi adalah properti arsitektur, ukuran masukan, dan mesin; F-1 nol mengubah dua yang
pertama. Jadi angka latensi terbit **dipertahankan**, dan `efficiency_run03.csv` disimpan
sebagai catatan pengukuran beserta konfoundingnya, bukan sebagai pengganti.

---

## 6. Provenansi

- `xai_metrics_run03.csv` dan `xai_depth_sweep_run03.csv` membawa `run_id`, `commit_sha`,
  `input_size`, `checkpoint`, `checkpoint_mtime` di tiap baris.
- `ensemble.csv` diregenerasi dengan `input_size` plus `probs_mtime_{backbone}` per
  backbone. Nol angka berubah: 24 baris, seluruh kolom numerik bit-identik dengan versi
  terkomit, selisih maksimum 0,0. Timestamp-nya langsung memperlihatkan kedua rezim berasal
  dari tanggal berbeda (5 Agustus lawan 22 Agustus), yang memang gunanya kolom itu.
- Timestamp masukan dicatat per backbone, bukan satu nilai agregat, karena `max()` atas
  ketiganya akan menyembunyikan persis ketidakcocokan yang kolom itu ada untuk memunculkan.
- Nol berkas lama ditimpa. Seluruh keluaran masuk `artifacts/results/run03/`;
  `artifacts/results/xai/xai_metrics.csv` tetap utuh sebagai jejak historis.

---

## 7. Putusan gate F-4

- XAI Layer-CAM plus pointing/IoU/Dice/energy untuk checkpoint baru, memakai
  `fixed_display_samples.json` yang sama — §2, §3. **Lolos.**
- Efisiensi: params/FLOPs terverifikasi tidak berubah, latensi diukur ulang — §5.
  **Lolos, dengan konfounding yang dinyatakan.**
- Statistik empat-arm Friedman/Nemenyi/DeLong **tidak** diregenerasi. Sesuai keputusan
  pra-registrasi: checkpoint legacy nol disentuh, jadi tetap sah. Penggantinya —
  pernyataan pemisahan checkpoint di kedua naskah dan kedua laporan — sudah dikerjakan di
  commit `36ae6e9`. **Lolos.**
- Seluruh CSV hilir membawa `run_id`, `commit_sha`, `input_size`, `checkpoint_mtime` —
  §6. **Lolos.**
- Nol angka naskah Track 1 yang masih merujuk checkpoint lama — **belum.** F-2 sudah
  memutuskan tabel terbit tetap primer dan run03 dilaporkan sebagai analisis sensitivitas,
  dan naskah menyatakan rezimnya di tiap klaim. Yang belum masuk naskah adalah hasil §4:
  klaim eksplainabilitas Track 1 sekarang memikul syarat protokol yang sejajar dengan
  syarat yang F-2 pasang pada klaim ekuivalensi. **Terbuka, dan ini satu-satunya butir
  F-4 yang belum tertutup.**

---

## 8. Yang tidak boleh disimpulkan dari sini

- **Bukan pernyataan bahwa nested CV merusak eksplainabilitas.** Sel `uf100` berbeda dari
  resep terbit dalam empat hal sekaligus (`f1_summary.md` §2b), dan keempatnya bergerak
  bersamaan di sini persis seperti di sana.
- **Bukan pernyataan bahwa densenet201 tidak melokalisasi.** Ia melokalisasi pada 0,3500 di
  `features.0.relu0`, jauh di atas kebetulan. Yang runtuh adalah situs kanoniknya.
- **Pembekuan BatchNorm tidak teruji terpisah**, sama seperti di F-1. Dua backbone yang
  ambruk kebetulan juga dua backbone yang menerimanya, dan dua bukan sampel.
- **Latensi run03 bukan angka model.** Lihat §5.
- Tiga backbone, satu dataset, satu ukuran masukan, 60 sampel per fold. Interval
  kepercayaan untuk pointing accuracy pada n=60 lebar, dan tidak dihitung di sini.
