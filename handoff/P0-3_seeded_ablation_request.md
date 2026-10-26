# Permintaan P0-3 — ablasi fusi Track 1 dengan radiomics seed-tetap (7 backbone)

Revisi 2026-10-21 setelah tinjauan agen kolaborator (tiga celah ditutup: keluaran per kasus, provenance, jalur checkpoint). **Harus dijalankan di mesin yang punya `data/`, `artifacts/checkpoints/`, torch, xgboost, scikit-learn dan GPU.**

## Masalah yang ditutup
Tabel I naskah Track 1 menguji 21 pasangan (arm fusi vs radiomics-only). Radiomics-only bukan satu model: AUC-nya 0,9301–0,9343 antar-backbone, padahal arm itu tidak menyentuh CNN. Dengan kode sekarang (`733856b`: `random_state=42` di `feature_selection.py:101`; `early_fusion.py:64` default 42; tanpa blok `xgboost` di config; fold dari `folds.json` beku) arm itu **harus** identik di tujuh backbone. Sebaran 0,0042 membuktikan 35 baris kanonik dibuat dengan kode tanpa seed. Penguat: baris kanonik itu tidak memuat `run_id`/`commit_sha`/`input_size`, jadi Tabel I tidak bisa diatribusikan ke commit mana pun. Rerun menutup kedua cacat itu, dan hanya berguna bila ketiga celah di bawah juga tertutup.

Ini analisis sensitivitas **post hoc** atas ablasi yang ada, bukan pencarian fusi baru; pra-registrasi run03 bagian 5 tetap tertutup.

## Sebelum menjalankan
1. **Commit patch lebih dulu** (`src/stage_03b_fusion.py`, `configs/config_p03_seeded.yaml`), supaya `commit_sha` di keluaran menunjuk ke kode yang benar-benar dipakai.
2. **Junction untuk checkpoint** (jangan alihkan `paths.checkpoints` ke folder kosong: backbone tanpa `fold0_best.pt` di-skip, hasilnya CSV kosong). Pada Windows cmd, ganti `<REPO>`:
```
mkdir "<REPO>\artifacts\checkpoints_rerun"
cd /d "<REPO>\artifacts"
for %M in (convnext_tiny densenet121 densenet201 googlenet inception_resnet_v2 inceptionv3 xception) do mklink /J "checkpoints_rerun\%M" "checkpoints\%M"
```
CNN dibaca lewat junction; checkpoint fusi intermediate yang baru ditulis ke `checkpoints_rerun\{backbone}_fusion_intermediate` (folder nyata). Pohon `checkpoints\` tidak tersentuh. Tetap cadangkan `artifacts\checkpoints\*_fusion_intermediate` dan `artifacts\results\fusion` sebagai jaring kedua.
3. Jangan menyentuh `artifacts/splits/folds.json` dan `artifacts/results/preds/`.

## Patch yang ada di repo ini (tambahan murni, mati bila config tidak menyalakannya)
- `save_persample: true` di config menulis `fusion/persample_{backbone}.csv`: `patient_id`, `nodule_idx`, (`scan_id` bila ada), `fold`, `label`, `prob_cnn`, `prob_radiomics`, `prob_fusion_early`, `prob_fusion_late`, `prob_fusion_intermediate` per nodul, urutan sama dengan DeLong pooled. Dibungkus try/except; kegagalan hanya menulis peringatan `[persample]` di log.
- `run_id` di config menambahkan kolom `run_id`, `commit_sha`, `input_size`, `checkpoint_mtime` ke `ablation_summary.csv`, `delong_fusion.csv`, dan semua `persample_*.csv`. `checkpoint_mtime` = waktu ubah `fold0_best.pt` checkpoint CNN arm A. Kolom `gpu_device` dan `torch_version` ikut ditambahkan (2026-10-26), supaya perangkat tercatat di keluaran, bukan hanya diatestasi. Dengan `config.yaml` biasa tidak ada kolom tambahan (keluaran lama tidak berubah).
- Sudah lolos `py_compile` dan uji fungsi `_provenance`; **belum dijalankan penuh** (tidak ada torch di mesin penulis).

## Menjalankan
```
python -m src.stage_03b_fusion --config configs/config_p03_seeded.yaml
```
Keluaran di `artifacts/results/rerun_seeded/fusion/`. Hasil lama di `artifacts/results/fusion/` tidak tersentuh. 7 backbone × 5 fold fusi intermediate (maks. 50 epoch, patience 10); waktu belum diketahui.

## Uji terima (batalkan rerun bila gagal)
Tujuh baris `radiomics_only` harus cocok **sampai digit terakhir** dengan `radiomics_only_seeded.csv` (`artifacts/results/size_baseline/` di mesin kolaborator):

| | nilai |
|---|---|
| rata-rata fold ± SD | 0,9326 ± 0,0156 (0,93255 ± 0,01559) |
| pooled | 0,9320 (0,93195) |
| per fold | 0,9347 · 0,9531 · 0,9052 · 0,9309 · 0,9388 |
| fitur terpilih | 14–27 dari 1130 (`mutual_info_classif`) |
| kohort | 1366 nodul, 607 pasien |

Jika meleset walau sedikit, seed bukan satu-satunya sumber nondeterminisme: hentikan dan laporkan, jangan menyetel apa pun agar cocok. Cek juga: nilai `fs_method` pada baris radiomics; jumlah baris (`ablation_summary` 7 × 5 arm × 5 fold, `delong_fusion` 7 × 3); dan `best_single_arm` yang bukan radiomics di backbone mana pun.

## Yang dikirim balik
Isi `artifacts/results/rerun_seeded/fusion/`: `ablation_summary.csv`, `delong_fusion.csv`, tujuh `persample_*.csv`, plus log run (baris `fs_method=` dan `[persample]`).

## Yang dilakukan penulis setelah file diterima
Mengganti Tabel I (p DeLong, p Holm, AUC rata-rata) dan angka terkait di abstrak, IV-A, Discussion, Limitations; menjalankan Holm atas 21 uji baru (`stage_12a` A2 diarahkan ke file baru); menghapus kalimat Limitations "recomputing Table I … was not done"; menyatakan di naskah bahwa fusi early dan intermediate dilatih ulang dan bahwa analisis ini post hoc.

## Apa yang boleh dan tidak boleh berubah (supaya tidak ada panik atau penyetelan)
- **Klaim utama tidak berisiko.** Di `delong_fusion.csv` lama, `fusion_significantly_better` bernilai False di 33 dari 33 baris. p DeLong late fusion 0,204 · 0,335 · 0,377 · 0,679 · 0,799 · 0,888 · 0,902, jauh dari tepi mana pun. Pembanding yang diperketat tidak bisa menciptakan kemenangan fusi yang tidak ada.
- **Yang bisa bergeser adalah ketajaman temuan negatif** ("early/intermediate signifikan lebih buruk"). Setelah Holm (Tabel I sekarang), yang paling dekat 0,05 di sisi signifikan: InceptionV3 early 0,044, GoogLeNet early 0,039, DenseNet121 early 0,031; di sisi tidak signifikan: DenseNet201 early 0,070, DenseNet121 intermediate 0,076, InceptionV3 intermediate 0,087. (p mentah 0,064 untuk ConvNeXt-Tiny intermediate dan 0,0148 untuk Inception-ResNet-v2 early sudah tidak signifikan setelah Holm, 0,512 dan 0,133, jadi bukan yang di tepi.)
- Bila satu-dua p bergeser, naskah mengikuti angka baru. Tidak ada pelatihan ulang kedua dan tidak ada penyetelan agar hasil terlihat lebih baik.
- Fusi intermediate dilatih ulang, jadi nilainya dapat berbeda dari Tabel I lama walau seed radiomics sama. Selisih itu dilaporkan.

## Catatan kohort (belum terverifikasi)
`_load_merged` membuang nodul berkunci (`patient_id`, `nodule_idx`) ganda. Agen kolaborator menyebut 42 nodul; naskah menulis 25 nodul dikeluarkan dari 1391 menjadi 1366. Keduanya konsisten bila 42 dihitung di seluruh 2138 nodul (termasuk median 3) dan 25 di antaranya berlabel biner. Mohon konfirmasi dari log `_load_merged` (peringatan "Dropping … nodules").

Jika waktu tidak cukup sebelum 30 Okt 2026, naskah tetap valid dengan pernyataan keterbatasan yang sekarang.
