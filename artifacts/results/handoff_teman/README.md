# Serahan untuk kolaborator Track 1 — 21 Oktober 2026

Lima CSV, dua keperluan. Semuanya baca-salin; nol hasil terbit yang ditimpa untuk membuatnya.

Kohort seluruh berkas di sini: **1366 nodul, 607 pasien**. Bukan 1391. `_load_merged`
(`src/stage_03b_fusion.py:38`) membuang 42 nodul berkunci `(patient_id, nodule_idx)` ganda,
karena `radiomics.parquet` tidak punya `scan_id` untuk memilah pasien multi-scan. Semua
arm memakai `_load_merged` yang sama, jadi pembandingnya adil.

## 1. Pembanding radiomics seed-tetap

| berkas | isi |
|---|---|
| `radiomics_only_seeded.csv` | satu baris ringkasan |
| `radiomics_only_seeded_probs.csv` | 1366 baris, `patient_id,nodule_idx,fold,label,rad_prob` |

Dihasilkan `src/stage_09j_radiomics_seeded.py`, commit `59d2fda`, `run_id=2026-10-16-radiomics-seeded`.

```
pooled      0.93195
mean-fold   0.93255 +- 0.01559
per fold    0.9347 0.9531 0.9052 0.9309 0.9388
terpilih    14..27 dari 1130 (mutual_info_classif)
```

Kedua angka itu dihitung ulang dari `..._probs.csv` setelah ditulis, bukan dari array di
memori, lewat assert di dalam stage tersebut. Jadi tabel apa pun yang dibangun di atas
file probabilitas ini akan mereproduksi AUC yang sama.

Resepnya identik dengan arm `radiomics_only` di `stage_03b_fusion`: `_select_fold_features`
lalu `train_early_fusion_xgboost` dengan parameter default (`configs/config.yaml` nol blok
`xgboost`), seed 42, fold dari `artifacts/splits/folds.json` yang beku.

**Kenapa ini ada.** Tabel I menerbitkan `radiomics_only` tujuh kali, satu per backbone,
0.9301..0.9343. Arm itu nol menyentuh CNN, jadi ketujuh angka tidak bisa berbeda karena
backbone — mereka berbeda karena seleksi fitur belum ter-seed saat run itu dijalankan.
Commit `733856b` ("fix: seed mutual_info_classif so feature selection is reproducible")
menutup itu, dan dengan kode pasca-commit tersebut ketujuh baris wajib identik. Nilai
ter-seed 0.93255 jatuh 0.0002 dari rerata sebaran terbit (0.9324), jadi sebaran 0.0042 itu
derau seed, bukan efek backbone.

Tabel II/Fig. 2 sekarang memakai 0.9336202, yang adalah refit `radiomics_only` milik
densenet121 (`run02/delong_run02.csv`, `run03/tost_run03.csv` kolom `auc_b`). Ganti dengan
0.93195 di atas supaya ketiga backbone diuji melawan satu pembanding yang sama.

## 2. Arsip pra-nested-CV, untuk tabel selisih bias seleksi

| berkas | asal |
|---|---|
| `leaky_pre_nestedcv_ablation_summary.csv` | `artifacts/results/_leaky_pre_nestedcv/fusion/` |
| `leaky_pre_nestedcv_delong_fusion.csv` | idem |
| `post_nestedcv_ablation_summary.csv` | `artifacts/results/fusion/ablation_summary.csv` |

Arsip pra-nested-CV tidak ada di working tree D:; disalin dari
`lung-nodule-fusion-xai-backup-F`, mtime 4 Agustus 2026. Pada kedua berkas ablasi, baris
dengan `condition` selain kosong atau `plain` adalah arm Rev1 5a/5c dan harus disaring
sebelum dibandingkan.

Selisih per backbone, rata-rata fold, fold yang sama di kedua run:

| backbone | cnn_only | radiomics_only | fusion_early | fusion_intermediate | fusion_late |
|---|---|---|---|---|---|
| convnext_tiny | +0.0000 | +0.0034 | +0.0015 | **-0.0040** | +0.0007 |
| densenet201 | +0.0000 | -0.0036 | -0.0003 | **-0.0281** | -0.0010 |
| googlenet | +0.0000 | +0.0011 | -0.0007 | **-0.0275** | +0.0001 |
| inception_resnet_v2 | -0.0000 | +0.0005 | -0.0011 | **-0.0379** | +0.0006 |
| inceptionv3 | +0.0000 | +0.0017 | +0.0003 | **-0.0098** | +0.0011 |
| xception | +0.0000 | +0.0020 | +0.0003 | **-0.0218** | +0.0009 |
| densenet121 | **+0.0637** | +0.0019 | -0.0045 | -0.0081 | +0.0095 |

`fusion_intermediate` enam backbone tanpa densenet121: rerata **-0.0215**, rentang
-0.0379..-0.0040. Itu mereproduksi angka 0.0210 yang terbit di
`docs/laporan/LAPORAN_TRACK1_FUSION_XAI.md:132`.

Dua hal wajib ikut dilaporkan:

**densenet121 keluar dari rerata.** `cnn_only`-nya bergerak +0.0637, dan perbaikan
nested-CV di `_train_fusion_fold` secara struktural tidak bisa menyentuh `cnn_only` — arm
itu nol melewati fungsi tersebut, dan enam backbone lain bergerak +0.0000 persis. Jadi
checkpoint densenet121 berganti antara dua run, bukan terkena efek seleksi. Konsisten
dengan peringatan provenance untuk `preds/densenet121_fold*.npz` di
`_baseline_pre_rev2/README.md`. Laporkan enam, catat yang ketujuh di catatan kaki.

**`radiomics_only` bergerak -0.0036..+0.0034.** Laporan sekarang menulis "tiga arm lain
bergerak di bawah 0.0005"; maksimalnya 0.0036. Dan arm itu nol alasan bergerak sama
sekali, karena ia tidak melewati `_train_fusion_fold`. Pergerakan itu adalah derau seleksi
fitur tak ter-seed yang sama dengan §1 di atas, terlihat dari arah yang berbeda.

## Reproduksi

```
python -m src.stage_09j_radiomics_seeded --self-check
python -m src.stage_09j_radiomics_seeded
```

Butuh `pyarrow` dan `xgboost`. Kalau data di drive lain, arahkan `paths.interim` (folder
`labels.csv`) dan `paths.features` (`radiomics.parquet`) di config lalu `--config <file>`.

Tabel selisih dihitung langsung dari ketiga CSV ablasi di folder ini, nol kode tambahan
yang dibutuhkan.
