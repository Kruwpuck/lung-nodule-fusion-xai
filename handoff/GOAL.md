# GOAL

Jalankan Langkah 3 dari `docs/revisi/rev2/PLAN_TRACK1_REV2.md`: latih dan skor arm
fusi yang sudah direbalans (`concat`, `branch_norm`, `gmu`) pada 3 backbone terpilih,
di bawah dua kondisi regularizer, lalu laporkan apakah ada arm fusi yang akhirnya
mengalahkan radiomics-only.

Nol kode baru pada `src/`. Kodenya sudah selesai sejak Rev1 tugas 5a, 5b dan 5c.
Yang dijalankan hanya dua proses dengan dua berkas konfigurasi yang sudah disiapkan.

## Perintah

```
.venv\Scripts\python.exe -m src.stage_03b_fusion --config configs/config_l3_plain.yaml
.venv\Scripts\python.exe -m src.stage_03b_fusion --config configs/config_l3_reg.yaml
```

Kondisi A (`config_l3_plain.yaml`): `modality_dropout_rate: 0.0`, `aux_loss_weight: 0.0`.
Kondisi B (`config_l3_reg.yaml`): `modality_dropout_rate: 0.2`, `aux_loss_weight: 0.3`.

Kedua konfigurasi memakai `tracks.track1.backbones: [convnext_tiny, densenet201, densenet121]`
dan `paths.results` yang berbeda. Pemisahan `paths.results` itu wajib dan bukan gaya:
`src/stage_03b_fusion.py:271` menulis `ablation_summary.csv` ke `cfg.paths.results/fusion/`
tanpa memberi nama unik per kondisi, jadi `paths.results` yang sama membuat proses kedua
menimpa hasil proses pertama tanpa peringatan.

Jalankan berurutan, bukan paralel. Satu GPU RTX 3060 12 GB, dua proses training
serentak akan kehabisan memori.

## Kriteria selesai

Kelimanya harus terpenuhi. Kalau salah satu gagal, run ini belum selesai.

1. `artifacts/results/l3_plain/fusion/ablation_summary.csv` ada dan berisi **105 baris**
   (3 backbone x 5 fold x 7 arm). Ketujuh arm itu: `cnn_only`, `radiomics_only`,
   `fusion_intermediate`, `fusion_intermediate_branch_norm`, `fusion_intermediate_gmu`,
   `fusion_early`, `fusion_late`.
2. `artifacts/results/l3_reg/fusion/ablation_summary.csv` ada dan berisi **105 baris**,
   dengan tiga arm intermediate bersufiks `_moddrop_auxloss`.
3. Kedua `delong_fusion.csv` ada, dan kolom `delong_p` tidak berisi satu pun NaN.
4. Nol nilai NaN pada kolom `auc` di kedua `ablation_summary.csv`.
5. `pytest tests/ -q` tetap **9 failed, 138 passed**. Angka itu baseline yang diukur
   pada commit `c1b14d0` sebelum run ini dimulai. Boleh membaik, tidak boleh memburuk.

## Pertanyaan yang dijawab run ini

Setelah bug `input_size` diperbaiki dan nested CV dipasang, apakah `branch_norm` atau
`gmu` mengalahkan radiomics-only? Angka pembanding ada di
`artifacts/results/fusion/ablation_summary.csv` (hasil Langkah 1, 7 backbone, arm lama):
di sana radiomics-only sekitar 0.929 sampai 0.933 dan tidak satu pun arm fusi menang
signifikan. `fusion_late` seri, `fusion_intermediate` dan `fusion_early` kalah signifikan.

Kalau nol dari sekian pasang tetap mendukung fusi setelah perbaikan resolusi, nested CV,
dan gating, temuan negatif Track 1 justru menjadi jauh lebih kuat. Itu hasil yang sah dan
bukan kegagalan run. Jangan mengejar angka yang lebih tinggi dengan mengubah protokol.

## Batasan

- Dilarang menyentuh: `.venv` selain menjalankannya, dan `data/raw/`.
- Dilarang menjalankan Track 2 apa pun. `track2_sweep.enabled: false` di kedua konfigurasi.
- Maks iterasi: 15. Setelah itu loop berhenti sendiri dan menulis `handoff/BLOCKED.md`.
- Satu proses training pada satu waktu.

## Larangan keras

- Dilarang mengedit atau menghapus berkas di `tests/`.
- Dilarang mengubah definisi metrik, ambang kriteria selesai, atau isi berkas GOAL ini.
- Dilarang mengubah `configs/config.yaml`. Langkah 3 memakai dua konfigurasi terpisah
  justru supaya konfigurasi utama tetap utuh.
- Dilarang `git push`, `git reset --hard`, `git clean -fdx`, `rm -rf`, `Remove-Item -Recurse`,
  dan flag `--force` dalam bentuk apa pun.
- Dilarang menghapus atau menimpa hasil lama di `artifacts/results/fusion/`,
  `artifacts/results/_baseline_pre_rev2/`, `artifacts/results/_leaky_pre_nestedcv/`,
  atau checkpoint mana pun di `artifacts/checkpoints/`.
- Dilarang menambah dependensi baru tanpa alasan tertulis di ledger.

## Catatan yang sudah diketahui, jangan diteliti ulang

- `pymrmr` tidak terpasang. Seleksi fitur yang benar-benar berjalan adalah
  `mutual_info_classif`, dan kolom `fs_method` sudah melaporkannya dengan jujur.
  Ini bukan bug dan bukan tugas run ini.
- Nested CV sudah aktif tanpa syarat di `_train_fusion_fold`, bukan di balik flag config.
- Aktivasi gate GMU tersimpan di `model.last_branch_norms` sebagai `img_gate` dan
  `rad_gate`, tetapi `stage_03b_fusion.py` belum menuliskannya ke CSV mana pun.
  PLAN rev2 bagian 5 memintanya untuk manuskrip. Itu butuh kode baru, jadi berada
  di luar kriteria selesai run ini. Catat sebagai pekerjaan lanjutan, jangan kerjakan.
