# GOAL run03 — fine-tuning bertahap + ensembling backbone

```
RUN_ID           = 2026-08-22-run03
LEDGER           = runs/2026-08-22-run03/ledger.md
PRA-REGISTRASI   = handoff/PREREG_run03.md      <- baca ini lebih dulu, mengikat
SUMBER TUGAS     = docs/revisi/rev3/implement2.md
GPU_HOST         = PC ini (RTX 3060)
MAX_ITER         = 20
RESERVED_ENV     = artifacts/results/_baseline_pre_rev2/, artifacts/results/_leaky_pre_nestedcv/,
                   artifacts/splits/folds.json, artifacts/checkpoints/{backbone}/,
                   artifacts/results/preds/, .venv
```

## Tujuan

Menaikkan mutu cabang CNN, yang selama ini jadi hambatan fusi, lewat dua mekanisme
yang belum pernah dijalankan: two-stage fine-tuning dengan BatchNorm dibekukan, dan
ensembling backbone per fold luar. **Bukan mekanisme fusi baru** — pencarian fusi
sudah ditutup; yang dikejar mutu masukannya.

Seluruh prediksi, aturan keputusan, tabel pembanding, dan larangan ada di
`handoff/PREREG_run03.md` dan **tidak boleh diubah setelah hasil dilihat**. Dokumen
ini hanya urutan tahap dan gate-nya.

---

## Tahap dan gate

### F-0 — pra-registrasi  ✔ SELESAI
`handoff/PREREG_run03.md` ditulis sebelum satu pun GPU jalan.

### F-1 — two-stage fine-tuning
Berkas: `src/training/finetune.py` (4 fungsi + `--self-check`) dan
`src/stage_10b_finetune.py` (`--backbone --fold --unfreeze {0,10,20,100}`).

- [ ] `--self-check` hijau sebelum satu pun sel jalan
- [ ] Sel percontohan `convnext_tiny fold 0 unfreeze 10` dijalankan sendirian sampai
      selesai dan diperiksa (jumlah parameter trainable, BN eval, `inner_val_auc`
      berbeda dari `outer_auc`) sebelum 59 sisanya diantre
- [ ] Tabel penuh 3 backbone × 4 varian × 5 fold = **60 baris** tercetak
- [ ] Varian terbaik per backbone ditentukan dari **inner-val**, bukan fold luar
- [ ] **Keempat varian dilaporkan**, termasuk yang kalah
- [ ] Tiga selisih dilaporkan terpisah: 100% lawan baseline lama (denda protokol),
      10%/20% lawan 100% (efek fine-tuning murni), terbaik lawan baseline lama
      (efek gabungan)
- [ ] Kalau nol varian menaikkan `cnn_only` di atas sel 100%: laporkan, lanjut ke F-3
      tanpa fine-tuning, sel 100% jadi checkpoint yang dibawa

### F-2 — turunkan ulang ablasi fusi
Berkas: `configs/config_run03.yaml` + `cnn_ckpt_subdir` di `src/stage_03b_fusion.py`.

- [ ] **Kontrol negatif diperiksa lebih dulu**, sebelum membaca angka fusi mana pun:
      `radiomics_only` dalam ±0,005 dari tabel 4b pra-registrasi
- [ ] Tabel penuh 3 backbone × 5 arm × 5 fold
- [ ] DeLong tercetak dengan p-value: tiap arm fusi lawan `radiomics_only` dan lawan
      `cnn_only` baru

### F-3 — ensembling backbone
Berkas: `src/stage_10c_ensemble.py`.

- [ ] **Urutan kasus antar backbone dalam fold yang sama diverifikasi identik dan
      buktinya dicetak** sebelum satu pun probabilitas dirata
- [ ] AUC ensemble per fold tercetak
- [ ] DeLong lawan backbone tunggal terbaik dan lawan `radiomics_only`

Ensembling **fold** tidak mungkin: tiap fold punya pasien validasi berbeda, nol kasus
punya prediksi dari kelima model. Yang sah ensembling backbone dalam fold luar sama.

### F-4 — turunkan ulang metrik hilir
- [ ] XAI: Layer-CAM + pointing accuracy/IoU/Dice/energy untuk checkpoint baru,
      memakai `artifacts/xai/fixed_display_samples.json` yang sama (dibaca saja)
- [ ] Efisiensi: params/FLOPs tidak berubah, latensi diukur ulang
- [ ] **Statistik empat-arm Friedman/Nemenyi/DeLong TIDAK diregenerasi.** Checkpoint
      legacy tidak disentuh, jadi tetap sah. Gantinya: **pernyataan pemisahan
      checkpoint di kedua naskah dan kedua laporan** — Track 2 memakai checkpoint
      legacy, Track 1 memakai run03, berikut alasannya
- [ ] Seluruh CSV hilir bawa `run_id`, `commit_sha`, `input_size`, `checkpoint_mtime`
- [ ] Nol angka naskah Track 1 yang masih merujuk checkpoint lama

---

## Perangkap yang sudah diketahui

1. **`apply_bn_eval` harus di DALAM loop epoch.** `model.train()` mengembalikan BN ke
   mode train tiap epoch. Dipanggil sekali sebelum loop = pembekuan batal di epoch
   pertama, senyap. Ada assert-nya di `--self-check`.
2. **Unfreeze per child module**, bukan slice `named_parameters()` — slice
   per-parameter bisa membuka weight tanpa bias.
3. **`Wait-Stage` di `loop/loop.ps1:88`** hanya cocok dengan
   `src.stage_0(3b_fusion|3d_merge_l3|8a_run02_probs|8b_run02_xai)`. Stage baru
   `src.stage_10b_finetune` dan `src.stage_10c_ensemble` **tidak dikenali**, jadi
   driver akan memutar iterasi kosong tanpa menunggu training selesai. Regex wajib
   diperluas sebelum 60 sel diantre.
4. **`tests/` masih deny.** Unit test untuk bug BatchNorm tidak bisa ditulis ke sana;
   gantinya `--self-check` ber-assert di modulnya sendiri.
5. **Verifikasi PID hidup dari sesi baru** sebelum melaporkan status training. Pada
   run01 lima iterasi berturut "gagal" ternyata batas pemakaian API, bukan agent mati.

---

## Kalau hasilnya negatif

Status tetap `done`, bukan gagal. Pencarian performa ditutup permanen. Dilaporkan apa
adanya: dua mekanisme terakhir yang belum dicoba pun tidak membalikkan kesimpulan.
Itu **memperkuat** klaim "radiomics setara atau mengungguli fusi pada n≈1400", karena
menunjukkan kesimpulan bertahan setelah upaya perbaikan yang wajar habis.

Agent dilarang memperluas ruang pencarian atas inisiatif sendiri.

---

## Yang tetap dikerjakan berbarengan, tidak menunggu GPU

Penulisan ulang `paper/track2/main.tex:241` beserta §6.3 laporan Track 2 (koreksi
klaim keterpisahan antar-arm) dan penulisan positioning korpus. Keduanya koreksi
kesalahan pelaporan yang berdiri sendiri dan nol ketergantungan pada checkpoint.
