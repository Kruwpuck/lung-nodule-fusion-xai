# Ledger run03 — 2026-08-22-run03

GOAL: `handoff/GOAL4.md` — fine-tuning bertahap + ensembling backbone untuk menaikkan
cabang CNN. Pra-registrasi lengkap: `handoff/PREREG_run03.md`.

Append-only. Satu baris per aksi, baris baru selalu di bawah. Jangan menyunting baris
lama, jangan mengurutkan ulang, jangan meringkas.

Format:

```
## <ISO8601> | lvl <n> | <AKSI> | <HASIL> | <sha atau ->
```

`AKSI` salah satu dari: `EXEC` `TRAIN` `SEARCH` `BRIEF` `RESEARCH` `HUMAN-RESEARCH`
`BLOCKED` `RESUME`

---

## 2026-08-22T00:00:00Z | lvl 1 | EXEC | F-0 selesai: handoff/PREREG_run03.md ditulis sebelum satu pun GPU jalan. Empat sel unfreeze {0,10,20,100} ditetapkan di muka, sel 100% ditandai kontrol denda protokol bukan kandidat. Tabel pembanding dipatok dari disk (run02/t0_checkpoint_sensitivity.csv pooled; fusion/ablation_summary.csv rerata-fold); angka brief tidak dipakai karena tidak cocok pada cara agregasi mana pun, dan nilai 0,8959 yang brief beri untuk densenet121 adalah nilai densenet201 di disk. Toleransi kontrol negatif F-2 disetel 0,005 karena sebaran radiomics_only antar backbone 0,9301-0,9343 sudah 0,0042. Regenerasi statistik empat-arm Track 2 dibatalkan, diganti pernyataan pemisahan checkpoint di kedua naskah. Loop belum diarmed: kode F-1 belum ada. | -

## 2026-08-22T00:05:00Z | lvl 1 | EXEC | Catatan untuk F1c: Wait-Stage di loop/loop.ps1:88 hanya cocok dengan src.stage_0(3b_fusion|3d_merge_l3|8a_run02_probs|8b_run02_xai). Stage baru src.stage_10b_finetune tidak dikenali, jadi driver akan memutar iterasi kosong tanpa menunggu training selesai. Regex wajib diperluas sebelum 60 sel diantre. | -

## 2026-08-22T09:50:00Z | lvl 1 | EXEC | F1a selesai: src/training/finetune.py (freeze_all, unfreeze_top_modules per child module, apply_bn_eval, build_param_groups) plus --self-check hijau. Empat assert lolos, termasuk kontrol bahwa model.train() memang mengembalikan BN ke mode train. Granularitas nyata diukur dan dicatat di PREREG 2a: convnext_tiny 8 child (uf10 = 1 modul = 51,4% bobot), densenet 12 child (uf10 = 2 modul = 31,1%/38,6%). pytest 9 failed / 144 passed, tidak berubah. | d2f65b2

## 2026-08-22T09:55:00Z | lvl 1 | EXEC | F1b selesai: src/stage_10b_finetune.py. Nested CV dengan GroupShuffleSplit(test_size=0.15, random_state=fold) identik stage_03b_fusion.py:187; fold luar dinilai sekali di akhir. Checkpoint ke checkpoints/run03/, prediksi ke results/run03/preds/, nol timpaan jalur lama. Smoke test 1 epoch per tahap di scratchpad lolos, nol berkas masuk repo. | 82a313a

## 2026-08-22T10:00:00Z | lvl 1 | EXEC | Sel percontohan convnext_tiny fold0 uf10 penuh: inner_val 0,8064, outer_merged 0,8719, outer_full 0,8773, 20+20 epoch, 52 detik. Dua cacat ditemukan dan diperbaiki: (1) convnext_tiny memuat NOL modul BatchNorm (LayerNorm sepanjang jaringan), jadi perlakuan pembekuan BN hanya berlaku pada densenet121/201 -- dicatat PREREG 2b plus kolom n_batchnorm; (2) kolom lr_epoch cuma menampilkan grup head sehingga LR diskriminatif tak pernah terlihat -- diganti lr_head + lr_body_top, terverifikasi 1e-4 dan 3,8e-5. Keluaran percontohan skema lama dipindah ke _run03_pilot_superseded, bukan dihapus. Rerun reproduksi bit-identik. | 15bc8a4

## 2026-08-22T10:05:00Z | lvl 1 | TRAIN | F1c dimulai: 60 sel (3 backbone x 4 varian x 5 fold) berurutan di latar, log runs/2026-08-22-run03/f1_train.log. Estimasi 1-3 jam, bukan berhari-hari seperti dugaan rencana -- epoch convnext 1,3 detik. Resume per sel lewat pemeriksaan npz+checkpoint. | 15bc8a4

## 2026-08-22T10:30:00Z | lvl 1 | EXEC | P4 selesai selagi GPU jalan (nol ketergantungan checkpoint): paper/track2/main.tex:241 ditulis ulang dengan chi2=44,52 dan DUA pasang tak signifikan (0,2733 dan 0,2280) disajikan sebagai temuan granularitas label; Limitations butir 7 diperluas dengan mekanisme penimpaan prediksi; laporan Track 2 par 6.3, tabel status, par 8 butir 12, par 9 butir 6, par 10 dan Lampiran diperbarui jadi tertutup; laporan Track 1 par 8.10 dapat butir E baru (bukti pernah ada lalu ditimpa -- mode kegagalan berbeda dari butir A-D). latexmk track2 bersih, bibtex terverifikasi lewat "Database file #1: refs.bib". | 36ae6e9

## 2026-08-22T11:28:12Z | lvl 1 | TRAIN | F1c selesai: 60/60 sel dalam 97 menit, nol duplikat, tiap sel lengkap 5 fold. | 36ae6e9

## 2026-08-22T11:40:00Z | lvl 1 | EXEC | F1d selesai: artifacts/results/run03/f1_summary.md. HASIL NEGATIF, gate F-1 butir terakhir berlaku. Pemenang inner-val adalah sel kontrol 100% pada KETIGA backbone; inner-val monoton 0% < 10% < 20% < 100% tanpa kecuali. Efek fine-tuning murni negatif di 9 dari 9 sel kandidat (-0,0004 sampai -0,0902). Denda protokol sel100 lawan baseline rerata-fold: convnext -0,0636, densenet201 -0,0603, densenet121 -0,0340; lawan cnn_last run02 -0,0339/-0,0528/-0,0105. Dicatat sebagai BATAS ATAS, bukan estimasi bersih: sel 100% berbeda dari baseline dalam empat hal sekaligus (seleksi inner-val, data latih -15%, jadwal dua tahap, pembekuan BN pada densenet saja). Pembekuan BN tidak menjelaskan pola denda karena convnext yang nol BN justru duduk di antara kedua densenet. Fine-tuning TIDAK diadopsi. | 36ae6e9
