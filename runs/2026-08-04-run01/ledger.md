# Ledger — run 2026-08-04-run01

Append-only. Satu baris per aksi, baris baru selalu di bawah. Jangan menyunting
baris lama, jangan mengurutkan ulang, jangan meringkas.

Format:

```
## <ISO8601> | lvl <n> | <AKSI> | <HASIL> | <sha atau ->
```

`AKSI` salah satu dari: `EXEC` `SEARCH` `BRIEF` `RESEARCH` `HUMAN-RESEARCH` `BLOCKED` `RESUME`

GOAL: Langkah 3 rev2, arm fusi rebalanced pada 3 backbone, dua kondisi regularizer.
Kriteria selesai lengkap ada di `handoff/GOAL.md`.

---

## 2026-08-04T00:00:00Z | lvl 1 | RESUME | scaffold loop selesai; baseline pytest 9 failed 138 passed; konfigurasi l3_plain dan l3_reg diverifikasi memuat 3 backbone dan 3 arm dengan paths.results terpisah | c1b14d0

## 2026-08-04T00:10:00Z | lvl 1 | EXEC | langkah 1/3 dimulai: `.venv\Scripts\python.exe -m src.stage_03b_fusion --config configs/config_l3_plain.yaml` dijalankan background, proses python.exe hidup (PID terlihat di tasklist), log baru berisi 2 baris warning awal (dropping 42 nodule ambigu, pymrmr tidak terpasang jadi fs_method mutual_info_classif). Belum selesai, training makan waktu berjam-jam, keluar iterasi ini. | -

## 2026-08-04T16:39:00Z | lvl 1 | RESUME | langkah 1/3 masih berjalan: PID 23608 hidup (tasklist), mem 1.586.712 K, CPU time 0:01:08. l3_plain.log fold 0 convnext_tiny: n_train=1074 n_val=292 fs_method=mutual_info_classif selected 50 to 23 fitur, XGBoost early_fusion terlatih, checkpoint artifacts/checkpoints/convnext_tiny_fusion_intermediate/fold0_best.pt baru ditulis. cudnnException warning (non-fatal, fallback plan) muncul berulang tapi training lanjut. artifacts/results/l3_plain belum berisi CSV output (belum ada fold selesai penuh). Tidak memulai proses baru, keluar iterasi ini. | -
