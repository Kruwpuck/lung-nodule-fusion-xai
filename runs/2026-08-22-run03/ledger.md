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
