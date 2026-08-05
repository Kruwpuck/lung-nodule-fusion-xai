# Ledger run02 — 2026-08-04-run02

GOAL: handoff/GOAL2.md — kuantifikasi keunggulan gabungan `fusion_late`
(AUC setara radiomics, unggul atas cnn_only, unggul pada explainability).

Satu baris per iterasi, format:
`## <ISO8601> | lvl <n> | <AKSI> | <HASIL> | <sha atau ->`

## 2026-08-05T08:37:03Z | lvl 1 | EXEC | batas pemakaian API tercapai (tunggu ke-1); level dan iterasi tidak diubah | -

## 2026-08-05T08:57:06Z | lvl 1 | EXEC | batas pemakaian API tercapai (tunggu ke-2); level dan iterasi tidak diubah | -

## 2026-08-05T09:15:00Z | lvl 1 | EXEC | langkah 1/3 selesai: stage_08a_run02_probs menulis 3 npz x 1366 baris (convnext_tiny, densenet201, densenet121) | 5220afb

## 2026-08-05T09:25:00Z | lvl 1 | EXEC | langkah 2/3 selesai: stage_08b_run02_xai menulis xai_metrics_fusion.csv (12 baris), xai_samples_used.csv, 3 shap_beeswarm png, shap_provenance.csv | 5220afb

## 2026-08-05T09:30:00Z | lvl 1 | EXEC | langkah 3/3 selesai: stage_08c_run02_gates, keenam gate LULUS, exit 0 | 5220afb

Keluaran `python -m src.stage_08c_run02_gates --check`, apa adanya:

```
  LULUS  G-0 verifikasi T-0: 3 backbone, diharapkan 3; kolom hilang []; sel kosong 0
         convnext_tiny: late_best 0.9300 vs rad 0.9336 (p 0.4555) | late_last 0.9268 (p 0.0246) | delta -0.0032
         densenet201: late_best 0.9363 vs rad 0.9336 (p 0.4901) | late_last 0.9360 (p 0.5122) | delta -0.0003
         densenet121: late_best 0.9319 vs rad 0.9336 (p 0.6793) | late_last 0.9241 (p 0.0144) | delta -0.0078
  LULUS  G-1 DeLong fusion vs cnn_only: 3 uji, diharapkan 3; fusion signifikan lebih baik 3/3; p = [0.0, 0.0, 0.0]
  LULUS  G-2 metrik XAI fusion: 3 backbone x 4 metrik, sel kosong 0; sampel cocok dengan fixed_display_samples.json: True
         sampel tetap: LIDC-IDRI-0075 nodule_idx=1
         sampel tetap: LIDC-IDRI-0164 nodule_idx=10
         sampel tetap: LIDC-IDRI-0194 nodule_idx=1
         sampel tetap: LIDC-IDRI-0469 nodule_idx=0
         sampel tetap: LIDC-IDRI-0491 nodule_idx=1
         sampel tetap: LIDC-IDRI-0732 nodule_idx=1
  LULUS  G-3 selisih XAI: 24 baris, diharapkan 24; nodul beda kelas keputusan [0, 2, 5, 6]
         convnext_tiny pointing_acc cnn 0.7288 -> fusion 0.6780 (selisih -0.0508)
         densenet121 pointing_acc cnn 0.5763 -> fusion 0.5593 (selisih -0.0169)
         densenet201 pointing_acc cnn 0.6949 -> fusion 0.6780 (selisih -0.0169)
  LULUS  G-4 SHAP beeswarm: 3/3 figure ada, provenance lengkap: True
  LULUS  G-5 tabel gabungan: 7 baris kriteria, sel kosong 0

Keenam gate lulus.
```

p-value G-1 yang tercetak 0.0 adalah pembulatan. Nilai persisnya:
convnext_tiny 1.009e-12, densenet201 1.362e-10, densenet121 2.969e-11.

## Tiga temuan yang harus masuk paper apa adanya

**T-0 bertahan, tapi dengan syarat yang wajib ditulis.** Seleksi checkpoint
menyumbang paling banyak 0.0078 AUC, dan nol backbone yang peringkatnya terbalik.
Tapi dengan checkpoint tanpa seleksi, `fusion_late` jadi signifikan LEBIH BURUK
dari radiomics pada convnext_tiny (p 0.0246) dan densenet121 (p 0.0144). Hanya
densenet201 yang tetap setara secara statistik (p 0.5122) sekaligus nominal lebih
tinggi. Jadi klaim "setara dengan radiomics" hanya aman kalau checkpointnya
dinyatakan, atau kalau densenet201 yang dipakai sebagai model utama.

**T-1 kuat tanpa syarat.** `fusion_late` mengalahkan `cnn_only` pada ketiga
backbone, signifikan, p terkecil 1.0e-12. Ini klaim paling kokoh dari seluruh run.

**Metrik XAI fusion sedikit di BAWAH cnn_only, bukan di atas.** Selisih pointing
accuracy -0.0508, -0.0169, -0.0169 di himpunan 60 nodul; identik persis di
himpunan 6 nodul karena n_disagree nol di situ. Penyebabnya arsitektural:
`fusion_late` mewarisi cabang citra arm A utuh, jadi peta CAM-nya hanya berbeda
pada nodul yang kelas keputusan fusi berbeda dari kelas keputusan CNN, dan pada
nodul seperti itu peta yang menjelaskan keputusan fusi melokalisasi sedikit lebih
buruk. Klaim "fusion unggul dalam XAI" TIDAK didukung angka dan tidak boleh
ditulis. Yang didukung angka: fusion mempertahankan kemampuan penjelasan spasial
cabang citra pada tingkat yang praktis sama, DAN satu-satunya arm yang sekaligus
punya SHAP fitur. Radiomics nol peta spasial, cnn_only nol SHAP.
