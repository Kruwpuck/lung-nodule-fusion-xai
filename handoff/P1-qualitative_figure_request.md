# Permintaan P1 — grid Grad-CAM kualitatif Track 2 (satu sesi GPU bersama P0-3)

**Skrip belum pernah dijalankan end-to-end** (mesin penulis tidak punya checkpoint, patch, mask, PyYAML). Hanya `py_compile` dan `--dry-run` untuk bagian CSV yang diuji. Baca dulu seluruh `src/stage_09t_fig_qualitative.py` sebelum menjalankan.

## Yang dihasilkan
Satu figur (PNG 300 dpi + PDF vektor) dan `t2_qualitative_selection.csv`, semuanya di subdirektori baru `artifacts/results/track2rev/qualitative/`. Tidak ada berkas yang sudah ada yang ditimpa. Checkpoint, patch, dan mask dibuka read-only; tidak ada `torch.save` dan tidak ada tulis ke `artifacts/checkpoints/` atau `artifacts/patches/`.

## Pilihan yang ditetapkan sebelum melihat peta
- Backbone: ConvNeXt-Tiny, DenseNet121, DenseNet201 (tiga backbone yang dilatih ulang nested-CV dan punya angka faithfulness di naskah), GoogLeNet (kasus situs degenerate: band rule memilih `features.17`, Dropout 1x1, peta nol).
- Nodul (aturan tetap, sama di semua panel): malignant yang diprediksi malignant oleh ke-12 backbone dengan mask terkecil (idx 52, LIDC-IDRI-0164 nodul 10) dan terbesar (idx 32, LIDC-IDRI-0075 nodul 1); satu kasus gagal: malignant yang diprediksi benign oleh keempat backbone dengan probabilitas malignant rata-rata terendah (idx 55, LIDC-IDRI-0147 nodul 4). Ketiganya dari 60 nodul fold-0 milik `stage_09e` (`_samples(cfg)`), **bukan** slot S1-S6 dari `stage_07b_fixed_samples.py`.
- Situs: band, canonical, dan best-in-candidate-set dari `depth_curve_summary.csv`, situs yang sama dengan tabel naskah. Bila dua jenis situs adalah modul yang sama, petanya identik. GoogLeNet punya kolom keempat, `features.13` (di luar himpunan kandidat; pointing accuracy 0,767 di Fig. 1 naskah), karena di dalam himpunan kandidat canonical dan best sama (`features.15`, 0,500) dan band memilih `features.17` (Dropout 1x1, peta nol, 0,000). Istilahnya: 0,000 itu skor situs band, bukan situs canonical. DenseNet121/201 memakai satu modul untuk ketiganya.
- Peta Layer-CAM kelas prediksi, diskalakan per peta ke [0,1], alpha 0,5, colorbar tetap 0-1, kontur mask konsensus hijau.

## Urutan
1. `git` commit patch lebih dulu (supaya `commit_sha` benar), termasuk `src/stage_03b_fusion.py` (kolom `gpu_device`, `torch_version` baru di provenance) dan `src/stage_09t_fig_qualitative.py`.
2. `.venv/Scripts/python.exe -m src.stage_09t_fig_qualitative --dry-run` — kirim keluarannya: daftar path checkpoint dan `mtime`, nodul terpilih, path keluaran. Semua `mtime` harus ada (tidak "MISSING").
3. `.venv/Scripts/python.exe -m src.stage_09t_fig_qualitative`, lalu `--self-check`.
4. P0-3 opsional. Bila dijalankan, pakai `handoff/config_rerun_seeded.yaml` (salinan versi `titip/` dengan `run_id` dan `track1_fusion.save_persample: true` ditambahkan; tanpa keduanya patch provenance dan dump per kasus tidak aktif). Jalankan (`handoff/P0-3_seeded_ablation_request.md`).

## Yang dikirim balik
`fig_t2_qualitative_print.{png,pdf}`, `t2_qualitative_selection.csv`, keluaran `--dry-run` dan `--self-check`, dan folder `artifacts/results/rerun_seeded/fusion/` dari P0-3. Checkpoint tidak dikirim.

## Aturan naskah
Figur ini hanya masuk naskah bila halaman cukup; tabel center map dan kalimat Limitations tentang kelas prediksi didahulukan. Bila tidak muat, figur dirujuk dari repositori.
