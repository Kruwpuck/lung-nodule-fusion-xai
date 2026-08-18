# Baseline pra-Rev2

Snapshot hasil eksperimen **sebelum** satu pun perbaikan Rev2 dijalankan.
Dibekukan 4 Agustus 2026, ditarik dari mesin remote `100.98.9.120`.

Folder ini adalah titik referensi wajib. Jangan ditimpa, jangan dihapus.

## Untuk apa

1. Mengukur dampak tiap perbaikan Rev2 terhadap angka, satu per satu.
2. Menghitung selisih bias seleksi saat kebocoran *early stopping* diperbaiki
   lewat nested CV. Selisih itu diharapkan **negatif**, dan justru itu yang
   akan dilaporkan sebagai kontribusi metodologis.
3. Menjalankan uji DeLong berpasangan sebelum lawan sesudah tanpa perlu
   melatih ulang model lama. Ini alasan `preds/*.npz` ikut diarsipkan:
   prediksi mentah per kasus adalah satu-satunya cara mendapatkan uji
   berpasangan yang sah setelah *checkpoint* lama ditimpa.

## Isi

| Lokasi | Jumlah | Keterangan |
|---|---|---|
| `fusion/ablation_summary.csv` | 175 baris | 7 backbone x 5 arm x 5 fold |
| `fusion/delong_fusion.csv` | 21 baris | fusi lawan arm tunggal terbaik, kolom `fusion_significantly_better` bernilai False pada seluruh baris |
| `preds/*.npz` | 149 berkas | prediksi mentah per fold, seluruh backbone dan seluruh granularitas label |
| `xai/xai_metrics.csv` | 12 baris | metrik Grad-CAM dan Layer-CAM per backbone |
| `xai/xai_densenet121.png` | 1 berkas | panel CAM, versi *working tree* remote (278389 byte), berbeda dari versi ter-*commit* (283112 byte) |

Salinan kedua CSV juga ada di repo lokal pada jalur yang sama, sebagai cadangan
kalau mesin remote hilang. Berkas `.npz` hanya ada di remote karena ukurannya.

## Angka kunci yang dibekukan

Rata-rata AUC per arm, dipool lintas 7 backbone dan 5 fold:

| Arm | AUC |
|---|---|
| radiomics_only | 0.9313 |
| fusion_intermediate | 0.9269 |
| fusion_early | 0.9179 |
| fusion_late | 0.9171 |
| cnn_only | 0.7853 |

`cnn_only` per backbone, terurut:

| Backbone | AUC |
|---|---|
| densenet201 | 0.6432 |
| googlenet | 0.7555 |
| xception | 0.7711 |
| convnext_tiny | 0.7888 |
| inceptionv3 | 0.8112 |
| inception_resnet_v2 | 0.8586 |
| densenet121 | 0.8686 |

Angka `densenet201` 0.6432 berhadapan dengan AUC *standalone*-nya 0.8988 adalah
gejala bug resolusi yang belum diperbaiki saat snapshot ini diambil. Gerbang
verifikasi Rev2 langkah 1 adalah kembalinya angka ini mendekati 0.8988.

## Peringatan provenance: `preds/densenet121_fold*.npz` dan checkpoint DenseNet121

Kelima `preds/densenet121_fold*.npz` di *working tree* remote **berbeda dari versi
ter-commit** sejak 29 Juli 2026, dan arsip ini menyalin versi *working tree*.
Artinya berkas di sini bukan yang tercatat di git, dan tidak boleh diperlakukan
sebagai salinan `git show` dari titik mana pun.

Penyebabnya terlacak, dan bukan korupsi. `artifacts/checkpoints/densenet121/`
berisi *checkpoint* dua angkatan:

| Fold | `fold*_best.pt` | Angkatan | AUC standalone |
|---|---|---|---|
| 0 | 14 Juli | pra-`input_size` | 0.8018 |
| 1 | 14 Juli | pra-`input_size` | 0.8272 |
| 2 | 28 Juli | pasca-`input_size` | 0.8659 |
| 3 | 28 Juli | pasca-`input_size` | 0.8911 |
| 4 | 14 Juli | pra-`input_size` | 0.7804 |

`input_size: 96` baru masuk pada commit `0b54376`, 28 Juli 09:15:32. Sesi latih
28 Juli (09:31 sampai 10:00) melanjutkan tiap *fold* DenseNet121 dari *epoch* 49
dengan `epochs: 50`, jadi hanya satu *epoch* tersisa. Untuk *fold* 2 dan 3 *epoch*
terakhir itu memperbaiki AUC sehingga `best.pt` ditimpa bobot rezim 96 piksel;
untuk *fold* 0, 1, dan 4 tidak, sehingga `best.pt` tetap bobot rezim 64 piksel
yang kini dievaluasi pada 96 piksel. DenseNet121 satu-satunya *backbone* dengan
*checkpoint* campuran, enam lainnya seragam 28 Juli atau lebih baru.

Konsekuensi: AUC DenseNet121 0.8333 pada `summary_binary.csv` **bukan** ukuran
kemampuan arsitekturnya. Rata-rata tiga *fold* basi 0.8031 lawan dua *fold* segar
0.8785; angka segar itu sejajar dengan enam *backbone* lain. Dua *fold* kolaps
(sensitivitas 0.0864 pada *fold* 1 dan 0.1196 pada *fold* 4) keduanya berada di
himpunan basi. Ini artefak infrastruktur, bukan instabilitas pelatihan, sehingga
mendiagnosisnya lewat *fine-tuning* akan salah sasaran.

Perlu dicatat: korupsi *checkpoint* OneDrive dan pemulihan dari *epoch* 31 yang
tercatat pada `run_all_log.txt` **tidak** menyentuh DenseNet121. Kejadian itu
mengenai `inceptionv3`, `xception`, `convnext_tiny`, dan `inception_resnet_v2`.
Kedua masalah berdiri sendiri.

## Dua cacat yang sudah diketahui pada baseline ini

1. **Bug resolusi input belum diperbaiki.** *Checkpoint* dilatih pada 96 piksel
   lalu dievaluasi pada 64 piksel. Seluruh kolom `cnn_only` tidak sahih.
2. **Seleksi fitur bukan mRMR.** `pip show pymrmr` di remote mengembalikan
   `Package(s) not found`. Artinya `src/radiomics/feature_selection.py:65`
   selama ini jatuh ke `mutual_info_classif` secara senyap, hanya menulis
   `logger.warning`. Setiap penyebutan mRMR pada manuskrip Track 1 harus
   dikoreksi menjadi *mutual information*, atau `pymrmr` dipasang lalu seluruh
   cabang radiomics dijalankan ulang.

## Verifikasi integritas

Snapshot diambil dari *working tree* remote, bukan dari `git show`. Kedua CSV
sudah dibandingkan terhadap `origin/main` dan isinya identik setelah normalisasi
akhiran baris, jadi status `M` yang terlihat di `git status` saat itu murni
akibat CRLF, bukan data yang berbeda.
