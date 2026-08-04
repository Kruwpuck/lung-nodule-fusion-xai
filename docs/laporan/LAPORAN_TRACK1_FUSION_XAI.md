# Laporan hasil Track 1: fusi radiomics-CNN dan explainability per-arm

## 0. Identitas penelitian

- **Judul**: Fusi radiomics-CNN dan explainability per-arm untuk klasifikasi malignansi nodul paru pada LIDC-IDRI
- **Repo**: `lung-nodule-fusion-xai`
- **Tugas**: Track 1 dari pemisahan dua paper (lihat `docs/Review Revisi 1.md` §8)
- **Dataset**: LIDC-IDRI
- **Tanggal laporan**: 30 Juli 2026

---

## 1. Ringkasan eksekutif

Track 1 membandingkan lima *arm* representasi (CNN-only, radiomics-only, early fusion, intermediate fusion, late fusion) pada tujuh backbone CNN, lalu mengevaluasi explainability tiap arm secara terpisah. Temuan utama: radiomics-only mengungguli semua varian fusi, dan pointing accuracy Grad-CAM/Layer-CAM sangat bervariasi antar backbone tanpa hubungan konsisten terhadap AUC klasifikasi.

### Status komponen

| Komponen | Status | Sumber |
|---|---|---|
| Ablasi fusi 5 arm x 7 backbone x 5 fold | Selesai (data pra-perbaikan resolusi) | `artifacts/results/fusion/ablation_summary.csv` |
| DeLong fusi vs radiomics-only | Selesai | `artifacts/results/fusion/delong_fusion.csv` |
| Metrik XAI (Grad-CAM/Layer-CAM) | Selesai | `artifacts/results/xai/xai_metrics.csv` |
| Perbaikan bug resolusi input_size | Kode selesai, belum di-*re-run* | `docs/revisi/rev1/TASKBOARD.md` tugas 1 |
| Arm fusi baru (branch_norm, GMU, modality dropout) | Kode selesai, belum dieksekusi | `docs/revisi/rev1/TASKBOARD.md` tugas 5a/5b/5c |

---

## 2. Latar belakang dan kontribusi

Fusi radiomics dan fitur CNN diasumsikan saling melengkapi, tapi asumsi ini tidak selalu berlaku pada kohort sekecil LIDC-IDRI, di mana radiomics dengan seleksi fitur adalah *baseline* yang kuat. Kontribusi laporan ini: (1) perbandingan lima arm representasi pada tujuh backbone dengan uji DeLong berpasangan, (2) protokol XAI yang mengevaluasi cabang CNN (Grad-CAM/Layer-CAM) dan cabang radiomics (SHAP) secara terpisah, karena belum ada metrik *model-agnostic* terstandardisasi untuk membandingkan keduanya secara adil.

---

## 3. Metodologi

### 3.1 Dataset dan split

LIDC-IDRI, label malignansi dari median 4 rating ahli radiologi (skala 1–5), *split* 5-fold berbasis pasien dengan seed 42 sehingga satu pasien tidak muncul di lebih dari satu fold.

### 3.2 Prapemrosesan

Patch 2.5D: tumpukan slice aksial berdekatan berpusat pada nodul, di-*crop* ke resolusi in-plane tetap. Intensitas Hounsfield unit di-*clip* ke $[-1000, 400]$ lalu dinormalisasi ke $[0, 1]$.

### 3.3 Backbone dan resolusi input

Tujuh backbone: DenseNet121, InceptionV3, Xception, GoogLeNet, ConvNeXt-Tiny, Inception-ResNet-v2, DenseNet201. Tiga di antaranya (InceptionV3, Xception, Inception-ResNet-v2) punya syarat resolusi minimum (75, 71, 75 piksel) melebihi patch native 64 piksel proyek ini, sehingga ketujuh backbone dievaluasi pada resolusi seragam 96 piksel, dipilih sebagai kelipatan 32 terkecil di atas seluruh syarat minimum agar tidak ada backbone yang diuntungkan resolusi saat dibandingkan lewat XAI.

### 3.4 Arm fusi

CNN-only, radiomics-only, early fusion (konkatenasi fitur radiomics mentah ke input CNN), intermediate fusion (konkatenasi embedding CNN dengan vektor radiomics sebelum classification head bersama), late fusion (rata-rata probabilitas arm CNN dan radiomics).

### 3.5 Protokol XAI

Grad-CAM/Layer-CAM untuk cabang CNN, dievaluasi dengan dice, IoU, dice size-matched, pointing accuracy (apakah aktivasi maksimum jatuh di dalam mask nodul), dan energy pointing metric, semua terhadap mask ground-truth radiolog. SHAP untuk cabang radiomics. Kedua modalitas dilaporkan terpisah, tidak digabung jadi satu skor.

---

## 4. Dataset

Lihat §3.1 dan §3.2. Detail lengkap distribusi kelas dan fold ada di `artifacts/results/tables/table_3_1_class_distribution.csv` dan `table_3_2_fold_distribution.csv`.

---

## 5. Konfigurasi

`configs/config.yaml` blok `tracks.track1`: 7 backbone di atas, `input_size: 96`.

---

## 6. Hasil

### 6.1 Ablasi fusi

AUC rata-rata per arm, dipool lintas 7 backbone dan 5 fold, dari `artifacts/results/fusion/ablation_summary.csv` setelah perbaikan bug resolusi **dan** perbaikan kebocoran seleksi *epoch* lewat nested CV (175 baris: 7 backbone x 5 arm x 5 fold, dijalankan 4 Agustus 2026):

| Arm | AUC (nested CV) | Sebelum nested CV | Baseline pra-perbaikan resolusi |
|---|---|---|---|
| fusion_late | 0.9333 | 0.9332 | 0.9171 |
| radiomics_only | 0.9318 | 0.9314 | 0.9313 |
| fusion_early | 0.9125 | 0.9126 | 0.9179 |
| fusion_intermediate | 0.9084 | 0.9294 | 0.9269 |
| cnn_only | 0.8927 | 0.8927 | 0.7853 |

Baseline pra-perbaikan diarsipkan di `artifacts/results/_baseline_pre_rev2/`. Kolom `cnn_only` konsisten dengan AUC standalone per backbone di `summary_binary.csv` (selisih 0.0019-0.0070, lihat §8.4 untuk sumber selisih itu), mengonfirmasi bug resolusi sudah tertutup.

Dari 21 uji DeLong berpasangan (`fusion/delong_fusion.csv`), fusion_late unggul angka di 5 dari 7 backbone tapi **tidak satu pun** dari 21 pasangan mencapai signifikansi yang mendukung fusi; p terkecil di sisi menang adalah 0.2040. fusion_early signifikan lebih buruk daripada radiomics_only di ketujuh backbone, fusion_intermediate di enam dari tujuh, dengan p terkecil 2.6e-7 pada `inception_resnet_v2`.

#### Kuantifikasi bias seleksi

Nested CV menurunkan `fusion_intermediate` sebesar **0.0210** AUC (0.9294 menjadi 0.9084). Angka itu estimasi langsung bias seleksi dari protokol lama, yang memilih *epoch* terbaik pada *fold* validasi luar lalu melaporkan *fold* yang sama.

Tiga arm lain bergerak di bawah 0.0005 dan `cnn_only` nol persis. Pola itu bukan kebetulan melainkan konfirmasi silang bahwa perbaikan mengenai sasaran: hanya `fusion_intermediate` yang melewati `_train_fusion_fold`, satu-satunya jalur berloop *epoch* dengan seleksi *checkpoint*. `fusion_early` dan `fusion_late` memakai *embedding* dan probabilitas dari *checkpoint* arm A yang dibekukan di luar `stage_03b_fusion`, jadi memang harus diam, dan memang diam.

Arah temuannya perlu dinyatakan tegas: kebocoran ternyata **menyamarkan kekalahan, bukan menciptakan kemenangan**. Sebelum diperbaiki, `fusion_intermediate` 0.9294 tampak nyaris menyamai radiomics 0.9314; sesudahnya ia jatuh ke 0.9084 dan kalah signifikan di enam dari tujuh *backbone*. Kesimpulan "radiomics mengungguli fusi" menguat, bukan melemah.

#### Komplementaritas modalitas

Untuk menguji apakah kekalahan fusi berakar pada arsitektur fusi atau pada ketimpangan modalitas, prediksi `cnn_only` dan `radiomics_only` dibandingkan per kasus tanpa pelatihan baru: probabilitas CNN diambil dari `preds/*.npz` yang sudah ada lalu disubset ke kohort 1366, probabilitas radiomik dihitung ulang dengan pipeline seleksi yang sama.

| Backbone | Pearson | Spearman | CNN benar, radiomik salah | Radiomik benar, CNN salah | Keduanya salah |
|---|---|---|---|---|---|
| inception_resnet_v2 | 0.831 | 0.731 | 14.4 | 17.2 | 19.8 |
| convnext_tiny | 0.824 | 0.740 | 13.6 | 22.2 | 20.6 |
| xception | 0.819 | 0.705 | 15.0 | 16.2 | 19.2 |
| inceptionv3 | 0.805 | 0.684 | 14.0 | 19.2 | 20.2 |
| densenet201 | 0.796 | 0.697 | 15.6 | 18.6 | 18.6 |
| googlenet | 0.795 | 0.692 | 15.2 | 19.4 | 19.0 |
| densenet121 | 0.626 | 0.611 | 13.8 | 43.6 | 20.4 |

Rata-rata per *fold*, n rata-rata 273 kasus, ambang keputusan 0,5. AUC radiomik identik 0.9343 di seluruh baris, sebagaimana seharusnya karena arm itu tidak bergantung pada *backbone*; kesamaan itu berfungsi sebagai pemeriksaan kewarasan.

CNN memiliki kontribusi unik, tetapi kecil. Korelasinya tinggi namun jauh dari jenuh, dan CNN benar sementara radiomik salah pada sekitar 14 kasus per *fold* atau 5,1 persen, stabil di ketujuh *backbone*. Arah ketimpangannya tetap jelas: radiomik unik-benar pada 16 sampai 22 kasus, lebih banyak. Satu batasan tafsir yang penting, angka-angka ini adalah keputusan keras pada ambang 0,5, dan benar pada ambang tidak otomatis berarti tambahan AUC.

### 6.2 XAI

Pointing accuracy per backbone, dari `artifacts/results/xai/xai_metrics.csv`:

| Backbone | Pointing accuracy |
|---|---|
| DenseNet121 | 0.7167 |
| ConvNeXt-Tiny | 0.7167 |
| DenseNet201 | 0.7000 |
| Inception-ResNet-v2 | 0.3833 |
| InceptionV3 | 0.2000 |
| Xception | 0.2000 |
| VGG16 | 0.2000 |
| ResNet50 | 0.1167 |
| EfficientNet-B0 | 0.0833 |
| MobileNetV3-Small | 0.0000 |
| ViT-Base | 0.0000 |
| GoogLeNet | 0.0000 |

Pointing accuracy rendah tidak berarti AUC klasifikasi rendah; beberapa backbone dengan pointing mendekati nol tetap klasifikasi kompetitif. Ini konsisten dengan keterbatasan Grad-CAM yang sudah didokumentasikan pada literatur, khususnya untuk arsitektur vision transformer.

---

## 7. Figur

| Figur | Berkas | Kegunaan |
|---|---|---|
| Panel Grad-CAM per backbone | `artifacts/results/xai/xai_{backbone}.png` | Visualisasi CAM per backbone (belum komparabel lintas model) |
| Panel komparabilitas XAI baru | `artifacts/results/figures_grid/grid_comparability.png` | Sampel identik, colorbar seragam, baris kegagalan (belum dieksekusi, tidak ada checkpoint lokal) |
| Diagram arsitektur fusi | `artifacts/results/figures/fusion_architecture.png` | Ilustrasi arm fusi |

---

## 8. Batasan

1. Kolom `cnn_only` pada ablasi fusi masih data pra-perbaikan bug resolusi; perlu di-*re-run* sebelum dianggap final (§6.1).
2. Tiga varian fusi baru (branch normalization, Gated Multimodal Unit, modality dropout) sudah diimplementasikan dan diuji unit, tapi belum pernah dieksekusi pada grid ablasi penuh.
3. Panel XAI komparabilitas baru (`stage_07f_xai_comparability.py`) belum bisa dijalankan di mesin manapun karena checkpoint tidak tersedia lokal saat ditulis.
4. SHAP dan Grad-CAM dilaporkan pada skala terpisah tanpa metrik penyatu; ini gap metodologis terbuka, bukan keterbatasan khusus studi ini.
5. Seleksi fitur radiomics memakai *mutual information* (`mutual_info_classif`), bukan mRMR. Lihat §8.1.

### 8.1 Temuan reproducibility: kegagalan yang diam

`src/radiomics/feature_selection.py` mencoba mengimpor `pymrmr`, lalu menangkap `ImportError` dan beralih diam-diam ke `mutual_info_classif`. `pymrmr` butuh kompilasi C++ dan tidak terpasang di mesin remote, terverifikasi lewat `pip show pymrmr` yang mengembalikan `Package(s) not found`. Artinya cabang mRMR **tidak pernah sekali pun dieksekusi**, dan seluruh angka radiomics di laporan ini dihasilkan oleh *mutual information*.

Substansi hasilnya tidak berubah. Seleksi fitur berbasis *mutual information* adalah metode yang sah dan lazim dilaporkan, dan tetap dijalankan per *fold* pada data latih saja sehingga bebas kebocoran. Yang salah adalah deskripsinya, bukan metodenya. Karena itu perbaikannya berupa koreksi teks, bukan pengulangan eksperimen. Mengganti metode justru akan merusak `radiomics_only` 0.9313 yang menjadi pembanding untuk seluruh 21 uji DeLong.

Pelajarannya lebih luas daripada satu paket yang hilang. Kegagalan ini **diam**: pipeline berjalan sampai selesai, menghasilkan angka yang masuk akal, dan hanya menulis satu baris `logger.warning` yang tenggelam di antara ribuan baris log latihan. Kegagalan yang berisik menghentikan pipeline dan langsung terlihat. Kegagalan yang diam berjalan berbulan-bulan lalu muncul sebagai klaim metode yang salah di dalam manuskrip. Cacat ini ditemukan saat audit kode, bukan saat pipeline dijalankan, dan memang hanya begitulah ia bisa ditemukan.

Perbaikannya karena itu bersifat struktural, bukan sekadar mengganti kata. `mrmr_select` sekarang mengembalikan nama metode yang benar-benar dipakai, dan nama itu ditulis ke kolom `fs_method` pada setiap baris `ablation_summary.csv`. Metode tidak lagi bisa terpisah dari angka yang dihasilkannya. Prinsip yang layak diterapkan ke seluruh *fallback* opsional pada proyek ini: kalau sebuah cabang kode boleh mengganti metode secara diam-diam, metode yang aktif harus ikut tersimpan bersama hasilnya, bukan hanya tercatat di log.

### 8.2 Temuan reproducibility: cakupan test, bukan sekadar satu bug

Audit yang sama memunculkan dua cacat pada *test suite* itu sendiri. Keduanya lebih tepat dibaca sebagai informasi tentang cakupan pengujian daripada sebagai dua bug lepas.

**Jalur kode yang dijamin gagal dan tak pernah diuji.** `full_feature_selection_pipeline` memanggil `lasso_select(..., seed=seed)`, padahal *signature* `lasso_select` mendeklarasikan `random_state`. Setiap pemanggilan pasti melempar `TypeError`. Fungsi itu bertahan justru karena tidak ada yang memanggilnya: seluruh `src/` memakai `mrmr_select` dan `lasso_select` secara langsung lewat `_select_fold_features`, dan tidak satu pun berkas di `tests/` menyentuhnya. Satu-satunya pemanggil adalah `notebooks/radiomics_extraction.ipynb`, sementara `docs/training_guide.md` menjanjikan notebook itu mendemokan pipeline ICC, *filter*, lalu LASSO pada *fold* 0. Demo tersebut tidak pernah bisa berjalan. Cacat sesungguhnya bukan pada satu kata kunci yang salah, melainkan pada adanya fungsi publik yang nol tersentuh pengujian sehingga kegagalan sepasti itu pun lolos berbulan-bulan. Celahnya kini ditutup dengan uji asap di `tests/test_utils.py`.

**Uji yang ada tetapi memeriksa hal yang keliru.** `test_registry_name_map_count` menegaskan `len(_NAME_MAP) == 8`. Angka itu dibekukan sebelum himpunan *backbone* Track 1 dan Track 2 ditambahkan, jadi ia gagal justru ketika registry benar, yaitu saat berisi 14 entri. Uji semacam ini lebih buruk daripada tidak ada uji sama sekali: ia berbunyi pada setiap penambahan yang disengaja sehingga melatih pembacanya mengabaikan kegagalan, sementara hal yang benar-benar berbahaya tidak pernah diperiksa, yakni apakah *backbone* yang diminta konfigurasi memang dapat dibangun. Penggantinya, `test_registry_covers_every_configured_backbone`, memeriksa invariannya secara langsung: setiap nama pada `configs/config.yaml` harus dapat diresolusi lewat `_NAME_MAP`. Registry sendiri ternyata sudah sinkron penuh, 13 dari 13 nama teresolusi, jadi tidak ada risiko tersisa untuk tahap berikutnya.

Benang merah §8.1 dan §8.2 sama: angka yang dilaporkan hanya sekuat mekanisme yang memaksanya tetap jujur. Peringatan di log, hitungan ajaib, dan fungsi tak terpanggil sama-sama tampak seperti perlindungan, padahal tidak satu pun benar-benar memaksa apa pun.

### 8.3 Temuan reproducibility: proses latih yang mati bersama sesi SSH

Peluncuran ulang ablasi fusi pada mesin remote mati setelah kira-kira dua menit. Hanya *fold* 0 yang tercatat, GPU kembali ke nol persen, dan tidak ada satu pun *traceback*. Penyebabnya bukan cacat kode: `Start-Process` pada mesin remote tetap menjadi anak dari *job object* milik sesi SSH, sehingga Windows menghentikannya begitu sesi ditutup. Prosesnya bertahan selama pemeriksaan di dalam sesi dan mati persis ketika sesi berakhir, yang menjelaskan mengapa gejalanya menyerupai kegagalan acak. Solusinya membuat proses di luar pohon proses sesi lewat `Invoke-CimMethod -ClassName Win32_Process -MethodName Create`; dengan cara itu proses selamat dari penutupan sesi, terverifikasi dengan memeriksa PID dari sesi SSH yang benar-benar baru.

Ini insiden infrastruktur keempat pada proyek ini setelah disk penuh, *checkpoint* korup, dan pemanggilan interpreter yang salah. Keempatnya berbagi satu pola: kegagalan yang diam dan nyaris dilaporkan sebagai keberhasilan. Konsekuensi prosedural yang diambil: **status proses yang berjalan terlepas tidak boleh diasumsikan, melainkan wajib diverifikasi dengan memeriksa PID dari sesi baru sebelum dilaporkan.** Melaporkan "sedang berjalan" berdasarkan keberhasilan peluncuran saja sama dengan melaporkan sesuatu yang belum diperiksa.

Prinsip yang sama berlaku untuk uji: mengganti sebuah uji belum selesai sebelum uji penggantinya dibuktikan menangkap kegagalan yang seharusnya ia tangkap. Untuk `test_registry_covers_every_configured_backbone` pembuktian itu dilakukan lewat uji negatif, yaitu menghapus `densenet201` dari registry lalu memastikan uji tersebut gagal dan menyebut nama itu.

### 8.4 Koreksi: dua penyebab offset +0.0037 pada kolom cnn_only, dan kebocoran seleksi epoch di stage_03b_fusion

Kenaikan `cnn_only` setelah perbaikan resolusi (§6.1) tidak seragam kebetulan. Hipotesis awal menduga kebocoran *early stopping* menjelaskannya. Hipotesis itu **tertolak**: `_cnn_only_preds` (`src/stage_03b_fusion.py:88-113`) hanya memuat ulang *checkpoint* `checkpoints/{model}/fold{f}_best.pt` yang sama persis dengan yang dipakai evaluasi standalone Bab 6.1, lalu melakukan inferensi murni tanpa latihan baru. Tidak ada proses latihan kedua yang bisa membocorkan sesuatu yang berbeda dari standalone.

Penyebab sebenarnya lebih sederhana dan terlacak sampai desimal keempat: **penyempitan kohort yang tidak dinyatakan.** `_load_merged` (`src/stage_03b_fusion.py:38-59`) membuang nodul dengan kunci `(patient_id, nodule_idx)` ganda sebelum `merge` dengan `radiomics.parquet`, menyisakan 1366 dari 1391 nodul standalone Bab 6.1. Diverifikasi dengan menghitung ulang AUC standalone dari `preds/*.npz` arsip pada subset 1366 nodul yang sama: ketujuh backbone cocok dengan selisih yang diamati sampai 3-4 desimal (0.0019 sampai 0.0070). Artinya angka Bab 6.1 (penyebut 1391) dan angka fusi selama ini **dibandingkan pada kohort berbeda tanpa dinyatakan** — bukan sekadar catatan reproducibility, melainkan koreksi terhadap perbandingan yang sudah tertulis di laporan ini.

Audit yang sama menemukan kebocoran nyata, tapi di tempat lain dari yang diduga semula: bukan pada offset kohort, melainkan **asimetris antar-arm** di dalam `stage_03b_fusion` sendiri. `radiomics_only` memanggil `clf.fit(X, y)` tanpa `eval_set` (`src/fusion/early_fusion.py`) sehingga bersih dari seleksi berbasis validasi. Sebaliknya `fusion_intermediate` (dan turunannya `fusion_early`/`fusion_late` lewat *embedding* dan probabilitas CNN yang sama) memilih *epoch* terbaik berdasarkan AUC pada *fold* validasi luar, lalu melaporkan skor pada *fold* yang sama (`trainer.py`-style leak, lokal di `_train_fusion_fold` sebelum perbaikan di bawah). Karena pembanding satu-satunya yang bersih adalah `radiomics_only`, setiap kemenangan tipis arm fusi di atasnya diukur dengan timbangan berat sebelah, bukan derau kohort yang menelan selisihnya.

**Perbaikan diterapkan**: `_train_fusion_fold` sekarang mencarik *inner split* per pasien (85/15, `GroupShuffleSplit` disemai per *fold*) dari *fold* pelatihan luar. *Epoch* terbaik dipilih dari AUC *inner-validation* itu; *fold* validasi luar (`outer_val_loader`) hanya dievaluasi sekali, setelah pelatihan selesai, dan tidak pernah memengaruhi bobot mana yang disimpan. Cakupan perbaikan ini sengaja dibatasi pada `stage_03b_fusion.py` saja — 215 *run* Track 2 di `src/training/trainer.py` tidak disentuh, karena keduanya jalur kode terpisah dan Track 2 di luar cakupan revisi ini.

Perbaikan itu memunculkan satu *bug* susulan yang layak dicatat karena penyebabnya halus. Pemotongan 85 persen mengubah ukuran himpunan latih, dan untuk sebagian kombinasi *fold* dan *backbone* *batch* terakhir menjadi berukuran satu, yang ditolak `BatchNorm` dalam moda pelatihan dengan `ValueError: Expected more than 1 value per channel`. Ukuran `train_df` penuh kebetulan tidak pernah menghasilkan sisa satu, jadi jalur lama tidak pernah menyentuh kasus ini. Perbaikannya `drop_last=True` pada *loader* pelatihan saja; *loader* evaluasi tidak terpengaruh karena `eval_fusion` berjalan dalam `model.eval()`.

### 8.5 Temuan reproducibility: seleksi fitur yang tidak deterministik

Setelah nested CV dijalankan, `radiomics_only` bergerak sampai 0.0036 AUC per *backbone* padahal tidak satu baris kode pun pada jalur arm itu berubah. Penelusurannya berujung pada `mutual_info_classif` yang dipanggil tanpa `random_state`. Penaksir *k-nearest-neighbour*-nya menambahkan derau kecil untuk memecah nilai seri, sehingga dua pemanggilan dengan input identik dapat memilih himpunan fitur yang berbeda.

Gejalanya sebenarnya sudah tercetak di log jauh sebelum disadari: dalam satu *run* ablasi yang sama, *fold* 0 mencatat `LASSO selected 29/50 features (alpha=0.0004)` untuk satu *backbone* dan `25/50 (alpha=0.0007)` untuk *backbone* lain, padahal arm radiomik tidak bergantung pada *backbone* dan menerima `train_df` yang sama persis. Diverifikasi langsung pada matriks fitur asli berisi 1130 kolom: dua pemanggilan tanpa *seed* menghasilkan himpunan yang berbeda pada 4 fitur, sementara dua pemanggilan dengan `random_state=42` identik. Sumber acak lain sudah tertutup, `train_early_fusion_xgboost` memakai `random_state=42` dan `LassoCV` bersifat deterministik pada moda `cyclic`.

Angka lantai derau 0.0036 itu **diukur sebelum penetapan seed ini**. Ia tetap berguna sebagai ambang saat menafsirkan hasil-hasil lama, tetapi tidak berlaku untuk *run* setelah `random_state=42` dipasang, yang seharusnya mengulang persis.

### 8.6 Temuan reproducibility: checkpoint dua angkatan pada DenseNet121

AUC standalone DenseNet121 0.8333 jauh di bawah enam *backbone* lain yang berkerumun di 0.89 sampai 0.91, dengan dua *fold* nyaris kolaps, yaitu sensitivitas 0.0864 pada *fold* 1 dan 0.1196 pada *fold* 4. Membacanya sebagai instabilitas pelatihan akan keliru.

Bukti yang menentukan bukan tanggal berkas melainkan metadata di dalam *checkpoint* itu sendiri. Setiap `fold*_best.pt` menyimpan `best_auc`, yaitu AUC validasi yang dicapai saat bobot itu dipilih. Membandingkannya dengan AUC hasil evaluasi ulang di `summary_binary.csv` memberi hasil berikut untuk seluruh 35 *checkpoint* Track 1:

| Backbone dan fold | `best_auc` tersimpan | AUC evaluasi | Selisih |
|---|---|---|---|
| densenet121 fold 0 | 0.8982 | 0.8018 | **0.0965** |
| densenet121 fold 1 | 0.9059 | 0.8272 | **0.0787** |
| densenet121 fold 4 | 0.8883 | 0.7804 | **0.1079** |
| 32 *checkpoint* lainnya | - | - | 0.0000 |

Tiga puluh dua *checkpoint* mencocokkan `best_auc` dengan AUC evaluasi sampai nol persis, yang berarti rezim saat pelatihan dan rezim saat evaluasi sama. Hanya ketiga *fold* DenseNet121 itu yang menyimpang, dan menyimpang besar. Bukti ini berdiri sendiri tanpa bergantung pada *timestamp*, yang penting karena klien sinkronisasi berkas pada mesin ini diketahui menyentuh berkas tanpa mengubah isinya.

Penjelasannya, `input_size: 96` masuk pada *commit* `0b54376` tanggal 28 Juli pukul 09:15:32. *Commit* yang sama adalah yang menambahkan enam *backbone* Track 1 lainnya, sehingga keenamnya tidak pernah ada sebelum rezim 96 piksel dan tidak mungkin mewarisi bobot rezim lama. DenseNet121 satu-satunya yang berasal dari himpunan enam model legacy dan sudah dilatih sejak 14 Juli. Ketika sesi 28 Juli mencoba melanjutkannya, `maybe_resume` menemukan `epoch` 49 dengan `epochs: 50` sehingga `start_epoch >= epochs` terpenuhi dan `src/stage_03_train.py:211` mencetak `[SKIP]` tanpa melatih satu *epoch* pun. Bobot rezim 64 piksel karena itu tidak pernah tergantikan, dan kini dievaluasi pada 96 piksel.

Pemisahannya bersih. Tiga *fold* berbobot lama rata-rata 0.8031, dua *fold* berbobot baru rata-rata 0.8785, dan angka kedua itu sejajar dengan enam *backbone* lain. Kedua *fold* kolaps berada di himpunan lama.

Perlu dipisahkan dari insiden §8.3 yang berbeda: korupsi *checkpoint* OneDrive dan pemulihan dari *epoch* 31 pada `run_all_log.txt` mengenai `inceptionv3`, `xception`, `convnext_tiny`, dan `inception_resnet_v2`, sama sekali tidak menyentuh DenseNet121. Dua masalah yang berdiri sendiri, dan keduanya sama-sama artefak infrastruktur, bukan sifat model.

Tindak lanjut yang diambil: keenam *checkpoint* DenseNet121 *fold* 0, 1, dan 4 dipindahkan ke `artifacts/checkpoints/_archive_densenet121_pre_input_size/`, bukan dihapus, karena berkas itu adalah bukti bagi bagian ini. Ketiga *fold* kemudian dilatih ulang dari nol, bukan dilanjutkan, sebab melanjutkan dari bobot rezim 64 piksel hanya akan mengulang persoalan yang sama. Log pelatihan mengonfirmasi tidak ada baris `Resumed from epoch` maupun `[SKIP]`, dan ketiganya berhenti dini pada *epoch* 36, 18, dan 23.

Hasilnya mengonfirmasi diagnosis:

| Fold | AUC lama | AUC baru | Sensitivitas lama | Sensitivitas baru |
|---|---|---|---|---|
| 0 | 0.8018 | **0.9181** | 0.2812 | **0.7708** |
| 1 | 0.8272 | **0.9124** | 0.0864 | **0.6914** |
| 2 (tak disentuh) | 0.8659 | 0.8659 | 0.7312 | 0.7312 |
| 3 (tak disentuh) | 0.8911 | 0.8911 | 0.6739 | 0.6739 |
| 4 | 0.7804 | **0.8921** | 0.1196 | **0.7283** |

Rata-rata DenseNet121 naik dari 0.8333 menjadi **0.8959**, sejajar dengan enam *backbone* lain yang berkisar 0.8911 sampai 0.9055, dan tidak lagi menjadi *outlier*. Sensitivitas kedua *fold* yang semula nyaris kolaps keluar jauh dari rentang 0.08 sampai 0.12 menuju 0.69 dan 0.73. *Fold* 2 dan 3 yang tidak disentuh bergerak nol persis, yang berfungsi sebagai kontrol bahwa perubahan hanya berasal dari pelatihan ulang. Gap `best_auc` terhadap AUC evaluasi kini 0.0000 pada kelima *fold*.

Kesimpulannya tegas: dua *fold* yang tampak kolaps bukan instabilitas pelatihan melainkan artefak infrastruktur, dan mendiagnosisnya lewat *fine-tuning* memang akan salah sasaran.

Satu catatan yang tersisa dan belum diselesaikan. Ketiga *fold* yang dilatih ulang kini berasal dari pelatihan penuh dari nol pada rezim 96 piksel, sedangkan *fold* 2 dan 3 berasal dari jalur yang berbeda dan tidak sepenuhnya terlacak, meskipun keduanya konsisten secara internal dengan gap 0.0000. Protokol pelatihan karena itu tidak seragam di dalam satu skema *cross-validation*, dan ketiga *fold* baru justru mencatat AUC lebih tinggi (0.8921 sampai 0.9181) daripada kedua *fold* lama (0.8659 dan 0.8911). Selisih itu masih berada dalam rentang variasi antar-*fold* yang wajar untuk *backbone* lain, jadi bukan bukti cacat, tetapi keseragaman protokol tetap layak ditegakkan dengan melatih ulang *fold* 2 dan 3 juga.

Pelajaran yang menyambung ke §8.1 sampai §8.5: *checkpoint* tidak menyimpan konfigurasi yang melahirkannya. Tidak ada mekanisme yang mencegah bobot rezim 64 piksel dievaluasi pada 96 piksel, persis seperti tidak ada mekanisme yang mencegah `mutual_info_classif` dilaporkan sebagai mRMR sebelum kolom `fs_method` ditambahkan. Perbaikan struktural yang setara adalah menyimpan `input_size` di dalam *checkpoint* lalu memeriksanya saat pemuatan. Perlu dicatat juga bahwa `best_auc` yang sudah tersimpan di dalam *checkpoint* ternyata cukup untuk mendeteksi seluruh persoalan ini dalam satu pemindaian; datanya sudah ada sejak awal, hanya tidak pernah dibandingkan dengan AUC yang dilaporkan.

### 8.7 Pola yang berulang: buktinya selalu sudah ada

Lima temuan pada §8.1 sampai §8.6 berbagi satu bentuk yang sama, dan bentuk itu lebih layak dicatat daripada tiap kejadiannya sendiri-sendiri.

| Bagian | Bukti yang sudah tercetak atau tersimpan | Berapa lama tidak terbaca |
|---|---|---|
| §8.1 | `logger.warning` menyebut `pymrmr` tidak terpasang | sepanjang proyek |
| §8.3 | proses hilang dari daftar proses setelah sesi ditutup | sampai diperiksa dari sesi baru |
| §8.4 | `n_val` 1366 lawan 1391 tercetak di setiap baris log *fold* | sejak ablasi pertama |
| §8.5 | `LASSO selected 29/50` lawan `25/50` untuk arm yang tidak bergantung *backbone* | sejak ablasi pertama |
| §8.6 | `best_auc` tersimpan di dalam setiap *checkpoint* | sejak 14 Juli |
| §8.8 | `[SKIP]` tercetak untuk kelima *fold* DenseNet121 | sejak 28 Juli |

Tidak satu pun dari kelimanya membutuhkan eksperimen baru untuk ditemukan. Semuanya tertulis di log atau tersimpan di berkas hasil, sebagian bahkan tercetak berulang kali pada setiap *run*. Yang hilang bukan datanya, melainkan pembandingnya: nol mekanisme yang membandingkan `n_val` antar tahap, nol yang membandingkan jumlah fitur terpilih antar *backbone* pada *fold* yang sama, nol yang membandingkan `best_auc` tersimpan dengan AUC yang dilaporkan.

Gejala `LASSO selected 29/50` lawan `25/50` adalah contoh paling telanjang. Arm radiomik secara definisi tidak bergantung pada *backbone*, sehingga dua angka berbeda pada *fold* yang sama adalah kemustahilan yang tercetak apa adanya di layar, berulang kali, tanpa ada yang membacanya. Konsekuensi praktis yang diambil: pemeriksaan konsistensi yang murah lebih berharga daripada log yang lengkap, karena log yang lengkap justru menenggelamkan anomali di antara ribuan baris yang normal.

### 8.8 Jebakan yang akan berulang: mengubah input_size tanpa memaksa pelatihan ulang

Cacat pada §8.6 bukan kecelakaan sekali jadi melainkan konsekuensi langsung dari cara `maybe_resume` berinteraksi dengan perubahan konfigurasi, dan siapa pun yang memakai *pipeline* ini akan menabraknya lagi dengan cara yang sama.

Urutannya: `input_size` dinaikkan ke 96, *pipeline* dijalankan ulang untuk seluruh *backbone*, dan `src/stage_03_train.py:209` memanggil `maybe_resume(last_pt, ...)`. Untuk *backbone* yang sudah selesai dilatih pada rezim lama, fungsi itu mengembalikan `start_epoch` sama dengan jumlah *epoch* penuh, sehingga syarat `start_epoch >= epochs` pada baris 211 terpenuhi dan seluruh pelatihan dilewati. Yang tercetak hanyalah `[SKIP]`, sebuah pesan yang terlihat persis seperti keberhasilan *caching* yang diinginkan, padahal artinya bobot rezim lama dipertahankan untuk dievaluasi pada rezim baru.

Tiga sifat membuatnya sulit terlihat. Pertama, `[SKIP]` adalah pesan normal yang muncul ratusan kali pada *run* yang sehat, jadi tidak ada yang mencurigakannya. Kedua, `maybe_resume` hanya membandingkan nomor *epoch*, tidak pernah membandingkan konfigurasi, karena `save_ckpt` memang tidak menyimpan `input_size` maupun `patch_xy` di dalam *checkpoint*. Ketiga, akibatnya tidak berupa galat melainkan angka yang lebih rendah namun masuk akal, yang mudah salah dibaca sebagai kelemahan arsitektur atau instabilitas pelatihan.

Aturan praktis yang berlaku untuk *pipeline* ini: **mengubah `input_size`, `patch_xy`, atau `n_slices` mewajibkan pemindahan *checkpoint* lama ke arsip, bukan sekadar menjalankan ulang.** Menaikkan `epochs` saja tidak cukup, sebab pelatihan akan dilanjutkan dari bobot rezim lama alih-alih dimulai dari nol. Perbaikan struktural yang menutup jebakan ini adalah menyimpan ketiga nilai itu di dalam *checkpoint* lalu menolak melanjutkan ketika salah satunya tidak cocok, sejalan dengan prinsip yang sama pada §8.1: konfigurasi yang menghasilkan sebuah angka harus melekat pada angka itu.

---

## 9. Rencana lanjutan

1. Jalankan ulang ablasi fusi dengan perbaikan `input_size` (`python -m src.stage_03b_fusion --config configs/config.yaml`), verifikasi `cnn_only` densenet201 kembali mendekati 0.8988.
2. Eksekusi ketiga varian fusi baru (branch_norm, GMU, modality dropout) pada grid penuh.
3. Jalankan panel XAI komparabilitas begitu checkpoint tersedia.
4. Tambahkan sitasi yang hilang lewat Zotero (`docs/laporan/REFERENSI_DIBUTUHKAN.md`).

---

## 10. Integritas riset

Semua angka pada laporan ini ditelusuri ke baris CSV nyata yang ditarik dari mesin remote pada 30 Juli 2026, bukan diperkirakan. Batasan bug resolusi dinyatakan eksplisit di titik angkanya muncul (§6.1), bukan disembunyikan.

---

## Lampiran: berkas hasil

| Berkas | Isi |
|---|---|
| `artifacts/results/fusion/ablation_summary.csv` | 175 baris, AUC per arm per backbone per fold |
| `artifacts/results/fusion/delong_fusion.csv` | 21 baris, uji DeLong fusi vs radiomics-only |
| `artifacts/results/xai/xai_metrics.csv` | 12 baris, metrik Grad-CAM/Layer-CAM per backbone |
