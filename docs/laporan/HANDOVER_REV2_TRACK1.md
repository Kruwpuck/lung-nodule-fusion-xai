# Handover Rev2 Track 1: perbaikan kebocoran, artefak checkpoint, dan status menuju Step 2

Dokumen ini menyerahkan keadaan pekerjaan Rev2 Track 1 per 4 Agustus 2026. Isinya apa yang
sudah dikerjakan beserta bukti angkanya, apa yang belum, dan apa yang harus diketahui
sebelum melanjutkan. Pembaca yang dituju adalah orang yang meneruskan pekerjaan ini tanpa
mengikuti sesi sebelumnya.

Laporan hasil lengkap ada di `docs/laporan/LAPORAN_TRACK1_FUSION_XAI.md`. Dokumen ini
tidak menggantikannya, melainkan merangkum keputusan dan keadaan operasional.

> **Dibaca setelah 5 Agustus 2026? Mulai dari sini.** Dokumen ini bertanggal 4 Agustus.
> Sehari sesudahnya, run `2026-08-04-run02` mengubah kesimpulan Track 1 untuk satu arm.
> Seluruh perbaikan dan angka operasional di bawah tetap berlaku, tetapi kalimat
> "radiomics-only mengungguli seluruh varian fusi" sekarang **terlalu luas**. Yang
> dikalahkan radiomics adalah fusi berparameter, yaitu `fusion_early` dan
> `fusion_intermediate`. `fusion_late` setara dengannya, dan menang telak atas `cnn_only`.
> Klaim final ada di §6.3 laporan Track 1; jangan mengutip dokumen ini sebagai kesimpulan.

---

## 1. Ringkasan satu paragraf

Empat perbaikan dijalankan berurutan: bug resolusi input, kebocoran seleksi *epoch* pada
ablasi fusi, seleksi fitur yang tidak deterministik, dan *checkpoint* DenseNet121 dari dua
rezim resolusi berbeda. Setiap perbaikan diukur terhadap arsip *baseline*, dan tiga di
antaranya mengubah angka secara berarti. Kesimpulan utama tidak berubah, malah menguat:
radiomics-only mengungguli varian fusi **berparameter**, dan nol dari 21 uji DeLong
mendukung fusi. Kebocoran ternyata menyamarkan kekalahan, bukan menciptakan kemenangan.
Batas cakupannya penting: ke-21 uji itu membandingkan tiap arm fusi terhadap arm tunggal
terbaik, dan tidak satu pun **mendukung** fusi — tetapi tidak terbedakan secara statistik
berbeda artinya dengan kalah. `fusion_late` termasuk yang tidak terbedakan, dan run
`2026-08-04-run02` kemudian menguji arm itu tersendiri (§6.3 laporan Track 1).

Step 2 (dua tahap *fine-tuning* dengan BatchNorm dibekukan) **belum dimulai** dan menunggu
keputusan.

---

## 2. Keadaan angka saat ini

### 2.1 Ablasi fusi, 175 baris

Sumber: `artifacts/results/fusion/ablation_summary.csv`, 7 *backbone* x 5 arm x 5 *fold*.

| Arm | AUC final | Sebelum nested CV | Baseline pra-Rev2 |
|---|---|---|---|
| fusion_late | 0.9349 | 0.9332 | 0.9171 |
| radiomics_only | 0.9324 | 0.9314 | 0.9313 |
| fusion_early | 0.9119 | 0.9126 | 0.9179 |
| fusion_intermediate | 0.9098 | 0.9294 | 0.9269 |
| cnn_only | 0.9018 | 0.8927 | 0.7853 |

Kolom `fs_method` terisi `mutual_info_classif` pada 175 dari 175 baris, nol kosong.

### 2.2 Uji DeLong, 21 pasangan

Sumber: `artifacts/results/fusion/delong_fusion.csv`. Kolom
`fusion_significantly_better` bernilai `False` pada seluruh 21 baris.

fusion_late unggul angka pada 5 dari 7 backbone, tetapi p terkecil di sisi menang adalah
0.2040 (`densenet201`). fusion_early kalah signifikan pada ketujuh backbone.
fusion_intermediate kalah signifikan pada enam dari tujuh, dengan p terkecil 2.6e-7 pada
`inception_resnet_v2`.

### 2.3 AUC standalone per backbone Track 1

Sumber: `artifacts/results/summary_binary.csv`, baris `track == "track1"`.

| Backbone | AUC rata-rata |
|---|---|
| convnext_tiny | 0.9055 |
| inceptionv3 | 0.8992 |
| densenet201 | 0.8988 |
| inception_resnet_v2 | 0.8986 |
| googlenet | 0.8962 |
| densenet121 | 0.8940 |
| xception | 0.8911 |

Rentangnya sempit, 0.0144 dari terendah ke tertinggi. DenseNet121 sempat tercatat 0.8333
dan tampak sebagai *outlier*; angka itu artefak, lihat bagian 3.4.

---

## 3. Empat perbaikan yang dijalankan

### 3.1 Metode seleksi fitur salah dilaporkan

`src/radiomics/feature_selection.py` mengimpor `pymrmr`, menangkap `ImportError`, lalu
beralih ke `mutual_info_classif` sambil hanya menulis satu baris `logger.warning`.
`pymrmr` tidak terpasang di mesin remote, jadi cabang mRMR tidak pernah dieksekusi sekali
pun. Seluruh angka radiomics dihasilkan oleh *mutual information*.

Perbaikannya bersifat teks, bukan eksperimen, karena yang salah adalah deskripsinya bukan
metodenya. Seleksi berbasis mutual information sah dan tetap dijalankan per fold pada
data latih saja. Yang ditambahkan adalah jaminan struktural: `mrmr_select` sekarang
mengembalikan nama metode yang benar-benar berjalan, dan nama itu ditulis ke kolom
`fs_method` pada setiap baris hasil.

*Commit* `3bb42b4`. Rincian di §8.1 laporan.

### 3.2 Kebocoran seleksi epoch pada ablasi fusi

`_train_fusion_fold` memilih epoch terbaik berdasarkan AUC pada fold validasi luar,
lalu melaporkan skor pada fold yang sama. Kebocoran ini **asimetris antar-arm**, dan itu
bagian yang penting. `radiomics_only` bersih karena `train_early_fusion_xgboost` memanggil
`clf.fit` tanpa `eval_set`. Karena pembanding satu-satunya yang bersih justru adalah
radiomics, setiap kemenangan tipis arm fusi di atasnya diukur dengan timbangan berat
sebelah.

Perbaikannya memakai *inner split* per pasien, yaitu `GroupShuffleSplit` 85/15 yang
disemai per fold, diambil dari fold pelatihan luar. Epoch terbaik dipilih dari AUC
*inner-validation*, dan fold validasi luar hanya dievaluasi sekali setelah pelatihan
selesai. Cakupannya sengaja dibatasi pada `src/stage_03b_fusion.py`; 215 *run* Track 2 di
`src/training/trainer.py` tidak disentuh.

Hasilnya menurunkan `fusion_intermediate` sebesar 0.0210 AUC. Tiga arm lain bergerak
di bawah 0.0005 dan `cnn_only` nol persis, yang berfungsi sebagai konfirmasi silang bahwa
perbaikan mengenai sasaran: hanya `fusion_intermediate` yang melewati `_train_fusion_fold`.

Commit `31a4ecd`, dengan perbaikan susulan `da7e0cb` untuk `drop_last=True` pada *loader*
pelatihan. Pemotongan 85 persen mengubah ukuran himpunan latih sehingga sebagian kombinasi
fold dan backbone menghasilkan *batch* terakhir berukuran satu, yang ditolak
`BatchNorm` dalam moda pelatihan.

### 3.3 Seleksi fitur tidak deterministik

Setelah nested CV, `radiomics_only` bergerak sampai 0.0036 AUC per backbone padahal nol
baris kode pada jalur arm itu berubah. Penyebabnya `mutual_info_classif` yang dipanggil
tanpa `random_state`; penaksir *k-nearest-neighbour*-nya menambahkan derau untuk memecah
nilai seri.

Verifikasi pada matriks fitur asli berisi 1130 kolom: dua pemanggilan tanpa *seed*
menghasilkan himpunan berbeda pada 4 fitur, sementara dua pemanggilan dengan
`random_state=42` identik. Sumber acak lain sudah tertutup, `train_early_fusion_xgboost`
memakai `random_state=42` dan `LassoCV` deterministik pada moda `cyclic`.

Angka lantai derau 0.0036 diukur **sebelum** seed dipasang. Ia berguna sebagai ambang
untuk menafsirkan hasil lama, tetapi tidak berlaku untuk run setelah `733856b`.

### 3.4 Checkpoint DenseNet121 dari dua rezim resolusi

AUC standalone DenseNet121 0.8333 jauh di bawah enam backbone lain, dengan dua fold
nyaris kolaps (sensitivitas 0.0864 pada fold 1 dan 0.1196 pada fold 4). Membacanya
sebagai instabilitas pelatihan keliru.

Bukti yang menentukan bukan tanggal berkas melainkan `best_auc` yang tersimpan di dalam
setiap checkpoint. Dibandingkan dengan AUC hasil evaluasi ulang, 32 dari 35 checkpoint
Track 1 cocok sampai 0.0000. Hanya DenseNet121 fold 0, 1, dan 4 yang menyimpang, dengan
selisih 0.0965, 0.0787, dan 0.1079.

Penyebabnya, `input_size: 96` masuk pada commit `0b54376` yang sekaligus menambahkan
enam backbone Track 1 lainnya. Keenamnya karena itu tidak pernah ada sebelum rezim 96
piksel. DenseNet121 satu-satunya yang berasal dari himpunan model legacy dan sudah dilatih
sejak 14 Juli. Ketika *pipeline* dijalankan ulang, `maybe_resume` menemukan epoch 49
dengan `epochs: 50` sehingga syarat `start_epoch >= epochs` pada
`src/stage_03_train.py:211` terpenuhi dan baris berikutnya mencetak `[SKIP]` tanpa
melatih satu epoch pun.

Kelima fold dilatih ulang dari nol dalam dua tahap, dengan checkpoint lama dipindahkan
ke `artifacts/checkpoints/_archive_densenet121_pre_input_size/` (10 berkas, tidak dihapus
karena menjadi bukti bagi §8.6 laporan).

| Fold | AUC lama | AUC baru | Sensitivitas lama | Sensitivitas baru |
|---|---|---|---|---|
| 0 | 0.8018 | 0.9181 | 0.2812 | 0.7708 |
| 1 | 0.8272 | 0.9124 | 0.0864 | 0.6914 |
| 2 | 0.8659 | 0.8632 | 0.7312 | 0.6989 |
| 3 | 0.8911 | 0.8841 | 0.6739 | 0.6957 |
| 4 | 0.7804 | 0.8921 | 0.1196 | 0.7283 |

Tahap pertama menyentuh fold 0, 1, dan 4 saja; selama itu fold 2 dan 3 bergerak nol
persis, yang menjadi kontrol bahwa lonjakan ketiga fold lain berasal dari pelatihan
ulang. Tahap kedua melatih ulang fold 2 dan 3 demi keseragaman protokol, dan keduanya
turun tipis (0.0027 dan 0.0069). Penurunan itu diperlakukan sebagai angka yang lebih
jujur, sebab angka lama berasal dari jalur yang tidak dapat ditelusuri.

Ablasi fusi kemudian dijalankan ulang khusus untuk DenseNet121, ditulis ke direktori
terpisah lalu digabungkan, sehingga 150 baris milik enam backbone lain tidak tersentuh.

Commit `84c48c0`, `1994b4a`, `fd2de16`.

---

## 4. Analisis komplementaritas modalitas

Pertanyaan yang dijawab: apakah kekalahan fusi berakar pada arsitektur fusi atau pada
ketimpangan modalitas. Dijawab tanpa GPU baru, dengan mengambil probabilitas CNN dari
`preds/*.npz` yang sudah ada lalu menghitung ulang probabilitas radiomik memakai pipeline
seleksi yang sama.

| Backbone | Pearson | Spearman | CNN benar, radiomik salah | Radiomik benar, CNN salah |
|---|---|---|---|---|
| inception_resnet_v2 | 0.831 | 0.731 | 14.4 | 17.2 |
| convnext_tiny | 0.824 | 0.740 | 13.6 | 22.2 |
| xception | 0.819 | 0.705 | 15.0 | 16.2 |
| densenet121 | 0.812 | 0.712 | 13.4 | 21.8 |
| inceptionv3 | 0.805 | 0.684 | 14.0 | 19.2 |
| densenet201 | 0.796 | 0.697 | 15.6 | 18.6 |
| googlenet | 0.795 | 0.692 | 15.2 | 19.4 |

Rata-rata per fold, n rata-rata 273, ambang keputusan 0,5.

CNN punya kontribusi unik tetapi kecil. Korelasinya tinggi namun jauh dari jenuh, dan CNN
benar sementara radiomik salah pada sekitar 14 kasus per fold atau 5,1 persen, stabil di
ketujuh backbone. Arah ketimpangannya tetap jelas: radiomik unik-benar pada 16 sampai 22
kasus, lebih banyak.

Satu batasan tafsir yang harus dijaga, angka-angka ini adalah keputusan keras pada ambang
0,5. Benar pada ambang tidak otomatis berarti tambahan AUC. Data ini **tidak** membuktikan
bahwa memperkuat CNN akan menolong, dan juga tidak membuktikan sebaliknya.

Baris DenseNet121 di tabel ini adalah hasil hitung ulang setelah pelatihan ulang. Sebelum
itu ia mencatat Pearson 0.626 dan radiomik-unik-benar 43.6, dua kali lipat backbone
lain; kedua anomali hilang setelah checkpoint-nya benar, yang menjadi bukti tambahan
bahwa penyimpangannya berasal dari bobot rezim salah.

---

## 5. Pola yang berulang, dan mengapa penting untuk penerus

Enam temuan reproducibility pada §8.1 sampai §8.8 laporan berbagi satu bentuk. Bukti untuk
setiap temuan **sudah tercetak atau tersimpan** sebelum ditemukan, sebagian bahkan
berulang pada setiap run.

| Bagian laporan | Bukti yang sudah ada | Tidak terbaca sejak |
|---|---|---|
| §8.1 | `logger.warning` menyebut `pymrmr` tidak terpasang | awal proyek |
| §8.3 | proses hilang dari daftar proses setelah sesi SSH ditutup | insiden pertama |
| §8.4 | `n_val` 1366 lawan 1391 tercetak di setiap baris log fold | ablasi pertama |
| §8.5 | `LASSO selected 29/50` lawan `25/50` untuk arm yang tidak bergantung backbone | ablasi pertama |
| §8.6 | `best_auc` tersimpan di dalam setiap checkpoint | 14 Juli |
| §8.8 | `[SKIP]` tercetak untuk kelima fold DenseNet121 | 28 Juli |

Nol dari enam membutuhkan eksperimen baru untuk ditemukan. Yang hilang bukan datanya
melainkan pembandingnya. Konsekuensi praktis bagi penerus: pemeriksaan konsistensi yang
murah lebih berharga daripada log yang lengkap, sebab log lengkap justru menenggelamkan
anomali di antara ribuan baris normal.

---

## 6. Aturan operasional yang wajib dipatuhi

### 6.1 Mengubah resolusi mewajibkan pengarsipan checkpoint

Mengubah `input_size`, `patch_xy`, atau `n_slices` **wajib** disertai pemindahan
checkpoint lama ke arsip. Menjalankan ulang saja tidak cukup, dan menaikkan `epochs`
juga tidak cukup, sebab pelatihan akan dilanjutkan dari bobot rezim lama alih-alih dimulai
dari nol. `maybe_resume` hanya membandingkan nomor epoch dan tidak pernah membandingkan
konfigurasi, karena `save_ckpt` memang tidak menyimpan ketiga nilai itu.

Gejalanya bukan galat melainkan angka lebih rendah yang masuk akal, dan pesan yang muncul
hanyalah `[SKIP]`, yang terlihat persis seperti *caching* yang sehat.

### 6.2 Verifikasi PID sebelum melaporkan status proses terlepas

`Start-Process` pada mesin remote tetap menjadi anak dari *job object* milik sesi SSH,
sehingga Windows menghentikannya begitu sesi ditutup. Prosesnya bertahan selama
pemeriksaan di dalam sesi dan mati persis ketika sesi berakhir, sehingga gejalanya
menyerupai kegagalan acak.

Gunakan `Invoke-CimMethod -ClassName Win32_Process -MethodName Create` dengan
`CurrentDirectory` dan jalur log absolut. Jalur log relatif menyebabkan berkas log ditulis
ke direktori kerja default WMI, bukan repo. Status proses **wajib** diverifikasi dengan
memeriksa PID dari sesi SSH yang benar-benar baru sebelum dilaporkan.

### 6.3 Uji negatif ketika mengganti sebuah uji

Mengganti sebuah uji belum selesai sebelum uji penggantinya dibuktikan menangkap kegagalan
yang seharusnya ia tangkap. Untuk `test_registry_covers_every_configured_backbone`
pembuktian dilakukan dengan menghapus `densenet201` dari registry lalu memastikan uji
tersebut gagal dan menyebut nama itu.

Prinsip yang sama berlaku untuk hipotesis. Dua uji sintetis pertama untuk determinisme
`mutual_info_classif` menyimpulkan "deterministik walau tanpa seed", dan keduanya salah
karena datanya terlalu mudah; nilai MI terpisah jauh sehingga derau 1e-10 tidak membalik
peringkat. Baru pengujian pada matriks 1130 fitur asli memberi jawaban yang benar.

---

## 7. Reproducibility

### 7.1 Lingkungan

Eksekusi berlangsung di mesin remote `100.98.9.120`, repo di
`C:\Users\Adaptive Network\Documents\Lung Cancer\lung-nodule-fusion-xai`, GPU NVIDIA RTX
3060 12 GB. Interpreter yang benar adalah `.venv\Scripts\python.exe` (Python 3.11.9,
pandas 2.2.2), **bukan** `python` polos yang teresolusi ke Python 3.11 sistem tanpa
dependensi proyek. Kesalahan interpreter pernah menghasilkan bukti yang tidak sahih pada
sesi ini.

`pylidc` terpasang di remote dan tidak terpasang di mesin lokal, yang menjelaskan 9
kegagalan uji lokal di `tests/test_data_loading.py`. Kegagalan itu nol kaitannya dengan
perubahan Rev2. Suite lokal: 138 lulus, 9 gagal.

### 7.2 Konfigurasi kunci

Dari `configs/config.yaml`:

| Parameter | Nilai |
|---|---|
| `tracks.track1.input_size` | 96 |
| `data.patch_xy` | 64 |
| `data.n_slices` | 3 |
| `data.n_folds` | 5 |
| `train.epochs` | 50 |
| `train.early_stopping_patience` | 10 |
| `train.lr` | 1e-4 |
| `train.batch_size` | 16 |
| `track1_fusion.fusion_arms` | `["concat"]` |
| `track1_fusion.modality_dropout_rate` | 0.0 |
| `track1_fusion.aux_loss_weight` | 0.0 |

`input_size: 96` dipilih sebagai kelipatan 32 terkecil di atas *floor* semua backbone
(InceptionV3 dan Inception-ResNet-v2 mensyaratkan 75, Xception 71).

### 7.3 Perintah

Semua dijalankan dari akar repo di remote dengan interpreter *venv*:

```
python -m src.stage_03_train --config configs/config.yaml --model densenet121 --fold 0 --task binary
python -m src.stage_04_evaluate --config configs/config.yaml --task binary
python -m src.stage_03b_fusion --config configs/config.yaml
```

Membatasi ablasi ke satu backbone dilakukan dengan menyalin konfigurasi dalam memori,
mengganti `tracks.track1.backbones` menjadi satu nama, mengarahkan `paths.results` ke
direktori sementara, lalu menggabungkan baris hasilnya. Menjalankan `stage_03b_fusion`
dengan konfigurasi terbatas tanpa mengalihkan `paths.results` akan **menimpa** 175 baris
`ablation_summary.csv` dengan 25 baris saja.

### 7.4 Pembagian data

Lima fold dibagi per `patient_id`, disemai 42, sehingga nol pasien menyumbang kasus ke
lebih dari satu fold. Pembagian ini dibekukan dan tidak diubah selama Rev2, agar 215
run Track 2 tetap sahih.

Kohort ablasi fusi berisi 1366 nodul, sedangkan evaluasi standalone 1391. Selisih 25
berasal dari `_load_merged` yang membuang 42 baris berkunci `(patient_id, nodule_idx)`
ganda milik pasien multi-scan. Penyempitan ini sebelumnya tidak dinyatakan, sehingga angka
Bab 6.1 dan angka fusi pernah dibandingkan pada penyebut berbeda. Selisih AUC yang
ditimbulkannya berkisar 0.0019 sampai 0.0070.

### 7.5 Uji statistik

Uji DeLong berpasangan, satu per backbone per varian fusi terhadap `radiomics_only`,
totalnya 21. Ambang 0,05. Aturan keputusan ditetapkan sebelum hasil dilihat: fusi
dilaporkan sebagai *headline* hanya bila p terhadap arm tunggal terbaik di bawah 0,05.

### 7.6 Artefak yang dibekukan

`artifacts/results/_baseline_pre_rev2/` adalah titik referensi wajib dan tidak boleh
ditimpa atau dihapus. Isinya `ablation_summary.csv` dan `delong_fusion.csv` pra-Rev2, 150
berkas `preds/*.npz`, dan `xai/xai_metrics.csv`. README di dalamnya memuat peringatan
provenance untuk `preds/densenet121_fold*.npz`, yang berbeda dari versi ter-commit sejak
29 Juli.

`artifacts/results/_leaky_pre_nestedcv/` menyimpan hasil sebelum nested CV, dipakai untuk
menghitung selisih bias seleksi 0.0210.

---

## 8. Yang belum dikerjakan

### 8.1 Step 2 dan Step 3, menunggu keputusan

Step 2 adalah dua tahap fine-tuning dengan BatchNorm dibekukan; Step 3 menjalankan arm
`branch_norm`, `gmu`, dan *modality dropout* yang kodenya sudah ada tetapi belum pernah
dieksekusi pada grid penuh. Urutan sudah diputuskan: nested CV mendahului keduanya, dan
itu sudah selesai.

Batas yang disepakati: Step 2 dan Step 3 adalah upaya sah terakhir. Bila setelah cabang
CNN diperbaiki dan fusi direbalans hasilnya tetap 0 dari 21, maka "radiomics mengungguli
fusi berparameter" adalah temuan final dan dilaporkan apa adanya.

Catatan per 5 Agustus 2026: batas itu kini hanya mengikat arm fusi berparameter. Untuk
`fusion_late`, run `2026-08-04-run02` sudah menjalankan pengujian tersendiri dan hasilnya
ada di §6.3 laporan Track 1, jadi arm itu tidak lagi menunggu Step 2 maupun Step 3.

Data komplementaritas pada bagian 4 relevan untuk keputusan ini dan sebaiknya dibaca lebih
dulu.

### 8.2 Uji DeLong sebelum lawan sesudah memakai preds arsip

Diantrikan, tidak mendesak. Tujuannya mengukur seberapa jauh bug resolusi mendistorsi
kesimpulan tanpa pelatihan ulang. Perlu diperhatikan bahwa `preds/densenet121_fold*.npz`
di arsip punya provenance tidak pasti, sehingga uji ini paling sahih untuk enam backbone
lainnya.

### 8.3 Perbaikan struktural yang diusulkan tetapi belum dikerjakan

Menyimpan `input_size`, `patch_xy`, dan `n_slices` di dalam checkpoint lalu menolak
melanjutkan ketika salah satunya tidak cocok. Ini menutup jebakan pada bagian 6.1 secara
permanen, sejalan dengan prinsip yang sama pada kolom `fs_method`.

### 8.4 Sisa dari Rev1

Sweep LR SGD (tugas 7), panel XAI komparabilitas (tugas 8), dan *re-run* stabilitas (tugas
6) masih berstatus `done-code` dan belum pernah dieksekusi.

### 8.5 Naskah

Build LaTeX kedua manuskrip belum pernah diverifikasi karena `latexmk` tidak ada di mesin
lokal. Sitasi yang dibutuhkan didaftar di `docs/laporan/REFERENSI_DIBUTUHKAN.md` dan harus
ditambahkan lewat Zotero, termasuk rujukan Rev2: Raghu et al. Transfusion (arXiv:1902.07208),
Howard dan Ruder ULMFiT (arXiv:1801.06146), Peng et al. OGM-GE (CVPR 2022), Baltatzis et al.
(MLMI 2021, arXiv:2108.05386), dan Demircioğlu (DOI 10.1186/s13244-021-01115-1).

`docs/laporan/RESEARCH_JOURNEY_REPORT_FILLED.md` baris 92 dan 187 masih memuat teks lama
yang menyebut mRMR lebih dulu. Berkas itu keluaran `stage_07f_journey_report.py` yang sudah
diperbaiki, jadi akan terkoreksi sendiri ketika dibangkitkan ulang.

`paper/track1/main.tex` masih memuat angka pra-Rev2 pada Tabel `tab:fusion` dan pernyataan
bahwa ablasi belum dijalankan ulang. Keduanya kini tidak akurat dan perlu diperbarui ke
angka pada bagian 2.1.

---

## 9. Riwayat commit sesi ini

Dari yang terlama:

| Commit | Isi |
|---|---|
| `3bb42b4` | laporkan metode seleksi fitur yang benar-benar berjalan |
| `64eefdd` | uji untuk jalur seleksi yang tak pernah diuji, invarian registry |
| `10a11bc` | ablasi ulang dengan perbaikan `input_size` |
| `31a4ecd` | tutup kebocoran *early stopping* dengan inner split per pasien |
| `da7e0cb` | `drop_last` untuk menghindari galat BatchNorm pada batch berukuran satu |
| `733856b` | semai `mutual_info_classif` agar seleksi fitur dapat diulang |
| `e5448a1` | hasil nested CV, kuantifikasi bias seleksi 0.0210 |
| `8df41d6` | koreksi diagnosis checkpoint DenseNet121, catat pola berulang |
| `84c48c0` | latih ulang DenseNet121 fold 0/1/4 |
| `b8ced86` | catat jebakan `input_size` tanpa pelatihan ulang |
| `1994b4a` | latih ulang fold 2/3 agar protokol seragam |
| `fd2de16` | ablasi ulang DenseNet121 pada checkpoint baru |

Seluruhnya sudah didorong ke `origin/main`. Pohon kerja bersih.

---

## 10. Koreksi yang dibuat terhadap pernyataan sendiri

Dicatat karena penerus mungkin menemukan versi lama di riwayat percakapan atau pesan
commit awal.

Bukti pertama tentang `pymrmr` diambil dari interpreter yang salah, sehingga tidak
membuktikan apa pun tentang pipeline. Diverifikasi ulang di dalam venv, kesimpulannya
tidak berubah.

Klaim bahwa `paper/track1/main.tex` menyebut mRMR keliru; berkas itu tidak pernah
menyebutnya, dan `docs/laporan/RESEARCH_JOURNEY_REPORT.md:175` justru sudah mengungkap
*fallback* itu dengan benar.

Hipotesis bahwa selisih 0.0037 pada `cnn_only` berasal dari kebocoran early stopping
tertolak. Kebocoran itu ada di kedua jalur sehingga simetris dan tidak dapat menghasilkan
selisih. Penyebab sebenarnya penyempitan kohort, terverifikasi eksak sampai desimal
keempat.

Mekanisme checkpoint DenseNet121 yang semula dijelaskan sebagai "satu epoch tersisa
sehingga `best.pt` hanya tertimpa pada fold yang kebetulan membaik" keliru. Yang terjadi
adalah `[SKIP]` total tanpa satu epoch pun berjalan. Provenance fold 2 dan 3 karena itu
bukan warisan-*resume* melainkan tidak terlacak, dan itulah alasan keduanya ikut dilatih
ulang.

---

## 11. Langkah pertama bagi penerus

Baca bagian 2 untuk angka terkini, lalu bagian 4 untuk data yang menentukan bentuk Step 2.
Setelah itu putuskan bersama pemilik pekerjaan apakah Step 2 dijalankan atau hasil negatif
dilaporkan apa adanya. Jangan memulai Step 2 tanpa keputusan itu.

Sebelum menjalankan apa pun di remote, patuhi bagian 6.2 tentang verifikasi PID, dan
bagian 6.1 bila menyentuh resolusi.
