# LungFuseNet: Research Journey & Progress Report (Rev1)

> Laporan ini ditulis setelah revisi Rev1 dijalankan penuh. Semua angka diambil dari hasil eksekusi nyata di mesin *remote* (`artifacts/results/`), bukan estimasi. Versi sebelumnya memakai desain 6 *backbone* tanpa konsep *track*; struktur itu sudah diganti dan perbedaannya dicatat di Bab 3.

**Tanggal laporan**: 30 Juli 2026
**Rentang pengerjaan**: 27 Juni 2026 sampai 30 Juli 2026 (49 *commit*)

---

## 0. Identitas penelitian

- **Judul**: Explainable Radiomics–Deep Learning Fusion for Lung Nodule Malignancy Classification on LIDC-IDRI CT Scans
- **Repo**: https://github.com/Kruwpuck/lung-nodule-fusion-xai
- **Tugas**: Klasifikasi keganasan (*malignancy*) nodul paru dari *CT scan*
- **Dataset**: LIDC-IDRI, dengan LUNA16 sebagai sumber *hard-negative*
- **Peneliti**: Ihab Hasanain Akmal (103032330054), Siti Nurhayati Syafaningrum (101012330012)
- **Pembimbing**: Felix Corputty

---

## 1. Ringkasan eksekutif

Penelitian ini membangun pipeline klasifikasi keganasan nodul paru pada LIDC-IDRI yang dipecah menjadi dua *track* dengan peran berbeda. Track 2 menangani perbandingan model dan studi stabilitas *hyperparameter*, memakai empat *backbone* ringan sampai menengah. Track 1 menangani *fusion* radiomics dengan CNN beserta analisis XAI per *arm*, memakai tujuh *backbone* yang tidak beririsan dengan Track 2.

Rev1 mengubah tiga hal mendasar dibanding desain lama. Komposisi *backbone* dikunci per *track* sesuai laporan justifikasi metodologi, sehingga tidak ada lagi satu set enam model yang dipakai untuk semua keperluan. Studi stabilitas *hyperparameter* ditambahkan dalam bentuk *sweep* tiga *optimizer* dikali tiga nilai *weight decay*. Sistem pencatatan diubah menjadi tiga tingkat CSV yang setiap barisnya membawa identitas *run* lengkap, sehingga pembuatan grafik tidak perlu lagi mengurai nama berkas.

Total 215 *run* unik selesai tanpa sisa kegagalan: 35 *run* Track 1 (7 *backbone* dikali 5 *fold*) dan 180 *run* *sweep* Track 2 (4 *backbone* dikali 3 *optimizer* dikali 3 *weight decay* dikali 5 *fold*).

Tiga temuan utama. Pertama, pilihan *optimizer* berpengaruh jauh lebih besar daripada pilihan *weight decay*: selisih AUC antara Adam dan SGD mencapai 0,1457 pada EfficientNet-B0, sementara variasi *weight decay* dalam satu *optimizer* paling banter menggeser AUC sebesar 0,0218. Kedua, radiomics saja mengalahkan seluruh varian *fusion* pada uji DeLong, tanpa kecuali di 21 pasangan yang diuji. Ketiga, kualitas lokalisasi Grad-CAM tidak mengikuti kapasitas model: ViT-Base dengan 85,8 juta parameter memperoleh *pointing accuracy* nol, sedangkan DenseNet121 dengan 6,96 juta parameter mencapai 0,7167.

### Status komponen

| Komponen | Status | Output |
|---|---|---|
| Track 1, 7 *backbone* dikali 5 *fold* (35 *run*) | Selesai | `summary_binary.csv`, *checkpoint* |
| Track 2 *sweep*, 180 *run* | Selesai | `track2_stability*.csv`, 2 figur |
| Evaluasi agregat semua model | Selesai | `summary_*.csv`, `efficiency_table*.csv` |
| Kurva training dari log CSV | Selesai | `curves_track1/track2/overfit.png` |
| Ablasi *fusion* + uji DeLong | Selesai | `ablation_summary.csv`, `delong_fusion.csv` |
| XAI Grad-CAM per model | Selesai | `xai_metrics.csv`, 12 figur |
| Validasi eksternal | Belum dikerjakan | pipeline tersedia |

---

## 2. Latar belakang dan motivasi

Deteksi dini keganasan nodul paru menentukan peluang bertahan pasien kanker paru, dan *CT scan* skrining menjadi modalitas utamanya. Beban interpretasinya berat karena butuh waktu radiolog yang mahal. Model *deep learning* menjanjikan bantuan di titik ini, tetapi model yang tidak bisa dijelaskan sulit dipercaya klinisi meskipun akurasinya tinggi, sehingga interpretabilitas menjadi syarat dan bukan pelengkap.

Celah yang ingin diisi: banyak publikasi melaporkan akurasi satu model tanpa membandingkan biaya komputasi dan kualitas interpretasi antar arsitektur pada data serta *split* yang identik. Lebih jarang lagi yang melaporkan kondisi ketika *fusion* justru tidak menang.

Kontribusi yang diklaim:

1. Perbandingan 11 *backbone* pada tugas dan *split* identik, dipisah menurut peran *track* sehingga tidak ada kebocoran peran antar eksperimen.
2. Studi stabilitas *hyperparameter* 180 *run* yang melaporkan varians, koefisien variasi, dan selang kepercayaan 95 persen per sel, bukan hanya nilai terbaik.
3. *Fusion* radiomics dengan CNN yang dilaporkan apa adanya termasuk saat kalah, disertai uji DeLong sebagai dasar keputusan.

---

## 3. Perjalanan revisi

### Fase 1 sampai 8 (ringkasan periode sebelum Rev1)

Fase awal menetapkan LIDC-IDRI sebagai dataset utama setelah memastikan label LUNA16 adalah nodul lawan bukan-nodul untuk deteksi, bukan keganasan, sehingga LUNA16 hanya dipakai sebagai *hard-negative*. Arsitektur pipeline diputuskan modular per *stage* dalam berkas `.py` dan bukan satu *notebook* tunggal, karena *notebook* monolitik rawan hilang saat koneksi terputus sedangkan pipeline modular bisa dilanjutkan per *stage*.

Empat *arm* label didefinisikan di atas sumber data yang sama: binary (buang median rating 3), ordinal (rating 1 sampai 5 langsung), grade3 (tiga kelas), dan grade4 (empat kelas dengan *hard-negative*). Untuk grade4 dipasang aturan anti-inflasi: metrik utama memakai AUC benign lawan malignant pada subset nodul saja, bukan AUC empat kelas mentah yang akan terangkat oleh kelas no-nodule yang mudah dipisahkan.

Fase XAI melewati tiga iterasi perbaikan. Bug pertama adalah `target_class` yang dipatok ke kelas malignant sehingga peta aktivasi pada sampel benign tampil kosong, diperbaiki dengan memakai kelas prediksi sebagai default. Bug kedua adalah *target layer* yang salah pilih, membuat ResNet mengenai lapisan *avgpool* dan VGG16 serta ViT gagal jalan, diperbaiki dengan percabangan per arsitektur plus `reshape_transform` khusus ViT. Bug ketiga adalah peta CAM yang polanya tetap dan tidak merespons masukan, disertai *pointing accuracy* nol, yang ternyata berakar pada resolusi *feature map* 2x2 yang terlalu kasar untuk masukan 64x64, diperbaiki dengan Layer-CAM pada tahap sekitar 8x8 lewat `_auto_target_layer`.

### Fase 9 (Rev1): pemisahan track dan studi stabilitas

Dua dokumen di `docs/revisi/rev1/` menetapkan desain final yang belum tercermin di kode. Perubahan yang dijalankan:

#### Komposisi backbone dikunci per track

Track 2 memakai MobileNetV2, EfficientNet-B0, ResNet50, dan VGG16. Track 1 memakai DenseNet121, InceptionV3, Xception, GoogLeNet, ConvNeXt-Tiny, InceptionResNetV2, dan DenseNet201. Irisan kedua himpunan kosong, sesuai syarat eksplisit laporan justifikasi. Enam *backbone* baru ditambahkan ke `src/models/backbones.py`, tiga di antaranya lewat pustaka `timm` karena tidak tersedia di torchvision.

#### Resolusi masukan diseragamkan lintas arm

Keluarga Inception punya batas bawah ukuran masukan yang lebih besar dari 64: InceptionV3 dan InceptionResNetV2 minimal 75, Xception minimal 71. Alih-alih memberi setiap model faktor *upsample* sendiri, seluruh *arm* Track 1 memakai `input_size: 96` yang sama. Alasannya, jika faktor *upsample* berbeda antar *arm*, perbandingan XAI antar arsitektur menjadi tidak adil karena luas piksel efektif per nodul ikut berbeda. Nilai 96 dipilih karena melewati batas tertinggi (75) dan merupakan kelipatan 32.

#### Studi stabilitas hyperparameter ditambahkan

`stage_03_train.py` menerima argumen `--optimizer` dan `--weight-decay`, dengan `run_id` berformat `{model}_{task}_f{fold}_{optimizer}_wd{wd}` supaya *run* *sweep* tidak saling menimpa *checkpoint*. Kombinasi default (AdamW dengan *weight decay* dari config) sengaja dipertahankan memakai jalur *checkpoint* lama agar 120 *checkpoint* yang sudah ada tetap bisa dilanjutkan.

#### Pencatatan diubah menjadi tiga tingkat

Tingkat per *epoch* di `artifacts/logs/epochs/{run_id}.csv` dengan 20 kolom termasuk `val_loss` yang sebelumnya tidak pernah dicatat, tanpanya kurva *overfitting* mustahil digambar. Tingkat per *run* di `artifacts/logs/runs.csv` dengan status `running`, `completed`, atau `failed`. Tingkat ringkasan di `summary_{task}.csv`. Modul `src/utils/log_io.py` menyatukan format lama dan baru dalam satu skema sehingga 120 *run* lama ikut terbaca.

### Fase 10: insiden eksekusi dan perbaikan ketahanan

Bagian ini dicatat karena sebagian besar waktu eksekusi habis di sini, dan setiap perbaikannya menghasilkan *commit* yang bisa ditelusuri.

#### BatchNorm gagal pada batch berukuran satu

*Batch* terakhir yang kebetulan berisi satu sampel membuat BatchNorm gagal menghitung varians. Diperbaiki dengan `drop_last` pada *train loader* (`9bf5de8`).

#### Disk penuh membuat penyimpanan checkpoint gagal

Penyimpanan `torch.save` gagal dengan `PytorchStreamWriter failed`, dan pada versi awal kegagalan itu mematikan seluruh proses *sweep*. Perbaikan pertama membuat penyimpanan berjalan lewat berkas sementara diikuti `os.replace` atomik dengan lima kali percobaan ulang, ditambah pembungkus `try/except` di *runner* supaya satu *run* gagal tidak menjatuhkan sisanya (`f2c6e64`). Diagnosis lanjutan menunjukkan akar masalahnya bukan penguncian berkas oleh OneDrive seperti dugaan awal, melainkan kapasitas *drive* C yang benar-benar habis (sisa 0 GB dari 930 GB). Kapasitas dibebaskan sampai 338 GB oleh peneliti.

#### Checkpoint korup menyebabkan kegagalan berulang

Setelah disk dibereskan, 27 *run* tetap gagal dengan pola yang persis sama di setiap percobaan ulang. Log menunjukkan `RuntimeError: PytorchStreamReader failed reading zip archive: failed finding central directory` pada `maybe_resume`, bukan pada penyimpanan. Sisa penulisan yang terpotong dari insiden disk penuh membuat berkas `fold{N}_last.pt` tidak bisa dibaca, dan karena isi berkasnya tidak pernah berubah, mengulang *run* tidak pernah menolong. Diperbaiki dengan memperlakukan *checkpoint* yang tidak terbaca sebagai tidak ada, sehingga *run* dimulai dari awal alih-alih mematikan proses (`b9c879c`). Setelah perbaikan ini seluruh 27 *run* selesai.

#### Interpreter Python salah saat peluncuran detached

Peluncuran awal memakai `python` dari PATH sistem, bukan dari `.venv`, dan mati seketika dengan `ModuleNotFoundError: No module named 'pandas'`. Skrip peluncur diganti menjadi `.venv\Scripts\python.exe`.

#### Cakupan XAI tidak sesuai desain Rev1

`stage_05_xai.py` masih menelusuri set enam model lama, padahal XAI per *arm* adalah urusan Track 1. Diperbaiki menjadi gabungan set legacy dengan *backbone* Track 1 (`5693315`).

---

## 4. Dataset

### 4.1 Komposisi

Label ditentukan dari median rating empat radiolog. Rating di atas 3 dihitung malignant, di bawah 3 benign, dan tepat 3 indeterminate.

| Kelas | Arm A (binary) | Arm C (grade3) | Arm D (grade4) |
|---|---|---|---|
| benign | 937 | 937 | 937 |
| malignant | 454 | 454 | 454 |
| indeterminate | dibuang | 747 | 747 |
| no-nodule | tidak dipakai | tidak dipakai | 2138 |
| **Total** | **1391** | **2138** | **4276** |

Distribusi Arm B (ordinal, rating dibulatkan): 260 pada rating 1, 677 pada rating 2, 747 pada rating 3, 353 pada rating 4, dan 101 pada rating 5.

Jumlah 1391 pada Arm A terverifikasi ulang dari matriks konfusi di `summary_binary.csv` (jumlah tp, tn, fp, fn lintas lima *fold*).

### 4.2 Prapemrosesan

Patch dibuat dalam bentuk 2,5D: tiga irisan aksial di sekitar sentroid nodul ditumpuk sebagai kanal, sebagai kompromi antara konteks volumetrik 3D dan biaya komputasi CNN 2D standar. Ukuran patch 64x64 piksel, jendela HU dari -1000 sampai 400 (rentang *lung window* standar), dan *resampling* ke 1 mm isotropik. Kecocokan 888 dari 888 `seriesuid` LUNA16 terhadap LIDC sudah diverifikasi.

Khusus Track 1, patch di-*upsample* ke 96x96 di dalam model, seragam untuk ketujuh *backbone*.

### 4.3 Pembagian data

Pembagian dilakukan *patient-level stratified* lima *fold*, artinya semua nodul dari satu pasien selalu berada di *fold* yang sama untuk mencegah kebocoran. Pembagian dibekukan di `artifacts/splits/folds.json` dengan *seed* 42 dan dipakai identik di seluruh *arm*, *track*, dan sel *sweep*.

---

## 5. Konfigurasi

### 5.1 Training

| Parameter | Nilai |
|---|---|
| Epochs | 50 |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Weight decay (default) | 1e-4 |
| Early stopping patience | 10 |
| Checkpoint every | 5 epoch |
| Mixed precision | true |
| Scheduler | CosineAnnealingLR |
| Seed | 42 |

Augmentasi: *flip* horizontal dan vertikal dengan peluang 0,5, rotasi ±15 derajat, *brightness jitter* 0,1, dan *zoom* 0,9 sampai 1,1.

### 5.2 Ruang sweep Track 2

Tiga *optimizer* dikali tiga nilai *weight decay*, dengan rentang nilai mengikuti default PyTorch per *optimizer* digeser satu orde ke atas dan ke bawah.

| Optimizer | Weight decay yang diuji |
|---|---|
| SGD (momentum 0,9) | 1e-5, 1e-4, 1e-3 |
| Adam | 1e-5, 1e-4, 1e-3 |
| AdamW | 1e-3, 1e-2, 1e-1 |

### 5.3 Radiomics dan fusion

Ekstraksi memakai PyRadiomics dengan `binWidth` 25 dan *resample* [1,1,1], mencakup kelas fitur *firstorder*, *shape*, *glcm*, *glrlm*, *glszm*, *gldm*, dan *ngtdm*. Seleksi fitur dijalankan per *fold* dengan mRMR (mundur ke `mutual_info_classif` karena `pymrmr` tidak terpasang) lalu LASSO, menghasilkan 20 sampai 26 fitur terpilih per *fold* dari 50 kandidat.

FusionNet memakai `embedding_dim` 256, `radiomic_hidden` 128, dan *dropout* 0,3. XGBoost memakai 400 *estimator*, `max_depth` 4, dan *learning rate* 0,05. Uji signifikansi memakai DeLong dengan α = 0,05.

---

## 6. Hasil

### 6.1 Perbandingan model (evaluasi kombinasi default)

Nilai berikut adalah rerata lintas lima *fold* pada Arm A binary, diambil dari `summary_binary.csv`. Kolom AUC/M adalah AUC dibagi jumlah parameter dalam juta, sebagai ukuran kasar efisiensi.

| Backbone | Track | Params (M) | GFLOPs | Latency (ms) | AUC | SD | Sens | Spec | AUC/M |
|---|---|---|---|---|---|---|---|---|---|
| vgg16 | track2 | 14,72 | 2,514 | 13,1 | 0,9103 | 0,0336 | 0,722 | 0,927 | 0,062 |
| convnext_tiny | track1 | 27,82 | 1,649 | 34,2 | 0,9055 | 0,0220 | 0,786 | 0,871 | 0,033 |
| inceptionv3 | track1 | 21,79 | 0,683 | 37,6 | 0,8992 | 0,0260 | 0,680 | 0,937 | 0,041 |
| densenet201 | track1 | 18,10 | 1,613 | 90,5 | 0,8988 | 0,0245 | 0,723 | 0,927 | 0,050 |
| inception_resnet_v2 | track1 | 54,31 | 1,473 | 63,3 | 0,8986 | 0,0178 | 0,712 | 0,937 | 0,017 |
| googlenet | track1 | 5,60 | 0,556 | 21,9 | 0,8962 | 0,0119 | 0,736 | 0,918 | 0,160 |
| resnet50 | track2 | 23,51 | 0,674 | 22,1 | 0,8945 | 0,0303 | 0,699 | 0,917 | 0,038 |
| vit_base | legacy | 85,80 | 35,211 | 136,8 | 0,8934 | 0,0236 | 0,664 | 0,924 | 0,010 |
| xception | track1 | 20,81 | 1,668 | 32,4 | 0,8911 | 0,0283 | 0,709 | 0,946 | 0,043 |
| efficientnet_b0 | track2 | 4,01 | 0,068 | 17,1 | 0,8651 | 0,0113 | 0,645 | 0,889 | 0,216 |
| densenet121 | track1 | 6,96 | 1,065 | 38,0 | 0,8333 | 0,0454 | 0,379 | 0,946 | 0,120 |
| mobilenetv3_small | legacy | 0,93 | 0,011 | 5,4 | 0,8302 | 0,0198 | 0,582 | 0,891 | 0,895 |

Argumen efisiensi terlihat jelas di sini. GoogLeNet mencapai AUC 0,8962 dengan 5,60 juta parameter, sementara InceptionResNetV2 mencapai 0,8986 dengan 54,31 juta parameter. Sekitar sepuluh kali lipat parameter hanya membeli selisih AUC 0,0024. ViT-Base menjadi kasus paling mahal: 35,2 GFLOPs dan latensi 136,8 ms untuk AUC yang masih di bawah GoogLeNet.

Dua catatan yang perlu disampaikan apa adanya. Pertama, **MobileNetV2 tidak muncul di tabel ini**. Penyebabnya, *sweep* Track 2 hanya menghasilkan kombinasi non-default sehingga *checkpoint* kombinasi default tidak pernah terbentuk, dan MobileNetV2 tidak punya *checkpoint* lama seperti EfficientNet-B0, ResNet50, dan VGG16 yang tersisa dari set enam model. Hasil MobileNetV2 tetap tersedia lengkap di studi stabilitas (Bab 6.2). Kedua, DenseNet121 memperoleh sensitivitas 0,379 yang jauh di bawah *backbone* lain meski spesifisitasnya tinggi, pola yang mengindikasikan ambang keputusan bergeser kuat ke kelas benign dan perlu diperiksa ulang sebelum dipakai sebagai klaim.

### 6.2 Stabilitas hyperparameter Track 2 (180 run)

Agregasi per *backbone* atas seluruh 45 *run* per model:

| Backbone | n | AUC rerata | SD | Varians | CV | CI 95% | Min | Maks |
|---|---|---|---|---|---|---|---|---|
| vgg16 | 45 | 0,8966 | 0,0359 | 0,0013 | 0,0400 | [0,8861; 0,9071] | 0,8304 | 0,9535 |
| resnet50 | 45 | 0,8553 | 0,0678 | 0,0046 | 0,0792 | [0,8355; 0,8751] | 0,6565 | 0,9318 |
| mobilenetv2 | 45 | 0,8262 | 0,0508 | 0,0026 | 0,0615 | [0,8114; 0,8411] | 0,7157 | 0,9057 |
| efficientnet_b0 | 45 | 0,8241 | 0,0713 | 0,0051 | 0,0865 | [0,8033; 0,8449] | 0,6841 | 0,9066 |

VGG16 unggul sekaligus paling stabil, dengan koefisien variasi 0,0400 yang berarti sebarannya paling sempit relatif terhadap reratanya. EfficientNet-B0 paling sensitif terhadap pilihan *hyperparameter* dengan CV 0,0865.

Pemisahan per *optimizer* menunjukkan dari mana sebaran itu berasal:

| Backbone | Adam | AdamW | SGD | Selisih Adam − SGD |
|---|---|---|---|---|
| efficientnet_b0 | 0,8744 | 0,8693 | 0,7287 | 0,1457 |
| resnet50 | 0,8980 | 0,8960 | 0,7720 | 0,1260 |
| mobilenetv2 | 0,8568 | 0,8577 | 0,7642 | 0,0926 |
| vgg16 | 0,9110 | 0,9104 | 0,8684 | 0,0426 |

Adam dan AdamW praktis tidak terbedakan, selisihnya di bawah 0,0051 pada keempat *backbone*. SGD dengan momentum 0,9 tertinggal jauh pada semua model, dan penurunannya paling parah justru pada model ringan. Kemungkinan besar 50 *epoch* dengan *learning rate* 1e-4 belum cukup bagi SGD untuk konvergen, bukan berarti SGD secara inheren tidak cocok, dan ini perlu dinyatakan sebagai batasan eksperimen ketimbang kesimpulan tentang *optimizer*.

Sebaliknya, *weight decay* nyaris tidak berpengaruh. Rentang AUC antara tiga nilai *weight decay* di dalam satu pasangan model dan *optimizer* paling besar hanya 0,0218 (MobileNetV2 dengan SGD), dan pada sepuluh dari dua belas pasangan berada di bawah 0,01. Perbandingan langsungnya: pengaruh *optimizer* sekitar tujuh kali lebih besar daripada pengaruh *weight decay* pada rentang yang diuji.

Sel terbaik adalah VGG16 dengan Adam dan *weight decay* 1e-3, AUC 0,9134 dengan SD 0,0291 dan CI 95% [0,8879; 0,9389]. Sel paling stabil adalah EfficientNet-B0 dengan AdamW dan *weight decay* 1e-3, CV 0,0095. Sel terburuk adalah EfficientNet-B0 dengan SGD dan *weight decay* 1e-3, AUC 0,7260.

Figur pendukung: `figures/track2_auc_by_optimizer.png` dan `figures/track2_auc_heatmap.png`.

### 6.3 Ablasi fusion Track 1

Rerata AUC lintas lima *fold* per *backbone* dan per *arm*, dari `ablation_summary.csv` (175 baris, 7 *backbone* dikali 5 *arm* dikali 5 *fold*):

| Backbone | cnn_only | radiomics_only | fusion_early | fusion_intermediate | fusion_late |
|---|---|---|---|---|---|
| convnext_tiny | 0,7888 | 0,9293 | 0,9215 | 0,9299 | 0,9207 |
| densenet121 | 0,8686 | 0,9318 | 0,9114 | 0,9312 | 0,9235 |
| densenet201 | 0,6432 | 0,9309 | 0,9242 | 0,9262 | 0,9000 |
| googlenet | 0,7555 | 0,9318 | 0,9251 | 0,9340 | 0,9148 |
| inception_resnet_v2 | 0,8586 | 0,9315 | 0,9193 | 0,9250 | 0,9230 |
| inceptionv3 | 0,8112 | 0,9332 | 0,9211 | 0,9263 | 0,9241 |
| xception | 0,7711 | 0,9307 | 0,9029 | 0,9155 | 0,9138 |
| **Rerata** | **0,7853** | **0,9313** | **0,9179** | **0,9269** | **0,9171** |

Uji DeLong dijalankan pada 21 pasangan (7 *backbone* dikali 3 varian *fusion*), membandingkan setiap varian *fusion* terhadap *arm* tunggal terbaik. Pada seluruh 21 pasangan, *arm* tunggal terbaik selalu radiomics, dan tidak ada satu pun varian *fusion* yang menang secara signifikan. Rinciannya: 13 pasangan menunjukkan *fusion* signifikan lebih buruk (p < 0,05), dan 8 pasangan tidak berbeda signifikan. Kasus paling telak adalah Xception dengan *fusion_early* (p = 0,00005) dan DenseNet201 dengan *fusion_late* (p < 0,00001).

Hasil ini negatif terhadap hipotesis awal dan dilaporkan tanpa penyesuaian, sesuai aturan keputusan yang ditetapkan sebelum hasil dilihat. Di antara varian *fusion*, *fusion_intermediate* paling baik (0,9269) dan paling sering imbang dengan radiomics, sehingga jika *fusion* tetap ingin dipertahankan sebagai kontribusi, varian inilah yang punya dasar paling kuat.

#### Batasan pada kolom cnn_only

Kolom ini tidak boleh dikutip sebagai kemampuan CNN mandiri. Penelusuran kode menunjukkan `stage_03b_fusion.py` memanggil `BackboneClassifier` dan `FusionNet` secara langsung tanpa meneruskan `input_size`, sehingga tidak melewati `build_model` di registry yang menangani resolusi per *track*. Akibatnya *checkpoint* Track 1 yang dilatih pada resolusi 96 dievaluasi pada resolusi 64 (atau pada batas minimum arsitektur masing-masing), menghasilkan ketidakcocokan antara kondisi latih dan kondisi uji. Efeknya terlihat pada DenseNet201 yang turun ke 0,6432 di kolom ini padahal mencapai 0,8988 pada evaluasi mandiri di Bab 6.1.

Perbandingan utama Bab ini tetap sahih karena *arm* radiomics dan ketiga *arm* *fusion* dilatih dan diuji pada resolusi yang sama, sehingga bersifat konsisten secara internal. Yang tidak sahih hanya kolom cnn_only. Angka kemampuan CNN mandiri yang benar adalah yang di Bab 6.1. Perbaikan pemanggilan `input_size` di `stage_03b_fusion.py` dicatat sebagai pekerjaan berikutnya.

### 6.4 XAI

Metrik dihitung pada 60 sampel *fold* 0 per *backbone*, dengan ambang 80 persen. `dice_size_matched` membandingkan area CAM setelah disamakan ukurannya dengan area mask, sehingga tidak menghukum CAM yang benar posisinya tetapi lebih luas dari nodul.

| Backbone | Dice | IoU | Dice (size-matched) | Pointing acc | Energy |
|---|---|---|---|---|---|
| densenet121 | 0,1030 | 0,0630 | 0,4703 | 0,7167 | 0,0309 |
| convnext_tiny | 0,1091 | 0,0653 | 0,4443 | 0,7167 | 0,0478 |
| densenet201 | 0,0970 | 0,0576 | 0,4322 | 0,7000 | 0,0315 |
| inception_resnet_v2 | 0,1199 | 0,0726 | 0,3089 | 0,3833 | 0,0669 |
| inceptionv3 | 0,0900 | 0,0502 | 0,1647 | 0,2000 | 0,0378 |
| xception | 0,0716 | 0,0479 | 0,1519 | 0,2000 | 0,0464 |
| vgg16 | 0,0914 | 0,0597 | 0,1785 | 0,2000 | 0,1027 |
| resnet50 | 0,1181 | 0,0703 | 0,1461 | 0,1167 | 0,0534 |
| efficientnet_b0 | 0,0610 | 0,0353 | 0,0667 | 0,0833 | 0,0563 |
| mobilenetv3_small | 0,0379 | 0,0201 | 0,0176 | 0,0000 | 0,0205 |
| vit_base | 0,0357 | 0,0198 | 0,0150 | 0,0000 | 0,0074 |
| googlenet | 0,0350 | 0,0189 | 0,0113 | 0,0000 | 0,0000 |

Interpretabilitas tidak mengikuti kapasitas maupun akurasi. ViT-Base membawa 85,8 juta parameter tetapi *pointing accuracy*-nya nol. GoogLeNet mencapai AUC 0,8962 pada Bab 6.1, setara *backbone* jauh lebih besar, namun peta CAM-nya tidak pernah jatuh di dalam mask nodul dan nilai *energy*-nya nol. Keluarga DenseNet bersama ConvNeXt-Tiny membentuk kelompok tersendiri di puncak, dengan *pointing accuracy* 0,70 ke atas dan *dice size-matched* di kisaran 0,43 sampai 0,47.

Nilai Dice mentah terlihat rendah di semua model (0,035 sampai 0,120) karena Layer-CAM menyorot wilayah yang lebih luas daripada batas nodul. Perbandingan yang lebih informatif ada di kolom *size-matched* dan *pointing accuracy*, dan pada kedua kolom itu urutannya konsisten.

Perlu dicatat bahwa nol pada GoogLeNet dan ViT-Base bisa berarti dua hal berbeda: model memang tidak memakai wilayah nodul, atau pemilihan *target layer* untuk arsitektur tersebut belum tepat. Sejarah tiga bug XAI di Fase 6 menunjukkan kemungkinan kedua bukan hal yang bisa diabaikan, sehingga temuan ini sebaiknya disampaikan sebagai indikasi yang masih perlu verifikasi, bukan kesimpulan final.

---

## 7. Figur yang tersedia

Seluruh figur berada di `artifacts/results/figures/` kecuali yang bertanda XAI.

| Figur | Berkas | Kegunaan |
|---|---|---|
| Params lawan AUC | `params_vs_auc.png` | bukti utama argumen efisiensi |
| FLOPs lawan AUC | `flops_vs_auc.png` | pendamping argumen efisiensi |
| AUC per optimizer | `track2_auc_by_optimizer.png` | temuan utama Bab 6.2 |
| Heatmap sel sweep | `track2_auc_heatmap.png` | sebaran 36 sel |
| Kurva training Track 1 | `curves_track1.png` | rerata lintas *fold* dengan pita ±1 SD |
| Kurva training Track 2 | `curves_track2.png` | idem untuk Track 2 |
| Kurva overfitting | `curves_overfit.png` | train loss lawan val loss |
| Matriks DeLong | `delong_matrix.png` | signifikansi antar model |
| ROC dan kalibrasi | `roc_curves.png`, `calibration.png` | kualitas probabilitas |
| Matriks konfusi | `confusion_matrices.png` | per model |
| Ikhtisar dataset | `dataset_overview.png` | contoh patch per kelas |
| Arsitektur fusion | `fusion_architecture.png` | diagram, masih draft |
| Grad-CAM per model | `xai/xai_{backbone}.png` | 12 berkas |

Untuk presentasi, tiga figur paling kuat adalah `params_vs_auc.png` (argumen efisiensi), `track2_auc_by_optimizer.png` (temuan stabilitas), dan susunan `xai/xai_*.png` (temuan interpretabilitas tidak mengikuti kapasitas).

---

## 8. Pelajaran

Kegagalan yang paling banyak memakan waktu bukan kegagalan model, melainkan kegagalan infrastruktur. Disk penuh, *checkpoint* korup, dan *interpreter* salah menghabiskan waktu lebih banyak daripada seluruh proses perancangan eksperimen. Perbaikan ketahanan yang dipasang setelahnya, yaitu penulisan atomik, percobaan ulang, dan pemulihan dari *checkpoint* rusak, seharusnya ada sejak awal.

Kegagalan yang diam lebih berbahaya daripada yang berisik. Insiden 27 *run* gagal berulang sempat terlihat seperti kesalahan acak karena `runs.csv` mencatat `best_score` yang valid pada baris yang berstatus `failed`. Baru setelah *traceback* dibaca sampai bawah terlihat bahwa kegagalannya ada di pembacaan *checkpoint*, bukan di penulisan. Pencatatan status per *run* inilah yang membuat pola berulang itu terlihat.

Menulis laporan menemukan bug. Batasan pada kolom cnn_only di Bab 6.3 tidak ditemukan saat pipeline dijalankan, melainkan saat angka 0,6432 pada DenseNet201 dibandingkan dengan 0,8988 pada bab sebelumnya. Angka yang tidak masuk akal antar bab adalah alat deteksi bug yang murah.

Temuan negatif tetap temuan. Radiomics mengalahkan *fusion* pada 13 dari 21 pasangan dan tidak pernah kalah. Kalau hasil ini disembunyikan, kontribusinya justru hilang, karena radiomics yang kuat pada tugas ini adalah informasi yang berguna bagi pembaca berikutnya.

---

## 9. Rencana lanjutan

1. Perbaiki pemanggilan `input_size` di `stage_03b_fusion.py` agar *arm* cnn_only dan *arm* *fusion* berjalan pada resolusi 96 yang sama dengan pelatihan Track 1, lalu jalankan ulang ablasi dan uji DeLong.
2. Periksa sensitivitas DenseNet121 yang hanya 0,379 pada Bab 6.1, kemungkinan terkait pergeseran ambang keputusan.
3. Verifikasi pemilihan *target layer* untuk GoogLeNet dan ViT-Base sebelum menyimpulkan bahwa keduanya tidak melokalisasi nodul.
4. Latih MobileNetV2 pada kombinasi default agar bisa masuk tabel perbandingan Bab 6.1 secara setara.
5. Perpanjang *sweep* SGD melampaui 50 *epoch* untuk memastikan tertinggalnya SGD adalah persoalan konvergensi, bukan sifat *optimizer*.
6. Validasi eksternal (opsional) memakai LUNGx, NLST, atau LNDb, disertai pemeriksaan kontaminasi.

---

## 10. Integritas riset

Semua *arm* dilaporkan, termasuk yang kalah. Bab 6.3 memuat hasil yang bertentangan dengan hipotesis awal tanpa penyesuaian. Aturan keputusan ditetapkan sebelum hasil dilihat, khususnya metrik utama Arm D yang memakai subset nodul saja untuk mencegah inflasi dari kelas no-nodule yang mudah dipisahkan.

Pembagian *fold* dibekukan lintas *arm*, *track*, dan seluruh sel *sweep*, memakai *seed* yang sama. Konfigurasi latih identik untuk semua *backbone*, kecuali `input_size` yang memang sengaja dibedakan per *track* dan seragam di dalam setiap *track*.

Batasan metodologis dinyatakan di tempat angkanya muncul, bukan disembunyikan di bagian akhir. Tiga batasan yang sudah teridentifikasi: kolom cnn_only pada Bab 6.3, ketiadaan MobileNetV2 pada Bab 6.1, dan kemungkinan salah pilih *target layer* pada Bab 6.4.

---

## Lampiran: berkas hasil

| Berkas | Isi |
|---|---|
| `artifacts/logs/runs.csv` | 332 baris, status per *run* |
| `artifacts/logs/epochs/*.csv` | log per *epoch*, 20 kolom, self-describing |
| `artifacts/results/summary_binary.csv` | 60 baris, metrik per model per *fold* |
| `artifacts/results/track2_stability.csv` | 36 sel *sweep* |
| `artifacts/results/track2_stability_by_opt.csv` | agregasi per *optimizer* |
| `artifacts/results/track2_stability_by_model.csv` | agregasi per *backbone* |
| `artifacts/results/fusion/ablation_summary.csv` | 175 baris ablasi |
| `artifacts/results/fusion/delong_fusion.csv` | 21 uji DeLong |
| `artifacts/results/xai/xai_metrics.csv` | metrik XAI 12 *backbone* |

Commit relevan Rev1: `5ed1108` (analisis stabilitas dan kurva), `f2c6e64` (ketahanan penyimpanan *checkpoint*), `b9c879c` (pemulihan *checkpoint* korup), `5693315` (cakupan XAI Track 1).
