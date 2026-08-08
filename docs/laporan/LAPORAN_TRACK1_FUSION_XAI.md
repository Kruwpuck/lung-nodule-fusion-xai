# Laporan hasil Track 1: fusi radiomics-CNN dan explainability per-arm

## 0. Identitas penelitian

- **Judul**: Fusi radiomics-CNN dan explainability per-arm untuk klasifikasi malignansi nodul paru pada LIDC-IDRI
- **Repo**: `lung-nodule-fusion-xai`
- **Tugas**: Track 1 dari pemisahan dua paper (lihat `docs/Review Revisi 1.md` §8)
- **Dataset**: LIDC-IDRI
- **Tanggal laporan**: 30 Juli 2026 (diperbarui 8 Agustus 2026 dengan hasil run `2026-08-04-run02`, lihat §6.3)

---

## 1. Ringkasan eksekutif

Track 1 membandingkan lima *arm* representasi (CNN-only, radiomics-only, early fusion, intermediate fusion, late fusion) pada tujuh backbone CNN, lalu mengevaluasi explainability tiap arm secara terpisah.

**Klaim final** (menggantikan kesimpulan lama "radiomics-only mengungguli semua varian fusi", yang sudah tidak akurat setelah perbaikan bug resolusi dan kuantifikasi run `2026-08-04-run02`):

> `fusion_late` mengalahkan CNN-sendirian secara signifikan tanpa syarat seleksi *checkpoint* (p < 1e-9 pada ketiga backbone di kedua rezim *checkpoint*; p < 1e-11 bila hanya rezim tanpa seleksi yang dikutip), setara dengan radiomics dalam AUC, mempertahankan penjelasan spasial pada tingkat yang praktis sama (selisih pointing accuracy ≤0.05, identik persis pada himpunan enam nodul tetap), dan satu-satunya *arm* yang menyediakan penjelasan spasial dan penjelasan fitur sekaligus.

Klaim kesetaraan dengan radiomics bertumpu pada DenseNet201 sebagai model utama (§6.3.2); dua backbone pendukung punya ketergantungan *checkpoint* yang dinyatakan eksplisit di §8.5. Temuan sekunder yang tidak berubah: pointing accuracy Grad-CAM/Layer-CAM sangat bervariasi antar backbone tanpa hubungan konsisten terhadap AUC klasifikasi.

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

Kelima *arm* di bawah ini diaudit baris demi baris terhadap kodenya pada 8 Agustus 2026 (lihat §8.6 untuk apa yang ditemukan audit itu). Nama fungsi dicantumkan supaya deskripsi dan implementasi tidak bisa lagi berpisah diam-diam.

| Arm | Yang benar-benar dijalankan | Sumber |
|---|---|---|
| `cnn_only` | Muat ulang *checkpoint* backbone, inferensi murni pada fold, tanpa pelatihan baru. | `_cnn_only_preds` (`src/stage_03b_fusion.py`) |
| `radiomics_only` | XGBoost pada vektor radiomik terpilih. Seleksi per fold: filter *mutual information* menyisakan 50 kandidat teratas, lalu LASSO ber-*cross-validation* memilih himpunan akhir, keduanya di-*fit* pada fold latih saja. Tanpa `eval_set`, jadi bebas seleksi berbasis validasi. | `_select_fold_features` + `train_early_fusion_xgboost` |
| `fusion_early` | Konkatenasi **embedding CNN** dengan vektor radiomik terpilih menjadi satu matriks fitur, lalu XGBoost di atasnya. | `build_early_fusion_features` (`src/fusion/early_fusion.py`) |
| `fusion_intermediate` | `img_proj` memproyeksikan embedding CNN ke 256-d dan `rad_branch` memproyeksikan vektor radiomik ke 128-d (masing-masing Linear + ReLU + Dropout 0.3), keduanya dikonkatenasi jadi 384-d lalu masuk *classification head* bersama, dilatih *end-to-end*. | `FusionNet` (`src/models/fusion_net.py`) |
| `fusion_late` | Rata-rata probabilitas, $0.5 \cdot p_\text{CNN} + 0.5 \cdot p_\text{radiomics}$. Nol parameter baru yang dilatih. | `average_fusion` (`src/fusion/late_fusion.py`) |

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

AUC rata-rata per arm, dipool lintas 7 backbone dan 5 fold, dari `artifacts/results/fusion/ablation_summary.csv` setelah perbaikan bug resolusi (175 baris: 7 backbone x 5 arm x 5 fold, ditarik 4 Agustus 2026):

| Arm | AUC rata-rata | Baseline pra-perbaikan |
|---|---|---|
| fusion_late | 0.9332 | 0.9171 |
| radiomics_only | 0.9314 | 0.9313 |
| fusion_intermediate | 0.9294 | 0.9269 |
| fusion_early | 0.9126 | 0.9179 |
| cnn_only | 0.8927 | 0.7853 |

Baseline pra-perbaikan diarsipkan di `artifacts/results/_baseline_pre_rev2/`. Kolom `cnn_only` kini konsisten dengan AUC standalone per backbone di `summary_binary.csv` (selisih 0.0019-0.0070, lihat §8.4 untuk sumber selisih itu), mengonfirmasi bug resolusi sudah tertutup.

Dari 21 uji DeLong berpasangan (`fusion/delong_fusion.csv`), fusion_late unggul angka di 5 dari 7 backbone tapi **tidak satu pun** dari 21 pasangan mencapai signifikansi yang mendukung fusi (p terkecil di sisi menang: 0.4387). fusion_early signifikan lebih buruk daripada radiomics_only di ketujuh backbone.

#### Batasan: seleksi epoch bocor, belum tertutup di angka ini

Angka `fusion_intermediate` (dan turunannya `fusion_early`/`fusion_late` lewat *embedding* dan probabilitas CNN yang sama) di atas masih memakai protokol lama: *epoch* terbaik dipilih berdasarkan AUC pada *fold* validasi luar, lalu *fold* yang sama dilaporkan sebagai skor akhir (`src/stage_03b_fusion.py:207-215` sebelum perbaikan nested CV). `radiomics_only` satu-satunya arm yang bersih dari kebocoran ini karena `train_early_fusion_xgboost` tidak memakai `eval_set`. Artinya keunggulan tipis fusion_late di atas radiomics_only diukur dengan timbangan berat sebelah dan belum bisa ditafsirkan sampai nested CV (§8.4) dijalankan ulang. Pertanyaan itu kini terjawab lewat *sensitivity check* rezim *checkpoint* di §6.3 dan §8.5, yang menggantikan §6.1 sebagai hasil utama untuk `fusion_late`.

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

### 6.3 Keunggulan gabungan `fusion_late` (run `2026-08-04-run02`, commit `5220afb`)

Bagian ini adalah hasil utama Track 1 versi terkini. Seluruh angkanya berasal dari satu run tunggal sehingga tidak lagi tersebar antar arsip, dan setiap baris mencantumkan rezim *checkpoint* yang dipakai.

#### 6.3.1 Tabel keunggulan gabungan

Tabel di bawah adalah tabel utama Track 1, disalin apa adanya dari `artifacts/results/run02/combined_advantage_table.md`. Angka dalam kurung adalah rentang antar tiga backbone (`convnext_tiny`, `densenet201`, `densenet121`).

| Kriteria | cnn_only | radiomics_only | fusion_late |
|---|---|---|---|
| AUC gabungan 5 fold (rerata 3 backbone) | 0.8944 (0.8907–0.8965) | 0.9336 (0.9336–0.9336) | 0.9327 (0.9300–0.9363) |
| AUC dengan *checkpoint* tanpa seleksi (T-0) | 0.8806 (0.8725–0.8888) | 0.9336 (0.9336–0.9336) (tidak terpengaruh) | 0.9290 (0.9241–0.9360) |
| DeLong vs `fusion_late` (p, 3 backbone) | 0.0000 (1.0e-12–1.4e-10); signifikan 3/3 | 0.5417 (0.4555–0.6793); signifikan 0/3 | — |
| Peta salience spasial (Grad-CAM/Layer-CAM) | ada | mustahil secara struktural | ada (diwarisi dari cabang citra) |
| Pointing accuracy (60 nodul fold 0, rerata 3 backbone) | 0.6667 (0.5763–0.7288) | tidak terdefinisi | 0.6384 (0.5593–0.6780) |
| SHAP fitur radiomik | tidak ada | ada | ada |
| **Penjelasan spasial DAN fitur sekaligus** | **tidak** | **tidak** | **ya** |

Baris terakhir adalah penutup argumen: dua *arm* lain masing-masing hanya menutupi satu sisi explainability, dan hanya `fusion_late` yang menutupi keduanya tanpa membayar dengan AUC.

#### 6.3.2 DenseNet201 sebagai model utama

Klaim kesetaraan AUC dengan `radiomics_only` dilaporkan dengan DenseNet201 sebagai model utama. Alasannya tunggal dan dapat diperiksa: DenseNet201 adalah **satu-satunya backbone yang tetap setara dengan radiomics tanpa keuntungan seleksi *checkpoint*** (p = 0.5122 pada *checkpoint* `last`, dengan AUC nominal lebih tinggi, 0.9360 vs 0.9336). ConvNeXt-Tiny dan DenseNet121 tetap dilaporkan sebagai bukti pendukung, tetapi kesetaraan keduanya hanya bertahan pada rezim `best`; lihat §8.5.

| Backbone | AUC `fusion_late` (best) | p vs radiomics (best) | AUC `fusion_late` (last) | p vs radiomics (last) | Setara tanpa seleksi? |
|---|---|---|---|---|---|
| **DenseNet201** | 0.9363 | 0.4901 | **0.9360** | **0.5122** | **ya** |
| ConvNeXt-Tiny | 0.9300 | 0.4555 | 0.9268 | 0.0246 | tidak |
| DenseNet121 | 0.9319 | 0.6793 | 0.9241 | 0.0144 | tidak |

#### 6.3.3 Kemenangan atas `cnn_only` bebas syarat

Uji DeLong `fusion_late` vs `cnn_only` dijalankan pada kedua rezim *checkpoint*. Kemenangan `fusion_late` bertahan signifikan pada ketiganya di kedua rezim, dengan p pada rezim tanpa seleksi justru lebih kecil:

| Backbone | p (rezim `best`) | p (rezim `last`) |
|---|---|---|
| ConvNeXt-Tiny | 1.009e-12 | 3.903e-12 |
| DenseNet201 | 1.362e-10 | 7.366e-12 |
| DenseNet121 | 2.969e-11 | 3.775e-15 |

Alasan struktural mengapa klaim ini kebal terhadap kebocoran seleksi *checkpoint* yang dibahas di §8.4: kedua *arm* memakai *checkpoint* CNN yang sama, sehingga keuntungan seleksi masuk ke kedua sisi perbandingan dan saling meniadakan. Perbandingan yang sensitif terhadap rezim *checkpoint* hanyalah `fusion_late` vs `radiomics_only`, karena `radiomics_only` bersih dari seleksi apa pun (§8.4). Batas p yang aman ditulis dalam manuskrip mengikuti nilai terbesar dari sel yang dikutip. Untuk kedua rezim sekaligus, nilai terbesarnya 1.362e-10 sehingga batas yang benar adalah **p < 1e-9**. Untuk rezim `last` saja, nilai terbesarnya 7.366e-12 sehingga batasnya **p < 1e-11**. Jangan menulis p < 1e-11 untuk kedua rezim sekaligus — DenseNet201 rezim `best` melanggarnya.

#### 6.3.4 Explainability: fusi tidak merusak lokalisasi

Metrik Grad-CAM/Layer-CAM dihitung pada dua himpunan sampel: himpunan enam nodul tetap dari `artifacts/xai/fixed_display_samples.json` (tidak pernah dipilih ulang) dan himpunan 60 nodul fold 0 (59 nodul dengan mask valid) untuk resolusi statistik yang lebih halus.

| Backbone | Himpunan | Pointing accuracy `cnn_only` | Pointing accuracy `fusion_late` | Selisih | Nodul beda keputusan |
|---|---|---|---|---|---|
| ConvNeXt-Tiny | 6 nodul tetap | 1.0000 | 1.0000 | 0.0000 | 0 |
| DenseNet201 | 6 nodul tetap | 0.3333 | 0.3333 | 0.0000 | 0 |
| DenseNet121 | 6 nodul tetap | 0.5000 | 0.5000 | 0.0000 | 0 |
| ConvNeXt-Tiny | 60 nodul fold 0 | 0.7288 | 0.6780 | −0.0508 | 5 |
| DenseNet201 | 60 nodul fold 0 | 0.6949 | 0.6780 | −0.0169 | 6 |
| DenseNet121 | 60 nodul fold 0 | 0.5763 | 0.5593 | −0.0169 | 2 |

Temuan ini positif dan dilaporkan sebagai temuan, bukan sebagai kekurangan: **menambahkan modalitas radiomik tidak merusak kemampuan lokalisasi spasial cabang citra.** Selisih terbesar 0.0508 dan identik persis pada himpunan enam nodul tetap.

Penjelasannya arsitektural dan dapat diperiksa langsung dari definisi *arm*. `fusion_late` adalah rata-rata probabilitas, $0.5 \cdot p_\text{CNN} + 0.5 \cdot p_\text{rad}$, tanpa jaringan baru yang dilatih; ia mewarisi cabang citra `cnn_only` secara utuh. Karena peta CAM dihitung terhadap kelas keputusan, peta `fusion_late` **identik dengan peta `cnn_only` pada setiap nodul yang kelas keputusan kedua *arm*-nya sama**, dan hanya berbeda pada nodul yang keputusan fusinya berbeda dari keputusan CNN. Jumlah nodul semacam itu kecil: 5, 6, dan 2 dari 59 pada ketiga backbone, dan nol pada himpunan enam nodul tetap — itulah sebabnya selisihnya persis nol di sana. Selisih kecil pada himpunan 60 nodul karena itu berasal dari segelintir nodul saja, bukan dari degradasi menyeluruh.

Klaim "fusi unggul dalam XAI" **tidak didukung angka dan tidak boleh ditulis**. Yang didukung angka: fusi mempertahankan penjelasan spasial pada tingkat yang praktis sama, sambil menjadi satu-satunya *arm* yang juga punya penjelasan tingkat fitur.

#### 6.3.5 SHAP cabang radiomik

Tiga figur beeswarm (`shap_beeswarm_{backbone}.png`, 23 fitur, fold 0, `fs_method = mutual_info_classif`) secara arsitektural **identik antar backbone**, karena cabang radiomik `fusion_late` tidak menerima masukan apa pun dari cabang CNN. Fakta ini dicatat eksplisit di kolom `identical_across_backbones` pada `artifacts/results/run02/shap_provenance.csv`, bukan disembunyikan dengan menampilkan tiga figur seolah-olah berbeda.

#### 6.3.6 Provenance angka

Seluruh angka §6.3 berasal dari run `2026-08-04-run02` pada commit `5220afb`, kolom `run_id` dan `commit_sha` tersimpan di setiap baris CSV sumbernya.

| Angka | Berkas sumber | Rezim *checkpoint* |
|---|---|---|
| AUC per *arm*, uji DeLong | `artifacts/results/run02/delong_run02.csv` | kolom `ckpt_kind` (`best` / `last`) |
| Sensitivitas T-0 | `artifacts/results/run02/t0_checkpoint_sensitivity.csv` | keduanya, berdampingan |
| Metrik XAI dan selisihnya | `artifacts/results/run02/xai_fusion_vs_cnn.csv` | `best` |
| Sampel XAI yang dipakai | `artifacts/results/run02/xai_samples_used.csv` | — |
| Provenance SHAP | `artifacts/results/run02/shap_provenance.csv` | `best` |
| Vektor probabilitas mentah | `artifacts/results/run02/probs/{backbone}.npz` | kedua rezim, kolom terpisah |
| Tabel gabungan §6.3.1 | `artifacts/results/run02/combined_advantage_table.{csv,md}` | dinyatakan per baris |

Rezim `best` memakai `checkpoints/{model}/fold{f}_best.pt`, yaitu *checkpoint* yang dipilih `stage_03_train` berdasarkan AUC pada fold yang sama dengan yang dilaporkan. Rezim `last` memakai `fold{f}_last.pt`, *checkpoint* akhir pelatihan tanpa seleksi apa pun. Rezim `last` adalah *sensitivity check* batas bawah, bukan nested CV penuh — nested CV akan menuntut pelatihan ulang 3 backbone × 5 fold dan tidak dijalankan.

---

## 7. Figur

| Figur | Berkas | Kegunaan |
|---|---|---|
| Panel Grad-CAM per backbone | `artifacts/results/xai/xai_{backbone}.png` | Visualisasi CAM per backbone (belum komparabel lintas model) |
| Panel komparabilitas XAI baru | `artifacts/results/figures_grid/grid_comparability.png` | Sampel identik, colorbar seragam, baris kegagalan (belum dieksekusi, tidak ada checkpoint lokal) |
| Diagram arsitektur fusi (Fig 11) | `artifacts/results/figures/fusion_architecture.png` | Kelima arm, `fusion_late` disorot sebagai arm utama. Label DRAFT sudah dibuang dan backbone dikoreksi ke DenseNet201 (8 Agustus 2026) |
| Bukti spasial dan fitur berdampingan (Fig 14) | `artifacts/results/run02/fig14_spatial_and_feature.png` | Layer-CAM enam nodul tetap plus SHAP cabang radiomik. Satu figure, bukan tiga (§6.3.5) |

---

## 8. Batasan

1. Kolom `cnn_only` pada ablasi fusi masih data pra-perbaikan bug resolusi; perlu di-*re-run* sebelum dianggap final (§6.1).
2. Tiga varian fusi baru (branch normalization, Gated Multimodal Unit, modality dropout) sudah diimplementasikan dan diuji unit, tapi belum pernah dieksekusi pada grid ablasi penuh.
3. Panel XAI komparabilitas baru (`stage_07f_xai_comparability.py`) belum bisa dijalankan di mesin manapun karena checkpoint tidak tersedia lokal saat ditulis.
4. SHAP dan Grad-CAM dilaporkan pada skala terpisah tanpa metrik penyatu; ini gap metodologis terbuka, bukan keterbatasan khusus studi ini.
5. Seleksi fitur radiomics memakai *mutual information* (`mutual_info_classif`), bukan mRMR. Lihat §8.1.
6. Kesetaraan `fusion_late` dengan `radiomics_only` bergantung pada rezim *checkpoint* untuk dua dari tiga backbone. Lihat §8.5.

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

### 8.5 Batasan: kesetaraan dengan radiomics bergantung pada rezim *checkpoint* pada dua backbone

Ini keterbatasan paling material dari klaim final dan dinyatakan di sini secara eksplisit, bukan diserahkan pada pembaca untuk menemukannya sendiri.

Pada rezim `best` — *checkpoint* yang dipilih berdasarkan AUC fold yang sama dengan yang dilaporkan — `fusion_late` setara secara statistik dengan `radiomics_only` pada ketiga backbone (p = 0.4555, 0.4901, 0.6793). Pada rezim `last`, yang membuang keuntungan seleksi itu, kesetaraan **runtuh pada dua backbone**: `fusion_late` menjadi signifikan **lebih buruk** daripada `radiomics_only` pada ConvNeXt-Tiny (p = 0.0246) dan DenseNet121 (p = 0.0144). Hanya DenseNet201 yang bertahan setara (p = 0.5122) sekaligus nominal lebih tinggi.

Asimetrinya berasal dari §8.4: `radiomics_only` tidak pernah menikmati seleksi berbasis validasi sama sekali, sehingga ia adalah satu-satunya pembanding yang rezim *checkpoint*-nya tidak berubah antar kolom. Ketika keuntungan seleksi dicabut dari sisi fusi saja, selisihnya terlihat. Besar keuntungan seleksi itu sendiri terbatas, paling banyak 0.0078 AUC (`delta_late` pada `t0_checkpoint_sensitivity.csv`), dan tidak ada satu pun backbone yang urutan peringkat *arm*-nya terbalik.

Konsekuensi untuk penulisan manuskrip: klaim kesetaraan dengan radiomics harus selalu menyebutkan model utamanya (DenseNet201) atau menyebutkan rezim *checkpoint*-nya. Klaim kesetaraan yang digeneralisasi ke ketiga backbone tanpa syarat tidak akan bertahan ketika diperiksa. Perhatikan bahwa batasan ini **tidak menyentuh** klaim terkuat, yaitu kemenangan atas `cnn_only`, dengan alasan yang dijelaskan di §6.3.3.

### 8.6 Temuan reproducibility: kegagalan senyap ketujuh, dan audit deskripsi kelima arm

**Bibliografi terbit kosong sementara sitasinya benar.** Saat manuskrip Track 1 pertama kali di-*build*, `latexmk` gagal dengan `! LaTeX Error: Something's wrong--perhaps a missing \item.` Gejala itu menunjuk ke berkas `.tex`, dan di sana tidak ada yang salah. Penyebab sebenarnya ada dua tingkat di bawahnya: `.latexmkrc` menyetel `ensure_path('BIBINPUTS', '..')`, sedangkan `latexmk` menjalankan `bibtex` dari dalam `$out_dir`, sehingga `'..'` teruraikan dari direktori yang berbeda dengan yang dipakai `pdflatex`. `bibtex` melapor `I couldn't open database file refs.bib` di `main.blg` — berkas yang tidak dibaca siapa pun — lalu tetap menghasilkan `main.bbl` berisi `\begin{thebibliography}` tanpa satu pun `\bibitem`. LaTeX kemudian gagal pada lingkungan kosong itu, dengan pesan yang menyesatkan.

Ini insiden kegagalan senyap **ketujuh** pada proyek ini, setelah *fallback* mRMR yang tidak pernah berjalan (§8.1), fungsi publik yang dijamin gagal dan tak pernah diuji (§8.2), uji yang membekukan angka yang salah (§8.2), disk penuh, *checkpoint* korup, dan proses latih yang mati bersama sesi SSH (§8.3). Polanya persis sama: komponen yang gagal menulis peringatan ke saluran yang tidak dibaca, lalu menyerahkan keluaran yang bentuknya sah tapi isinya kosong ke tahap berikutnya. Yang membedakan kasus ini hanya keberuntungan bahwa LaTeX kebetulan menolak bibliografi kosong. Seandainya dokumen ini tidak punya sitasi sama sekali, *build* akan **berhasil** dan menerbitkan PDF tanpa daftar pustaka tanpa satu pun keluhan.

Perbaikannya memakai jalur mutlak, bukan menambah satu tingkat `..`, karena jumlah tingkat yang benar bergantung pada program mana yang sedang dijalankan `latexmk` — asumsi yang justru menyebabkan cacat ini. `paper/track2/.latexmkrc` identik dan diperbaiki bersamaan, sebelum Track 2 sempat menemukannya sendiri.

**Audit deskripsi kelima arm.** Pemeriksaan silang antara laporan ini dan manuskrip menemukan bahwa §3.4 mendeskripsikan `fusion_early` sebagai "konkatenasi fitur radiomics mentah ke input CNN", padahal `build_early_fusion_features` menggabungkan **embedding CNN** dengan vektor radiomik terpilih lalu melatih XGBoost. Deskripsi itu salah sejak draf awal dan tidak pernah tertangkap karena tidak ada apa pun yang mengikat prosa ke kode. Karena satu deskripsi salah berarti yang lain patut dicurigai, keempat arm sisanya ikut diaudit baris demi baris; keempatnya benar, dan `fusion_intermediate` ditambahi dimensi proyeksi yang sebelumnya tidak disebut. Tabel §3.4 sekarang mencantumkan nama fungsi setiap arm, sehingga deskripsi yang menyimpang dari kodenya bisa diperiksa dalam hitungan detik, bukan ditemukan kebetulan saat menulis paper.

**Celah cakupan test: `_trim_white`.** `src/stage_08d_run02_fig14.py` memuat `_trim_white`, yang memotong bingkai putih PNG SHAP dengan ambang intensitas. Fungsi ini punya cabang nyata (larik yang seluruhnya putih dikembalikan apa adanya) dan nol uji, karena `tests/` masuk daftar tolak izin pada sesi penulisan ini. Dicatat di sini sebagai celah cakupan, bukan dianggap tidak ada — persis seperti `full_feature_selection_pipeline` di §8.2, yang juga bertahan lama justru karena tidak ada yang memanggil maupun mengujinya. Bedanya, kali ini celahnya diketahui sejak menit pertama, dan satu-satunya gerbang yang menjaganya sekarang adalah pemeriksaan visual figure. Uji asap yang dibutuhkan kecil: satu larik putih seluruhnya, satu larik dengan blok gelap di tengah, periksa bentuk keluarannya.

---

## 9. Rencana lanjutan

1. Jalankan ulang ablasi fusi dengan perbaikan `input_size` (`python -m src.stage_03b_fusion --config configs/config.yaml`), verifikasi `cnn_only` densenet201 kembali mendekati 0.8988.
2. Eksekusi ketiga varian fusi baru (branch_norm, GMU, modality dropout) pada grid penuh.
3. Jalankan panel XAI komparabilitas begitu checkpoint tersedia.
4. Tambahkan sitasi yang hilang lewat Zotero (`docs/laporan/REFERENSI_DIBUTUHKAN.md`). Daftar konkretnya ada di §9.1.
5. Tulis uji asap untuk `_trim_white` (§8.6) begitu `tests/` bisa disentuh lagi.

### 9.1 Sitasi yang dibutuhkan manuskrip Track 1

Per 8 Agustus 2026, `paper/refs.bib` hanya memuat satu citekey yang relevan, `prabhavalkarHybridPETCTRadiomics2026`; entri satunya sisa Zotero yang tidak berhubungan. Lima belas klaim di manuskrip menunggu kuncinya. Semuanya sudah ditandai di `paper/track1/main.tex` dengan makro `\CITE{...}` yang tercetak merah di PDF, dan seluruhnya padam sekaligus dengan mengganti `\draftnotestrue` jadi `\draftnotesfalse`. Penanda sengaja dibuat terlihat, bukan komentar, supaya tidak bisa lolos ke *submit* tanpa disadari.

| # | Bagian | Klaim yang butuh sitasi |
|---|---|---|
| 1 | Related Work | 2–3 studi radiomics LIDC-IDRI yang menopang rentang AUC 0.79–0.94 |
| 2 | Related Work | Kompetisi modalitas / *greedy multimodal learning* |
| 3 | Related Work | Taksonomi early/intermediate/late fusion |
| 4 | Related Work | Gated Multimodal Unit |
| 5 | Metodologi | Makalah dataset LIDC-IDRI (Armato dkk.) |
| 6 | Metodologi | Konvensi agregasi label median LIDC |
| 7 | Metodologi | ConvNeXt |
| 8 | Metodologi | DenseNet |
| 9 | Metodologi | XGBoost |
| 10 | Metodologi | PyRadiomics / IBSI |
| 11 | Metodologi | Uji DeLong (1988) |
| 12 | Metodologi | Layer-CAM, dengan Grad-CAM sebagai pendahulunya |
| 13 | Metodologi | *Pointing game* (Zhang dkk.) |
| 14 | Metodologi | *Energy-based pointing game* / Score-CAM |
| 15 | Metodologi | SHAP (Lundberg & Lee 2017) dan TreeSHAP |

Aturan yang berlaku selama menunggu: **jangan mengarang citekey.** Kalau kunci belum ada di `refs.bib`, penandanya dibiarkan sampai ekspor Zotero berikutnya. `refs.bib` adalah auto-export Better BibTeX dan tidak boleh disunting tangan.

---

## 10. Integritas riset

Semua angka pada laporan ini ditelusuri ke baris CSV nyata yang ditarik dari mesin remote pada 30 Juli 2026, bukan diperkirakan. Batasan bug resolusi dinyatakan eksplisit di titik angkanya muncul (§6.1), bukan disembunyikan.

Untuk angka §6.3, setiap baris CSV sumber menyimpan `run_id` dan `commit_sha`-nya sendiri, dan setiap perbandingan AUC menyimpan kolom `ckpt_kind` sehingga rezim *checkpoint* tidak pernah terpisah dari angkanya — penerapan langsung dari prinsip §8.1. Dua hasil yang tidak menguntungkan klaim penelitian ini dilaporkan apa adanya di titik angkanya muncul: runtuhnya kesetaraan dengan radiomics pada dua backbone tanpa seleksi *checkpoint* (§8.5), dan pointing accuracy `fusion_late` yang sedikit di bawah `cnn_only` alih-alih di atasnya (§6.3.4).

---

## Lampiran: berkas hasil

| Berkas | Isi |
|---|---|
| `artifacts/results/fusion/ablation_summary.csv` | 175 baris, AUC per arm per backbone per fold |
| `artifacts/results/fusion/delong_fusion.csv` | 21 baris, uji DeLong fusi vs radiomics-only |
| `artifacts/results/xai/xai_metrics.csv` | 12 baris, metrik Grad-CAM/Layer-CAM per backbone |
| `artifacts/results/run02/delong_run02.csv` | 12 baris, DeLong `fusion_late` vs `cnn_only` dan vs `radiomics_only`, dua rezim *checkpoint* |
| `artifacts/results/run02/t0_checkpoint_sensitivity.csv` | 3 baris, AUC `best` vs `last` per backbone (§8.5) |
| `artifacts/results/run02/xai_fusion_vs_cnn.csv` | 24 baris, selisih metrik XAI per backbone per himpunan sampel |
| `artifacts/results/run02/combined_advantage_table.md` | tabel utama §6.3.1 |
| `artifacts/results/run02/shap_provenance.csv` | 3 baris, provenance figur SHAP termasuk `identical_across_backbones` |
