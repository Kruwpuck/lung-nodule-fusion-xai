# Laporan Justifikasi Metodologi — LungFuseNet
### Model Backbone, Variasi Hyperparameter, Metrik Evaluasi, dan Skema Pelabelan Dataset

**Disusun berdasarkan**: hasil query Scopus AI (Deep Research) dan verifikasi silang literatur, Juli 2026.
**Gaya sitasi**: IEEE.

---

## Ringkasan Proses Validasi

Sebelum masuk ke hasil final, berikut catatan proses seleksi — penting dicantumkan di metodologi skripsi karena menunjukkan proses seleksi yang disengaja, bukan asal pilih:

| Keputusan awal | Ditemukan masalah saat cross-check | Keputusan final |
|---|---|---|
| DenseNet121 sebagai salah satu backbone "heavyweight" Track 2 | Parameter DenseNet121 (~8M) secara kuantitatif lebih dekat ke kelompok lightweight (<10–15M) daripada heavyweight (20M+) [1] | Diganti VGG16 (~138M, jelas masuk kategori 20M+) |
| Capsule Net (CapsNet) sebagai salah satu backbone Track 1 | Metode Grad-CAM/SHAP belum matang untuk arsitektur CapsNet — confidence "low" pada studi interpretability CapsNet dibanding CNN konvensional [13]–[16] | Diganti GoogLeNet (xAI konvensional, kompatibel langsung dengan Grad-CAM) |

---

## 1. Justifikasi Model per Track

### 1.1 Track 2 — Model Comparison + Hyperparameter/Stability Study

Empat backbone dipilih berdasarkan dua kriteria: (a) representasi dua filosofi arsitektur kontras (efficient/mobile-oriented vs. classic deep convolutional), dan (b) kontras parameter count yang tegas secara kuantitatif untuk mendukung narasi "light vs. heavy".

| Model | Kategori | Parameter | Justifikasi |
|---|---|---|---|
| **MobileNetV2** | Lightweight | ~3.4M | Akurasi tinggi (95–99%), inferensi cepat, stabilitas training tinggi, banyak dipakai pada studi klasifikasi medis dengan sumber daya terbatas [1], [2] |
| **EfficientNet-B0** | Lightweight | ~5.3M | Trade-off akurasi/efisiensi terbaik pada studi pembanding arsitektur CNN, akurasi hingga 98.8% [3], [4] |
| **ResNet50** | Heavyweight | ~25.6M | Backbone heavyweight paling banyak digunakan pada literatur, transfer learning kuat, robust untuk berbagai tugas CT [5] |
| **VGG16** | Heavyweight | ~138M | Representasi arsitektur "plain deep convolutional" klasik, kontras tegas terhadap grup lightweight secara parameter count, dipakai luas sebagai baseline heavyweight pada studi CT [6] |

### 1.2 Track 1 — Fusion Radiomics-CNN + xAI per-Arm

Empat backbone dipilih untuk keragaman filosofi arsitektur (mendukung narasi "pola fitur penting berbeda per arsitektur" di analisis xAI), dan **kematangan metode explainability** menjadi kriteria eksplisit (bukan hanya akurasi).

| Model | Filosofi Arsitektur | Kematangan xAI | Justifikasi |
|---|---|---|---|
| **DenseNet121** | Dense connectivity, feature reuse | Extensive | AUC 0.954–0.989 pada fusion model, Grad-CAM memberikan lokalisasi lesi yang fokus dan konsisten karena konektivitas padat [7]–[9] |
| **InceptionV3** | Multi-scale feature extraction | Extensive | Akurasi hingga 96.7%, AUC ~0.999 pada fusion model; ekstraksi fitur multi-skala mendukung kualitas heatmap Grad-CAM dan atribusi SHAP [10], [11] |
| **Xception** | Depthwise separable convolution | Frequent | Sering dipasangkan dengan InceptionV3 untuk ekstraksi fitur multi-level; konvolusi depthwise-separable mendukung efisiensi sekaligus interpretabilitas peta Grad-CAM [10], [11] |
| **GoogLeNet** | Inception module klasik (multi-scale awal) | Konvensional, kompatibel langsung | Kompetitif secara akurasi, arsitektur konvensional sehingga Grad-CAM dapat diterapkan tanpa modifikasi tambahan [10], [11] |

**Catatan eksklusi CapsNet**: Capsule Net awalnya dipertimbangkan karena performanya kompetitif dan potensi mempertahankan hierarki spasial [12], namun studi eksplisit tentang penerapan Grad-CAM/SHAP pada CapsNet menunjukkan metode saliency/post-hoc yang ada dirancang untuk arsitektur CNN dan **tidak dapat diterapkan langsung** pada CapsNet tanpa modifikasi signifikan, dengan tingkat kepercayaan riset yang rendah dibanding CNN konvensional [13]–[16]. Karena Track 1 menjadikan xAI sebagai komponen inti (bukan pelengkap), risiko implementasi ini dianggap terlalu besar untuk timeline penelitian, sehingga CapsNet dieksklusi.

**Non-overlap check**: {MobileNetV2, EfficientNet-B0, ResNet50, VGG16} ∩ {DenseNet121, InceptionV3, Xception, GoogLeNet} = ∅. Tidak ada model yang sama antar track.

---

## 2. Parameter Variant Track 2

### 2.1 Optimizer

Trio **SGD+momentum, Adam, AdamW** dikonfirmasi representatif — literatur medis menyebutnya sebagai kombinasi standar yang dibandingkan secara luas [17], meski beberapa studi memperluas perbandingan ke RMSProp, Nadam, AdaGrad, dan AdaBelief [17], [18]. Rasionalnya:
- **Adam**: konvergensi cepat, tuning minimal, cocok untuk dataset kecil/noisy [17]
- **SGD+momentum**: generalisasi lebih baik, robust terhadap domain shift, unggul pada dataset kecil [17]
- **AdamW**: decoupled weight decay memperbaiki generalisasi dan mengurangi overfitting, terutama pada dataset imbalanced [17]

Trio ini dipertahankan sebagai desain final Track 2 karena representatif dan cukup untuk kontras stabilitas 3-arah tanpa memperbesar scope eksperimen secara berlebihan.

### 2.2 Weight Decay

**Tidak ditemukan ambang default numerik universal** yang eksplisit disepakati pada literatur untuk SGD/Adam/AdamW [19]–[22] — ini adalah gap yang perlu diakui secara transparan di laporan metodologi, bukan disembunyikan. Yang dapat dipertanggungjawabkan dari literatur:
- Weight decay adalah komponen regularisasi fundamental untuk generalisasi, khususnya pada SGD, dan kurang efektif pada Adam standar tanpa modifikasi seperti AdamW [20], [21]
- AdamW dengan decoupled weight decay terbukti superior dibanding Adam+L2 regularization standar [21], [22]

**Rekomendasi desain**: karena tidak ada nilai default "resmi" di literatur, digunakan nilai default implementasi standar per-optimizer (PyTorch: SGD/Adam = 1e-4, AdamW = 1e-2, mengikuti desain decoupled-nya) sebagai titik tengah, dengan variasi order-of-magnitude (×10) di atas dan di bawah — konvensi ini perlu dicantumkan sebagai **keputusan desain peneliti** (bukan hasil sitasi langsung), didukung prinsip regularisasi dari [20]–[22].

### 2.3 Parameter Tambahan yang Dipertimbangkan (opsional)

Selain optimizer dan weight decay, literatur secara konsisten menyoroti hyperparameter berikut sebagai faktor stabilitas training dan pembanding light-vs-heavy [23]–[25]:

| Hyperparameter | Relevansi Stabilitas | Relevansi Light vs Heavy |
|---|---|---|
| Learning rate | ✓ (paling kritis) | ✓ |
| Batch size | ✓ | ✓ |
| Arsitektur (jumlah layer, filter) | — | ✓ |
| Learning rate schedule | ✓ | — |
| Momentum | ✓ | — |
| Regularisasi (dropout, L2 tambahan) | ✓ | ✓ |

Jika scope eksperimen 180-run (4 model × 3 optimizer × 3 weight decay × 5 fold) dirasa kurang "menjual", **learning rate schedule** adalah kandidat tambahan paling kuat secara literatur karena berinteraksi langsung dengan pilihan optimizer [23]–[25].

---

## 3. Evaluation Metric per Track

### 3.1 Track 2 — Metrik Stabilitas

Metrik utama yang didukung literatur untuk kuantifikasi stabilitas training lintas fold cross-validation [28], [29]:
- **AUC Variance**: sebaran nilai AUC antar fold; makin rendah, makin stabil
- **Coefficient of Variation (CV)**: rasio standar deviasi terhadap rata-rata AUC, memungkinkan perbandingan ternormalisasi antar model
- **ANOVA / bootstrap non-parametrik**: dekomposisi varians inter-model dan intra-model untuk pengujian statistik formal [29]
- **Confidence Interval 95%** untuk AUC/akurasi, dilaporkan berdampingan dengan mean dan SD

Literatur secara konsisten melaporkan model lightweight (mis. EfficientNet-B0) menunjukkan AUC variance dan CV lebih rendah dibanding model heavyweight (mis. ResNet varian) — pola ini relevan langsung sebagai hipotesis awal untuk narasi "kondisi apa light vs heavy menang" di Track 2 [28].

### 3.2 Track 1 — Metrik Kontribusi Modalitas & xAI

**Temuan penting (gap riset)**: penelusuran literatur **tidak menemukan metrik model-agnostic yang terstandardisasi** untuk membandingkan kontribusi fitur secara adil lintas SHAP (radiomics) dan Grad-CAM (CNN) [26], [27]. Sebagian besar studi fusion melaporkan performa keseluruhan (AUC, akurasi) setelah fusion, tanpa mendekomposisi kontribusi tiap modalitas secara kuantitatif [26].

**Implikasi untuk Track 1**: ini justru bisa menjadi **kontribusi/novelty** penelitian kalian, bukan hanya keterbatasan — mengembangkan atau mengadaptasi metrik kontribusi modalitas (mis. pendekatan ablation berbasis occlusion/zero-out per modalitas, dibandingkan dengan performa fusion penuh) mengisi gap yang secara eksplisit diakui pada literatur [26], [27]. Pendekatan yang dapat diadaptasi:
- Pelaporan AUC per-arm + DeLong test (sudah ada di desain existing)
- Analisis SHAP untuk cabang radiomics dan Grad-CAM/Layer-CAM untuk cabang CNN dilaporkan **terpisah per-arm**, dengan interpretasi kualitatif kontribusi relatif tiap modalitas berdasarkan perubahan performa saat satu modalitas di-ablasi

---

## 4. Klasifikasi Dataset — Justifikasi Threshold Median Rating

> **Catatan penting**: bagian ini **belum divalidasi lewat Scopus AI** pada sesi ini (belum dijalankan Langkah 1 dari prosedur pencarian yang direncanakan). Sitasi di bawah berasal dari penelusuran literatur independen (arXiv/jurnal) dengan metadata penulis terverifikasi — disarankan tetap menjalankan query Scopus AI yang sudah disiapkan sebelumnya untuk cross-check tambahan sebelum difinalisasi di laporan skripsi.

### 4.1 Konvensi yang Digunakan

Aturan **median rating radiolog > 3 = malignant, < 3 = benign, = 3 dieksklusi/indeterminate** adalah konvensi yang dipakai berulang pada studi klasifikasi nodul paru berbasis LIDC-IDRI:

- Al-Shabi *et al.* menggunakan median rating dari minimal tiga radiolog; kasus dengan median >3 dikategorikan malignant, <3 benign, dan median tepat 3 dieksklusi — menghasilkan 848 nodul (442 benign, 406 malignant) [30].
- Studi ProCAN oleh peneliti yang sama menggunakan skema identik dengan hasil dataset yang sama persis (848 nodul, 442 benign, 406 malignant), mengonfirmasi konsistensi konvensi ini lintas studi mereka [31].

### 4.2 Justifikasi Klinis untuk Kelas "Indeterminate"

Le Folgoc *et al.* secara eksplisit mencatat bahwa skor malignancy subjektif pada LIDC-IDRI berkisar dari 1 (benign) hingga 5 (malignant), dengan **skor 3 secara eksplisit merepresentasikan ketidakpastian tinggi dari radiolog itu sendiri** — dan skor ini yang lazim dipakai sebagai ambang biner ketika label harus dibinerkan [32]. Ini memberi dasar langsung bahwa mengeluarkan/memisahkan median=3 sebagai kelas tersendiri bukan pilihan sembarangan, melainkan pengakuan eksplisit terhadap zona ketidakpastian yang sudah teridentifikasi dalam desain dataset itu sendiri.

### 4.3 Apakah Perlu Mengikuti "Label Asli" Dataset?

LIDC-IDRI **tidak menyediakan label benign/malignant biner "asli"** untuk mayoritas nodul — yang tersedia hanyalah skor subjektif 1–5 dari tiap radiolog per nodul [32]. Median>3/<3/=3 yang dipakai penelitian kalian **sudah merupakan interpretasi standar** yang dipakai berulang di komunitas riset terhadap skor mentah tersebut [30], [31] — bukan penyimpangan dari suatu label resmi yang lebih otoritatif. Pendekatan ini dinilai lebih konservatif secara klinis dibanding memaksakan biner tanpa mengakui ketidakpastian pada rating menengah.

---

## Daftar Referensi (IEEE)

[1] M. Mahmoud, Y. Wen, X. Pan, and Y. Guan, "Evaluation of recent lightweight deep learning architectures for lung cancer CT classification," *Frontiers in Oncology*, 2025.

[2] T. Wang, Z. Huang, H. Wang, and W. Zhao, "Rock Thin Slice Lithology Identification Based on MobileNetV2," *J. Jilin Univ. (Earth Sci. Ed.)*, 2024.

[3] S. Sharma and K. Guleria, "Deep Learning Models for Image Classification: Comparison and Applications," in *Proc. 2022 2nd Int. Conf. Advance Computing and Innovative Technologies in Engineering (ICACITE)*, 2022.

[4] N. Muzoğlu, A. M. Halefoğlu, M. O. Avci, and B. S. B. Yarman, "Detection of COVID-19 and its pulmonary stage using Bayesian hyperparameter optimization and deep feature selection methods," *Expert Systems*, 2023.

[5] Z. Chen, "Transfer Learning and Batch Dynamics for Medical Image Classification: A Comparative Study of Modern CNN Architectures," in *Proc. 2025 IEEE 8th Int. Conf. Information Systems and Computer Aided Education (ICISCAE)*, 2025.

[6] V. Sudharsanan and D. B. David, "Improving the accuracy in classifying the chest X-ray for detecting COVID-19 using DenseNet121 in comparison with VGG-16," *AIP Conf. Proc.*, 2025.

[7] P. P. Sarangi, A. Panigrahi, and R. Dash, "Melanoma Detection Using Transfer Learning: A Comparative Study of Pretrained CNN Models," *Lecture Notes in Networks and Systems*, 2026.

[8] A. Al Mamun Sheikh, M. L. Bhuiyan, S. M. Mahdi, and A. Sattar, "A Lightweight D-CNN for the Classification and Detection of Brain Tumors with XAI Integration," in *Proc. 2026 IEEE 2nd Int. Conf. Quantum Photonics, Artificial Intelligence and Networking (QPAIN)*, 2026.

[9] Y. Tang, J. Yang, Y. Luo, and W. Fang, "A fusion model of deep learning and conventional features based on computed tomography angiography of carotid plaque for predicting the risk of acute ischemic stroke," *Quantitative Imaging in Medicine and Surgery*, 2026.

[10] M. Ardiansyah and M. K. Putro, "Comparative Study of Pretrained CNN Architectures for Tomato Leaf Disease Diagnosis Using Transfer Learning," in *Proc. Int. Conf. Information and Communications Technology (ICOIACT)*, 2025.

[11] Z. Li, J. Yang, X. Wang, and S. Zhou, "Establishment and Evaluation of Intelligent Diagnostic Model for Ophthalmic Ultrasound Images Based on Deep Learning," *Ultrasound in Medicine & Biology*, 2023.

[12] H. Liu, Z. Jiao, W. Han, and B. Jing, "Identifying the histologic subtypes of non-small cell lung cancer with computed tomography imaging: A comparative study of capsule net, convolutional neural network, and radiomics," *Quantitative Imaging in Medicine and Surgery*, 2021.

[13] J. Gu, "Interpretable Graph Capsule Networks for Object Recognition," in *Proc. 35th AAAI Conf. Artificial Intelligence (AAAI)*, 2021.

[14] S. Tawalbeh and J. Oramas, "Towards the characterization of representations learned via capsule-based network architectures," *Neurocomputing*, 2025.

[15] M. U. Haq, M. A. J. Sethi, and A. U. Rehman, "Capsule Network with Its Limitation, Modification, and Applications—A Survey," *Machine Learning and Knowledge Extraction*, 2023.

[16] A. Bondarenko, S. Tawalbeh, and J. Oramas, "Analyzing the Explanation and Interpretation Potential of Matrix Capsule Networks," *Communications in Computer and Information Science*, 2025.

[17] S. A. Adablanu, U. Barman, and D. Das, "15 Years of optimizers in medical deep learning: A systematic review," *Neuroscience Informatics*, 2026.

[18] K. A. Sayın, N. K. Gürsoy, T. Yolcu, and A. Gürsoy, "On the Synergy of Optimizers and Activation Functions: A CNN Benchmarking Study," *Mathematics*, 2025.

[19] X. Jia, X. Feng, H. Yong, and D. Meng, "Weight Decay With Tailored Adam on Scale-Invariant Weights for Better Generalization," *IEEE Trans. Neural Networks and Learning Systems*, 2024.

[20] F. D'Angelo, M. Andriushchenko, A. Varre, and N. Flammarion, "Why Do We Need Weight Decay in Modern Deep Learning?," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2024.

[21] G. Zhang, C. Wang, B. Xu, and R. Grosse, "Three mechanisms of weight decay regularization," in *Proc. 7th Int. Conf. Learning Representations (ICLR)*, 2019.

[22] P. Zhou, X. Xie, Z. Lin, and S. Yan, "Towards Understanding Convergence and Generalization of AdamW," *IEEE Trans. Pattern Analysis and Machine Intelligence*, 2024.

[23] S. Echamsi, E. Belouafi, A. El Bakali Kassimi, and A. Guennoun, "Optimizing Training Hyperparameters for Multilayer Perceptrons in Deep Learning," in *Proc. 2025 5th Int. Conf. Innovative Research in Applied Science, Engineering and Technology (IRASET)*, 2025.

[24] M. Wojciuk, Z. Swiderska-Chadaj, K. Siwek, and A. Gertych, "Improving classification accuracy of fine-tuned CNN models: Impact of hyperparameter optimization," *Heliyon*, 2024.

[25] J. B. Hopson, A. Flaus, C. J. McGinnity, and A. Hammers, "Deep Convolutional Backbone Comparison for Automated PET Image Quality Assessment," *IEEE Trans. Radiation and Plasma Medical Sciences*, 2024.

[26] M. S. Saravanan, L. Kartheesan, and S. N. Bhavanam, "A Hybrid Deep Learning Framework Integrating CNN and Radiomics Features for Automated Classification of Ovarian Tumors," in *Proc. 2025 3rd Int. Conf. Sustainable Computing and Smart Systems (ICSCSS)*, 2025.

[27] O. Davydko, V. Pavlov, P. Biecek, and L. Longo, "SRFAMap: A Method for Mapping Integrated Gradients of a CNN Trained with Statistical Radiomic Features to Medical Image Saliency Maps," *Communications in Computer and Information Science*, 2024.

[28] W.-C. Shia, "Deep learning-based classification of benign and malignant breast microcalcifications in mammography," *Scientific Reports*, 2025.

[29] M. Gasmi, H. Bendjenna, and A. Meraoumia, "A Robust ANOVA-Based Framework for Multi-Metric Comparison of Convolutional Neural Networks in Medical Imaging," in *Proc. Int. Conf. Recent Advances in Mathematics and Informatics (ICRAMI)*, 2025.

[30] M. Al-Shabi, K. Shak, and M. Tan, "3D Axial-Attention for Lung Nodule Classification," *International Journal of Computer Assisted Radiology and Surgery*, vol. 16, pp. 1–6, 2021.

[31] M. Al-Shabi, K. Shak, and M. Tan, "ProCAN: Progressive Growing Channel Attentive Non-Local Network for Lung Nodule Classification," *arXiv:2010.15417*, 2020.

[32] L. Le Folgoc, V. Baltatzis, A. Alansary, S. Desai, A. Devaraj, S. Ellis, O. E. Martinez Manzanera, F. Kanavati, A. Nair, J. Schnabel, and B. Glocker, "Bayesian analysis of the prevalence bias: learning and predicting from imbalanced data," *arXiv:2108.00250*, 2021.
