# Referensi yang dibutuhkan untuk paper/track1 dan paper/track2

**Berkas ini daftar kanonik.** Tabel §9.1 `LAPORAN_TRACK1_FUSION_XAI.md` adalah
cermin: ia mendaftar *klaim* mana yang menunggu sitasi, berkas ini mendaftar
*sumber* mana yang mengisinya. Kalau keduanya berbeda, yang benar berkas ini.

`paper/refs.bib` cuma punya 2 entri saat ini, dan hanya
`prabhavalkarHybridPETCTRadiomics2026` yang relevan dan sudah dipakai di
`paper/track1/main.tex`. Kalimat di manuskrip sengaja ditulis tanpa `\cite{}`
karena kuncinya belum ada di `refs.bib` — jangan mengarang citekey. Tambahkan
lewat Zotero (Better BibTeX auto-export), lalu isi `\cite{...}` di tempat yang
ditandai `\CITE{...}` (tercetak merah di PDF; seluruhnya padam sekaligus dengan
mengganti `\draftnotestrue` jadi `\draftnotesfalse`).

DOI pada dua bagian pertama dikutip dari `docs/Review Revisi 1.md` dan sudah
terverifikasi. **Bagian "Masih kosong" di bawah tidak punya DOI terverifikasi**
dan wajib dicek sendiri saat ditarik ke Zotero.

## Untuk paper/track1 (fusion + XAI)

- Astaraki et al., "A Comparative Study of Radiomics and Deep-Learning Based
  Methods for Pulmonary Nodule Malignancy Prediction in Low Dose CT Images,"
  Frontiers in Oncology 11:737368, 2021. DOI 10.3389/fonc.2021.737368.
  (radiomics vs deep learning precedent, dipakai di §Related Work dan
  §Discussion Track 1)
- Wu, Jastrzebski, Cho & Geras, "Characterizing and overcoming the greedy
  nature of learning in multi-modal deep neural networks," ICML 2022,
  PMLR 162:24043–24055. arXiv:2202.05306. (modality competition / greedy
  learning, dipakai di §Related Work Track 1)
- Bhattacharya et al., "Synergy vs. Noise: Performance-Guided Multimodal
  Fusion for Biochemical Recurrence-Free Survival in Prostate Cancer,"
  arXiv:2511.11452. (fusi modalitas lemah mendegradasi performa)
- Arevalo, Solorio, Montes-y-Gómez, González, "Gated Multimodal Units for
  Information Fusion," ICLR 2017 workshop, arXiv:1702.01992. (dasar
  implementasi GMU di 5b, dipakai kalau bagian follow-up GMU disebut
  secara teknis)

## Untuk paper/track2 (komparasi + stabilitas)

> **Catatan (20 Agustus 2026).** Empat entri di bawah ditulis untuk framing rev1
> (paper stabilitas). Track 2 sedang ditulis ulang di sekitar pertanyaan yang
> berbeda — lihat "Tambahan rev2 Track 2" di bawah. Keempatnya **dipertahankan**,
> bukan dibuang: sebagiannya masih relevan, dan penghapusan diam-diam bakal
> menghilangkan jejak kandidat yang sudah pernah diverifikasi.

- Kumar et al., "How to Fine-Tune Vision Models with SGD," arXiv:2211.09359.
  (learning-rate mismatch SGD vs Adam, dipakai di §Discussion Track 2)
- Rosenfeld et al., "A Constructive Prediction of the Generalization Error
  Across Scales," arXiv:1909.12673. (resep learning rate SGD standar)
- Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets,"
  JMLR 7:1–30, 2006. (dasar metodologi Friedman + Nemenyi)
- Lei, Li, Shen, Zhang, Shan, "CLIP-Lung: Textual Knowledge-Guided Lung
  Nodule Malignancy Prediction," MICCAI 2023, LNCS 14226:403–412,
  DOI 10.1007/978-3-031-43990-2_38. (precedent granularitas label
  binary vs 3-class)

## Untuk kedua paper

- Armato et al., "The Lung Image Database Consortium (LIDC) and Image
  Database Resource Initiative (IDRI)," Medical Physics 38(2):915–931, 2011.
  DOI 10.1118/1.3528204. (sitasi dataset LIDC-IDRI, wajib di kedua paper)

## Masih kosong: 11 dari 15 klaim Track 1 belum punya kandidat

Empat sumber di atas menutup empat baris §9.1 (radiomics LIDC-IDRI, modality
competition, GMU, dataset LIDC-IDRI). Sebelas sisanya belum punya kandidat sama
sekali, dan seluruhnya berupa makalah metode standar yang mestinya cepat ditarik
dari Zotero. Didaftar di sini supaya kekosongannya terlihat, bukan ditemukan
belakangan saat manuskrip sudah mau dikirim.

| §9.1 | Klaim | Petunjuk pencarian |
|---|---|---|
| 3 | Taksonomi early/intermediate/late fusion | Survei fusi multimodal; Baltrušaitis dkk. sering dipakai sebagai rujukan taksonomi |
| 6 | Konvensi agregasi label median LIDC | Konvensi dari literatur LIDC-IDRI, bukan satu makalah kanonik — mungkin perlu 2 rujukan praktik |
| 7 | ConvNeXt | Liu dkk., "A ConvNet for the 2020s", CVPR 2022 |
| 8 | DenseNet | Huang dkk., "Densely Connected Convolutional Networks", CVPR 2017 |
| 9 | XGBoost | Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016 |
| 10 | PyRadiomics / IBSI | van Griethuysen dkk. 2017 (PyRadiomics); Zwanenburg dkk. (IBSI) — dua rujukan terpisah |
| 11 | Uji DeLong | DeLong, DeLong & Clarke-Pearson, Biometrics 1988 |
| 12 | Layer-CAM, dengan Grad-CAM sebagai pendahulu | Jiang dkk. (LayerCAM); Selvaraju dkk. (Grad-CAM) — dua rujukan terpisah |
| 13 | Pointing game | Zhang dkk., excitation backprop / top-down attention |
| 14 | Energy-based pointing game / Score-CAM | Wang dkk., Score-CAM, CVPRW 2020 |
| 15 | SHAP dan TreeSHAP | Lundberg & Lee, NeurIPS 2017; Lundberg dkk. untuk TreeSHAP — kemungkinan dua rujukan |

**Peringatan.** Kolom petunjuk di atas ditulis dari ingatan umum, **bukan
disalin dari sumber terverifikasi** seperti dua bagian sebelumnya. Judul, tahun,
maupun venue-nya bisa meleset. Perlakukan sebagai kata kunci pencarian Zotero,
dan ambil keterangan bibliografisnya dari Zotero, bukan dari tabel ini.

Perhatikan juga bahwa empat baris menyebut kemungkinan dua rujukan sekaligus,
jadi 11 klaim ini bisa berujung pada sekitar 15 entri `refs.bib`.

---

## Tambahan rev2 (19 Agustus 2026)

Revisi rev2 menaikkan penanda `\CITE{}` di `paper/track1/main.tex` dari 15 jadi 31.
Sumber tambahannya berasal dari `docs/revisi/rev2/Track1 v2.md` §5, yang mencantumkan
DOI lengkap. Bagian ini menyalinnya ke daftar kanonik supaya tidak ada dua daftar
yang bisa menyimpang diam-diam — masalah yang sama dengan §8.10 laporan Track 1.

**Peringatan yang berlaku sebelum apa pun ditarik ke Zotero:** sampai 19 Agustus 2026
`bibtex` diveto diam-diam oleh `latexmk`, sehingga entri yang ditambahkan ke `refs.bib`
tidak akan pernah muncul di PDF dan tidak ada yang mengeluh. Bug itu sudah diperbaiki
(§8.10 laporan Track 1). Verifikasi setelah ekspor bukan "build berhasil", melainkan:

```
grep "Database file" paper/track1/build/main.blg
```

### Uji ekuivalensi dan margin (§6.4 laporan, §Statistical protocol manuskrip)

- Lakens D. Equivalence Tests: A Practical Primer for t Tests, Correlations, and
  Meta-Analyses. *Social Psychological and Personality Science* 2017;8(4):355–362.
  DOI 10.1177/1948550617697177. (prinsip TOST; dasar kalimat "gagal menolak bukan bukti
  kesetaraan")
- Liu J-P, Ma M-C, Wu C-Y, Tai J-Y. Tests of equivalence and non-inferiority for
  diagnostic accuracy based on the paired areas under ROC curves. *Statistics in
  Medicine* 2006;25(7):1219–1238. DOI 10.1002/sim.2358. (formulasi paired-AUC yang
  dipakai `tost_auc`)
- Lin H, Huang C, Wang W, Luo J, Yang X, Liu Y. Measuring Interobserver Disagreement
  in Rating Diagnostic Characteristics of Pulmonary Nodule Using the Lung Imaging
  Database Consortium and Image Database Resource Initiative. *Academic Radiology*
  2017;24(4):401–410. **[FLAG: DOI belum diverifikasi; PII Elsevier
  S1076-6332(17)30009-0. Cek halaman penerbit sebelum submit.]** (menopang justifikasi
  margin klaim 1: disagreement rating malignansi 0.2144)
- Baltatzis V, Bintsi K-M, Le Folgoc L, dkk. The Pitfalls of Sample Selection: A Case
  Study on Lung Nodule Classification. *PRIME @ MICCAI 2021*, LNCS 12928:201–211.
  DOI 10.1007/978-3-030-87602-9_19. (menopang justifikasi margin klaim 2, sekaligus
  dipakai di Limitations untuk menolak perbandingan AUC lintas-makalah)

### Posisi kontribusi (§Related Work, §Discussion manuskrip)

- Astaraki M, Yang G, Zakko Y, Toma-Dasu I, Smedby Ö, Wang C. A Comparative Study of
  Radiomics and Deep-Learning Based Methods for Pulmonary Nodule Malignancy Prediction
  in Low Dose CT Images. *Frontiers in Oncology* 2021;11:737368.
  DOI 10.3389/fonc.2021.737368. (preseden terkuat: radiomics 0.921, DL 0.824,
  deep-feature 0.936, hybrid 0.938 — keunggulan hybrid cuma 0.017)
- Rudin C. Stop explaining black box machine learning models for high stakes decisions
  and use interpretable models instead. *Nature Machine Intelligence* 2019;1(5):206–215.
  DOI 10.1038/s42256-019-0048-x. (tulang punggung normatif argumen interpretability-first)
- Shen S, Han SX, Aberle DR, Bui AA, Hsu W. An Interpretable Deep Hierarchical Semantic
  Convolutional Neural Network for Lung Nodule Malignancy Classification (HSCNN).
  *Expert Systems with Applications* 2019;128:84–95. DOI 10.1016/j.eswa.2019.01.048.
  (analog paling dekat: AUC 0.856 lawan 0.847 black-box, dibenarkan lewat interpretability)
- LaLonde R, Torigian D, Bagci U. Encoding Visual Attributes in Capsules for Explainable
  Medical Diagnoses (X-Caps). *MICCAI* 2020, LNCS 12261:294–304.
  DOI 10.1007/978-3-030-59710-8_29. **[FLAG: nomor halaman perlu dicek.]**

### Faithfulness (Limitations manuskrip; fase GOAL3)

- Rong Y, Leemann T, Borisov V, Kasneci G, Kasneci E. A Consistent and Efficient
  Evaluation Strategy for Attribution Methods (ROAD). *ICML* 2022, PMLR 162:18770–18795.
  arXiv:2202.00449. **[Tanpa DOI; kutip sebagai proceedings/arXiv.]**
- Draelos RL, Carin L. Use HiResCAM instead of Grad-CAM for faithful explanations of
  convolutional neural networks. arXiv:2011.08891. **[FLAG: preprint, nomor arXiv perlu
  dicocokkan dengan PDF-nya sebelum submit.]** (HiResCAM sudah terpasang di
  `src/xai/gradcam_utils.py:163,177`, jadi ini sitasi untuk kode yang sudah ada)

### Standar pelaporan (Limitations manuskrip)

- Mongan J, Moy L, Kahn CE Jr. Checklist for Artificial Intelligence in Medical Imaging
  (CLAIM). *Radiology: Artificial Intelligence* 2020;2(2):e200029.
  DOI 10.1148/ryai.2020200029. Pembaruan 2024: Tejani AS, Klontzas ME, Gatti AA, dkk.
  *Radiol Artif Intell* 2024;6(4):e240300. DOI 10.1148/ryai.240300.
- Collins GS, Moons KGM, Dhiman P, dkk. TRIPOD+AI statement. *BMJ* 2024;385:e078378.
  DOI 10.1136/bmj-2023-078378.
- Varoquaux G, Cheplygina V. Machine learning for medical imaging: methodological
  failures and recommendations for the future. *npj Digital Medicine* 2022;5:48.
  DOI 10.1038/s41746-022-00592-y.

### Daftar DOI siap-tempel

Zotero → **Add Item by Identifier**, tempel seluruh blok sekaligus. Baris ber-DOI saja;
yang hanya punya arXiv atau perlu dicek manual didaftar terpisah di bawahnya.

```
10.1177/1948550617697177
10.1002/sim.2358
10.1007/978-3-030-87602-9_19
10.3389/fonc.2021.737368
10.1038/s42256-019-0048-x
10.1016/j.eswa.2019.01.048
10.1007/978-3-030-59710-8_29
10.1148/ryai.2020200029
10.1148/ryai.240300
10.1136/bmj-2023-078378
10.1038/s41746-022-00592-y
10.1118/1.3528204
10.1109/TIP.2021.3089943
10.1007/s11263-019-01228-7
10.1007/s11263-017-1059-x
10.1109/CVPRW50498.2020.00020
10.1158/0008-5472.CAN-17-0339
10.1148/radiol.2020191145
10.2307/2531595
10.1109/TPAMI.2018.2798607
10.1109/CVPR52688.2022.00806
10.1038/s41746-020-00341-z
```

Ditarik terpisah (tanpa DOI penerbit, pakai arXiv ID atau entri manual):

```
arXiv:2202.00449    ROAD (Rong dkk., ICML 2022, PMLR 162:18770-18795)
arXiv:2011.08891    HiResCAM (Draelos & Carin)  [FLAG: cek nomornya]
arXiv:1705.07874    SHAP (Lundberg & Lee, NeurIPS 2017)
arXiv:2202.05306    Greedy multimodal (Wu dkk., ICML 2022, PMLR 162:24043-24055)
arXiv:1702.01992    GMU (Arevalo dkk., ICLR 2017 workshop)
arXiv:2511.11452    Synergy vs. Noise (Bhattacharya dkk.)
Lin dkk. 2017, Academic Radiology 24(4):401-410, PII S1076-6332(17)30009-0  [FLAG: cek DOI]
Chen & Guestrin, XGBoost, KDD 2016
Liu dkk., A ConvNet for the 2020s, CVPR 2022
Huang dkk., Densely Connected Convolutional Networks, CVPR 2017
Demsar, JMLR 2006;7:1-30  [FLAG: cek volume/halaman]
```

**Aturan yang tetap berlaku.** `refs.bib` adalah auto-export Better BibTeX dan tidak
boleh disunting tangan; daftar di atas untuk ditarik lewat Zotero, bukan disalin ke
`.bib`. Setelah ekspor, ganti penanda `\CITE{...}` di `paper/track1/main.tex` dengan
`\cite{kunci}` yang benar-benar ada. **Jangan mengarang citekey.**

---

## Tambahan rev2 Track 2 (20 Agustus 2026)

Track 2 ditulis ulang: bukan lagi paper stabilitas rev1, melainkan analisis gabungan
**efisiensi model + kualitas eksplanasi** lintas keluarga arsitektur CNN pada LIDC-IDRI.
Sumbernya `docs/revisi/rev2/Track2 v2.md` §"Verified Citation List (with DOIs)" dan
§"Caveats". Disalin ke sini supaya daftar kanonik tetap satu, sama seperti bagian
Track 1 di atas.

Dikelompokkan menurut **klaim yang ditopang**, bukan alfabetis — supaya jelas entri mana
yang hilang kalau sebuah klaim dicoret. DOI di bawah sudah terverifikasi di sumbernya
kecuali yang ditandai `[FLAG]`.

### Arsitektur backbone (§Methods, tabel arsitektur)

- Szegedy C, Liu W, Jia Y, dkk. Going Deeper with Convolutions (GoogLeNet). *CVPR* 2015,
  pp. 1–9. DOI 10.1109/CVPR.2015.7298594.
- Szegedy C, Vanhoucke V, Ioffe S, Shlens J, Wojna Z. Rethinking the Inception
  Architecture for Computer Vision (InceptionV3). *CVPR* 2016, pp. 2818–2826.
  DOI 10.1109/CVPR.2016.308.
- Szegedy C, Ioffe S, Vanhoucke V, Alemi A. Inception-v4, Inception-ResNet and the Impact
  of Residual Connections on Learning. *AAAI* 2017;31(1):4278–4284.
  DOI 10.1609/aaai.v31i1.11231. **[FLAG: ada identifier tandingan di ACM DL,
  10.5555/3298023.3298188. DOI AAAI di atas yang kanonik — pastikan Zotero tidak menarik
  yang versi ACM, kalau iya buang dan tarik ulang lewat DOI AAAI.]**
- Chollet F. Xception: Deep Learning with Depthwise Separable Convolutions. *CVPR* 2017,
  pp. 1800–1807. DOI 10.1109/CVPR.2017.195.
- Huang G, Liu Z, van der Maaten L, Weinberger KQ. Densely Connected Convolutional
  Networks. *CVPR* 2017, pp. 2261–2269. DOI 10.1109/CVPR.2017.243. Versi jurnal:
  Convolutional Networks with Dense Connectivity, *IEEE TPAMI*,
  DOI 10.1109/TPAMI.2019.2918284. (dense skip connection sebagai mekanisme aliran gradien
  bersih ke layer CAM — dipakai di §Discussion, bukan cuma sitasi arsitektur)
- Liu Z, Mao H, Wu C-Y, Feichtenhofer C, Darrell T, Xie S. A ConvNet for the 2020s
  (ConvNeXt). *CVPR* 2022, pp. 11976–11986. DOI 10.1109/CVPR52688.2022.01167.
  **[FLAG: blok Track 1 di atas mencantumkan 10.1109/CVPR52688.2022.00806 untuk makalah
  yang sama. Kedua nomor tidak bisa dua-duanya benar — cek satu kali di IEEE Xplore,
  lalu samakan kedua blok. Jangan tarik dua entri ConvNeXt ke Zotero.]**

### Metode CAM dan metrik lokalisasi (§Methods XAI, §Results lokalisasi)

- Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: Visual
  Explanations from Deep Networks via Gradient-based Localization. *ICCV* 2017.
  DOI 10.1109/ICCV.2017.74. Versi jurnal *IJCV* 2020;128(2):336–359.
  DOI 10.1007/s11263-019-01228-7. (sumber primer; lihat juga "Celah yang harus dicari"
  di bawah — paper butuh kutipan pasasenya, bukan cuma entri bibliografinya)
- Jiang P-T, Zhang C-B, Hou Q, Cheng M-M, Wei Y. LayerCAM: Exploring Hierarchical Class
  Activation Maps for Localization. *IEEE TIP* 2021;30:5875–5888.
  DOI 10.1109/TIP.2021.3089943. (alasan CAM dihitung di tahap 8×8)
- Wang H, Wang Z, Du M, dkk. Score-CAM: Score-Weighted Visual Explanations for
  Convolutional Neural Networks. *CVPR Workshops* 2020, pp. 111–119.
  DOI 10.1109/CVPRW50498.2020.00020. (silang-uji bebas gradien)
- Draelos RL, Carin L. Use HiResCAM instead of Grad-CAM for faithful explanations of
  convolutional neural networks. arXiv:2011.08891. DOI 10.48550/arXiv.2011.08891.
  (teorema faithfulness: jaminan cuma berlaku untuk head GAP→satu FC; alasan prinsipil
  kenapa head keluarga Inception bisa berperilaku lain)
- Zhang J, Bargal SA, Lin Z, dkk. Top-Down Neural Attention by Excitation Backprop.
  *ECCV* 2016. DOI 10.1007/978-3-319-46493-0_33. Versi *IJCV* 2018;126(10):1084–1102.
  DOI 10.1007/s11263-017-1059-x. (asal metrik pointing game)
- Adebayo J, Gilmer J, Muelly M, Goodfellow I, Hardt M, Kim B. Sanity Checks for Saliency
  Maps. *NeurIPS* 2018. arXiv:1810.03292. **[FLAG: tidak ada DOI Crossref untuk NeurIPS
  2018 — kutip arXiv atau halaman proceedings.]** (uji randomisasi parameter/data yang
  wajib dijalankan pada GoogLeNet sebelum angka 0.0000 boleh diklaim)

### Pemilihan target layer dan kegagalannya (§Methods, §Limitations)

- *Grad-CAM: The impact of large receptive fields and other caveats.* *CVIU* 2025;258.
  DOI 10.1016/j.cviu.2025.104383. (organisasi spasial fitur tidak terjaga pada receptive
  field besar sehingga eksplanasi bisa kehilangan makna; membahas **empty map**,
  rektifikasi, dan efek GAP-vs-flatten — jangkar terdekat untuk mode kegagalan Inception)
- Selvaraju dkk. dan Jiang dkk. di atas juga menopang klaim ini: keduanya
  mendokumentasikan ketergantungan kuat kualitas lokalisasi pada kedalaman layer, yang
  membuat analisis sensitivitas lintas kandidat target layer wajib, bukan opsional.

### Efisiensi dan Green AI (§Results efisiensi, §Discussion)

- Raghu M, Zhang C, Kleinberg J, Bengio S. Transfusion: Understanding Transfer Learning
  for Medical Imaging. *NeurIPS* 2019. arXiv:1902.07208. DOI 10.48550/arXiv.1902.07208.
  (sumber kanonik "kapasitas tidak membeli akurasi"; sebagian manfaat transfer berasal
  dari over-parametrization, bukan feature reuse canggih)
- Schwartz R, Dodge J, Smith NA, Etzioni O. Green AI. *CACM* 2020;63(12):54–63.
  DOI 10.1145/3381831.
- Strubell E, Ganesh A, McCallum A. Energy and Policy Considerations for Deep Learning in
  NLP. *ACL* 2019, pp. 3645–3650. DOI 10.18653/v1/P19-1355.
- *Radiology AI and sustainability paradox: environmental, economic, and social
  dimensions.* *Insights into Imaging* 2025. DOI 10.1186/s13244-025-01962-2.
  (versi domain-spesifik dari argumen Green AI)
- Perbandingan CNN vs ViT lintas modalitas medis, angka "22.01 accuracy points per
  million parameters" untuk EfficientNet-B0. arXiv:2507.21156. **[FLAG: preprint, belum
  peer-review. Kalau dipakai, sebut statusnya di teks; jangan jadi satu-satunya penopang
  klaim accuracy-per-parameter.]**

### Domain LIDC-IDRI (§Dataset, §Related Work)

- Armato SG III, McLennan G, Bidaut L, dkk. The Lung Image Database Consortium (LIDC) and
  Image Database Resource Initiative (IDRI). *Medical Physics* 2011;38(2):915–931.
  DOI 10.1118/1.3528204. (sudah terdaftar di "Untuk kedua paper"; dicatat ulang di sini
  supaya kelompok klaim ini lengkap, **bukan** entri baru)
- Saied M, Raafat M, Yehia S, Khalil MM. Efficient pulmonary nodules classification using
  radiomics and different artificial intelligence strategies. *Insights into Imaging*
  2023;14:91. DOI 10.1186/s13244-023-01441-6. (1.007 nodul LIDC-IDRI; AUROC terbaik
  justru dari CNN sederhana, 0.9600, di atas VGG-16 0.9539 dan VGG-19 0.9569, sementara
  akurasi terbaik 90.39% dari DenseNet-121 — instans se-domain "ringan ≥ berat".
  **Catatan:** draf awal salah atribusi ke "Naik & Edla"; penulis yang benar Saied dkk.)
- Sahu P, Yu D, Dasari M, Hou F, Qin H. A Lightweight Multi-Section CNN for Lung Nodule
  Classification and Malignancy Estimation. *IEEE JBHI* 2019;23(3):960–968.
  **[FLAG: DOI tidak diverifikasi independen di sumber. Cek di IEEE Xplore sebelum
  ditarik; kalau tidak ketemu, masukkan manual.]** (motivasi deployment mobile/portabel
  untuk backbone ringan)
- Zhu W, Liu C, Fan W, Xie X. DeepLung: Deep 3D Dual Path Nets for Automated Pulmonary
  Nodule Detection and Classification. *IEEE WACV* 2018. arXiv:1801.09555.
  **[FLAG: DOI 10.1109/WACV.2018.00079 tidak diverifikasi ulang di sumber.]**
  (FROC 84.2% lawan 83.4% dengan "hanya ¼ parameter" 3D ResNet — perhatikan metriknya
  FROC deteksi, bukan AUC klasifikasi; jangan disamakan)

### Benchmarking dan varians (§Limitations, pembelaan terhadap kritik reviewer)

- Qiu Y, Rivaz H, Xiao Y. Is Visual Explanation with Grad-CAM More Reliable for Deeper
  Neural Networks? *MLMI @ MICCAI 2023*, LNCS 14349:224–233.
  DOI 10.1007/978-3-031-45676-3_23. arXiv:2308.15172. **[FLAG: ambiguitas tahun —
  workshop 2023, volume Springer tercatat 2024. Pilih satu konvensi (mis. "MLMI 2023,
  LNCS 14349, 2024") dan pakai konsisten di seluruh manuskrip; tidak konsisten = umpan
  flag reviewer.]** (preseden se-bidang terdekat untuk klaim "arsitektur menentukan
  kualitas CAM": akurasi nyaris datar 83.2–88.2% sementara EHR AUC beda jauh,
  ResNet101 0.0319 dan VGG16 0.0243 lawan ViT 0.0145–0.0171. Bedanya: modalitas X-ray,
  metrik overlap kustom, CNN-vs-ViT alih-alih antar-keluarga CNN)
- Bouthillier X, Delaunay P, Bronzi M, dkk. Accounting for Variance in Machine Learning
  Benchmarks. *MLSys* 2021. arXiv:2103.03098. **[FLAG: tidak ada DOI Crossref bersih —
  kutip arXiv.]** (dasar untuk melaporkan varians dan menahan diri dari klaim ranking
  kuat dengan sedikit seed/model)
- "Tune It or Don't Use It" — budget tuning bisa membalik ranking arsitektur.
  arXiv:2108.13122. **[FLAG: hanya disebut sepintas di sumber; judul/penulis/venue belum
  diverifikasi. Perlakukan sebagai kata kunci pencarian, bukan entri siap tarik.]**

### Celah yang harus dicari: 2 klaim baru belum punya kandidat

Dua klaim di bawah muncul dari temuan sesi ini dan **tidak ada di brief** `Track2 v2.md`.
Didaftar seperti bagian "Masih kosong" di atas supaya kekosongannya terlihat sekarang,
bukan saat manuskrip mau dikirim. **Tidak ada DOI atau citekey yang dikarang di sini.**

| Klaim | Status | Petunjuk pencarian |
|---|---|---|
| Rekomendasi kanonik "jelaskan di layer konvolusi terakhir" | **Tertutup sumbernya** — Selvaraju dkk. Grad-CAM (ICCV 2017, DOI 10.1109/ICCV.2017.74; IJCV 2020, DOI 10.1007/s11263-019-01228-7) adalah sumber primer dan sudah ada di daftar | Yang **belum** ada: pasase spesifik tempat rekomendasi itu dinyatakan (nomor halaman/bagian, kalimatnya). Kontribusi Track 2 bertumpu persis di situ, jadi kutipannya harus ditunjuk tepat, bukan disitasi selevel makalah |
| Kegagalan senyap pemilihan target layer otomatis di tooling XAI | **Belum ada kandidat** | Sesi ini menemukan heuristik yang lazim dipakai bisa jatuh ke layer tanpa extent spasial, menghasilkan peta identik-nol yang tetap memberi nilai metrik yang kelihatan wajar. Cari: laporan implementasi Grad-CAM yang gagal tanpa error, atau bukti bahwa pilihan target layer mengubah angka lokalisasi yang sudah dipublikasi secara material. Kata kunci: *silent failure*, *degenerate/empty CAM*, *target layer sensitivity*, *reproducibility of saliency benchmarks*, isu repo `pytorch-grad-cam` / `keras-vis` |

Paper CVIU 2025 (DOI 10.1016/j.cviu.2025.104383) membahas **empty map** dan merupakan
jangkar terdekat yang sudah ada, tapi **penutup sebagian saja**: ia menjelaskan kenapa
peta bisa kosong secara mekanistik (receptive field besar, rektifikasi), bukan bahwa
pemilihan layer otomatis gagal diam-diam sambil tetap melaporkan metrik. Jangan
diperlakukan seolah menutup klaim kedua.

### Daftar DOI siap-tempel (Track 2)

Zotero → **Add Item by Identifier**, tempel seluruh blok sekaligus. Hanya DOI baru untuk
Track 2; yang sudah ada di blok Track 1 (10.1109/TIP.2021.3089943,
10.1007/s11263-019-01228-7, 10.1007/s11263-017-1059-x, 10.1109/CVPRW50498.2020.00020,
10.1118/1.3528204) **tidak diulang** — kalau blok Track 1 sudah ditarik, kelimanya sudah
ada di Zotero.

```
10.1109/CVPR.2015.7298594
10.1109/CVPR.2016.308
10.1609/aaai.v31i1.11231
10.1109/CVPR.2017.195
10.1109/CVPR.2017.243
10.1109/TPAMI.2019.2918284
10.1109/CVPR52688.2022.01167
10.1109/ICCV.2017.74
10.1007/978-3-319-46493-0_33
10.1016/j.cviu.2025.104383
10.48550/arXiv.1902.07208
10.48550/arXiv.2011.08891
10.1145/3381831
10.18653/v1/P19-1355
10.1186/s13244-025-01962-2
10.1186/s13244-023-01441-6
10.1007/978-3-031-45676-3_23
```

Ditarik terpisah (arXiv-only, DOI belum diverifikasi, atau entri manual):

```
arXiv:1810.03292    Sanity Checks (Adebayo dkk., NeurIPS 2018)  [FLAG: tak ada DOI Crossref]
arXiv:2103.03098    Accounting for Variance (Bouthillier dkk., MLSys 2021)  [FLAG: tak ada DOI Crossref]
arXiv:2507.21156    CNN vs ViT medis + accuracy-per-parameter  [FLAG: preprint, belum peer-review]
arXiv:2108.13122    "Tune It or Don't Use It"  [FLAG: judul/venue belum diverifikasi]
10.1109/WACV.2018.00079    DeepLung (Zhu dkk., WACV 2018; arXiv:1801.09555)  [FLAG: DOI belum diverifikasi ulang]
Sahu dkk. 2019, IEEE JBHI 23(3):960-968  [FLAG: DOI tidak diverifikasi; cek IEEE Xplore]
```

**Aturan yang sama tetap berlaku.** `refs.bib` auto-export Better BibTeX — jangan disunting
tangan; daftar di atas untuk ditarik lewat Zotero. **Jangan mengarang citekey**, termasuk
untuk dua celah di §"Celah yang harus dicari" yang memang belum punya sumber.
