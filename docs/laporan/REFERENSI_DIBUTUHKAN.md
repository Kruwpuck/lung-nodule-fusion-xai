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
