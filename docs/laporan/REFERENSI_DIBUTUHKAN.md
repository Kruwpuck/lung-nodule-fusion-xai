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
