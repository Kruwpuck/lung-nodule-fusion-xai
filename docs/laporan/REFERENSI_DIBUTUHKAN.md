# Referensi yang dibutuhkan untuk paper/track1 dan paper/track2

`paper/refs.bib` cuma punya 2 entri saat ini, dan hanya
`prabhavalkarHybridPETCTRadiomics2026` yang relevan dan sudah dipakai di
`paper/track1/main.tex`. Kalimat di bawah ini sengaja ditulis tanpa `\cite{}`
karena kuncinya belum ada di `refs.bib` — jangan mengarang citekey. Tambahkan
lewat Zotero (Better BibTeX auto-export), lalu isi `\cite{...}` di tempat yang
ditandai.

DOI di bawah dikutip dari `docs/Review Revisi 1.md`.

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
