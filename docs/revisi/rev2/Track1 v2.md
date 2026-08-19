# Related Work, Contribution Positioning, and Citation List for a Radiomics–CNN Fusion Lung Nodule Study with Explainability

## TL;DR
- Your headline result — handcrafted radiomics matching or beating radiomics–CNN fusion on ~1,400 LIDC-IDRI nodules — is NOT an anomaly: it is directly precedented by Astaraki et al. (2021), whose hybrid model (AUROC 0.938) barely exceeded conventional radiomics (0.921) and clearly beat end-to-end deep learning (0.824), and by systematic reviews showing fusion gains are small and inconsistent in small-sample imaging.
- An "explainability-first" contribution (statistically equivalent AUC + uniquely dual spatial-plus-feature-level explanations) IS publishable, but only if you replace "failure to reject DeLong" with a formal non-inferiority/equivalence test (pre-specified margin, TOST), add a faithfulness metric beyond pointing-game localization (ROAD or deletion/insertion), and ideally add a small clinician reader study.
- Recommended venue ladder: Radiology: Artificial Intelligence or Journal of Imaging Informatics in Medicine (JIIM) as primary targets; Computers in Biology and Medicine, Diagnostics, or Frontiers in Oncology as faster fallbacks; the MICCAI iMIMIC interpretability workshop for an early, XAI-focused version.

## Key Findings

**1. Precedent for radiomics matching/beating fusion.** The single strongest precedent is Astaraki et al. (2021, *Frontiers in Oncology*), on an open database of exactly 1,297 manually delineated LIDC nodules: fine-tuned conventional radiomics reached AUROC 0.921±0.010, end-to-end deep learning only 0.824±0.021, deep-feature radiomics 0.936±0.011, and the hybrid 0.938±0.010. The hybrid's advantage over radiomics alone was ~0.017 AUC — the same order of magnitude as your 0.9333 vs 0.9318 gap. Their framing: end-to-end DL "outperforms conventional radiomics out of the box" only before tuning; after tuning "the conventional and deep-feature based models achieved comparable results," and the hybrid is merely "the most promising." This is precisely the "fusion ≈ radiomics, both ≫ CNN-alone" pattern you report. Systematic-review evidence (the HNSCC review; a PET/SPECT review of 226 studies) consistently finds DL only "slightly" superior to handcrafted radiomics and that handcrafted-radiomics studies often have the highest methodological quality. The narrative-review consensus is that radiomics "is well suited to limited-cohort studies and supports transparent, feature-level interpretation," whereas DL "excels in large-scale datasets" — so neither is universally optimal.

**2. Positioning the contribution.** Arguing for a model on interpretability grounds despite non-superior accuracy has clear precedent: Rudin (2019, *Nature Machine Intelligence*) is the canonical normative argument, stating that "trying to explain black box models, rather than creating models that are interpretable in the first place, is likely to perpetuate bad practice and can potentially cause great harm to society. The way forward is to design models that are inherently interpretable." HSCNN (Shen et al. 2019) is a LIDC-specific example whose validation malignancy AUC of 0.856 barely exceeds a black-box 3D CNN (0.847) and is justified by radiologist-interpretable semantic outputs; X-Caps (LaLonde et al., MICCAI 2020) predicts radiologist-defined visual attributes at accuracy on par with black-box counterparts. To make an equivalence claim credible reviewers will demand: (a) formal equivalence/non-inferiority testing with a pre-specified margin; (b) faithfulness metrics, not just localization; (c) ideally clinician evaluation of explanations.

**3. Literature gap.** I found no study that quantitatively compares explainability ACROSS fusion arms (early/intermediate/late vs CNN-only) for lung nodules, nor one that quantifies the trade-off whereby adding a tabular modality slightly degrades spatial localization. This is a defensible, specific gap for you to claim. The closest adjacent work is the multimodal "greedy learning" / modality-imbalance literature (Wu et al. ICML 2022; Peng et al. OGM-GE CVPR 2022), which shows joint training under-optimizes one modality — a plausible mechanism for your observation that fusion's pointing accuracy dips slightly below CNN-only — but that literature does not measure explanation quality.

**4. Anticipated criticism.** The most likely reviewer attack is that "failure to reject DeLong ≠ equivalence." This is correct and well-documented: Lakens (2017) states that "researchers often incorrectly conclude an effect is absent based [on] a nonsignificant result" and that the recommended remedy is to "test for equivalence." You must run TOST/non-inferiority (Liu et al. 2006 give the paired-AUC method). Other attacks — Occam's razor, localization≠faithfulness, inherited CNN maps, small single-dataset sample — each have standard defenses detailed below.

## Details

### 1. Precedent for radiomics ≈ or > fusion, and the size of the fusion gain

**Direct LIDC/lung-CT precedents (report these AUCs):**

- **Astaraki et al. (2021)** — the key citation. On an open LIDC-derived set of exactly 1,297 nodules: best conventional radiomics AUROC **0.921±0.010**; best end-to-end DL **0.824±0.021**; deep-feature radiomics **0.936±0.011**; hybrid **0.938±0.010**. Conclusion: comparable radiomics vs deep-feature results after tuning; hybrid only marginally best. This mirrors your finding almost exactly and is your strongest "not an anomaly" anchor.
- **Contrasting (fusion-helps) studies** to cite for balance: Du et al. (2025, *Medical Physics*) report a Discriminant-Correlation-Analysis fusion "outperforms the single-feature model in all classification tasks"; Zheng et al. (2025, *Biomedical Signal Processing and Control*) report radiomics alone 97.0% vs deep features 90.3%, fusion 97.6% (note radiomics already beats DL here, and the fusion gain is only +0.6 pt); the three-way attention fusion I-VISTA study (2025) reports fusion AUC ~0.93 beating standalone arms; and an RGD graph-fusion study reports AUC 0.9629 on LIDC. Presenting these honestly shows the field is split and that small/negative fusion gains are common.

**Small-sample "does DL beat radiomics?" evidence:** The HNSCC systematic review (23 studies) found DL only "slightly superior" to handcrafted radiomics, with the highest methodological quality among handcrafted-radiomics studies. The PET/SPECT review (226 studies, 2020–2025) found the comparative performance of handcrafted radiomics, deep radiomics, DL and fusion "remains inconsistent across clinical applications." A hemorrhagic-stroke CT study found radiomics+deep-feature fusion gave numerically higher AUCs than radiomics alone but the gain was statistically significant in only one of several endpoints — a close methodological analogue to your "fusion never significantly beats radiomics."

**Interpretation for your paper:** The literature supports the claim that in the ~1,000–2,000-sample regime, handcrafted radiomics is a strong baseline that fusion frequently fails to beat by a statistically meaningful margin, and that end-to-end CNNs are the weakest arm without heavy tuning — exactly your ordering.

### 2. Positioning: is "equivalent AUC + unique dual explainability" a publishable contribution?

Yes, conditionally. Precedents for interpretability-first arguments:
- **Rudin (2019, Nat Mach Intell)** — argues interpretable models should be preferred for high-stakes decisions and challenges the assumption that black boxes are more accurate ("The way forward is to design models that are inherently interpretable"). Use this as your normative backbone.
- **HSCNN (Shen et al. 2019)** — LIDC validation malignancy AUC 0.856 vs black-box 3D CNN 0.847 (essentially equal; the paper notes the difference is significant by a paired t-test but the absolute gap is <0.01), justified by radiologist-interpretable semantic outputs. Near-perfect analogue to your argument.
- **X-Caps (LaLonde et al., MICCAI 2020)** — capsule model encoding visual attributes for explainable lung-nodule diagnosis at accuracy comparable to black-box models.

What reviewers demand for an equivalence-plus-explainability claim to be credible:
- **Formal equivalence / non-inferiority testing** with a pre-specified margin (see §4a).
- **Faithfulness metrics beyond localization** (ROAD; deletion/insertion curves) — because pointing accuracy measures where a map lands, not whether the map reflects the model's actual computation.
- **Clinician evaluation** of the explanations (even a small reader study) strengthens acceptance, per CLAIM and multimodal-XAI guidance.
- **Reporting-standard adherence** (CLAIM, TRIPOD+AI) signals rigor and is increasingly expected by radiology-AI venues.

### 3. The literature gap you can claim

- **No cross-arm explainability comparison for lung nodules.** I found no LIDC study that quantitatively compares explanation quality across fusion strategies (early/intermediate/late) against a CNN-only baseline. Your Layer-CAM pointing-accuracy comparison across arms, plus the observation that radiomics-only is structurally incapable of spatial maps while fusion uniquely provides both spatial and SHAP feature-level explanations, is a genuinely novel framing.
- **No quantification of the "tabular modality degrades localization" trade-off.** The balanced-multimodal-learning literature (Wu et al. ICML 2022; Peng et al. OGM-GE CVPR 2022) explains WHY a jointly trained modality can be under-optimized, but does not measure explanation/localization quality as the degraded quantity. You can position your <0.05 pointing-accuracy drop as the first quantification of an explainability cost of fusion. State explicitly that the evidence base here is thin — this strengthens, not weakens, a gap claim.

### 4. Anticipated criticism and published defenses

**(a) "Non-significant DeLong ≠ equivalence."** Correct, and the single biggest risk. Absence of evidence is not evidence of absence. Lakens (2017) frames the fix precisely: "researchers often incorrectly conclude an effect is absent based [on] a nonsignificant result," and the remedy is the two one-sided tests (TOST) procedure, in which "when both these one-sided tests can be statistically rejected, we can conclude that … the observed effect falls within the equivalence bounds and is close enough to zero to be practically equivalent." Defense: run TOST — or a non-inferiority test on paired AUCs (Liu, Ma, Wu & Tai 2006, *Statistics in Medicine*, give the exact standardized-difference method for paired ROC areas). Pre-specify a margin (e.g., ΔAUC = 0.02 or 0.03, justified clinically or by literature). Report the confidence interval on the AUC difference and show it lies within ±margin. This converts "we failed to find a difference" into "we demonstrated equivalence within margin δ."

**(b) Occam's razor — if fusion doesn't beat radiomics, why add CNN complexity?** Defense: reframe the deliverable as capability, not accuracy. Fusion is the only arm delivering BOTH spatial (Layer-CAM) and feature-level (SHAP) explanations simultaneously; radiomics-only cannot produce spatial maps, CNN-only cannot produce tabular feature attributions. Cite Rudin (2019) and the multimodal-XAI orchestration literature (npj Digit Med 2024) that value complementary explanation modalities. Also note robustness/missing-modality benefits from the fusion surveys (Huang et al. 2020; Baltrušaitis et al. 2019).

**(c) Pointing accuracy measures localization, not faithfulness.** Defense: add faithfulness metrics — ROAD (Rong et al. ICML 2022) and deletion/insertion curves — and consider HiResCAM (Draelos & Carin), which is provably more faithful than Grad-CAM for architectures without post-conv pooling. Acknowledge explicitly that localization and faithfulness are distinct axes; report both.

**(d) Fusion's spatial maps are "inherited" from the CNN branch and add nothing.** This is partially true and you should concede it: the spatial map originates in the CNN feature extractor. But your defensible claim is about the *combined* explanation object — spatial map + SHAP feature attribution + calibrated fused prediction — which neither unimodal arm provides. Frame the contribution as the joint explanation, and empirically show whether fusion's maps differ from CNN-only's maps (your <0.05 pointing drop already quantifies this: they are near-identical, which you should report honestly rather than overclaim novelty in the spatial channel).

**(e) Small sample / single dataset / no external validation.** The universal LIDC criticism. Defenses: (i) cite Baltatzis et al. (2021) on how LIDC sample-selection choices (truthing, thresholds) swing reported performance, and present your frozen patient-level stratified split and median-of-4 labeling as best-practice mitigations; (ii) cite Varoquaux & Cheplygina (2022) on small-sample optimism and evaluation pitfalls, and your nested-CV design as a guard against validation-overfitting/leakage; (iii) explicitly flag no external validation as a limitation and propose LUNA16 or an institutional cohort as future work. Reviewers accept single-dataset LIDC papers when the methodology is rigorous and limitations are stated.

### 5. Required citation list with DOIs (organized by section)

**Dataset**
- Armato SG III, McLennan G, Bidaut L, et al. The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A completed reference database of lung nodules on CT scans. *Medical Physics* 2011;38(2):915–931. DOI: 10.1118/1.3528204

**XAI methods and metrics**
- Selvaraju RR, Cogswell M, Das A, et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV* 2017 (extended in *IJCV* 2020;128:336–359). DOI: 10.1007/s11263-019-01228-7 (arXiv:1610.02391)
- Jiang P-T, Zhang C-B, Hou Q, Cheng M-M, Wei Y. LayerCAM: Exploring Hierarchical Class Activation Maps for Localization. *IEEE Transactions on Image Processing* 2021;30:5875–5888. DOI: 10.1109/TIP.2021.3089943
- Draelos RL, Carin L. Use HiResCAM instead of Grad-CAM for faithful explanations of convolutional neural networks. arXiv:2011.08891 (2020/2021). [FLAG: preprint; no journal DOI — cite arXiv ID; verify the arXiv number against the PDF before submission.]
- Chattopadhay A, Sarkar A, Howlader P, Balasubramanian VN. Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks. *WACV* 2018:839–847. DOI: 10.1109/WACV.2018.00097
- Zhang J, Bargal SA, Lin Z, Brandt J, Shen X, Sclaroff S. Top-Down Neural Attention by Excitation Backprop (introduces the Pointing Game). *International Journal of Computer Vision* 2018;126(10):1084–1102. DOI: 10.1007/s11263-017-1059-x (earlier ECCV 2016 version: DOI 10.1007/978-3-319-46493-0_33)
- Wang H, Wang Z, Du M, et al. Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks (uses the energy-based pointing game). *CVPR Workshops* 2020:111–119. DOI: 10.1109/CVPRW50498.2020.00020
- Lundberg SM, Lee S-I. A Unified Approach to Interpreting Model Predictions (SHAP). *Advances in Neural Information Processing Systems* 2017;30:4765–4777. [No DOI; NeurIPS proceedings. arXiv:1705.07874]
- Rong Y, Leemann T, Borisov V, Kasneci G, Kasneci E. A Consistent and Efficient Evaluation Strategy for Attribution Methods (ROAD). *ICML* 2022, PMLR 162:18770–18795. [Proceedings; no DOI. arXiv:2202.00449]

**Radiomics**
- van Griethuysen JJM, Fedorov A, Parmar C, et al. Computational Radiomics System to Decode the Radiographic Phenotype (PyRadiomics). *Cancer Research* 2017;77(21):e104–e107. DOI: 10.1158/0008-5472.CAN-17-0339
- Zwanenburg A, Vallières M, Abdalah MA, et al. The Image Biomarker Standardization Initiative: Standardized Quantitative Radiomics for High-Throughput Image-based Phenotyping. *Radiology* 2020;295(2):328–338. DOI: 10.1148/radiol.2020191145 (foundational preprint: Zwanenburg A, Leger S, Vallières M, Löck S. Image biomarker standardisation initiative. arXiv:1612.07003)
- Lambin P, Leijenaar RTH, Deist TM, et al. Radiomics: the bridge between medical imaging and personalized medicine (introduces the Radiomics Quality Score). *Nature Reviews Clinical Oncology* 2017;14(12):749–762. DOI: 10.1038/nrclinonc.2017.141

**LIDC baselines**
- Al-Shabi M, Shak K, Tan M. ProCAN: Progressive growing channel attentive non-local network for lung nodule classification. *Pattern Recognition* 2022;122:108309. DOI: 10.1016/j.patcog.2021.108309 (reported AUC 98.05%, accuracy 95.28% on LIDC-IDRI)
- Causey JL, Zhang J, Ma S, et al. Highly accurate model for prediction of lung nodule malignancy with CT scans (NoduleX). *Scientific Reports* 2018;8:9286. DOI: 10.1038/s41598-018-27569-w (reported AUC ~0.99)
- Astaraki M, Yang G, Zakko Y, Toma-Dasu I, Smedby Ö, Wang C. A Comparative Study of Radiomics and Deep-Learning Based Methods for Pulmonary Nodule Malignancy Prediction in Low Dose CT Images. *Frontiers in Oncology* 2021;11:737368. DOI: 10.3389/fonc.2021.737368 (radiomics 0.921, DL 0.824, deep-feature 0.936, hybrid 0.938)
- Shen W, Zhou M, Yang F, et al. Multi-crop Convolutional Neural Networks for lung nodule malignancy suspiciousness classification. *Pattern Recognition* 2017;61:663–673. DOI: 10.1016/j.patcog.2016.05.029 [FLAG: a variant DOI 10.1016/j.patcog.2016.07.030 circulates in secondary sources; verify against the ScienceDirect page.]
- Al-Shabi M, Lan BL, Chan WY, Ng K-H, Tan M. Lung nodule classification using deep Local-Global networks. *International Journal of Computer Assisted Radiology and Surgery* 2019;14(10):1815–1819. DOI: 10.1007/s11548-019-01981-7 (reported AUC 95.62%)
- Al-Shabi M, Lee HK, Tan M. Gated-Dilated Networks for Lung Nodule Classification in CT Scans. *IEEE Access* 2019;7:178827–178838. DOI: 10.1109/ACCESS.2019.2958663 (reported AUC >0.95)
- Shen S, Han SX, Aberle DR, Bui AA, Hsu W. An Interpretable Deep Hierarchical Semantic Convolutional Neural Network for Lung Nodule Malignancy Classification (HSCNN). *Expert Systems with Applications* 2019;128:84–95. DOI: 10.1016/j.eswa.2019.01.048 (malignancy AUC 0.856 vs 0.847 black-box baseline)

**Methodology**
- DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics* 1988;44(3):837–845. DOI: 10.2307/2531595
- Baltatzis V, Bintsi K-M, Le Folgoc L, et al. The Pitfalls of Sample Selection: A Case Study on Lung Nodule Classification. *PRIME @ MICCAI 2021*, LNCS 12928:201–211. DOI: 10.1007/978-3-030-87602-9_19
- Lakens D. Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses. *Social Psychological and Personality Science* 2017;8(4):355–362. DOI: 10.1177/1948550617697177
- Lakens D, Scheel AM, Isager PM. Equivalence Testing for Psychological Research: A Tutorial. *Advances in Methods and Practices in Psychological Science* 2018;1(2):259–269. DOI: 10.1177/2515245918770963
- Liu J-P, Ma M-C, Wu C-Y, Tai J-Y. Tests of equivalence and non-inferiority for diagnostic accuracy based on the paired areas under ROC curves. *Statistics in Medicine* 2006;25(7):1219–1238. DOI: 10.1002/sim.2358
- Demšar J. Statistical Comparisons of Classifiers over Multiple Data Sets. *Journal of Machine Learning Research* 2006;7:1–30. [No DOI; JMLR open access. FLAG: verify volume/pages.]

**Multimodal fusion**
- Arevalo J, Solorio T, Montes-y-Gómez M, González FA. Gated Multimodal Units for Information Fusion. arXiv:1702.01992 (ICLR 2017 Workshop). [FLAG: workshop/preprint; no DOI.]
- Baltrušaitis T, Ahuja C, Morency L-P. Multimodal Machine Learning: A Survey and Taxonomy. *IEEE Transactions on Pattern Analysis and Machine Intelligence* 2019;41(2):423–443. DOI: 10.1109/TPAMI.2018.2798607
- Wu N, Jastrzębski S, Cho K, Geras KJ. Characterizing and Overcoming the Greedy Nature of Learning in Multi-modal Deep Neural Networks. *ICML* 2022, PMLR 162:24043–24055. [Proceedings; no DOI.]
- Peng X, Wei Y, Deng A, Wang D, Hu D. Balanced Multimodal Learning via On-the-fly Gradient Modulation (OGM-GE). *CVPR* 2022:8238–8247. DOI: 10.1109/CVPR52688.2022.00806
- Huang S-C, Pareek A, Seyyedi S, Banerjee I, Lungren MP. Fusion of medical imaging and electronic health records using deep learning: a systematic review and implementation guidelines. *npj Digital Medicine* 2020;3:136. DOI: 10.1038/s41746-020-00341-z

**Reporting guidelines**
- Mongan J, Moy L, Kahn CE Jr. Checklist for Artificial Intelligence in Medical Imaging (CLAIM): A Guide for Authors and Reviewers. *Radiology: Artificial Intelligence* 2020;2(2):e200029. DOI: 10.1148/ryai.2020200029 (2024 update: Tejani AS, Klontzas ME, Gatti AA, et al. *Radiol Artif Intell* 2024;6(4):e240300. DOI: 10.1148/ryai.240300)
- Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 2024;385:e078378. DOI: 10.1136/bmj-2023-078378
- Varoquaux G, Cheplygina V. Machine learning for medical imaging: methodological failures and recommendations for the future. *npj Digital Medicine* 2022;5:48. DOI: 10.1038/s41746-022-00592-y

**Supporting (explainability-first argument)**
- Rudin C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence* 2019;1(5):206–215. DOI: 10.1038/s42256-019-0048-x
- LaLonde R, Torigian D, Bagci U. Encoding Visual Attributes in Capsules for Explainable Medical Diagnoses (X-Caps). *MICCAI* 2020, LNCS 12261:294–304. DOI: 10.1007/978-3-030-59710-8_29 [FLAG: verify page numbers.]

### 6. Venue recommendation

- **Radiology: Artificial Intelligence** (RSNA). Scope: AI in medical imaging with clinical grounding; strongly enforces CLAIM. Selective. Welcomes rigorous non-superiority/methodological work if clinically framed. Best "prestige-with-fit" target. Time to first decision typically a few months.
- **Journal of Imaging Informatics in Medicine (JIIM)** (formerly *Journal of Digital Imaging*, Springer). Scope: imaging informatics, ML methods, XAI. Receptive to comparative and explainability studies (it published the 226-study PET/SPECT fusion review). Good fit, moderate selectivity, reasonable turnaround.
- **Computers in Biology and Medicine** (Elsevier). Scope: computational methods in medicine. Welcomes rigorous benchmarking/fusion papers. Faster than the top imaging journals. Solid mid-tier target.
- **Diagnostics** or **Cancers** (MDPI). Scope: diagnostic methods / oncology. Fast (weeks), open access (APC), receptive to negative/non-superiority and XAI results if methodology is sound. Good if speed matters.
- **Frontiers in Oncology** (Cancer Imaging section). Published Astaraki et al. — an ideal topical match for a radiomics-vs-DL comparison. Open access, relatively fast.
- **Scientific Reports** (Nature Portfolio). Broad, judges on soundness not novelty; explicitly accepts non-superiority results. Good general-purpose home.
- **MICCAI iMIMIC workshop** (interpretability of medical imaging) and **SPIE Medical Imaging**. Best venues for an early, XAI-forward conference version emphasizing the cross-arm explanation comparison.
- **IEEE Transactions on Medical Imaging** and **Medical Image Analysis**: highest prestige but demand strong methodological novelty and usually external validation/SOTA; an equivalent-not-SOTA result would likely need a stronger methodological contribution (e.g., a novel fusion-explainability framework with faithfulness theory) to clear the bar. Consider only after adding equivalence testing + faithfulness + external validation.

## Recommendations
1. **Immediately re-run your primary comparison as a formal non-inferiority/equivalence test** (TOST on paired AUCs, Liu et al. 2006; principle from Lakens 2017) with a pre-registered margin (start at ΔAUC=0.03; tighten to 0.02 if the CI supports it). Report the AUC-difference CI. This is the single highest-leverage change; without it the central claim is rejectable on sight.
2. **Add at least one faithfulness metric** (ROAD; deletion/insertion) alongside your Layer-CAM pointing accuracy, and add HiResCAM as a faithfulness-oriented comparator. Explicitly separate "localization" from "faithfulness" in your claims.
3. **Concede the inheritance point (4d) honestly**: show quantitatively that fusion and CNN-only spatial maps are near-identical (your <0.05 gap), and pivot the novelty to the *joint* spatial+SHAP explanation object that only fusion provides.
4. **Frame the gap (§3) as your primary novelty**: the first cross-arm (early/intermediate/late/CNN-only) quantitative explainability comparison for LIDC nodules, plus the first quantification of fusion's small localization cost.
5. **Adhere to CLAIM and TRIPOD+AI** and cite Baltatzis (sample selection) and Varoquaux & Cheplygina (small-sample pitfalls) proactively in your Limitations to preempt reviewer 5(e).
6. **Venue staging:** submit an XAI-focused short version to MICCAI iMIMIC or SPIE; target the full paper at Radiology: AI or JIIM; fall back to Computers in Biology and Medicine, Frontiers in Oncology, or Diagnostics for speed. **Benchmark that changes this:** if you add external validation (LUNA16 or institutional) AND a novel methodological fusion-explainability contribution, elevate to IEEE TMI / Medical Image Analysis.

## Caveats
- The field is genuinely split: several LIDC studies report fusion clearly beating single modalities (Du 2025; I-VISTA 2025; RGD AUC 0.9629). Present these fairly; your contribution is that fusion gains are small and often non-significant, not that fusion never helps.
- Direct AUC comparisons across LIDC papers are unreliable because of differing sample-selection and label-truthing choices (Baltatzis 2021). Do not claim SOTA against numbers like NoduleX's 0.99 or ProCAN's 0.98 — those use different subsets and splits.
- Several key methods (SHAP, ROAD, Wu et al., Arevalo GMU, HiResCAM, Demšar) are conference-proceedings or preprints without journal DOIs; cite them by proceedings/arXiv ID.
- Items flagged [FLAG] (HiResCAM arXiv number, X-Caps page numbers, Demšar volume/pages, Multi-crop CNN DOI variant .05.029 vs .07.030) should be verified against the publisher page before submission. The Grad-CAM++, pointing-game (Zhang et al.), Score-CAM, CLAIM, TRIPOD+AI, HSCNN, and Local-Global/Gated-Dilated citations were verified through a dedicated research subagent rather than my own live searches; treat the [FLAG]ged subset as needing a final manual check.
- No external validation exists in your study; this is the most defensible reviewer criticism and should be stated as a limitation with a concrete future-work plan.