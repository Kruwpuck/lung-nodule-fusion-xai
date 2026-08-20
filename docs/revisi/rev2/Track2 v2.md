# Literature Positioning and Citations for a Model-Efficiency + Explainability Analysis of Lung-Nodule Malignancy Classification (LIDC-IDRI)

## TL;DR
- Both of the researcher's headline observations are well-supported by peer-reviewed precedent: (a) "capacity does not buy accuracy" on small medical datasets is the central finding of Raghu et al.'s *Transfusion* (NeurIPS 2019); and (b) "architecture, not size, governs CAM localization quality" is *partially* precedented (Qiu, Rivaz & Xiao, MLMI 2023 show CNN-family vs ViT and depth effects on Grad-CAM overlap), but a clean, quantitative pointing-game/IoU comparison across this many CNN families on lung CT with Layer-CAM appears to be a genuine gap the researcher can claim.
- An efficiency analysis (params/FLOPs/latency, accuracy-per-parameter, Pareto plots) is an accepted secondary contribution, and the Green-AI / deployment framing (Schwartz et al. 2020; Strubell et al. 2019; radiology sustainability literature) is a legitimate justification for favoring lightweight models. The *joint* efficiency-and-explainability-quality analysis is under-explored — this is the strongest novelty claim.
- The most dangerous reviewer attack is the GoogLeNet 0.0000 pointing accuracy, which is an extraordinary result that must be affirmatively ruled out as a target-layer/implementation artifact before it can be presented as an architectural property; the mechanistic CAM literature (HiResCAM faithfulness; Grad-CAM large-receptive-field caveats) both explains and threatens the family claim.

## Key Findings

**Q1 — "Capacity doesn't buy accuracy" is strongly precedented.** The canonical citation is Raghu, Zhang, Kleinberg & Bengio, *Transfusion: Understanding Transfer Learning for Medical Imaging* (NeurIPS 2019; arXiv:1902.07208; DOI 10.48550/arXiv.1902.07208). It shows "simple, lightweight models can perform comparably to ImageNet architectures," attributes part of the transfer benefit to "over-parametrization of standard models rather than sophisticated feature reuse," and isolates useful feature reuse to the lowest layers. This directly supports observation (a): an 8.5× parameter range yielding only a 0.0144 AUC spread, and a 28.6M model (ConvNeXt-Tiny) beating a 55.9M one (InceptionResNetV2).

**Q2 — "Architecture determines CAM quality" is partially precedented, with a real gap.** Qiu, Rivaz & Xiao (MLMI 2023, LNCS vol. 14349, pp. 224–233, DOI 10.1007/978-3-031-45676-3_23; arXiv:2308.15172) find, on pneumothorax chest X-rays, that "the effectiveness of GradCAM also varies among different network architectures" and that "deeper neural networks do not necessarily contribute to a strong improvement of pneumothorax diagnosis accuracy." This is the closest precedent to observation (b), but it uses custom overlap metrics (Diff_GradCAM and Effective-Heat-Ratio AUC, not pointing game/IoU/Dice), compares CNN-vs-ViT and depth rather than several CNN families, and does not use Layer-CAM or lung nodules.

**Q3 — Efficiency analysis is an accepted secondary contribution.** Standard metrics are parameter count, FLOPs/MACs, inference latency, memory, and derived measures like accuracy-per-million-parameters and Pareto (accuracy vs cost) plots.

**Q4 — Green-AI / deployment arguments are established.** Schwartz et al., *Green AI* (CACM 2020, DOI 10.1145/3381831) and Strubell et al. (ACL 2019, DOI 10.18653/v1/P19-1355) anchor the efficiency/carbon/inclusivity argument; radiology-specific sustainability reviews extend it to medical imaging.

**Q5 — The joint efficiency + explainability-quality analysis is a genuine gap.** No located peer-reviewed study jointly asks "can lightweight models be explained as well as heavyweight ones?" for lung nodules; adjacent work covers efficiency OR explainability, rarely both as co-equal axes.

**Q6 — Anticipated criticisms are real and each has a literature-backed defense**, but the GoogLeNet 0.0000 result is the one that must be pre-empted.

## Details

### Q1. Precedent for "capacity doesn't buy accuracy"

The theoretical framing the researcher should invoke:
- **Over-parameterization / transfer saturation:** *Transfusion* (Raghu et al., NeurIPS 2019, DOI 10.48550/arXiv.1902.07208) is the primary source. Its two large-scale tasks (retinal fundus, chest X-ray) show that ImageNet-scale architectures confer little benefit over small custom CNNs, and that meaningful feature reuse is concentrated in the lowest layers. This is the single most citable support for observation (a).
- **Sample-complexity / small-data argument (same domain):** Saied, Raafat, Yehia & Khalil, *Efficient pulmonary nodules classification using radiomics and different artificial intelligence strategies* (Insights into Imaging 2023, 14:91, DOI 10.1186/s13244-023-01441-6), on 1,007 LIDC-IDRI nodules from 551 patients, report verbatim that "the best accuracy reached 90.39% with DenseNet-121 model and the best AUROC was 96.0%, 95.39% and 95.69% with simple CNN, VGG-16 and VGG-19, respectively" — i.e., a *simple CNN* achieved the top AUROC over the deeper transfer-learned backbones. This is a same-domain, small-data instance of lightweight ≈ or > heavyweight. (Note: earlier drafts mis-attributed this paper to "Naik & Edla"; the correct authors are Saied et al.)
- **Lightweight CNNs on LIDC-IDRI:** Sahu et al., *A Lightweight Multi-Section CNN for Lung Nodule Classification and Malignancy Estimation* (IEEE J. Biomedical and Health Informatics 23(3):960–968, 2019) explicitly targets a compact, mobile-deployable model on LIDC-IDRI. (Exact DOI not independently verified — flag.)
- **A cleaner "lighter beats heavier" datapoint on lung CT:** Zhu et al., *DeepLung* (IEEE WACV 2018; arXiv:1801.09555) report a 3D dual-path net achieving higher FROC (84.2% vs 83.4%) with "only ¼ of the parameters" of a 3D ResNet on LIDC-IDRI — though the metric is detection FROC, not classification AUC. (WACV DOI 10.1109/WACV.2018.00079 — flag, not independently re-verified.)
- **Efficient-family evidence outside lung CT:** A comparative study of CNNs vs ViTs across medical modalities reports EfficientNet-B0 achieving "22.01 accuracy points per million parameters" — a vivid accuracy-per-parameter statistic, but it is a preprint (arXiv:2507.21156), so cite with the non-peer-reviewed caveat.

**Honest limitation:** I did not find a single peer-reviewed paper that reports, in one table, LIDC-IDRI nodule *classification* AUC AND parameter counts showing a lightweight model beating a heavyweight one. The evidence is assembled from several partial sources. State this explicitly rather than implying a perfect precedent exists.

### Q2. Precedent for "architecture family determines CAM quality," and the mechanistic why

**Quantitative cross-architecture CAM comparisons exist but are thin in medical imaging:**
- Qiu, Rivaz & Xiao (MLMI 2023, LNCS 14349, pp. 224–233, DOI 10.1007/978-3-031-45676-3_23; Springer lists the volume publication year as 2024) is the best same-field precedent. On SIIM-ACR pneumothorax X-rays (7,200 cases), across VGG16/19, ResNet18/34/50/101, and ViT-small/base/large, they measure two mask-overlap metrics (Diff_GradCAM; Effective Heat Ratio AUC). CNNs systematically produced better-localized maps than ViTs (e.g., EHR AUC: ResNet101 0.0319, VGG16 0.0243, vs ViTs 0.0145–0.0171), while diagnostic accuracy was nearly flat (83.2–88.2%). Their conclusion — accuracy and Grad-CAM quality are "not necessarily in synch" — is a direct analogue of observation (b).
- The pointing-game metric itself originates with Zhang et al., *Top-Down Neural Attention by Excitation Backprop* (ECCV 2016, DOI 10.1007/978-3-319-46493-0_33; extended IJCV 2018 version DOI 10.1007/s11263-017-1059-x).
- General-vision cross-method CAM benchmarks (FD-CAM, Integrated Grad-CAM, the Weighting Game) report pointing-game and energy-based pointing-game numbers but typically hold architecture fixed (VGG16/ResNet-50), so they support the metric methodology, not the family claim.

**Mechanistic explanations the researcher should marshal:**
- **Grad-CAM's spatial-correspondence assumption breaks with large effective receptive fields / output stride.** The CVIU 2025 paper *Grad-CAM: The impact of large receptive fields and other caveats* (DOI 10.1016/j.cviu.2025.104383) shows, on VinDr-CXR and ImageNet/MNIST-derived data, that "for models with large receptive fields, the feature spatial organization is not kept during the forward pass, which may render the explanations devoid of meaning," and discusses empty maps, rectification and GAP-vs-flatten effects. This is the key citation for why aggressive Inception-stem downsampling and multi-scale concatenation could degrade localization.
- **Feature-map resolution at the CAM target layer.** The original Grad-CAM paper (Selvaraju et al., IJCV 2020, DOI 10.1007/s11263-019-01228-7; ICCV 2017, DOI 10.1109/ICCV.2017.74) documents that "localization becomes progressively worse as we move to earlier convolutional layers" and that coarse final-layer maps limit pixel-accurate localization — motivating Layer-CAM.
- **Layer-CAM rationale.** Jiang et al., *LayerCAM* (IEEE TIP 2021, 30:5875–5888, DOI 10.1109/TIP.2021.3089943) is explicitly designed to recover fine localization from higher-resolution intermediate layers, which is why the researcher computes it at the 8×8 stage.
- **Dense connectivity / feature reuse (DenseNet).** Huang et al. (CVPR 2017, DOI 10.1109/CVPR.2017.243; journal version *Convolutional Networks with Dense Connectivity*, IEEE TPAMI, DOI 10.1109/TPAMI.2019.2918284) — dense skip connections preserve and propagate features across layers, a plausible mechanism for cleaner gradient flow to the CAM layer.
- **ConvNeXt design.** Liu et al., *A ConvNet for the 2020s* (CVPR 2022, DOI 10.1109/CVPR52688.2022.01167) — patchify stem, large depthwise kernels, LayerNorm.
- **HiResCAM faithfulness theorem.** Draelos & Carin (arXiv:2011.08891; DOI 10.48550/arXiv.2011.08891) prove Grad-CAM's gradient-averaging can "highlight locations the model did not actually use," and that gradient-based CAMs are provably faithful only for architectures ending in GAP-then-single-FC (the "CAM architecture," e.g., ResNet/DenseNet). Architectures deviating from this head (or with intermediate pooling / >1 FC layer) lose the faithfulness guarantee — a principled reason why Inception-family heads may behave differently.
- **Sanity checks.** Adebayo et al., *Sanity Checks for Saliency Maps* (NeurIPS 2018; arXiv:1810.03292) established that Grad-CAM passes model-parameter-randomization and data-randomization tests while Guided Grad-CAM fails — the researcher should run these on GoogLeNet in particular (see Q6). (No Crossref DOI exists for NeurIPS 2018 — cite arXiv or the proceedings page — flag.)
- **Score-CAM** (Wang et al., CVPR Workshops 2020, DOI 10.1109/CVPRW50498.2020.00020) — a gradient-free alternative useful as a cross-check on any architecture where gradient-based CAMs are suspect.
- **Known Grad-CAM-on-Inception pathologies.** Beyond the peer-reviewed caveats above, there is well-documented practitioner evidence (e.g., long-standing keras-vis GitHub issues reporting Grad-CAM behaving poorly on InceptionV3 relative to VGG) that corroborates the direction of observation (b). Present as informal support only, not citable evidence.

### Q3. Efficiency as a legitimate secondary contribution

Conventions to adopt:
- **Report:** parameter count, FLOPs/MACs, peak memory, and inference latency (with batch size and hardware stated).
- **Derived metrics:** accuracy-per-million-parameters and Pareto-frontier plots of accuracy vs computational cost are standard and appear across medical-imaging and vision efficiency papers.
- **Measurement tooling:** FLOPs/MACs are typically counted with libraries such as ptflops, fvcore, or thop; latency should be wall-clock, warm-started, averaged over many runs.
- **Standard caveats:** FLOPs is a poor proxy for wall-clock latency (memory-bandwidth-bound ops, kernel-launch overhead, depthwise-conv inefficiency), and latency is batch-size- and hardware-dependent — Xception/InceptionResNetV2, despite differing FLOPs, may not rank as expected on a given GPU. State the measurement environment explicitly.

### Q4. Green-AI and deployment arguments

- **Foundational:** Schwartz et al., *Green AI* (CACM 2020, 63(12):54–63, DOI 10.1145/3381831) — efficiency reduces carbon footprint and increases inclusivity ("deep learning study should not require the deepest pockets"); Strubell et al., *Energy and Policy Considerations for Deep Learning in NLP* (ACL 2019, pp. 3645–3650, DOI 10.18653/v1/P19-1355) — quantify training energy/carbon.
- **Radiology-specific:** *Radiology AI and sustainability paradox: environmental, economic, and social dimensions* (Insights into Imaging 2025, DOI 10.1186/s13244-025-01962-2) covers environmental/economic/social costs and mitigation (pruning, quantization, transfer/federated learning); a sustainable-histopathology benchmarking study (Scientific Reports / PMC11668825) proposes an "environmentally sustainable performance" (ESPer) metric integrating diagnostic performance and CO₂eq during training and inference.
- **Scale of the problem:** Medical imaging is estimated to contribute approximately 1% of global greenhouse-gas emissions (Picano et al., J. Clin. Med. 2022, reasoning from ~10 billion medical examinations per year worldwide), while the health-care sector as a whole accounts for roughly 4.6% — a figure the researcher can cite to motivate model-efficiency in imaging AI specifically.
- **Deployment framing:** point-of-care / mobile screening, edge/resource-constrained hardware, hospital IT constraints, and latency requirements all motivate lightweight backbones; several LIDC-IDRI lightweight-CNN papers (e.g., Sahu et al. 2019) cite mobile/portable deployment as the explicit motivation.

### Q5. The gap — joint efficiency AND explainability quality

Framing to claim: efficiency and explainability are each individually mature literatures, but the *conjunction* — whether small models localize as faithfully as large ones — is rarely studied, and not (in my located sources) for lung-nodule CT with Layer-CAM across multiple CNN families. Qiu et al. (2023) touch the accuracy-vs-CAM-quality relationship but not efficiency; the ESPer histopathology work touches accuracy-vs-carbon but not explainability. The researcher can honestly claim: "To our knowledge, no prior work jointly evaluates parameter efficiency and CAM localization quality across CNN families for lung-nodule malignancy classification." This is a defensible, appropriately hedged novelty statement rather than an absolute one.

### Q6. Anticipated criticisms and how to defend

- **(a) n=7 architectures is too small for a statistical claim about "families."** Defense: frame it as a controlled observational / hypothesis-generating study, not a powered statistical test; report per-fold variance from the nested CV; avoid the word "prove." The benchmarking-variance literature (Bouthillier et al., *Accounting for Variance in Machine Learning Benchmarks*, MLSys 2021, arXiv:2103.03098 — no clean Crossref DOI, flag) supports reporting variance and refraining from strong ranking claims with few seeds/models.
- **(b) Confounding — "family" is not a clean variable** (stem, depth, normalization, activation, pretraining recipe all co-vary). Defense: acknowledge explicitly; anchor the interpretation to a *specific mechanism* (effective receptive field / output stride at the CAM layer per the CVIU 2025 paper; CAM-architecture faithfulness per HiResCAM) rather than the vague label "family." The GoogLeNet-vs-DenseNet121 near-equal-size / opposite-CAM contrast (6.6M/0.0000 vs 8.0M/0.7167) is your cleanest quasi-controlled comparison — lead with it.
- **(c) The CAM target layer was chosen per-architecture and may itself drive the differences.** Defense: report a sensitivity analysis across candidate target layers; Selvaraju et al. and Jiang et al. both document strong layer-dependence, so this must be shown to be controlled, not assumed away.
- **(d) GoogLeNet's 0.0000 pointing accuracy looks like a bug or degenerate target-layer choice.** This is the single most likely reviewer attack and an extraordinary claim. Defense (mandatory): (i) run the Adebayo et al. sanity checks on the GoogLeNet maps; (ii) show maps at multiple candidate target layers to rule out a degenerate 8×8 auxiliary/stem choice; (iii) verify against a gradient-free method (Score-CAM) and/or HiResCAM to rule out a Grad-CAM/Layer-CAM gradient-averaging artifact specific to GoogLeNet's head; (iv) confirm the GAP-then-FC head structure and whether the inception auxiliary classifiers / channel concatenation are interfering. Present 0.0000 only after these are excluded, and label it as "degenerate localization under the tested configuration" rather than an immutable architectural constant.
- **(e) Single dataset, no external validation.** Defense: state as a limitation; note LIDC-IDRI is the field-standard reference database (Armato et al., *Medical Physics* 2011, 38(2):915–931, DOI 10.1118/1.3528204); propose external validation (e.g., LUNA16 subset, IQ-OTH/NCCD) as future work.
- **(f) Identical hyperparameters may unfairly favor some architectures ("a fair setup can be unfair").** Defense: cite the benchmarking literature showing tuning budget can flip architecture rankings (e.g., "Tune It or Don't Use It," arXiv:2108.13122; Bouthillier et al. 2021); report that the identical-training choice was deliberate (to isolate architecture) and acknowledge it may under-serve some backbones; ideally show that light per-architecture tuning does not change the qualitative ranking.

## Recommendations
1. **Anchor observation (a)** on Raghu et al. 2019 (10.48550/arXiv.1902.07208) plus the LIDC-IDRI radiomics study by Saied et al. (10.1186/s13244-023-01441-6); present the 8.5×-params / 0.0144-AUC result as a same-domain confirmation, not a novel discovery. Threshold to change: if a reviewer supplies a same-dataset counterexample where capacity clearly helps, soften to "on datasets of this scale."
2. **Position observation (b)** as *extending* Qiu et al. 2023 (10.1007/978-3-031-45676-3_23) from CNN-vs-ViT/depth to multiple CNN families with Layer-CAM on lung CT. Ground the mechanism in the CVIU 2025 receptive-field paper (10.1016/j.cviu.2025.104383) and the HiResCAM faithfulness argument (10.48550/arXiv.2011.08891).
3. **Before submission, resolve the GoogLeNet 0.0000** with sanity checks (Adebayo et al. 2018), multi-layer sensitivity, and a Score-CAM/HiResCAM cross-check. Do not submit the family claim until this is excluded as an artifact — it is the reviewer's easiest kill shot.
4. **Report efficiency** as params + FLOPs (ptflops/fvcore/thop) + measured latency/memory with stated hardware/batch size; add an accuracy-per-parameter column and a Pareto plot; state the FLOPs≠latency caveat.
5. **Frame the Green-AI contribution** with Schwartz et al. 2020 (10.1145/3381831), Strubell et al. 2019 (10.18653/v1/P19-1355), and the radiology sustainability review (10.1186/s13244-025-01962-2).
6. **Claim the gap** (Q5) in appropriately hedged language ("to our knowledge…"), backed by the observation that the efficiency and CAM-quality literatures rarely intersect.

## Verified Citation List (with DOIs)
**Architecture papers**
- GoogLeNet — Szegedy et al., "Going Deeper with Convolutions," CVPR 2015, pp. 1–9. DOI 10.1109/CVPR.2015.7298594
- InceptionV3 — Szegedy et al., "Rethinking the Inception Architecture for Computer Vision," CVPR 2016, pp. 2818–2826. DOI 10.1109/CVPR.2016.308
- Inception-ResNet-v2 / Inception-v4 — Szegedy et al., "Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning," AAAI 2017, 31(1):4278–4284. DOI 10.1609/aaai.v31i1.11231
- Xception — Chollet, "Xception: Deep Learning with Depthwise Separable Convolutions," CVPR 2017, pp. 1800–1807. DOI 10.1109/CVPR.2017.195
- DenseNet — Huang et al., "Densely Connected Convolutional Networks," CVPR 2017, pp. 2261–2269. DOI 10.1109/CVPR.2017.243; journal version "Convolutional Networks with Dense Connectivity," IEEE TPAMI. DOI 10.1109/TPAMI.2019.2918284
- ConvNeXt — Liu et al., "A ConvNet for the 2020s," CVPR 2022, pp. 11976–11986. DOI 10.1109/CVPR52688.2022.01167

**CAM / explainability methods**
- Grad-CAM — Selvaraju et al., ICCV 2017 (DOI 10.1109/ICCV.2017.74) and IJCV 2020 (DOI 10.1007/s11263-019-01228-7)
- Layer-CAM — Jiang et al., IEEE TIP 2021, 30:5875–5888. DOI 10.1109/TIP.2021.3089943
- HiResCAM — Draelos & Carin, arXiv:2011.08891. DOI 10.48550/arXiv.2011.08891
- Score-CAM — Wang et al., CVPR Workshops 2020, pp. 111–119. DOI 10.1109/CVPRW50498.2020.00020
- Pointing game / Excitation Backprop — Zhang et al., ECCV 2016 (DOI 10.1007/978-3-319-46493-0_33); IJCV 2018 (DOI 10.1007/s11263-017-1059-x)
- Sanity Checks for Saliency Maps — Adebayo et al., NeurIPS 2018; arXiv:1810.03292 (no Crossref DOI)
- Grad-CAM large-receptive-field caveats — CVIU 2025, vol. 258. DOI 10.1016/j.cviu.2025.104383

**Efficiency / Green AI / domain**
- Transfusion — Raghu et al., NeurIPS 2019; arXiv:1902.07208. DOI 10.48550/arXiv.1902.07208
- Green AI — Schwartz et al., CACM 2020, 63(12):54–63. DOI 10.1145/3381831
- Energy & Policy — Strubell et al., ACL 2019, pp. 3645–3650. DOI 10.18653/v1/P19-1355
- Radiology sustainability — Insights into Imaging 2025. DOI 10.1186/s13244-025-01962-2
- LIDC-IDRI radiomics/efficiency — Saied et al., Insights into Imaging 2023, 14:91. DOI 10.1186/s13244-023-01441-6
- LIDC-IDRI reference database — Armato et al., Medical Physics 2011, 38(2):915–931. DOI 10.1118/1.3528204
- Cross-architecture Grad-CAM in medical imaging — Qiu, Rivaz & Xiao, MLMI 2023, LNCS 14349:224–233. DOI 10.1007/978-3-031-45676-3_23

## Caveats
- **Citations whose details could not be fully verified:** Sahu et al. lightweight multi-section CNN JBHI DOI; DeepLung WACV 2018 DOI (10.1109/WACV.2018.00079, not re-verified); Bouthillier et al. 2021 has no clean Crossref DOI (cite arXiv:2103.03098); NeurIPS 2018 "Sanity Checks" has no Crossref DOI (cite arXiv:1810.03292); the EfficientNet accuracy-per-parameter figure and the CNN-vs-ViT medical study are arXiv preprints (arXiv:2507.21156), not peer-reviewed. The AAAI Inception-v4 paper is also indexed by ACM DL under 10.5555/3298023.3298188; the canonical publisher DOI is the AAAI one listed above.
- **The strongest gap in my sourcing** is a same-domain (lung CT), same-method (pointing game/IoU/Dice) cross-architecture CAM comparison — Qiu et al. 2023 is the nearest analogue but differs in modality, metric, and architecture set. Present observation (b) as an extension of a thin precedent, not a filled gap.
- **No located peer-reviewed paper** reports LIDC-IDRI nodule classification with AUC AND parameter counts showing lightweight-beats-heavyweight in one table; the supporting evidence is assembled from partial sources (Saied et al. for AUROC without full parameter tables; DeepLung for a "¼ the parameters" claim on detection FROC) and should be presented as such.
- **Grad-CAM-on-Inception failure reports from practitioner forums** are anecdotal corroboration only, not citable evidence.
- **The Qiu et al. MLMI paper** is peer-reviewed (MICCAI workshop, LNCS) but the volume publication year is listed by Springer as 2024 even though the workshop was held in 2023 — cite consistently (2023 workshop / 2024 volume) to avoid a reviewer flag.