# Joint master table -- Track 2 backbone comparison

`run_id` 2026-08-20-run04, `commit_sha` 2f7cea5. Every number here is recomputed from the CSVs in this directory.

## Subset membership

The three claims rest on three different sets of models, and the sets are not the
same size. Membership is a column in the CSV (`subsets`, plus the three integer
flags) so no reader has to infer it.

* **localization12** -- all twelve rows below. CAM input size varies (64, 96, 224 px), which is deliberate: at this stage input resolution is an architectural variable.
* **efficiency7** -- the 7 rows measured at a shared 96 px input. Only these have efficiency columns.
* **optimizer4** -- the optimizer sweep covers mobilenetv2, efficientnet_b0, resnet50, vgg16. 3 of those four appear among the twelve; **mobilenetv2 is in the optimizer sweep but not in this table at all**, so the optimizer subset is not nested inside the localization subset.

## Localization, twelve backbones

| backbone            | subsets                    |   cam_input_size | module_path                                   | cam_spatial          |   n_zero_cam |   pointing_acc |   pointing_acc_ci_lo |   pointing_acc_ci_hi |   dice |   dice_ci_lo |   dice_ci_hi |   dice_size_matched |   energy_mean |
|:--------------------|:---------------------------|-----------------:|:----------------------------------------------|:---------------------|-------------:|---------------:|---------------------:|---------------------:|-------:|-------------:|-------------:|--------------------:|--------------:|
| mobilenetv3_small   | localization12             |               64 | features.0.12.2                               | 2x2                  |            0 |         0.0000 |               0.0000 |               0.0602 | 0.0017 |       0.0000 |       0.0052 |              0.0018 |        0.0193 |
| efficientnet_b0     | localization12+optimizer4  |               64 | features.0.8.2                                | 2x2                  |            0 |         0.0000 |               0.0000 |               0.0602 | 0.0013 |       0.0000 |       0.0038 |              0.0015 |        0.0188 |
| densenet121         | localization12+efficiency7 |               96 | features.0.norm5                              | 3x3                  |            0 |         0.5833 |               0.4573 |               0.6994 | 0.0622 |       0.0401 |       0.0900 |              0.3475 |        0.0248 |
| resnet50            | localization12+optimizer4  |               64 | features.7.2                                  | 2x2                  |            0 |         0.0000 |               0.0000 |               0.0602 | 0.0030 |       0.0000 |       0.0074 |              0.0020 |        0.0198 |
| vgg16               | localization12+optimizer4  |               64 | features.0.30                                 | 2x2                  |            0 |         0.0000 |               0.0000 |               0.0602 | 0.0025 |       0.0000 |       0.0068 |              0.0019 |        0.0201 |
| vit_base            | localization12             |              224 | features.encoder.layers.encoder_layer_11.ln_1 | n/a (token sequence) |           53 |         0.0000 |               0.0000 |               0.0602 | 0.0357 |       0.0201 |       0.0557 |              0.0150 |        0.0074 |
| inceptionv3         | localization12+efficiency7 |               96 | features.14                                   | 4x4                  |            0 |         0.0333 |               0.0092 |               0.1136 | 0.0955 |       0.0624 |       0.1335 |              0.0456 |        0.0435 |
| xception            | localization12+efficiency7 |               96 | features.act4                                 | 3x3                  |            0 |         0.0667 |               0.0262 |               0.1593 | 0.0286 |       0.0060 |       0.0591 |              0.0338 |        0.0197 |
| googlenet           | localization12+efficiency7 |               96 | features.15                                   | 3x3                  |            0 |         0.5000 |               0.3774 |               0.6226 | 0.0901 |       0.0580 |       0.1282 |              0.2597 |        0.0309 |
| convnext_tiny       | localization12+efficiency7 |               96 | features.0.7.2                                | 3x3                  |            0 |         0.7167 |               0.5923 |               0.8149 | 0.1091 |       0.0749 |       0.1485 |              0.4443 |        0.0478 |
| inception_resnet_v2 | localization12+efficiency7 |               96 | features.repeat_1.19                          | 4x4                  |            0 |         0.1000 |               0.0466 |               0.2015 | 0.1175 |       0.0811 |       0.1571 |              0.0713 |        0.0462 |
| densenet201         | localization12+efficiency7 |               96 | features.0.norm5                              | 3x3                  |            0 |         0.7000 |               0.5749 |               0.8010 | 0.0970 |       0.0665 |       0.1331 |              0.4322 |        0.0315 |

Intervals: `pointing_acc` uses the Wilson score interval at 95% on 60 Bernoulli trials. `dice` uses a percentile bootstrap at 95% over the 60 per-sample rows, B=10000, seed 20260820. Bootstrap intervals for `iou`, `dice_size_matched` and `energy_mean` are in the CSV. The normal-approximation interval is used nowhere: several backbones score exactly zero and the normal interval degenerates there.

## Efficiency, seven backbones at 96 px

| backbone            |   efficiency_input_size |   params_M |   gflops |   latency_cpu_ms_median |   peak_mem_gpu_mb |   auc_mean |   auc_std |   auc_per_M_params |
|:--------------------|------------------------:|-----------:|---------:|------------------------:|------------------:|-----------:|----------:|-------------------:|
| mobilenetv3_small   |                         |            |          |                         |                   |            |           |                    |
| efficientnet_b0     |                         |            |          |                         |                   |            |           |                    |
| densenet121         |                      96 |      6.956 |   1.0649 |                  17.344 |            164.78 |   0.893976 |  0.022185 |           0.128519 |
| resnet50            |                         |            |          |                         |                   |            |           |                    |
| vgg16               |                         |            |          |                         |                   |            |           |                    |
| vit_base            |                         |            |          |                         |                   |            |           |                    |
| inceptionv3         |                      96 |     21.79  |   0.6828 |                  12.054 |            222.14 |   0.899223 |  0.025964 |           0.041268 |
| xception            |                      96 |     20.811 |   1.6683 |                  14.769 |             93.5  |   0.891076 |  0.028282 |           0.042818 |
| googlenet           |                      96 |      5.602 |   0.5558 |                   8.315 |            159.2  |   0.89616  |  0.011914 |           0.159971 |
| convnext_tiny       |                      96 |     27.82  |   1.6486 |                  12.703 |            125.53 |   0.905506 |  0.021974 |           0.032549 |
| inception_resnet_v2 |                      96 |     54.31  |   1.4733 |                  34.383 |            346.78 |   0.898637 |  0.017818 |           0.016546 |
| densenet201         |                      96 |     18.097 |   1.6132 |                  33.462 |            207.9  |   0.898808 |  0.024488 |           0.049666 |

**Footnote on the five empty rows.** mobilenetv3_small, efficientnet_b0, resnet50 and vgg16 were trained and profiled at a 64 px input; vit_base at 224 px. Cost figures -- GFLOPs, latency, peak memory -- scale with input size, so pasting their own measurements into this block would put numbers from three different input sizes into one column and invite exactly the comparison the column exists to support. They are left empty rather than dropped, because they still carry localization results. `params_M_any_res` in the CSV does hold a parameter count for all twelve; it is separated from the efficiency block because it was measured at each model's own input size and is used only as the x axis of the twelve-model scatter, never as a cost figure.

## Correlations

| subset                    | x                | y            |   n |   spearman_rho |   p_value | significant   | verdict                                                                          |
|:--------------------------|:-----------------|:-------------|----:|---------------:|----------:|:--------------|:---------------------------------------------------------------------------------|
| efficiency7 (96 px input) | params_M         | auc_mean     |   7 |         0.5000 |    0.2532 | False         | not significant at alpha=0.05 with n=7; no monotone association is demonstrated  |
| localization12            | params_M_any_res | pointing_acc |  12 |         0.1305 |    0.6860 | False         | not significant at alpha=0.05 with n=12; no monotone association is demonstrated |

Both are descriptive summaries of a small sample, reported with n attached.

## Silent-failure census

Of 86 candidate explanation sites across the twelve backbones, 17 emit an identically-zero CAM on at least one sample and 12 emit one on every sample. Sizes and backbones are in `silent_failure_census.csv`. The failure is silent because a constant-zero CAM still produces finite dice and energy values -- the `dice_emitted` column in that file -- so an automatic selector that lands on such a site reports a plausible number rather than an error.

One of the twelve headline rows is itself affected: the automatic resolver put vit_base at `encoder_layer_11.ln_1`, where the CAM is identically zero on 53 of 60 samples, and that row still reports a finite dice. Its `n_zero_cam` column in the master table is the flag.

## Depth-curve shapes

| backbone            |   n_sites | best_site_spatial   |   best_score | canonical_spatial   |   canonical_score |   gap_best_minus_canonical |   band_score | shape        |
|:--------------------|----------:|:--------------------|-------------:|:--------------------|------------------:|---------------------------:|-------------:|:-------------|
| mobilenetv3_small   |         7 | 32x32               |       0.0333 | 2x2                 |            0.0000 |                     0.0333 |       0.0000 | flat         |
| efficientnet_b0     |         7 | 16x16               |       0.1833 | 2x2                 |            0.0000 |                     0.1833 |       0.0833 | mid_peak     |
| densenet121         |         6 | 3x3                 |       0.5833 | 3x3                 |            0.5833 |                     0.0000 |       0.5833 | mid_peak     |
| resnet50            |         7 | 32x32               |       0.3833 | 2x2                 |            0.0000 |                     0.3833 |       0.1167 | shallow_peak |
| vgg16               |         7 | 64x64               |       0.4667 | 2x2                 |            0.0000 |                     0.4667 |       0.2000 | shallow_peak |
| vit_base            |        12 | 14x14               |       0.0667 | 14x14               |            0.0000 |                     0.0667 |       0.0000 | mid_peak     |
| inceptionv3         |         7 | 20x20               |       0.4333 | 4x4                 |            0.0333 |                     0.4000 |       0.2000 | mid_peak     |
| xception            |         8 | 45x45               |       0.3833 | 3x3                 |            0.0667 |                     0.3167 |       0.2000 | mid_peak     |
| googlenet           |         6 | 3x3                 |       0.5000 | 3x3                 |            0.5000 |                     0.0000 |       0.0000 | mid_peak     |
| convnext_tiny       |         6 | 3x3                 |       0.7167 | 3x3                 |            0.7167 |                     0.0000 |       0.7167 | mid_peak     |
| inception_resnet_v2 |         7 | 20x20               |       0.7333 | 4x4                 |            0.1000 |                     0.6333 |       0.3833 | mid_peak     |
| densenet201         |         6 | 3x3                 |       0.7000 | 3x3                 |            0.7000 |                     0.0000 |       0.7000 | mid_peak     |

Shape rule, applied to the data and not copied from any earlier report: sites ordered shallow to deep by exec_index_norm; score = pointing_acc; flat if max-min < 0.05; else shallow_peak if the best site is the first, deep_peak if it is the last, mid_peak otherwise. The 0.05 threshold is three of sixty samples.

## The near-equal-size pair, as it now stands

GoogLeNet (5.602 M parameters) and DenseNet121 (6.956 M) differ in size by 1.354 M, a ratio of 1.24, and come from different architecture families. The positioning brief proposed leading with them as a quasi-controlled contrast: near-equal capacity, opposite CAM quality.

Under the corrected resolver that contrast is largely gone. Pointing accuracy is 0.5000 [0.377, 0.623] for GoogLeNet against 0.5833 [0.457, 0.699] for DenseNet121, a gap of 0.0833 with overlapping Wilson intervals. The published numbers had a gap of 0.7167. Dice is 0.0901 against 0.0622. GoogLeNet's explanation site is now features.15 at 3x3; DenseNet121's is features.0.norm5 at 3x3.

Two similarly sized models from different families now behave similarly once both are explained at a site with spatial extent. That is consistent with the thesis that the explanation site, not the architecture family, governs localization quality -- the contrast in the published numbers was a property of where the CAM was taken, not of the family. It is a finding, not a loss.

## Optimizer axis, for context

From `track2_anova_eta_sq.csv`: optimizer eta^2 = 0.4533 (p = 1.1e-45), model eta^2 = 0.2060 (p = 4.64e-27), weight decay eta^2 = 0.0002 (p = 0.996). Measured on efficientnet_b0, mobilenetv2, resnet50, vgg16 only.
