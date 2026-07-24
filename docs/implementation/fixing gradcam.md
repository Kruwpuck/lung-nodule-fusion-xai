# Fixing Grad-CAM Interpretability for 2.5D LIDC-IDRI Lung Nodule Malignancy Classification

## TL;DR
- Your all-blue maps are a **target-class bug, not a model failure**: the CAM is hardcoded to explain class 1 (malignant) on samples that are label 0 (benign) and predicted benign, so there is no positive gradient for class 1 to highlight. Explain the **predicted class** (or, for figures, curate **true-positive high-confidence malignant** cases), and the maps will light up on the lesion.
- The corner/edge "hot" blobs on near-black patches are a **cropping/centering artifact** (axis-order z/y/x confusion, origin handling, or 2.5D slice mis-selection), amplified by Grad-CAM's coarse last-layer map being upsampled with bilinear interpolation; verify centering by overlaying the pylidc consensus-mask centroid before trusting any CAM.
- Add a **VGG16 branch (`model.features[-1]`)** and a **ViT branch with a `reshape_transform` that drops the CLS token**, ensure ResNet/DenseNet grab the last *spatial* map (`layer4[-1]` / `features[-1]`, never post-`avgpool`), quantify alignment with **IoU/Dice, the pointing game, and the energy-based pointing game**, and for small nodules prefer **HiResCAM or Layer-CAM** over vanilla Grad-CAM.

## Key Findings

1. **The blue-map cause is confirmed and standard.** Grad-CAM applies a ReLU to the class-weighted activation sum, so it only shows *positive* evidence for the requested class. If you request the malignant class on a nodule the network scores as benign, the class-1 gradient is weak/negative everywhere and the ReLU zeroes the map — producing an all-blue overlay. This is expected behavior, not a distortion.
2. **Convention: explain the predicted class for diagnostics; show true positives in the paper figure.** In `pytorch-grad-cam`, passing `targets=None` makes the library explain the highest-scoring category automatically. Fixing the target to malignant is only meaningful on cases the model actually predicts (or that are truly) malignant.
3. **Target layers must be the last spatial feature map.** Official recommendations: ResNet50 → `model.layer4[-1]`; VGG16 and DenseNet → `model.features[-1]`; MobileNetV3/EfficientNet-B0 → last block of `model.features`; ViT → a normalization layer inside the last transformer block *with* a `reshape_transform` that removes the CLS token.
4. **CAM upsampling in the library uses per-image min-max normalization + `cv2.resize` (bilinear default).** Edge artifacts come from the tiny source grid (e.g., 7×7 or 2×2 for a 64×64 input), not from the interpolation kernel per se.
5. **Quantitative alignment metrics are well-established and citable:** IoU/Dice of the thresholded CAM vs. mask, the pointing game (Zhang et al., ECCV 2016; IJCV 2018, DOI 10.1007/s11263-017-1059-x), and the energy-based pointing game (Wang et al., Score-CAM). Your `cam_in_nodule_fraction` is essentially the energy-based pointing game.
6. **For small lesions, gradient-averaging methods (Grad-CAM) over-smooth and over-expand.** HiResCAM (faithful, element-wise), Layer-CAM (fine-grained, shallow-layer capable), and Grad-CAM++ (better for small/multiple objects) are better suited; Score-CAM/Ablation-CAM/Eigen-CAM are gradient-free alternatives.

## Details

### A. Target class convention and sample selection

**Root cause of the all-blue maps.** Grad-CAM computes `L = ReLU(Σ_k α_k^c A^k)`, where `α_k^c` are gradients of the class-`c` score with respect to feature map `A^k`. The ReLU keeps only regions that *positively* support class `c`. When you force `c = malignant` on a benign, benign-predicted sample, there is little positive evidence for malignancy anywhere, so the map is near-zero everywhere → uniformly blue after colormapping. The HiResCAM authors (Draelos & Carin) describe the related pathology that gradient averaging can even highlight regions the model did not use; the all-blue case is simpler: you are asking "where is the malignancy evidence?" on an image the model thinks has none.

**Which class to explain.** There are three defensible conventions, and the right one depends on the question:
- **Explain the predicted (top-1) class** — the standard diagnostic default. This answers "why did the model decide what it decided?" In `pytorch-grad-cam`, `targets=None` selects the highest-scoring category per image automatically. This is what you want for model debugging and for most XAI figures.
- **Explain a fixed class (e.g., always malignant)** — valid only when you want to compare "malignancy evidence" across cases, and only informative on cases where that class has support. On benign/benign-predicted samples it *correctly* produces empty maps; that is not a bug but it is misleading to show in a "does the model look at the lesion?" figure.
- **Explain the ground-truth class** — useful for error analysis (e.g., show where the model *should* have looked on a false negative), but do not present it as the model's rationale.

For a **binary malignancy head with a single sigmoid output**, note the `pytorch-grad-cam` maintainer's guidance (Discussion #297): the single output *is already* the class-1 score, so use a custom `BinaryClassifierOutputTarget` (multiply by +1 for the positive class, −1 for the negative class) rather than `ClassifierOutputTarget(1)`, which assumes ≥2 logits.

**Which samples to display in the paper.** Best practice for an XAI figure that argues "the model localizes the lesion":
- Use **true positives** (ground truth = malignant *and* predicted malignant), ideally **high-confidence** ones (high predicted probability), so the CAM is well-formed and the claim is clean.
- Explain the **predicted class** on those samples.
- Optionally add a small number of **instructive failure cases** (false positives/negatives) in a separate panel for honest error analysis — but label them as such.
- Avoid cherry-picking a single lucky map; show several, and pair them with the quantitative metrics in Section D so reviewers see the CAMs are representative rather than curated.

### B. Correct CAM upsampling / resizing

**What the library actually does.** In `pytorch_grad_cam/utils/image.py`, `scale_cam_image` performs, per image: (1) min-max normalization `img = (img - img.min())/(img.max() + 1e-7)` to [0,1], then (2) resize to the input size with `cv2.resize` for 2D maps (or `scipy.ndimage.zoom` for higher-dim). `cv2.resize` defaults to **bilinear** interpolation. The map is computed at the target layer's spatial resolution and upsampled to the full input size inside `compute_cam_per_layer`.

**Recommendations to avoid distortion/edge artifacts:**
- **Interpolation kernel:** Bilinear (the library/Grad-CAM paper default) is the standard and is fine. Bicubic can overshoot and create ringing (values outside the true range) near sharp edges; if you use it, re-clip to [0,1]. The original Grad-CAM paper uses bilinear.
- **`align_corners`:** `cv2.resize` does not expose `align_corners`; if you instead upsample with `torch.nn.functional.interpolate`, use `mode='bilinear', align_corners=False` — the modern default. `align_corners=True` biases the sampling grid toward the corner pixels and can shift/stretch a coarse map, which for a 2×2 or 7×7 source grid meaningfully misplaces the hot region. Prefer `align_corners=False` for geometric consistency.
- **Normalization:** Normalize *before* colormapping (as the library does). Be aware that per-image min-max means an essentially empty map (benign case) still gets stretched to [0,1], which can manufacture a spurious "hot" spot from numerical noise — another reason the corner artifact appears on near-black patches. Consider a global/fixed scale or a small-activation guard (e.g., skip normalization when `max < ε`) when displaying benign cases.
- **Root cause of the edge blob:** the source CAM for a 64×64 input at `layer4` of ResNet50 is only 2×2 (input 64 → /32); at DenseNet/EfficientNet last blocks it is similarly tiny. Upsampling a 2×2 grid to 64×64 makes any single hot cell bloom into a large corner/edge region. Combined with an off-center or padded crop, the hot cell lands on the padding. The fix is upstream (center the nodule; see E) and, optionally, use a higher-resolution method (Layer-CAM at a shallower layer) so the source grid is larger.

### C. Target layer + reshape_transform per architecture (copy-pasteable)

Official `pytorch-grad-cam` guidance ("Some common choices"): ResNet → `model.layer4[-1]`; VGG/DenseNet → `model.features[-1]`; MNASNet → `model.layers[-1]`; ViT → `model.blocks[-1].norm1`; Swin → `model.layers[-1].blocks[-1].norm1`. The maintainer's discussion thread (#459) lists the same for torchvision models. Key rule: **target the last layer that still has spatial (H×W) extent**, i.e., *before* global average pooling — never `avgpool`, `classifier`, or `fc`.

```python
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def get_target_layers(model, arch):
    arch = arch.lower()
    if arch == "resnet50":
        return [model.layer4[-1]]              # last residual block (last spatial map)
    if arch == "densenet121":
        return [model.features[-1]]            # final BN (features.norm5); last spatial map
    if arch == "vgg16":
        return [model.features[-1]]            # last conv block output (before avgpool/classifier)
    if arch == "mobilenet_v3":
        return [model.features[-1]]            # last inverted-residual/conv block
    if arch == "efficientnet_b0":
        return [model.features[-1]]            # final conv block
    if arch == "convnext":
        return [model.features[-1]]
    if arch == "vit_base":
        return [model.blocks[-1].norm1]        # timm ViT: norm before last attention block
    raise ValueError(f"no target layer branch for {arch}")
```

Notes per architecture:
- **VGG16:** `model.features` is a `Sequential` ending in the last conv (+ReLU) before the `avgpool`/`classifier`; `model.features[-1]` (the last ReLU/conv) is the correct choice. This is the branch your code is missing.
- **ResNet50:** `model.layer4[-1]` is the last Bottleneck block; its output is the 7×7 (for 224 input) spatial map. Do **not** use `model.avgpool`.
- **DenseNet121:** `model.features[-1]` is `norm5`, the final BatchNorm over the last dense block's spatial features — the correct last spatial map. Avoid indexing into `classifier`. (A common pitfall, seen in the library's own issue tracker, is targeting a specific `denselayer`/`conv` deep inside the nested blocks, which can raise shape errors — `features[-1]` is the safe target.)
- **MobileNetV3 / EfficientNet-B0 / ConvNeXt:** all expose `model.features`; the last element is the last spatial block. (Your existing branches for these are fine as long as they resolve to `features[-1]`.)
- **ViT-base:** the tokens are `(batch, 1 + N, C)`; the CLS token is index 0. You must (a) pick a layer *before* the final attention block's classification collapse (the library recommends the norm layer of the last block), and (b) pass a `reshape_transform`:

```python
import numpy as np

def vit_reshape_transform(tensor, height=14, width=14):
    # tensor: (B, 1+H*W, C) for ViT-base/16 at 224 -> 196 patches -> 14x14
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # (B, H, W, C) -> (B, C, H, W) to mimic a CNN feature map
    return result.transpose(2, 3).transpose(1, 2)

# For a non-standard input size, infer the grid from the token count:
def vit_reshape_auto(tensor):
    n = tensor.shape[1] - 1                 # drop CLS
    s = int(np.sqrt(n))
    r = tensor[:, 1:, :].reshape(tensor.size(0), s, s, tensor.size(2))
    return r.permute(0, 3, 1, 2)

cam = GradCAM(model=model,
              target_layers=[model.blocks[-1].norm1],
              reshape_transform=vit_reshape_auto)
```

The library explicitly warns: because ViT's final classification uses the CLS token computed in the *last* attention block, the gradient of the output with respect to the 14×14 spatial channels of that last layer can be zero — so target `blocks[-1].norm1` (before that block's attention) or an earlier block. If your ViT input is 64×64 with 16×16 patches, the grid is 4×4 (16 patches); set the reshape accordingly.

Binary-head target (for a single-logit malignancy model):
```python
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
targets = [BinaryClassifierOutputTarget(1)]   # +1 for malignant; use 0 -> multiplies by -1
# Or, to always explain the predicted class, pass targets=None.
```

### D. Quantitative CAM-vs-mask alignment metrics

All metrics below operate on the upsampled, [0,1]-normalized CAM `S` and the binary lesion mask `G` (your pylidc consensus mask, resampled to the patch grid).

**1. IoU / Dice of the high-activation region.** Threshold the CAM (commonly at a fixed fraction, e.g., top-20% of energy, or Otsu, or `S > τ` with τ=0.5) to get a binary map `S_b`, then:
- IoU (Jaccard) = `|S_b ∩ G| / |S_b ∪ G|`
- Dice = `2|S_b ∩ G| / (|S_b| + |G|)`
Report the threshold you used; sweep τ and report the best/curve to avoid threshold cherry-picking.

```python
import numpy as np
def dice_iou(cam, mask, pct=0.80):
    thr = np.quantile(cam, pct)           # top-20% activation
    s = cam >= thr
    g = mask.astype(bool)
    inter = np.logical_and(s, g).sum()
    dice = 2*inter / (s.sum() + g.sum() + 1e-7)
    iou  = inter / (np.logical_or(s, g).sum() + 1e-7)
    return dice, iou
```

**2. Pointing game (Zhang et al., "Top-down Neural Attention by Excitation Backprop", ECCV 2016; IJCV 126:1084–1102, 2018, DOI 10.1007/s11263-017-1059-x).** A "hit" is scored if the single maximum-activation pixel of the CAM falls inside the ground-truth region; accuracy = hits / (hits + misses) over the dataset. It is minimal-post-processing and measures spatial selectivity. Reference PyTorch implementation: `ruthcfong/pointing_game`.

```python
def pointing_hit(cam, mask):
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    return bool(mask[y, x])
```

**3. Energy-based pointing game (Wang et al., "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks", CVPRW 2020; arXiv:1910.01279).** Fraction of total CAM energy that lands inside the mask: `Proportion = Σ_{(i,j)∈G} S_ij / (Σ_{(i,j)∈G} S_ij + Σ_{(i,j)∉G} S_ij)`. This is exactly your `cam_in_nodule_fraction` — cite Score-CAM as the standard version and rename it accordingly. Reference implementation exists at `haofanwang/Score-CAM` (`utils/energyPointGame.py`).

```python
def energy_pointing_game(cam, mask):
    cam = cam - cam.min()
    return (cam * mask).sum() / (cam.sum() + 1e-7)
```

**Precedent for CAM-vs-mask alignment on lung CT (use to frame — not overclaim — your numbers):**
- **Feng, Yang, Laine & Angelini, "Discriminative Localization in CNNs for Weakly-Supervised Segmentation of Pulmonary Nodules," MICCAI 2017** (DOI 10.1007/978-3-319-66179-7_65; PMID 29308456; arXiv:1707.01086). This is the canonical LIDC-IDRI CAM-localization reference: it derives nodule activation maps (a CAM variant) from a slice-level classifier and evaluates the resulting segmentation against >50%-consensus radiologist masks on LIDC-IDRI, reporting Dice overlap on the order of ~0.55 (with per-true-positive Dice ~0.74) for the best weakly-supervised variant — competitive with a fully-supervised U-Net baseline. Confirm the exact table value ("TP Dice") against the PDF before quoting a precise number. Its classification task is *nodule presence*, not benign-vs-malignant, so cite it for the localization methodology rather than as a malignancy benchmark.
- **Häggström/Guehring et al., "Weakly supervised segmentation of tumor lesions in PET-CT hybrid imaging," Journal of Medical Imaging 8(5):054003** (DOI 10.1117/1.JMI.8.5.054003), on 453 oncological patients (~50% lung cancer): a useful cautionary datapoint that **vanilla Grad-CAM over-expands and performs poorly for delineation** — "best median Dice score 0.47, interquartile range (IQR) 0.35 … However, GradCAM led to inferior results (median Dice score: 0.12, IQR 0.21) and was likely to ignore multiple instances within a given slice." ScoreCAM 0.47 (IQR 0.35), CAM 0.46 (IQR 0.35), GradCAM++ 0.42 (IQR 0.30); fully-supervised U-Net 0.72 (IQR 0.36).
- A recent explainable lung-cancer CT classifier (**Katar, Akbalik & Yildirim, "An Explainable Transformer-Based Framework for Lung Cancer Classification and Automated Radiology Report Generation from Multi-Slice CT Images," *Biomedicines* 14(5):1103**, DOI 10.3390/biomedicines14051103) reports, verbatim, "a mean IoU of 0.45 ± 0.14, a mean Dice of 0.59 ± 0.15, and a Pointing Game accuracy of 85.4%" for Grad-CAM vs. radiologist annotations (per-class IoU 0.48 SCLC vs. 0.42 NSCLC). **Verify independently before citing**: it is very recent, uses a *private* dataset (not LIDC), and its ground truth is a bounding box rather than a voxel nodule mask.

Report at least IoU/Dice + the energy-based pointing game; the pointing game is a robust, threshold-free complement. Because no single peer-reviewed paper perfectly combines LIDC + malignancy classification + Grad-CAM-vs-voxel-mask IoU, present your own per-architecture metrics as a genuine contribution.

### E. Debugging checklist for empty / mis-cropped patches

The corner-hot, near-black patches almost always mean the nodule is not where your crop thinks it is. Work through these in order:

1. **Axis order (the #1 culprit).** `pylidc`'s `scan.to_volume()` returns a NumPy array indexed `[i, j, k]` = `[row(y), col(x), slice(z)]`, and `ann.centroid` returns `(i, j, k)` in that same order (the pylidc docs plot the centroid with `plt.plot(j, i, '.r')` over `vol[:,:,int(k)]` — note x=j, y=i). SimpleITK, by contrast, exposes arrays as `[z, y, x]` and its physical points as `(x, y, z)`. If you load the volume with SimpleITK but take the centroid from pylidc (or vice versa) without swapping axes, your (row, col) and slice indices get transposed → the crop lands off-nodule. **Fix:** pick one library end-to-end, or explicitly map indices.

2. **Use pylidc's own bbox/centroid rather than reconstructing from `ImagePositionPatient`.** `ann.bbox(pad=...)` returns Python slice objects already in the volume's index space, and `ann.centroid` is in the same space — so you generally do **not** need to convert DICOM `ImagePositionPatient`/origin yourself. If you *do* go through world coordinates (SimpleITK `TransformPhysicalPointToIndex`), remember DICOM uses the LPS convention (x → patient left, y → posterior, z → superior) and that origin + direction + spacing all matter; a sign or ordering error puts the nodule at the wrong voxel. Note pylidc's own `bbox` pads are clipped to image borders, so a padded box near the pleura can be asymmetric.

3. **2.5D slice selection.** For 3 axial slices around the centroid, the center slice must be the centroid's *z* index in the same volume you crop from. Common bugs: (a) selecting slices by nodule-local mask index instead of the full-volume z index; (b) after resampling to 1 mm, the z index changes — recompute the centroid in resampled space; (c) `drop_ends`/consensus slice ranges shifting which slice is "central." Sanity check that the middle of your 3 slices has the largest nodule cross-sectional area.

4. **Resampling consistency.** Resample the CT volume and the mask with the **same** transform (image: linear/B-spline; mask: nearest-neighbor). If you resample to 1 mm but compute the crop window in original-spacing voxels, the 64 mm window becomes the wrong pixel count and can run off the lung. Compute the window in **mm**, then convert to pixels using the *resampled* spacing.

5. **Out-of-bounds / padding.** When the 64 mm window exceeds the volume near the pleura, pad with a constant (air ≈ −1000 HU or the volume min, as pylidc's own interpolation does with `fill_value=min`), not zero-after-normalization, and record a flag. Then **exclude or specially handle padded patches in the CAM figure**, because normalized padding is where the spurious corner activations appear.

6. **The definitive sanity check — overlay the consensus centroid.** For every patch, overlay the pylidc `consensus()` mask (or `ann.boolean_mask()` placed via `ann.bbox()`) and its centroid on the extracted patch and assert the centroid is within a few pixels of the patch center. Automate it: compute `|patch_center − mask_centroid|` and log any patch exceeding, say, 4 px. Any patch with an empty mask or an off-center centroid should be dropped before CAM analysis.

```python
import numpy as np, pylidc as pl
from pylidc.utils import consensus

ann = pl.query(pl.Annotation).first()
cmask, cbbox, _ = consensus([ann], clevel=0.5)   # boolean vol + bbox slices
# centroid within the consensus bbox, then shift to full-volume indices:
local_c = np.argwhere(cmask).mean(0)
full_c  = local_c + np.array([s.start for s in cbbox])
# assert your crop is centered on full_c (in i,j,k / row,col,slice order)
```

### F. Alternative CAM methods for small lesions

For lung nodules — often only a handful of pixels in a 64×64 patch, and tiny (2×2–7×7) at the last conv layer — vanilla Grad-CAM's gradient averaging + coarse map is a poor fit. Options, all available in `pytorch-grad-cam`:

| Method | Gradient-free? | Mechanism | Fit for small nodules |
|---|---|---|---|
| **Grad-CAM** (Selvaraju et al., ICCV 2017; arXiv:1610.02391) | No | Global-avg-pooled gradients × activations, ReLU | Baseline; over-smooths, over-expands small objects |
| **HiResCAM** (Draelos & Carin, arXiv:2011.08891) | No | Element-wise gradient × activation (no spatial averaging) | **Strong**: faithful, sharper, avoids highlighting unused regions; validated on chest CT abnormality classification |
| **Grad-CAM++** (Chattopadhay et al., WACV 2018; arXiv:1710.11063) | No | Positive higher-order (2nd-deriv) gradient weights | **Good** for small/multiple objects; better localization than Grad-CAM |
| **Layer-CAM** (Jiang et al., IEEE TIP 2021, DOI 10.1109/TIP.2021.3089943) | No | Per-pixel positive-gradient weighting; works at **shallow** layers | **Strong** for fine-grained/small lesions — larger source grid → less upsampling blur |
| **Score-CAM** (Wang et al., CVPRW 2020; arXiv:1910.01279) | **Yes** | Masks input with each activation map, weights by confidence change | Cleaner, less noisy; slow (many forward passes) |
| **Ablation-CAM** (Desai & Ramaswamy, WACV 2020) | **Yes** | Zero out each feature map, measure score drop | Robust to gradient noise; slow (k+1 forward passes) |
| **Eigen-CAM** (Muhammad & Yeasin, IJCNN 2020; arXiv:2008.00299) | **Yes** | First principal component of activations | Fast, robust, but **not class-discriminative** — weak for benign-vs-malignant |

**Recommendation for your paper.** Use **HiResCAM as your primary method** (faithful by construction and specifically validated on CT abnormality classification, where it reduced spurious out-of-organ attribution) and report **Layer-CAM at a shallower layer** as a higher-resolution complement for the smallest nodules; where Layer-CAM is applied to a layer followed by a stride>1/max-pool, expect a documented grid artifact, so prefer the layer just before downsampling. Keep **Grad-CAM++** as the closest drop-in upgrade to your current Grad-CAM. Add one **gradient-free method (Score-CAM or Ablation-CAM)** as a robustness cross-check, but avoid Eigen-CAM as a primary explainer because it is not class-discriminative — a liability when the whole point is separating malignant from benign evidence.

## Recommendations

**Stage 1 — Fix the bug and re-render (do first).**
1. Change the CAM call to explain the **predicted class** (`targets=None`) or a `BinaryClassifierOutputTarget` matching your head; stop hardcoding class 1 on benign samples.
2. Rebuild the figure from **true-positive, high-confidence malignant** cases (GT=malignant ∧ pred=malignant ∧ high p). The maps should now be red on the lesion. *Benchmark to proceed:* ≥80% of curated TP cases show the CAM maximum inside the nodule mask (pointing-game hit).

**Stage 2 — Fix the pipeline (do in parallel).**
3. Add the **VGG16 (`features[-1]`)** and **ViT (`blocks[-1].norm1` + reshape_transform)** branches; assert every target layer produces a spatial (H×W>1) map.
4. Run the **centering sanity check (E6)** on the entire dataset; drop or fix any patch whose mask centroid is >4 px from center or whose mask is empty. This should eliminate the near-black corner-hot patches. *Benchmark:* zero patches with empty masks in the analysis set.

**Stage 3 — Quantify and strengthen (for the paper).**
5. Report **IoU, Dice, pointing game, and energy-based pointing game** (rename `cam_in_nodule_fraction` → energy-based pointing game, cite Score-CAM) across the test set, per architecture, with the threshold stated.
6. Compare **Grad-CAM vs. HiResCAM vs. Layer-CAM (and one gradient-free method)**; adopt whichever maximizes energy-in-mask and pointing-game accuracy on your validation split. *Threshold that changes the recommendation:* if HiResCAM's energy-in-mask does not exceed Grad-CAM's by a meaningful margin, keep Grad-CAM++ for simplicity; if the smallest-nodule stratum has low pointing-game accuracy for all last-layer methods, switch to Layer-CAM at a shallower layer.

**Stage 4 — Report honestly.** Include a small failure-case panel; state the target-class convention, threshold, interpolation (bilinear, `align_corners=False`), and metric definitions in the methods so the CAMs are reproducible.

## Caveats
- **Grad-CAM localization ≠ faithfulness.** Pointing game and IoU/Dice measure whether the map overlaps the lesion, not whether the model truly used those pixels; HiResCAM is faithful by construction (exactly so only for architectures with a single fully-connected head after the target layer), Grad-CAM is not. Report both localization and, ideally, a faithfulness/perturbation metric (e.g., ROAD, built into `pytorch-grad-cam`).
- **ViT maps are inherently coarser and depend on the reshape/target layer choice**; a wrong layer yields zero-gradient (blank) maps — validate the ViT branch on a known-malignant case before trusting it.
- **Small-nodule resolution limit:** no CAM can exceed the spatial resolution of the feature map it reads; at a 2×2 last-layer grid for a 64×64 input, last-layer CAMs are fundamentally coarse regardless of interpolation. Shallow-layer methods mitigate but add noise.
- **Sourcing note on precedent:** the closest LIDC precedent (Feng et al., MICCAI 2017) evaluates *nodule-presence* CAM localization, not benign-vs-malignant; the lung-cancer CT/Grad-CAM paper reporting IoU 0.45 / Dice 0.59 / pointing-game 85.4% (Katar et al., *Biomedicines* 2026, DOI 10.3390/biomedicines14051103) uses a private dataset with bounding-box (not voxel-mask) ground truth and is very recent — verify it independently before citing. The PET/CT Grad-CAM Dice figures (Häggström/Guehring et al., J. Med. Imaging 8(5):054003) are on mixed malignancies, not LIDC nodules. No single peer-reviewed paper perfectly combines LIDC + malignancy classification + Grad-CAM-vs-voxel-mask IoU, so present your own metrics as a contribution.
- **Per-image min-max normalization can fabricate structure on empty maps** — guard against it when displaying benign/negative cases.