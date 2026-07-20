# Fixing a Broken Grad-CAM Pipeline for 2.5D Lung-Nodule Malignancy Classification (LIDC-IDRI)

## TL;DR
- The fixed-per-architecture heatmap pattern (densenet always-left, resnet50 always-bottom-right, `pointing_acc = 0.000` across all 6 backbones and 60 samples) is the signature of a **structural bug, not gradient saturation** — saturation produces noisy/random maps, not a stable spatial pattern that ignores input content. Rule out the structural bug FIRST.
- The two most likely structural culprits are (a) a **degenerate ~2×2 feature map** at the chosen target layer for a 64×64 input (output-stride-32 backbones), so a tiny grid is bilinearly upsampled to 64×64 producing a fixed blob whose location is pinned by the argmax cell; and (b) a **broken ViT reshape_transform** (hardcoded 14×14 = 196 tokens vs the actual 4×4 = 16 tokens for a 64×64/patch-16 ViT), which either crashes or silently scrambles geometry.
- Prioritized fix order: (1) diagnostics to confirm the hook fires and the CAM changes with input; (2) move the target layer shallower (Layer-CAM at an 8×8–16×16 stage) to fix resolution; (3) switch to HiResCAM / Score-CAM / Ablation-CAM to address gradient saturation from the over-confident (p≈0.95–1.00) model. Expect the biggest single win from steps 1–2.

## Key Findings

1. **A stable, input-independent spatial pattern is diagnostic of a structural bug.** In "Sanity Checks for Saliency Maps" (Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt & Been Kim, NeurIPS 2018, Adv. NIPS 31, pp. 9525–9536), the authors demonstrate that "some existing saliency methods are independent both of the model and of the data generating process. Consequently, methods that fail the proposed tests are inadequate for tasks that are sensitive to either data or model." A map that does not change with the input is not reflecting the model. Gradient saturation from an over-confident model degrades a CAM toward *noise*, not toward a *stable fixed blob*. The observed always-left / always-bottom-right behavior therefore points to the CAM being computed from a constant/near-constant tensor or a geometrically degenerate feature map — not primarily saturation.

2. **For a 64×64 input, the last conv feature map is ~2×2 for all five CNN backbones** (all are output-stride-32 by default: resnet50 `layer4`, vgg16 after 5 max-pools, densenet121 final dense block, efficientnet_b0 final conv, mobilenetv3_small final conv). A 2×2 grid has only four possible argmax cells; after bilinear upsampling to 64×64, this yields a coarse blob pinned to one of four quadrants — exactly consistent with a "fixed per architecture" corner pattern, and it cannot localize a nodule occupying <10% of the patch.

3. **The ViT path is almost certainly separately broken.** torchvision `vit_b_16` hard-asserts input size in `_process_input` (`torch._assert(h == self.image_size, "Wrong image height! Expected 224 but got 64!")`); timm `vit_base_patch16_224` hard-asserts in `PatchEmbed.forward` (`Input height (64) doesn't match model (224).`) unless created with `dynamic_img_size=True` or `img_size=64`. Even after reconfiguring to 64×64, the standard jacobgil reshape_transform hardcodes `height=width=14` (196 tokens) while a 64×64/patch-16 ViT produces only 4×4 = 16 spatial tokens (+1 CLS = 17). Reshaping 16 tokens into a 196-cell grid raises `RuntimeError: shape '[B,14,14,C]' is invalid for input of size B*16*C` — or, if masked, scrambles spatial layout — explaining vit_base's anomalous but still-nonlocalizing numbers (dice 0.036, pointing_acc 0.000).

4. **Gradient saturation is real but secondary here.** On a very confident prediction the class logit's gradient w.r.t. deep activations can become near-flat; per-image min-max normalization in pytorch-grad-cam (`scale_cam_image`: `img = img - np.min(img); img = img / (1e-7 + np.max(img))`) then amplifies tiny numerical differences. But saturation alone gives unstable/noisy maps, not the reported stable pattern. Gradient-free methods (Score-CAM, Ablation-CAM) and gradient×activation methods (HiResCAM, Layer-CAM) are the recommended mitigations once the structural bug is excluded.

5. **Literature converges on shallower layers + HiResCAM/Layer-CAM/Ablation-CAM for small lesions.** Peng-Tao Jiang, Chang-Bin Zhang, Qibin Hou, Ming-Ming Cheng & Yunchao Wei, "LayerCAM: Exploring Hierarchical Class Activation Maps for Localization," IEEE Transactions on Image Processing, vol. 30, pp. 5875–5888, 2021 (DOI 10.1109/TIP.2021.3089943), note that "due to the small spatial resolution of the final convolutional layer, such class activation maps often locate coarse regions... we propose a simple yet effective method, called LayerCAM." Rachel Lea Draelos & Lawrence Carin, "Use HiResCAM instead of Grad-CAM for faithful explanations of convolutional neural networks," arXiv:2011.08891 (2020), "propose HiResCAM, a novel class-specific explanation method that is guaranteed to highlight only the locations the model used to make each prediction... a generalization of CAM." Medical benchmarks (e.g., the Hemorica brain-hemorrhage CAM benchmark on EfficientNetV2-S) find intermediate layers and HiResCAM/Ablation-CAM outperform last-layer vanilla Grad-CAM for pixel-level Dice/IoU.

## Details

### 1. Target-layer resolution: the last conv layer is far too coarse for a 64×64 patch

Grad-CAM's final step upsamples the feature-map-resolution heatmap to the input size, assuming spatial correspondence. With a 64×64 input and standard ImageNet backbones (all output-stride 32 by default in timm — "Most networks are stride 32 by default"), the last feature map is only ~2×2:

| Backbone | Default target layer | Feature map @224 | Feature map @64 (÷32) | Recommended shallower target |
|---|---|---|---|---|
| resnet50 | layer4 | 7×7 | ~2×2 | layer3 (÷16 → 4×4) or layer2 (÷8 → 8×8) |
| vgg16 | last conv (features[-1]) | 7×7 | ~2×2 | block after 3rd/4th pool (÷8/÷16 → 8×8/4×4) |
| densenet121 | features.norm5 / last dense block | 7×7 | ~2×2 | denseblock3 transition (÷16 → 4×4) |
| efficientnet_b0 | last conv (features[-1]) | 7×7 | ~2×2 | block 4–5 (÷16 → 4×4) |
| mobilenetv3_small | last conv before pool | 7×7 | ~2×2 | an intermediate inverted-residual block (÷16 → 4×4) |
| vit_base | encoder last block ln_1 | 14×14 tokens | 4×4 tokens (16 + CLS) | fix reshape_transform to 4×4; consider an earlier block |

A 2×2 map is the core resolution failure: four cells cannot represent a small nodule, and min-max normalization forces one cell to 1.0 and another to 0.0, producing a stable corner blob. The Grad-CAM paper's own ablation (Selvaraju et al., ICCV 2017, appendix D.2) notes localization degrades at shallow layers because receptive fields shrink — but for a 64×64 input the tradeoff strongly favors shallower layers. LayerCAM (Jiang et al., IEEE TIP 2021) is explicitly built to make shallower layers usable by weighting each spatial location separately, and is merged into jacobgil/pytorch-grad-cam. **Recommendation: target a stage with an 8×8 feature map (output-stride 8) and use Layer-CAM there.** Alternatively, increase the input patch to 112×112 or 128×128 so the last conv map becomes 4×4 or 7×7 while keeping the last layer.

### 2. Gradient saturation on over-confident predictions

The initial hypothesis (p≈0.95–1.00 → near-flat gradients) is a documented phenomenon: the CAM literature repeatedly cites "gradient saturation" as a failure mode of gradient-based CAMs. Mechanistically, once the model is deep in the saturated region of the output nonlinearity, ∂y^c/∂A is tiny and roughly uniform across space, so Grad-CAM's channel weights carry little spatial signal.

Mitigations, by mechanism:
- **HiResCAM (Draelos & Carin, arXiv:2011.08891):** element-wise gradient×activation before summation; "guaranteed to highlight only the locations the model used to make each prediction" for single-FC-head architectures; sharper, more faithful maps than Grad-CAM in 3D medical imaging (their AxialNet chest-CT work).
- **Layer-CAM (Jiang et al., IEEE TIP 2021):** per-location positive-gradient weighting; strong at shallow layers.
- **Score-CAM (Haofan Wang, Zifan Wang, Mengnan Du, Fan Yang, Zijian Zhang, Sirui Ding, Piotr Mardziel & Xia Hu, "Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks," CVPR Workshops 2020, pp. 111–119, DOI 10.1109/CVPRW50498.2020.00020):** "gets rid of the dependence on gradients by obtaining the weight of each activation map through its forward passing score on target class... it also passes the sanity check." Gradient-free ⇒ bypasses saturation. Cost: k forward passes per image.
- **Ablation-CAM (Saurabh Desai & Harish G. Ramaswamy, IIT Madras, "Ablation-CAM: Visual Explanations for Deep Convolutional Network via Gradient-free Localization," WACV 2020, pp. 983–991, DOI 10.1109/WACV45572.2020.9093360):** "uses ablation analysis to determine the importance (weights) of individual feature map units w.r.t. class... this gradient-free approach works better than state-of-the-art Grad-CAM." Weights come from the class-score drop when each channel is ablated. Gradient-free ⇒ bypasses saturation. Cost: k+1 forward passes per image.
- **Eigen-CAM:** first principal component of activations — no class discrimination but robust and a useful sanity check.

**Use the raw pre-softmax/pre-sigmoid logit as the target** (the Grad-CAM paper recommends the pre-softmax score). For a binary sigmoid head, ensure the target scalar is the logit for the predicted class, not the squashed probability, or gradients will be even more saturated.

### 3. Per-image min-max normalization and low-signal amplification

pytorch-grad-cam's `scale_cam_image` normalizes each map as `img = img - np.min(img); img = img / (1e-7 + np.max(img))`. When the pre-normalization map is nearly flat (saturation or a constant hook tensor), this divides by a near-zero range and **amplifies floating-point noise into a full-contrast [0,1] map** that looks structured. Combined with a 2×2 grid, the result is a deterministic, high-contrast corner blob. Diagnostics should always inspect the **raw, pre-normalization** CAM range (`max − min`): a range near machine epsilon confirms the low-signal/flat case. There is no per-image normalization flag that fixes underlying flatness; the fix is a better target layer and/or a saturation-robust method. (The library's non-CNN handling is documented in the jacobgil ViT tutorial and README; there is no dedicated "all-blue map" issue thread, but the flat-input → epsilon-division amplification is inherent to the `scale_cam_image` code shown above.)

### 4. Diagnostic protocol — RULE OUT THE STRUCTURAL BUG FIRST

Because the pattern is stable and input-independent, run these before changing CAM method. A fixed spatial pattern means the pipeline is reading a tensor that does not depend on the input at that spatial location — i.e., a constant/wrong hook, a detached activation, or a degenerate/scrambled grid.

**D0 — Does the CAM change when the input changes?**
```python
cam_a = cam(input_tensor=x1, targets=t)
cam_b = cam(input_tensor=x2, targets=t)
print(np.abs(cam_a - cam_b).max())   # ~0 => structural bug (constant tensor)
```

**D1 — Raw activation varies across inputs (hook on the right module):**
```python
acts = {}
h = target_layer.register_forward_hook(lambda m,i,o: acts.__setitem__('a', o.detach()))
model(x1); a1 = acts['a'].clone()
model(x2); a2 = acts['a'].clone()
print(target_layer.__class__.__name__, a1.shape)   # confirm module + shape (expect ~2x2!)
print((a1 - a2).abs().mean())                       # ~0 => hook on a constant/wrong module
```
Confirm the resolved module is the intended conv (not `avgpool`, not a ReLU wrapper) and that its spatial size is what you expect. A 2×2 shape here is itself the smoking gun for the resolution failure.

**D2 — Gradient magnitude at the target layer is nonzero:**
```python
logit = model(x1)[0, pred]      # use raw logit, not probability
model.zero_grad(); logit.backward()
print(grads.abs().mean(), grads.abs().max())   # ~0 => saturation; large => not saturation
```
Register a full-backward hook (`register_full_backward_hook`) and verify it fires. If gradients are large but the CAM is still fixed, the bug is geometric (reshape/upsampling), not saturation.

**D3 — requires_grad / eval-mode checks:** ensure the input tensor has `requires_grad=True` where needed, that you are NOT inside `torch.no_grad()`, and that activations captured for the CAM are not `.detach()`ed before the backward pass. Confirm `model.eval()` so BatchNorm uses running stats (BN in train mode with batch size 1 can give unstable/near-constant normalized activations).

**D4 — ViT-specific:** for vit_base, (i) confirm the model actually accepts 64×64 — torchvision `vit_b_16` asserts `Wrong image height! Expected 224 but got 64!`; timm asserts `Input height (64) doesn't match model (224).` unless `dynamic_img_size=True` or `img_size=64` (which interpolates the pretrained positional embedding from the 14×14 grid to 4×4); (ii) fix reshape_transform to the true grid. A 64×64/patch-16 ViT yields 4×4 = 16 spatial tokens (+1 CLS = 17), NOT 14×14 = 196. Use a dynamic reshape (the pattern the jacobgil docs recommend — set height/width to match the actual token grid):
```python
def reshape_transform(tensor):
    act = tensor[:, 1:, :]                       # drop CLS
    side = int(act.shape[1] ** 0.5)              # 4 for a 64x64/patch16 ViT
    act = act.reshape(act.shape[0], side, side, act.shape[2])
    return act.transpose(2, 3).transpose(1, 2)
```

**D5 — 2.5D channel ordering:** the 3 stacked axial slices are fed to ImageNet-pretrained RGB weights. Verify slice order and normalization (HU window −1000..400 → the same mean/std convention used in training) are identical between training and CAM inference; a channel-order mismatch degrades but would not by itself create a fixed spatial pattern — still worth confirming.

**D6 — Sanity check (Adebayo et al., NeurIPS 2018):** re-run the CAM with randomly re-initialized top-layer weights (model-parameter randomization test). If the heatmap is unchanged, the CAM is not reflecting the model — confirming the structural bug.

### 5. Concrete recommendation and prioritized experiment order

**Stage 1 — Rule out the structural bug (do this first; likely root cause).**
1. Run D0/D1/D2. Expected findings: raw activations DO vary (D1 nonzero) but the feature map is ~2×2, and/or gradients are near-zero (D2). If D0≈0 while D1≠0, the bug is in CAM assembly (reshape/upsample/normalization), not the model.
2. Fix the ViT reshape_transform (D4) and confirm the ViT even runs at 64×64. This alone should change vit_base behavior.
3. Verify no `no_grad`/detach/BN-train issues (D3).

**Stage 2 — Fix resolution (largest expected gain for CNNs).**
4. Move the target layer to an output-stride-8 stage (8×8 map) and switch to **Layer-CAM** there. Re-measure Dice/IoU/pointing_acc.
5. If still coarse, increase the input patch to 112×112 or 128×128 (last conv becomes 4×4/7×7) and/or fuse Layer-CAM across two stages.
6. Threshold note: a top-20% activation threshold over a 64×64 patch on a nodule <10% of area sets an IoU ceiling near ~0.3 even for a perfect map; consider an adaptive/percentile threshold matched to nodule size and report pointing-game accuracy as the primary metric.

**Stage 3 — Address gradient saturation (over-confident model).**
7. Swap GradCAM → **HiResCAM** (drop-in, same target layer) and compare.
8. Run **Ablation-CAM** and **Score-CAM** (gradient-free) as saturation-robust references; if they localize while Grad-CAM does not, saturation is confirmed as a contributing factor.
9. Optionally target the raw logit and/or add SmoothGrad-style input-noise averaging for stability.

**Decision thresholds:** If after Stage 1+2 pointing_acc rises well above 0 and Dice/IoU climb into a sensible range for a small-lesion patch, resolution was the dominant issue. If gradient-free methods (Stage 3) localize but gradient-based ones remain flat, saturation is the residual issue and you should report HiResCAM or Ablation-CAM as the primary XAI method. Best overall bet for this setting (small CT nodule, very confident model): **Layer-CAM or HiResCAM at an 8×8 intermediate layer**, with Ablation-CAM as a gradient-free cross-check.

## Recommendations
1. **First, prove the CAM responds to input** (D0/D1) and confirm the target-layer feature-map size. A ~2×2 map and/or an input-invariant CAM confirms the structural/resolution bug over saturation.
2. **Fix the ViT reshape_transform and input geometry immediately** — it is almost certainly independently broken at 64×64 (hard asserts + hardcoded 14×14 grid).
3. **Re-target CNNs to an 8×8 (output-stride-8) stage and adopt Layer-CAM**; alternatively enlarge the patch to 112–128 px.
4. **Only then compare HiResCAM, Ablation-CAM, and Score-CAM** to quantify and mitigate gradient saturation; prefer the pre-softmax/pre-sigmoid logit as the target.
5. **Report pointing-game accuracy as primary** and use size-matched thresholds; document the small-lesion IoU ceiling explicitly in the paper.

## Caveats
- Exact default target layers differ slightly by library (torchvision vs timm) and by how the model was assembled; verify the resolved module name programmatically.
- The ~2×2 figure assumes standard output-stride-32 backbones and a 64×64 input; if the pipeline changed strides or input size, recompute.
- HiResCAM's faithfulness guarantee holds strictly only for single-FC-head architectures; for deeper heads it is approximate.
- Score-CAM/Ablation-CAM cost many forward passes per image — heavier for a 6-backbone × 5-fold study; batch them.
- The exact `RuntimeError` string for the 17→14×14 reshape mismatch is inferred from PyTorch's standard reshape behavior (16·C ≠ 196·C elements) rather than a verbatim logged instance; the assertion strings for torchvision/timm are verbatim from source.
- Some cited comparison magnitudes come from adjacent domains (chest X-ray, brain CT, fundus) rather than LIDC nodules specifically; treat cross-domain numbers as directional.