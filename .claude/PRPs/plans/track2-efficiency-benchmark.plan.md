# Plan: Track 2 — Lightweight-vs-Heavyweight Efficiency Benchmark

## Summary
Track 1 (fusion + XAI) is implemented. This plan covers the missing **Track 2**:
prove a small-param model can match a large-param one. Add 3 missing backbones,
a params/FLOPs measurement module, and the two "efficiency" scatter plots
(Params-vs-AUC, FLOPs-vs-AUC) that are the headline evidence.

## User Story
As a researcher, I want each candidate model benchmarked on the same split with
params/FLOPs/latency recorded, so that I can show efficiency wins with a scatter
plot and a DeLong-tested AUC comparison.

## Problem → Solution
Only 5 backbones exist (all mid-size), no efficiency measurement, no plots →
full lightweight↔heavyweight spectrum + `efficiency.py` metrics + benchmark table
+ scatter plots.

## Metadata
- **Complexity**: Medium
- **Source PRD**: `RESEARCH_PLAN.md` (Track 2, §2.1–2.6)
- **PRD Phase**: Track 2 — Model Comparison
- **Estimated Files**: 4 (2 new, 2 edited) + 1 notebook

---

## UX Design
Internal / research change — no user-facing UX. Output artifacts:
`results/efficiency_table.csv`, `results/params_vs_auc.png`, `results/flops_vs_auc.png`.

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 | `src/models/backbones.py` | 8–129 | Exact pattern for adding a backbone (conv-patch + emb_dim) |
| P0 | `src/evaluation/metrics.py` | all | `compute_metrics` / `bootstrap_ci` return shapes to reuse |
| P1 | `src/evaluation/statistical_tests.py` | all | `delong_test`, `build_ablation_table` to reuse for model compare |
| P1 | `configs/train.yaml` | 35–53 | `backbones:` list schema to extend |
| P2 | `src/training/trainer.py` | 117–183 | `run_kfold_cv` — how a benchmark run is driven |

## External Documentation

| Topic | Source | Key Takeaway |
|---|---|---|
| FLOPs count | `ptflops.get_model_complexity_info` | Returns (macs, params); FLOPs≈2×MACs. One call, no custom hooks. |
| ViT channel patch | torchvision `vit_b_16` | First layer is `conv_proj` (not `features[0][0]`); patch that for n_input_channels. |
| VGG channel patch | torchvision `vgg16` | First layer `features[0]`; classifier expects 224×224 → keep input 224. |

```
KEY_INSIGHT: ptflops already computes both params and FLOPs in one call.
APPLIES_TO: efficiency.py
GOTCHA: it needs input_res as (C,H,W); pass (n_slices, 224, 224) for 2.5D.
```

---

## Patterns to Mirror

### ADD_BACKBONE (conv-patch + emb_dim)
```python
# SOURCE: src/models/backbones.py:34-44 (efficientnet_b0 branch)
elif name == "efficientnet_b0":
    weights = "IMAGENET1K_V1" if pretrained else None
    model = tvm.efficientnet_b0(weights=weights)
    old_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        n_input_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size, stride=old_conv.stride,
        padding=old_conv.padding, bias=old_conv.bias is not None,
    )
    features = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
    emb_dim = 1280
```

### METRICS_RETURN
```python
# SOURCE: src/evaluation/metrics.py (compute_metrics)
m = compute_metrics(y_true, y_prob)   # dict: auc, accuracy, sensitivity, ...
lo, hi = bootstrap_ci(y_true, y_prob, n_iterations=2000)
```

### DELONG_COMPARE
```python
# SOURCE: src/evaluation/statistical_tests.py
z, p, delta = delong_test(y_true, y_prob_a, y_prob_b)
```

### TEST_STRUCTURE
```python
# SOURCE: tests/test_models.py:1-17
import pytest
torch = pytest.importorskip("torch", reason="torch not installed")
from src.models.backbones import BackboneClassifier
```

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `src/models/backbones.py` | UPDATE | Add `mobilenet_v3_small`, `vgg16`, `vit_b_16` branches |
| `src/evaluation/efficiency.py` | CREATE | params/FLOPs/latency measurement + plot helpers |
| `configs/train.yaml` | UPDATE | Add 3 backbones to `backbones:` list |
| `tests/test_efficiency.py` | CREATE | One `importorskip` test for count_params/flops |
| `notebooks/track2_benchmark.ipynb` | CREATE | Drives all backbones, builds table + 2 scatter plots |

## NOT Building
- ResNet-101, EfficientNet-B1 — spectrum already covered by the 3 added + existing 5. Add later if a data point is needed.
- Custom FLOPs counter — ptflops does it.
- Multi-seed automation — single seed first; loop the notebook cell manually if mean±std needed.
- New training loop — reuse `run_kfold_cv`.

---

## Step-by-Step Tasks

### Task 1: Add 3 backbones
- **ACTION**: Add branches to `build_2d_backbone` in `src/models/backbones.py`.
- **IMPLEMENT**:
  - `mobilenet_v3_small` → patch `model.features[0][0]`, `emb_dim = 576`.
  - `vgg16` → patch `model.features[0]`, take `model.features` + `AdaptiveAvgPool2d(1)` + `Flatten`, `emb_dim = 512`.
  - `vit_b_16` → patch `model.conv_proj` for n_input_channels; replace `model.heads` with `nn.Identity()`, features = model, emb_dim = 768. Requires 224×224 input.
- **MIRROR**: ADD_BACKBONE pattern above.
- **IMPORTS**: `import torchvision.models as tvm` (already lazy inside fn).
- **GOTCHA**: VGG/ViT need 224×224 input, not 64×64 — note in config `input_res: 224`. MobileNetV3-Small emb_dim is 576 (not 960).
- **VALIDATE**: `BackboneClassifier("mobilenet_v3_small").forward(torch.randn(2,3,224,224)).shape == (2,2)`.

### Task 2: Efficiency module
- **ACTION**: Create `src/evaluation/efficiency.py`.
- **IMPLEMENT**:
  - `count_params(model) -> int` — `sum(p.numel() for p in model.parameters())`. (stdlib, no dep)
  - `measure_flops(model, input_res) -> tuple[float,int]` — lazy `from ptflops import get_model_complexity_info`; return (gflops, params). Fallback: if ptflops missing, return (nan, count_params).
  - `measure_latency(model, input_res, n=50, device="cpu") -> float` — warm up 5, time n forward passes, return mean ms.
  - `build_efficiency_table(results: dict) -> pd.DataFrame` — columns: model, params_M, gflops, latency_ms, auc, auc_ci_low, auc_ci_high.
  - `plot_params_vs_auc(df, out_png)` and `plot_flops_vs_auc(df, out_png)` — matplotlib scatter, log-x, annotate points.
- **MIRROR**: METRICS_RETURN for auc/ci columns.
- **IMPORTS**: `import numpy as np, pandas as pd`; matplotlib + ptflops lazy inside fns.
- **GOTCHA**: FLOPs ≈ 2×MACs; ptflops returns MACs — multiply. Keep torch/ptflops imports lazy so tests run without them.
- **VALIDATE**: `count_params(BackboneClassifier("efficientnet_b0"))` ≈ 5.3e6.

### Task 3: Config
- **ACTION**: Append to `backbones:` in `configs/train.yaml`.
- **IMPLEMENT**: 3 entries with `input_mode: 2_5d`, `pretrained: true`, plus `input_res: 224` for vgg16/vit_b_16.
- **VALIDATE**: `yaml.safe_load` parses without error.

### Task 4: Test
- **ACTION**: Create `tests/test_efficiency.py`.
- **IMPLEMENT**: `importorskip("torch")`; assert `count_params` > 0 and returns int for a small `BackboneClassifier`; assert `build_efficiency_table` yields expected columns from a fake results dict.
- **MIRROR**: TEST_STRUCTURE.
- **VALIDATE**: `pytest tests/test_efficiency.py` passes or skips (no torch).

### Task 5: Benchmark notebook
- **ACTION**: Create `notebooks/track2_benchmark.ipynb` (Colab, GPU).
- **IMPLEMENT**: loop over 8 backbones → `run_kfold_cv` (or load Phase-1 checkpoints) → collect y_true/y_prob → `compute_metrics` + `bootstrap_ci` → `measure_flops`/`count_params`/`measure_latency` → `build_efficiency_table` → 2 scatter plots + `delong_test` on best-light vs best-heavy. Save all to `MyDrive/results/`.
- **GOTCHA**: keep a `SKIP_TRAINING = True` flag like phase1/phase2 notebooks so structure runs without GPU hours.
- **VALIDATE**: notebook JSON valid (`nbformat.read`).

---

## Testing Strategy

### Unit Tests
| Test | Input | Expected | Edge? |
|---|---|---|---|
| count_params returns int | `BackboneClassifier("mobilenet_v3_small")` | >0 int | no |
| efficiency table columns | fake results dict | has model,params_M,gflops,auc | no |
| new backbone forward | `randn(2,3,224,224)` | `(2,2)` | 224-input |

### Edge Cases Checklist
- [ ] ptflops not installed → `measure_flops` returns (nan, params), no crash
- [ ] n_input_channels != 3 on ViT (conv_proj patch)
- [ ] VGG/ViT fed 64×64 → assert-guard or documented 224 requirement

---

## Validation Commands

### Unit Tests
```bash
python3 -m pytest tests/ -q --tb=short
```
EXPECT: all pass, torch-dependent tests skip if torch absent.

### Notebook validity
```bash
python3 -c "import nbformat,glob; [nbformat.read(f,4) for f in glob.glob('notebooks/*.ipynb')]"
```
EXPECT: no exceptions.

### Config parse
```bash
python3 -c "import yaml; print(yaml.safe_load(open('configs/train.yaml'))['backbones'])"
```
EXPECT: 8 backbones listed.

---

## Acceptance Criteria
- [ ] 3 new backbones build and forward correctly
- [ ] `efficiency.py` measures params + FLOPs + latency
- [ ] Efficiency table + 2 scatter plots generated in notebook
- [ ] DeLong test compares lightest vs heaviest
- [ ] Tests pass/skip cleanly

## Completion Checklist
- [ ] Follows ADD_BACKBONE pattern exactly (conv patch + emb_dim)
- [ ] torch/ptflops/matplotlib imports lazy (tests run bare)
- [ ] Reuses `compute_metrics`, `bootstrap_ci`, `delong_test` — no reinvention
- [ ] No new heavy deps beyond ptflops
- [ ] Self-contained

## Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ViT conv_proj patch subtlety | Med | Med | Replace `heads` with Identity, keep conv_proj 3-ch, stack 2.5D→3ch |
| VGG overfits ~656 patches | High | Med | Heavy aug + dropout (RESEARCH_PLAN §caveat); report honestly |
| 224-input backbones slow on Colab T4 | Med | Low | SKIP_TRAINING flag; benchmark subset first |

## Notes
- MobileNetV3-Small emb_dim = 576, VGG16 = 512, ViT-B/16 = 768.
- ptflops is the only new dependency; add to `requirements.txt` when implementing.
- Track 1 already ships the shared eval/stat tooling — this plan only wires Track 2 on top.
