# Plan: Resumable Training Pipeline

## Summary
Wrap all existing src/ modules in a checkpoint-based, resumable pipeline.
Each stage writes its output to disk and skips on re-run if output already exists.
All metrics flush to CSV per epoch so plots/analysis never need retraining.

## User Story
As a researcher running on Colab Free, I want each pipeline stage to be independently
resumable, so that a runtime crash or session reset doesn't force me to restart
preprocessing or training from scratch.

## Problem → Solution
Existing code has all the logic (loader, trainer, XAI) but no stage scripts, no
CSVLogger, no resume logic, no orchestrator → wrap into 6 stage `.py` files
+ utils + single `config.yaml` + `run_local.sh` + `run_colab.ipynb`.

## Metadata
- **Complexity**: Large
- **Source PRD**: `TRAINING_PIPELINE_PLAN.md`
- **PRD Phase**: Full pipeline
- **Estimated Files**: 12 new, 1 updated

---

## UX Design
N/A — internal change. CLI: `python src/stage_03_train.py --config configs/config.yaml --model mobilenet_v3_small --fold 0`

---

## Mandatory Reading

| Priority | File | Lines | Why |
|---|---|---|---|
| P0 | `src/training/trainer.py` | all | `run_kfold_cv`, `EarlyStopping`, `train_one_epoch`, `evaluate` — stage_03 calls these |
| P0 | `src/data_loading/lidc_loader.py` | 31–150 | `load_and_split` — stage_00 calls this |
| P0 | `src/radiomics/extraction.py` | all | `extract_dataset_features` — stage_01 calls this |
| P1 | `src/models/backbones.py` | 8–129 | `BackboneClassifier` + all backbone names — registry wraps |
| P1 | `src/training/dataset.py` | all | `NoduleDataset2_5D` constructor args — stage_03 wires this |
| P1 | `src/xai/gradcam_utils.py` | all | `compute_gradcam` — stage_05 calls |
| P1 | `src/evaluation/metrics.py` | all | `compute_metrics`, `bootstrap_ci` — stage_04 calls |
| P2 | `configs/train.yaml` | all | Existing hyperparams to migrate into config.yaml |

---

## Patterns to Mirror

### CACHE_SKIP
```python
# SOURCE: TRAINING_PIPELINE_PLAN.md (blueprint)
def cached(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

def run(cfg):
    out = cfg['paths']['patches']
    if cached(out) and not cfg.get('force_rerun', False):
        print(f"[SKIP] {out}"); return
    # ... do work ...
    print(f"[DONE] {out}")
```

### CSV_LOGGER
```python
# SOURCE: TRAINING_PIPELINE_PLAN.md (blueprint)
class CSVLogger:
    def __init__(self, path, fieldnames):
        new = not os.path.exists(path)
        self.f = open(path, 'a', newline='')
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        if new:
            self.writer.writeheader()

    def log(self, row):
        self.writer.writerow(row)
        self.f.flush()   # flush every row — safe against hang

    def close(self):
        self.f.close()
```

### CHECKPOINT_SAVE_RESUME
```python
# SOURCE: TRAINING_PIPELINE_PLAN.md (blueprint)
def save_ckpt(path, model, optimizer, epoch, best_auc):
    import torch
    torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(), 'best_auc': best_auc}, path)

def maybe_resume(path, model, optimizer):
    import torch, os
    if os.path.exists(path):
        ck = torch.load(path, weights_only=True, map_location='cpu')
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optim_state'])
        return ck['epoch'] + 1, ck['best_auc']
    return 0, 0.0
```

### STAGE_CLI
```python
# SOURCE: TRAINING_PIPELINE_PLAN.md (blueprint)
import argparse, yaml

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/config.yaml')
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg)

if __name__ == '__main__':
    main()
```

### LOGGING_PATTERN
```python
# SOURCE: src/data_loading/lidc_loader.py:14-15
import logging
logger = logging.getLogger(__name__)
logger.info("Total nodules: %d", len(df))
```

---

## Files to Change

| File | Action | Justification |
|---|---|---|
| `src/utils/__init__.py` | CREATE | Package marker |
| `src/utils/io.py` | CREATE | `cached()` helper |
| `src/utils/logger.py` | CREATE | `CSVLogger` |
| `src/utils/seed.py` | CREATE | `fix_seed(seed)` |
| `src/models/registry.py` | CREATE | `build_model(name, cfg)` wrapper |
| `configs/config.yaml` | CREATE | Unified single-source config |
| `src/stage_00_preprocess.py` | CREATE | DICOM → patches (resumable) |
| `src/stage_01_radiomics.py` | CREATE | patches → radiomics parquet (resumable) |
| `src/stage_02_split.py` | CREATE | k-fold split → folds.json |
| `src/stage_03_train.py` | CREATE | train 1 model × 1 fold (resume + CSV log) |
| `src/stage_04_evaluate.py` | CREATE | load best.pt → metrics → summary.csv |
| `src/stage_05_xai.py` | CREATE | Grad-CAM + SHAP (reads best.pt) |
| `run_local.sh` | CREATE | Local orchestrator (loops model × fold) |
| `run_colab.ipynb` | CREATE | Colab orchestrator (Drive-backed artifacts) |
| `src/training/trainer.py` | UPDATE | Add `save_ckpt`, `maybe_resume`, optional CSV epoch logging |

## NOT Building
- MLflow / W&B — CSV is the spec
- Parallel multi-GPU — single GPU per fold is enough
- Docker — out of scope
- Hyperparameter search — manual config.yaml edits

---

## Step-by-Step Tasks

### Task 1: src/utils/ package
- **ACTION**: Create `src/utils/__init__.py` (empty), `io.py`, `logger.py`, `seed.py`.
- **IMPLEMENT**:
  - `io.py`: `cached(path) -> bool` = `os.path.exists(path) and os.path.getsize(path) > 0`
  - `logger.py`: `CSVLogger(path, fieldnames)` — append mode, check exists BEFORE open to know if header needed, `log(row)` flushes, `close()`.
  - `seed.py`: `fix_seed(seed=42)` — sets `random.seed`, `numpy.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic=True`.
- **MIRROR**: CSV_LOGGER pattern.
- **IMPORTS**: `csv, os` (io, logger); `random, numpy as np` (seed); `torch` lazy in seed.py.
- **GOTCHA**: Check `os.path.exists(path)` BEFORE `open(path, 'a')` — once opened for append the file exists even if empty.
- **VALIDATE**: `python3 -c "from src.utils.io import cached; import tempfile,os; f=tempfile.mktemp(); open(f,'w').write('x'); assert cached(f); os.unlink(f); assert not cached(f); print('OK')"`.

### Task 2: src/models/registry.py
- **ACTION**: Create thin name-mapping wrapper over `BackboneClassifier`.
- **IMPLEMENT**:
  ```python
  _NAME_MAP = {
      "mobilenetv3_small": "mobilenet_v3_small",
      "mobilenetv3_large": "mobilenet_v3_large",
      "efficientnet_b0":   "efficientnet_b0",
      "densenet121":       "densenet121",
      "convnext_tiny":     "convnext_tiny",
      "resnet50":          "resnet50",
      "vgg16":             "vgg16",
      "vit_base":          "vit_b_16",
  }

  def build_model(name: str, cfg: dict):
      from src.models.backbones import BackboneClassifier
      backbone = _NAME_MAP.get(name, name)
      n_slices = cfg.get('data', {}).get('n_slices', 3)
      return BackboneClassifier(backbone, n_input_channels=n_slices, pretrained=True)
  ```
- **GOTCHA**: config uses "mobilenetv3_small" (no underscore) but backbones.py key is "mobilenet_v3_small".
- **VALIDATE**: `python3 -c "from src.models.registry import _NAME_MAP; assert len(_NAME_MAP)==8; print('OK')"`.

### Task 3: configs/config.yaml
- **ACTION**: Create unified config; keep `radiomics_params.yaml` intact (PyRadiomics reads it directly).
- **IMPLEMENT**:
  ```yaml
  seed: 42
  force_rerun: false

  paths:
    raw: "./data/raw/LIDC-IDRI"
    interim: "./artifacts/patches"
    features: "./artifacts/features/radiomics.parquet"
    splits: "./artifacts/splits/folds.json"
    checkpoints: "./artifacts/checkpoints"
    logs: "./artifacts/logs"
    results: "./artifacts/results"

  data:
    n_slices: 3
    patch_xy: 64
    n_folds: 5

  train:
    epochs: 50
    batch_size: 16
    lr: 0.0001
    weight_decay: 0.0001
    early_stopping_patience: 10
    checkpoint_every: 5
    mixed_precision: true

  models:
    lightweight: ["mobilenetv3_small", "efficientnet_b0", "densenet121"]
    heavyweight: ["resnet50", "vgg16", "vit_base"]

  track1_fusion:
    enabled: true
    backbone: "mobilenetv3_small"

  radiomics:
    params_yaml: "configs/radiomics_params.yaml"
  ```
- **GOTCHA**: Colab paths will differ — the notebook cell overwrites `paths.interim` etc. to Drive paths before calling stage scripts.
- **VALIDATE**: `python3 -c "import yaml; cfg=yaml.safe_load(open('configs/config.yaml')); assert 'train' in cfg; print('OK')"`.

### Task 4: Update src/training/trainer.py
- **ACTION**: Add `save_ckpt` + `maybe_resume` as module-level functions; add optional `log_dir` param to `run_kfold_cv` that creates a `CSVLogger` per fold.
- **IMPLEMENT**:
  - Add after imports (but keep torch imports lazy per existing pattern):
    ```python
    def save_ckpt(path: str, model, optimizer, epoch: int, best_auc: float) -> None:
        import torch
        torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict(), 'best_auc': best_auc}, path)

    def maybe_resume(path: str, model, optimizer):
        import torch, os
        if not os.path.exists(path):
            return 0, 0.0
        ck = torch.load(path, weights_only=True, map_location='cpu')
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optim_state'])
        return ck['epoch'] + 1, ck['best_auc']
    ```
  - Add `log_dir: str = None` to `run_kfold_cv` signature.
  - Inside fold loop: if `log_dir` is set, create `CSVLogger(f"{log_dir}/{fold}.csv", ['epoch','train_loss','val_auc','time_sec'])` and call `csv.log(...)` each epoch.
- **MIRROR**: CHECKPOINT_SAVE_RESUME pattern.
- **GOTCHA**: `weights_only=True` requires PyTorch ≥ 2.0 — wrap in try/except with fallback to `torch.load(path)` if TypeError.
- **VALIDATE**: `python3 -m pytest tests/ -q --tb=short` — all 38+ still pass.

### Task 5: Six stage scripts
- **ACTION**: Create `src/stage_00_preprocess.py` through `src/stage_05_xai.py`.
- **IMPLEMENT**: Each follows STAGE_CLI + CACHE_SKIP pattern. Lazy imports inside `run()`.

  **stage_00** calls `lidc_loader.load_and_split`; output = `{interim}/labels.csv`.

  **stage_01** calls `radiomics.extraction.extract_dataset_features`; output = `{features}` parquet.

  **stage_02** reads labels.csv fold col → writes `{splits}/folds.json` mapping fold→row indices.

  **stage_03** (additional CLI args `--model`, `--fold`):
  - Reads labels.csv, builds train/val split by fold.
  - Calls `build_model`, wraps `train_one_epoch` + `evaluate` per epoch loop.
  - Uses `maybe_resume(last.pt)` at start; saves `last.pt` every `checkpoint_every` epochs, `best.pt` on AUC improvement.
  - Skips if `start_epoch >= cfg['train']['epochs']` (already complete).
  - Logs per epoch via `CSVLogger` to `{logs}/{model}_fold{fold}.csv`.

  **stage_04** loops all models × folds, loads `best.pt`, calls `evaluate` + `compute_metrics` + `bootstrap_ci` + `count_params` + `measure_flops`; writes `{results}/summary.csv`.

  **stage_05** loads best checkpoint for Track 1 backbone; runs `compute_gradcam` on 10 val samples; saves CAM PNGs to `{results}/xai/`.

- **MIRROR**: CACHE_SKIP + STAGE_CLI + LOGGING_PATTERN.
- **GOTCHA**: `NoduleDataset2_5D` — verify constructor signature in `src/training/dataset.py` before Task 5 (read it). Don't assume args.
- **VALIDATE**: Each script: `python3 src/stage_XX.py --config configs/config.yaml` prints `[SKIP]` or clean error (not ImportError).

### Task 6: Orchestrators
- **ACTION**: Create `run_local.sh` and `run_colab.ipynb`.
- **IMPLEMENT run_local.sh**:
  ```bash
  #!/bin/bash
  set -e
  CFG=configs/config.yaml
  python src/stage_00_preprocess.py --config $CFG
  python src/stage_01_radiomics.py  --config $CFG
  python src/stage_02_split.py      --config $CFG
  for model in mobilenetv3_small efficientnet_b0 densenet121 resnet50 vgg16 vit_base; do
    for fold in 0 1 2 3 4; do
      python src/stage_03_train.py --config $CFG --model $model --fold $fold
    done
  done
  python src/stage_04_evaluate.py --config $CFG
  python src/stage_05_xai.py      --config $CFG
  ```
- **IMPLEMENT run_colab.ipynb**: 7 cells. Cell 1 mounts Drive, creates Drive artifact dir, symlinks `./artifacts` → Drive. Cells 2–7 = one `!python src/stage_XX...` per stage. Cell 5 has `model = "mobilenetv3_small"` variable + loop comment. Colab Drive path = `/content/drive/MyDrive/Kanker Kanker apa yg Kanker/LungFuseNet_artifacts`.
- **GOTCHA**: `os.makedirs(drive_artifact_dir, exist_ok=True)` before `os.symlink` — symlink fails if target doesn't exist.
- **VALIDATE**: `bash -n run_local.sh` (syntax check); `python3 -c "import json; json.load(open('run_colab.ipynb')); print('OK')"`.

---

## Testing Strategy

### Unit Tests (add to tests/test_utils.py)
| Test | Input | Expected | Edge? |
|---|---|---|---|
| `cached()` non-empty file | real tmp file | True | no |
| `cached()` missing path | `/tmp/nope` | False | no |
| `cached()` 0-byte file | empty tmp file | False | yes |
| `CSVLogger` header once | 2 log() calls | 1 header + 2 data rows | no |
| `CSVLogger` append (resume) | open existing | no second header | yes |
| `fix_seed` reproducibility | seed=0, randn twice | equal tensors | no |
| `build_model` all 8 names | each name in _NAME_MAP | no ValueError | no |

### Edge Cases Checklist
- [ ] `cached()` with 0-byte file → False (truncated write guard)
- [ ] `maybe_resume` with missing `last.pt` → (0, 0.0), no crash
- [ ] stage_03 start_epoch >= epochs → prints [SKIP], exits cleanly
- [ ] CSVLogger second open on existing file → no duplicate header

---

## Validation Commands

### Tests
```bash
python3 -m pytest tests/ -q --tb=short
```
EXPECT: 38+ passed, torch tests skipped.

### Config
```bash
python3 -c "import yaml; cfg=yaml.safe_load(open('configs/config.yaml')); print(list(cfg.keys()))"
```
EXPECT: `['seed', 'force_rerun', 'paths', 'data', 'train', 'models', 'track1_fusion', 'radiomics']`

### Shell syntax
```bash
bash -n run_local.sh
```
EXPECT: no output.

### Notebook JSON
```bash
python3 -c "import json; json.load(open('run_colab.ipynb')); print('OK')"
```
EXPECT: `OK`.

### Stage import check
```bash
python3 -c "import src.stage_00_preprocess, src.stage_03_train; print('imports OK')"
```
EXPECT: `imports OK` (lazy imports mean no torch/pylidc needed at import time).

---

## Acceptance Criteria
- [ ] `src/utils/{io,logger,seed}.py` implemented and tested
- [ ] `src/models/registry.py` maps all 8 model names correctly
- [ ] `configs/config.yaml` parses cleanly
- [ ] `src/training/trainer.py` has `save_ckpt`, `maybe_resume`, optional CSV logging
- [ ] 6 stage scripts with `--config` CLI, cache-skip, lazy imports
- [ ] `run_local.sh` passes `bash -n`
- [ ] `run_colab.ipynb` valid JSON
- [ ] All existing tests pass

## Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `NoduleDataset2_5D` arg mismatch in stage_03 | Med | Med | Read dataset.py (Task 5 GOTCHA) |
| trainer.py edit breaks 38 existing tests | Low | Med | Run tests after every edit |
| Colab symlink fails if Drive dir missing | High | Low | `os.makedirs` before `os.symlink` |
| `weights_only=True` TypeError on old PyTorch | Med | Low | try/except fallback |

## Notes
- All stage script imports are lazy (inside `run()`) — `python src/stage_00 --config ...` starts instantly without torch/pylidc installed.
- `stage_02` (folds.json) is arguably redundant since labels.csv has `fold` column — keep it for explicit auditability, but it's the simplest stage.
- `run_local.sh` is sequential by design; parallelise with `&` + `wait` only if local multi-GPU is available later.
- Keep `configs/radiomics_params.yaml` unchanged — PyRadiomics reads it directly.
