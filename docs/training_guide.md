# Training Guide — Lung Nodule Fusion XAI

Two environments: **Google Colab** (GPU, no setup required) and **local Ubuntu LTS** (from `setup_env.sh`). Use Colab for all actual model training. Use local only for data prep, debugging, or if you have a local GPU.

---

## Quick Reference

| Step | Notebook | Needs GPU | Estimated Time (Colab T4) |
|---|---|---|---|
| 1 — Data pipeline | `data_pipeline.ipynb` | No | ~2–4 h (LIDC extraction) |
| 2 — Segmentation | `segmentation.ipynb` | No | ~30 min |
| 3 — Radiomics extraction | `radiomics_extraction.ipynb` | No | ~4–8 h |
| 4 — CNN benchmark | `phase1_cnn_benchmark.ipynb` | **Yes** | ~3–6 h per backbone |
| 5 — Fusion | `phase2_fusion.ipynb` | **Yes** | ~2–4 h |
| 6 — XAI | `xai_analysis.ipynb` | Yes (optional) | ~30 min |
| 7 — Evaluation | `evaluation.ipynb` | No | ~15 min |
| 8 — External validation | `external_validation.ipynb` | No | ~1 h |

> Steps 1–3 and 6–8 can run on Colab CPU. Steps 4–5 require GPU — use Colab T4 (free) or A100 (Colab Pro).

---

## Part 1 — Google Colab

### 1.1 One-time setup

**1. Download LIDC-IDRI from TCIA**

Go to [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/lidc-idri/) and download the full LIDC-IDRI dataset (~124 GB). Put the extracted folder on your Google Drive:

```
MyDrive/
└── LIDC-IDRI/
    ├── LIDC-IDRI-0001/
    ├── LIDC-IDRI-0002/
    └── ...
```

**2. Clone the repo to Colab**

Every notebook starts with this cell — it auto-clones if the path doesn't exist:

```python
from google.colab import drive
drive.mount('/content/drive')

import subprocess, os
REPO_PATH = '/content/lung-nodule-fusion-xai'
if not os.path.exists(REPO_PATH):
    subprocess.run([
        'git', 'clone',
        'https://github.com/YOUR_USERNAME/lung-nodule-fusion-xai.git',
        REPO_PATH
    ], check=True)
os.chdir(REPO_PATH)
```

Replace `YOUR_USERNAME` with your GitHub username after pushing the repo.

**3. Runtime type**

For steps 4–5 (CNN and fusion training):
- `Runtime` → `Change runtime type` → `T4 GPU` (free) or `A100` (Pro)

For all other steps: CPU is fine, leave as-is.

---

### 1.2 Persist outputs to Drive

Colab VMs are ephemeral. All output goes to `/content/drive/MyDrive/` so it survives session resets. The notebooks already handle this — intermediate data writes to:

```
MyDrive/lung_nodule_interim/     ← pylidc patches + masks (.npy)
MyDrive/lung_nodule_processed/   ← labels.csv, radiomic_features.parquet
MyDrive/checkpoints/             ← model weights per fold
MyDrive/results/                 ← metrics CSVs, plots
```

> If a Colab session disconnects mid-extraction, re-run the notebook from the top. The loaders check for existing files and skip already-processed nodules.

---

### 1.3 Step-by-step Colab run

#### Step 1 — Data pipeline (`data_pipeline.ipynb`)

Opens LIDC-IDRI DICOM, runs pylidc consensus at `clevel=0.5`, derives binary labels (median rating: ≤2 benign, ≥4 malignant, exclude score=3), splits into patient-level 5-fold CV.

Expected output:
```
Total nodules: ~656
Malignant: ~352 | Benign: ~304
Saved: MyDrive/lung_nodule_processed/labels.csv
```

If pylidc can't find scans, fix the `~/.pylidcrc` cell:
```python
cfg['pylidc'] = {'dicom_path': '/content/drive/MyDrive/LIDC-IDRI'}
```

---

#### Step 2 — Segmentation (`segmentation.ipynb`)

Validates consensus masks (Route A) and optionally runs lungmask (Route B). No training.

Run the ICC comparison cell to get `icc_route_comparison.csv` before proceeding to Step 3 — the feature selection pipeline uses it.

---

#### Step 3 — Radiomics extraction (`radiomics_extraction.ipynb`)

Extracts ~107–2000 features per nodule (Original + LoG σ=[1,2,3] + Wavelet). The full extraction on ~656 nodules takes **4–8 hours on Colab CPU**.

To avoid re-running: the extractor caches to `radiomic_features.parquet`. Set `force_rebuild=False` (default) to load cache on re-run.

After extraction, the notebook runs `ICC → mRMR → LASSO` on fold 0 as a demonstration. The full nested CV feature selection runs inside the fusion training loop (Step 5).

Expected output shape: `(656, ~1500)` before selection, `(656, ≤30)` after LASSO.

---

#### Step 4 — CNN benchmark (`phase1_cnn_benchmark.ipynb`)

**Requires GPU.**

To actually train, find this cell and set the flag:

```python
SKIP_TRAINING = False   # ← change from True to False
```

Trains 6 backbones (MobileNetV3, EfficientNet-B0, ResNet-50, DenseNet-121, ConvNeXt-Tiny, 3D-ResNet) with 5-fold CV, 50 epochs each, early stopping at patience=10.

Training config (`configs/train.yaml`):
```yaml
training:
  epochs: 50
  batch_size: 16       # fits T4 for 2.5D 64×64
  learning_rate: 1.0e-4
  early_stopping_patience: 10
  mixed_precision: true
```

For A100 (Colab Pro), increase `batch_size` to 32–64 for faster throughput.

Checkpoints save per fold:
```
MyDrive/checkpoints/mobilenet_v3_large/fold0_best.pt
MyDrive/checkpoints/mobilenet_v3_large/fold1_best.pt
...
```

Results save to:
```
MyDrive/results/phase1/phase1_metrics.csv
```

At the end of Step 4, pick the **top-2 backbones by mean cross-validated AUC**. Use those in Step 5.

---

#### Step 5 — Fusion (`phase2_fusion.ipynb`)

**Requires GPU.**

Set:
```python
SKIP_TRAINING = False
BEST_BACKBONE = 'mobilenet_v3_large'  # replace with actual top-1 from Step 4
N_RADIOMIC = 20  # replace with actual LASSO-selected feature count from Step 3
```

Trains 3 fusion variants:
- **Intermediate** (end-to-end `FusionNet`): CNN embedding + radiomic vector → shared dense layers
- **Early** (XGBoost): concatenated CNN deep features + radiomic features → XGBoost
- **Late** (probability averaging + stacking): independent CNN and XGBoost outputs averaged

Each variant runs full 5-fold CV. Results write to `MyDrive/results/phase2/ablation.csv`.

The DeLong test cell computes p-values for fusion vs. best single modality. **If p ≥ 0.05, report it honestly** — the XAI cross-validation is still the contribution.

---

#### Step 6 — XAI (`xai_analysis.ipynb`)

Load best checkpoint from Step 5 (intermediate fusion or best Phase 1 backbone):

```python
import torch
model.load_state_dict(torch.load('/content/drive/MyDrive/checkpoints/BEST/fold0_best.pt'))
```

Generates:
- Grad-CAM / Grad-CAM++ / Score-CAM overlays for 10 random test nodules
- TreeSHAP global beeswarm on XGBoost early-fusion branch
- Waterfall plot for 3 individual nodule explanations
- Spatial cross-check: Dice of Grad-CAM high-activation region vs. nodule mask

Key cell to watch:
```python
result = spatial_cross_check(shap_vals, feature_names, cam_maps, nodule_masks)
print(f"Spurious activation flag: {result['spurious_flag']}")
# Flag = True if Grad-CAM peak outside nodule >30% of cases
```

All plots save to `MyDrive/results/xai/`.

---

#### Step 7 — Evaluation (`evaluation.ipynb`)

Aggregates all fold results, produces:
- Full ablation table (9 model variants: 6 CNNs + radiomics-only + 3 fusion types)
- DeLong test for every fusion vs. best-single comparison
- Calibration curves + Brier scores
- Bootstrap 95% CI (n=2000 iterations)

Load results from saved CSVs:
```python
import pandas as pd
phase1 = pd.read_csv('/content/drive/MyDrive/results/phase1/phase1_metrics.csv')
phase2 = pd.read_csv('/content/drive/MyDrive/results/phase2/ablation.csv')
```

Output: `MyDrive/results/evaluation/ablation_table.csv`, `calibration_curves.png`.

---

#### Step 8 — External validation (`external_validation.ipynb`)

Requires NSCLC-Radiomics (TCIA) or an NLST subset. Upload to `MyDrive/external_data/`.

Runs the **frozen** best pipeline (no retraining) on external CT, reports AUC + ΔAUC vs. internal. Also runs Route B (lungmask) on external CTs and computes feature ICC.

If NSCLC-Radiomics is not yet available, the notebook runs with synthetic data to validate the pipeline code. Replace with real data before reporting final numbers.

---

## Part 2 — Local Ubuntu Environment

Use this for: data preprocessing, debugging individual modules, running unit tests, or if you have a local NVIDIA GPU.

### 2.1 Install

```bash
git clone https://github.com/YOUR_USERNAME/lung-nodule-fusion-xai.git
cd lung-nodule-fusion-xai
chmod +x setup_env.sh
./setup_env.sh
source .venv/bin/activate
```

`setup_env.sh` installs PyTorch 2.3 (CUDA 12.1), MONAI, PyRadiomics, SHAP, grad-cam, and all dependencies into `.venv/`.

For CPU-only machines (no GPU), install PyTorch CPU build instead:
```bash
pip install torch==2.3.0 torchvision==0.18.0
```

### 2.2 Configure pylidc

After placing LIDC-IDRI data locally (e.g. at `/data/LIDC-IDRI`):

```bash
python3 - <<'EOF'
import configparser, os
cfg = configparser.ConfigParser()
cfg['pylidc'] = {'dicom_path': '/data/LIDC-IDRI'}
with open(os.path.expanduser('~/.pylidcrc'), 'w') as f:
    cfg.write(f)
print("pylidc configured")
EOF
```

### 2.3 Run data pipeline locally

```bash
source .venv/bin/activate

python3 - <<'EOF'
from src.data_loading.lidc_loader import load_and_split

df = load_and_split(
    lidc_path='/data/LIDC-IDRI',
    interim_path='data/interim',
    processed_path='data/processed',
    n_folds=5,
    seed=42,
)
print(f"Nodules: {len(df)} | Malignant: {(df.label==1).sum()} | Benign: {(df.label==0).sum()}")
EOF
```

### 2.4 Run radiomics extraction locally

```bash
python3 - <<'EOF'
import pandas as pd
from src.radiomics.extraction import extract_dataset_features

df = pd.read_csv('data/processed/labels.csv')
features = extract_dataset_features(
    df,
    params_yaml='configs/radiomics_params.yaml',
    output_parquet='data/processed/radiomic_features.parquet',
)
print(f"Feature matrix: {features.shape}")
EOF
```

To parallelize across CPU cores:
```bash
# Split by fold and run in parallel (optional for faster extraction)
for fold in 0 1 2 3 4; do
    python3 -c "
import pandas as pd
from src.radiomics.extraction import extract_dataset_features
df = pd.read_csv('data/processed/labels.csv')
df = df[df.fold == $fold]
extract_dataset_features(df, output_parquet='data/processed/radiomic_features_fold${fold}.parquet')
" &
done
wait
```

### 2.5 Train on local GPU

```bash
python3 - <<'EOF'
import pandas as pd, torch, yaml
from torch.utils.data import DataLoader
from src.training.dataset import NoduleDataset2_5D
from src.training.trainer import run_kfold_cv
from src.models.backbones import BackboneClassifier

cfg = yaml.safe_load(open('configs/train.yaml'))
df = pd.read_csv('data/processed/labels.csv')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {device}")

def model_factory():
    return BackboneClassifier('mobilenet_v3_large', n_input_channels=3, pretrained=True)

def dataset_factory(sub_df, augment):
    ds = NoduleDataset2_5D(sub_df, augment=augment)
    return DataLoader(ds, batch_size=cfg['training']['batch_size'],
                      shuffle=augment, num_workers=4, pin_memory=True)

fold_results = run_kfold_cv(
    model_factory=model_factory,
    dataset_factory=dataset_factory,
    labels_df=df,
    epochs=cfg['training']['epochs'],
    device_str=device,
    checkpoint_dir='results/checkpoints/mobilenet_v3_large',
)
print("Done")
EOF
```

### 2.6 Run tests

```bash
# Pure-Python tests (no GPU/LIDC needed):
pytest tests/test_data_loading.py tests/test_preprocessing.py \
       tests/test_segmentation.py tests/test_evaluation.py tests/test_xai.py -v

# All tests (requires torch):
pytest tests/ -v

# With coverage:
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Common Issues

**`No module named 'pylidc'`**
→ Run `pip install pylidc==0.2.2` or use the Colab install cell.

**`No module named 'radiomics'`**
→ Run `pip install pyradiomics==3.1.0`.

**CUDA OOM on T4 during CNN training**
→ Lower `batch_size` in `configs/train.yaml` from 16 to 8. Or switch to A100 (Colab Pro).

**Colab session disconnects mid-extraction**
→ Re-run from the top — all extractors check for cache files and resume.

**`~/.pylidcrc` not found / wrong path**
→ Re-run the pylidc config cell with the correct DICOM path.

**Feature count after LASSO < 5**
→ Increase `mrmr_n` in `configs/train.yaml` or lower `icc_threshold` from 0.75 to 0.65.

**DeLong test p ≥ 0.05 for fusion**
→ Expected in some runs. Report it honestly; pivot contribution to XAI cross-validation and robustness analysis.

---

## Experiment Tracking (optional)

MLflow is included in requirements. To enable:

```bash
# Start MLflow UI
mlflow ui --port 5000 &

# In training scripts, add:
import mlflow
mlflow.set_experiment("lung-nodule-fusion")
mlflow.log_params(cfg)
mlflow.log_metric("fold_auc", auc, step=fold)
```

Access at `http://localhost:5000`.

---

## Checklist Before Reporting Results

- [ ] Patient-level splits confirmed (no patient in >1 fold)
- [ ] Feature scaler fit on training folds only (not test)
- [ ] DeLong test computed for all fusion vs. best-single comparisons
- [ ] Bootstrap 95% CI reported for all AUC values
- [ ] External validation run on frozen pipeline (no retraining)
- [ ] IBSI deviations documented (see `configs/radiomics_params.yaml` comments)
- [ ] PyRadiomics version pinned in `requirements.txt`
- [ ] Grad-CAM spatial cross-check run; spurious activation flag reported
