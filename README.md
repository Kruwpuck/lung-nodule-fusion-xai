# Lung Nodule Fusion XAI

**Explainable Radiomics-Deep Learning Fusion for Lung Nodule Malignancy Classification on LIDC-IDRI CT Scans**

A research project combining CNN deep features with handcrafted radiomic features for lung nodule malignancy classification, with explainability provided through Grad-CAM and SHAP.

---

## Key Features

- 🧠 **Multi-backbone CNN benchmark** — 6 architectures (MobileNetV3, EfficientNet-B0, ResNet-50, DenseNet-121, ConvNeXt-Tiny, 3D-ResNet)
- 🔗 **3 fusion strategies** — Early (XGBoost), Intermediate (end-to-end FusionNet), Late (probability averaging + stacking)
- 🔍 **Explainable AI** — Grad-CAM / Grad-CAM++ spatial maps + TreeSHAP feature attributions with spatial cross-check
- 📊 **Full 5-fold cross-validation** — Patient-level splits, DeLong tests, bootstrap 95% CI
- 🏥 **External validation pipeline** — Frozen pipeline evaluation on LUNA16 / NSCLC-Radiomics

---

## Project Structure

```
lung-nodule-fusion-xai/
├── configs/              # Training & radiomics configuration
├── data/
│   ├── raw/              # ← Place LIDC-IDRI dataset here
│   ├── interim/          # Auto-generated (patches, masks)
│   └── processed/        # Auto-generated (labels, features)
├── docs/                 # Training guide & documentation
├── notebooks/            # Jupyter notebooks (Colab-ready)
├── results/              # Auto-generated (metrics, plots, checkpoints)
├── src/                  # Source code
│   ├── data_loading/     # LIDC & LUNA16 data loaders
│   ├── evaluation/       # Metrics, statistical tests
│   ├── fusion/           # Early, intermediate, late fusion
│   ├── models/           # CNN backbones & FusionNet
│   ├── preprocessing/    # CT preprocessing pipeline
│   ├── radiomics/        # Feature extraction & selection
│   ├── segmentation/     # Consensus & auto-segmentation
│   ├── training/         # Dataset & trainer
│   └── xai/              # Grad-CAM & SHAP utilities
├── tests/                # Unit tests
├── setup_env.sh          # Linux/macOS setup
├── setup_env.bat         # Windows CMD setup
├── setup_env.ps1         # Windows PowerShell setup
└── requirements.txt
```

---

## Quick Start

### Linux / macOS

```bash
git clone https://github.com/Kruwpuck/lung-nodule-fusion-xai.git
cd lung-nodule-fusion-xai
chmod +x setup_env.sh
./setup_env.sh
source .venv/bin/activate
```

### Windows (CMD)

```cmd
git clone https://github.com/Kruwpuck/lung-nodule-fusion-xai.git
cd lung-nodule-fusion-xai
setup_env.bat
.venv\Scripts\activate
```

### Windows (PowerShell)

```powershell
git clone https://github.com/Kruwpuck/lung-nodule-fusion-xai.git
cd lung-nodule-fusion-xai
.\setup_env.ps1
.venv\Scripts\Activate.ps1
```

> The setup scripts automatically detect CUDA/GPU availability and install the appropriate PyTorch version. On Windows, `dcm2niix` is also downloaded automatically.

---

## Dataset Setup

1. **LIDC-IDRI (Required)** — Download from [TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/) (~124 GB)
   - Place in `data/raw/LIDC-IDRI/`
2. **LUNA16 (Optional)** — Download from [LUNA16 Grand Challenge](https://luna16.grand-challenge.org/Download/) (~90 GB)
   - Place in `data/raw/LUNA16/`

```
data/raw/
├── LIDC-IDRI/
│   ├── LIDC-IDRI-0001/
│   ├── LIDC-IDRI-0002/
│   └── ...
└── LUNA16/               # optional
    ├── subset0/ ... subset9/
    ├── annotations.csv
    └── candidates_V2.csv
```

> **Note**: `data/raw/` is git-ignored. Each collaborator must download datasets independently.

For detailed dataset placement, environment setup, CUDA configuration on Windows, and step-by-step training instructions, see **[Training Guide](docs/training_guide.md)**.

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Documentation

- 📖 **[Training Guide](docs/training_guide.md)** — Complete guide for Google Colab, Linux, and Windows environments
  - Dataset placement (LIDC-IDRI & LUNA16)
  - CUDA setup for Windows
  - Step-by-step pipeline execution
  - Common issues & troubleshooting

---

## License

TBD