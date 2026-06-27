#!/usr/bin/env bash
# setup_env.sh — Lung Nodule Fusion XAI
# Tested: Ubuntu 22.04 LTS / 24.04 LTS, CUDA 12.x
set -euo pipefail

PYTHON=${PYTHON:-python3.11}
CUDA_URL="https://download.pytorch.org/whl/cu121"

echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    build-essential git curl \
    libsm6 libxrender1 libxext6 libgomp1 \
    dcm2niix

echo "[2/6] Creating Python virtual environment..."
$PYTHON -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools

echo "[3/6] Installing PyTorch (CUDA 12.1)..."
pip install torch==2.3.0 torchvision==0.18.0 --index-url $CUDA_URL

echo "[4/6] Installing project dependencies..."
pip install \
    monai==1.3.2 \
    timm==1.0.3 \
    pydicom==2.4.4 \
    SimpleITK==2.3.1 \
    nibabel==5.2.1 \
    pylidc==0.2.2 \
    pyradiomics==3.1.0 \
    scikit-learn==1.5.0 \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    pandas==2.2.2 \
    pyarrow==16.1.0 \
    numpy==1.26.4 \
    scipy==1.13.0 \
    pymrmr==0.1.3 \
    shap==0.45.1 \
    grad-cam==1.5.3 \
    captum==0.7.0 \
    matplotlib==3.9.0 \
    seaborn==0.13.2 \
    "opencv-python==4.9.0.80" \
    scikit-plot==0.3.7 \
    lungmask==0.2.19 \
    pyyaml==6.0.1 \
    tqdm==4.66.4 \
    statsmodels==0.14.2 \
    mlflow==2.13.0 \
    pytest==8.2.2 \
    pytest-cov==5.0.0

echo "[5/6] Installing project as editable package..."
pip install -e .

echo "[6/6] Freezing environment..."
pip freeze > requirements.lock

echo ""
echo "Setup complete!"
echo "Activate with: source .venv/bin/activate"
echo "Run tests with: pytest tests/ -v"
