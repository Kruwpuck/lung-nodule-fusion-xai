@echo off
REM ============================================================================
REM  setup_env.bat  —  Lung Nodule Fusion XAI  (Windows CMD)
REM
REM  Automated environment setup for the lung-nodule-fusion-xai project.
REM  Detects CUDA/GPU availability, creates a virtual environment, installs
REM  pinned dependencies, and downloads dcm2niix.
REM
REM  Tested on: Windows 10/11, Python 3.10+, CUDA 12.x
REM  Usage:     Double-click or run from cmd:  setup_env.bat
REM ============================================================================
setlocal enabledelayedexpansion

set "CUDA_DETECTED=0"
set "PYMRMR_FAILED=0"
set "PYTHON_CMD="

echo.
echo ============================================================
echo   Lung Nodule Fusion XAI — Windows Environment Setup
echo ============================================================
echo.

REM --------------------------------------------------------------------------
REM  Step 1: Check Python
REM --------------------------------------------------------------------------
echo [1/10] Checking Python installation...

REM Try "python" first
python --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_CMD=python"
    goto :check_python_version
)

REM Try "py" launcher
py --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_CMD=py"
    goto :check_python_version
)

echo.
echo ERROR: Python is not installed or not found on PATH.
echo        Please download Python 3.10+ from:
echo        https://www.python.org/downloads/
echo.
echo        Make sure to check "Add Python to PATH" during installation.
echo.
exit /b 1

:check_python_version
REM Extract major.minor version and validate >= 3.10
for /f "tokens=2 delims= " %%V in ('!PYTHON_CMD! --version 2^>^&1') do set "PY_FULL_VER=%%V"
for /f "tokens=1,2 delims=." %%A in ("!PY_FULL_VER!") do (
    set "PY_MAJOR=%%A"
    set "PY_MINOR=%%B"
)

if !PY_MAJOR! lss 3 (
    echo ERROR: Python 3.10+ is required, but found Python !PY_FULL_VER!.
    echo        Download from: https://www.python.org/downloads/
    exit /b 1
)
if !PY_MAJOR! equ 3 if !PY_MINOR! lss 10 (
    echo ERROR: Python 3.10+ is required, but found Python !PY_FULL_VER!.
    echo        Download from: https://www.python.org/downloads/
    exit /b 1
)

echo        Found Python !PY_FULL_VER! using "!PYTHON_CMD!"
echo.

REM --------------------------------------------------------------------------
REM  Step 2: Create virtual environment
REM --------------------------------------------------------------------------
echo [2/10] Creating virtual environment (.venv)...

if exist ".venv\Scripts\activate.bat" (
    echo        .venv already exists, reusing it.
) else (
    !PYTHON_CMD! -m venv .venv
    if !errorlevel! neq 0 (
        echo ERROR: Failed to create virtual environment.
        exit /b 1
    )
)

call .venv\Scripts\activate.bat
echo        Virtual environment activated.
echo.

REM --------------------------------------------------------------------------
REM  Step 3: Upgrade pip
REM --------------------------------------------------------------------------
echo [3/10] Upgrading pip, wheel, and setuptools...
pip install --upgrade pip wheel setuptools >nul 2>&1
if !errorlevel! neq 0 (
    echo WARNING: pip upgrade had issues, continuing anyway...
) else (
    echo        Done.
)
echo.

REM --------------------------------------------------------------------------
REM  Step 4: Auto-detect CUDA and install PyTorch
REM --------------------------------------------------------------------------
echo [4/10] Detecting NVIDIA GPU and installing PyTorch...

nvidia-smi >nul 2>&1
if !errorlevel! equ 0 (
    set "CUDA_DETECTED=1"
    echo        NVIDIA GPU detected! Installing PyTorch with CUDA 12.1 support...
    pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
) else (
    echo        No NVIDIA GPU detected. Installing CPU-only PyTorch...
    pip install torch==2.3.0 torchvision==0.18.0
)

if !errorlevel! neq 0 (
    echo ERROR: PyTorch installation failed.
    exit /b 1
)
echo        PyTorch installed successfully.
echo.

REM --------------------------------------------------------------------------
REM  Step 5: Install project dependencies
REM --------------------------------------------------------------------------
echo [5/10] Installing project dependencies...

REM Install all packages EXCEPT pymrmr first
pip install ^
    monai==1.3.2 ^
    timm==1.0.3 ^
    pydicom==2.4.4 ^
    SimpleITK==2.3.1 ^
    nibabel==5.2.1 ^
    pylidc==0.2.2 ^
    pyradiomics==3.1.0 ^
    scikit-learn==1.5.0 ^
    xgboost==2.0.3 ^
    lightgbm==4.3.0 ^
    pandas==2.2.2 ^
    pyarrow==16.1.0 ^
    numpy==1.26.4 ^
    scipy==1.13.0 ^
    shap==0.45.1 ^
    grad-cam==1.5.3 ^
    captum==0.7.0 ^
    matplotlib==3.9.0 ^
    seaborn==0.13.2 ^
    opencv-python==4.9.0.80 ^
    scikit-plot==0.3.7 ^
    lungmask==0.2.19 ^
    pyyaml==6.0.1 ^
    tqdm==4.66.4 ^
    statsmodels==0.14.2 ^
    mlflow==2.13.0 ^
    pytest==8.2.2 ^
    pytest-cov==5.0.0

if !errorlevel! neq 0 (
    echo ERROR: Dependency installation failed.
    exit /b 1
)
echo        Core dependencies installed.

REM Try installing pymrmr separately (often fails on Windows)
echo        Attempting to install pymrmr...
pip install pymrmr==0.1.3 >nul 2>&1
if !errorlevel! neq 0 (
    set "PYMRMR_FAILED=1"
    echo.
    echo WARNING: pymrmr failed to install. This package requires C++ compilation.
    echo          Install Microsoft Visual C++ Build Tools from:
    echo          https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo          Then re-run: pip install pymrmr==0.1.3
    echo          Continuing without pymrmr...
) else (
    echo        pymrmr installed successfully.
)
echo.

REM --------------------------------------------------------------------------
REM  Step 6: Auto-download dcm2niix
REM --------------------------------------------------------------------------
echo [6/10] Downloading dcm2niix...

set "DCM2NIIX_URL=https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_win.zip"
set "DCM2NIIX_ZIP=dcm2niix_win.zip"

curl -L -o "!DCM2NIIX_ZIP!" "!DCM2NIIX_URL!" >nul 2>&1
if !errorlevel! neq 0 (
    echo WARNING: Failed to download dcm2niix. You can install it manually later.
    echo          URL: !DCM2NIIX_URL!
    goto :skip_dcm2niix
)

REM Extract to .venv\Scripts so dcm2niix is on PATH when venv is active
tar -xf "!DCM2NIIX_ZIP!" -C ".venv\Scripts" >nul 2>&1
if !errorlevel! neq 0 (
    echo WARNING: Failed to extract dcm2niix. You can install it manually later.
    goto :skip_dcm2niix
)

del "!DCM2NIIX_ZIP!" >nul 2>&1
echo        dcm2niix installed to .venv\Scripts\
echo.

:skip_dcm2niix

REM --------------------------------------------------------------------------
REM  Step 7: Install project as editable
REM --------------------------------------------------------------------------
echo [7/10] Installing project as editable package...
pip install -e . >nul 2>&1
if !errorlevel! neq 0 (
    echo WARNING: Editable install failed. Check setup.py / pyproject.toml.
) else (
    echo        Done.
)
echo.

REM --------------------------------------------------------------------------
REM  Step 8: Create data directories
REM --------------------------------------------------------------------------
echo [8/10] Creating data directories...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\interim" mkdir "data\interim"
if not exist "data\processed" mkdir "data\processed"
if not exist "results" mkdir "results"
echo        Created: data\raw, data\interim, data\processed, results
echo.

REM --------------------------------------------------------------------------
REM  Step 9: Freeze requirements
REM --------------------------------------------------------------------------
echo [9/10] Freezing installed packages...
pip freeze > requirements.lock
echo        Saved to requirements.lock
echo.

REM --------------------------------------------------------------------------
REM  Step 10: Summary
REM --------------------------------------------------------------------------
echo [10/10] Setup complete!
echo.
echo ============================================================
echo   SETUP SUMMARY
echo ============================================================
echo.
echo   Activate environment:  .venv\Scripts\activate
echo   Run tests:             pytest tests/ -v
echo.
if !CUDA_DETECTED! equ 1 (
    echo   GPU:  NVIDIA GPU detected — PyTorch installed with CUDA 12.1
) else (
    echo   GPU:  No NVIDIA GPU — PyTorch installed for CPU only
)
if !PYMRMR_FAILED! equ 1 (
    echo.
    echo   WARNING: pymrmr is NOT installed.
    echo            Install Visual C++ Build Tools first:
    echo            https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo            Then run: pip install pymrmr==0.1.3
)
echo.
echo ============================================================
echo.

endlocal
