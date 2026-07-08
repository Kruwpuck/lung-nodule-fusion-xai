#Requires -Version 5.1
<#
.SYNOPSIS
    Lung Nodule Fusion XAI — Windows PowerShell Environment Setup

.DESCRIPTION
    Automated environment setup for the lung-nodule-fusion-xai project.
    Detects CUDA/GPU availability, creates a Python virtual environment,
    installs pinned dependencies, and downloads dcm2niix.

    Tested on: Windows 10/11, PowerShell 5.1+/7+, Python 3.10+, CUDA 12.x

.USAGE
    Open PowerShell and run:
        .\setup_env.ps1

    If you get an execution policy error, run:
        Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
#>

$ErrorActionPreference = 'Stop'

# --- State variables ---
$CudaDetected = $false
$PymrmrFailed = $false
$PythonCmd = $null

# --- Helper functions ---
function Write-Step {
    param([string]$Step, [string]$Message)
    Write-Host "`n[$Step] " -ForegroundColor Cyan -NoNewline
    Write-Host $Message
}

function Write-Ok {
    param([string]$Message)
    Write-Host "       $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "       WARNING: $Message" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Message)
    Write-Host "       ERROR: $Message" -ForegroundColor Red
}

# ============================================================================
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Lung Nodule Fusion XAI — Windows Environment Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# --------------------------------------------------------------------------
#  Step 1: Check Python
# --------------------------------------------------------------------------
Write-Step "1/10" "Checking Python installation..."

# Try "python" first, then "py"
$pythonCandidates = @("python", "py")
foreach ($candidate in $pythonCandidates) {
    try {
        $verOutput = & $candidate --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $PythonCmd = $candidate
            break
        }
    } catch {
        # Command not found, try next
    }
}

if (-not $PythonCmd) {
    Write-Err "Python is not installed or not found on PATH."
    Write-Host ""
    Write-Host "       Please download Python 3.10+ from:" -ForegroundColor Yellow
    Write-Host "       https://www.python.org/downloads/" -ForegroundColor White
    Write-Host ""
    Write-Host "       Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    exit 1
}

# Parse and validate version (>= 3.10)
$verString = (& $PythonCmd --version 2>&1) -replace 'Python\s+', ''
$verParts = $verString.Split('.')
$major = [int]$verParts[0]
$minor = [int]$verParts[1]

if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
    Write-Err "Python 3.10+ is required, but found Python $verString."
    Write-Host "       Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

Write-Ok "Found Python $verString using '$PythonCmd'"

# --------------------------------------------------------------------------
#  Step 2: Create virtual environment
# --------------------------------------------------------------------------
Write-Step "2/10" "Creating virtual environment (.venv)..."

if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Ok ".venv already exists, reusing it."
} else {
    & $PythonCmd -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to create virtual environment."
        exit 1
    }
}

# Activate the virtual environment
& .\.venv\Scripts\Activate.ps1
Write-Ok "Virtual environment activated."

# --------------------------------------------------------------------------
#  Step 3: Upgrade pip
# --------------------------------------------------------------------------
Write-Step "3/10" "Upgrading pip, wheel, and setuptools..."

pip install --upgrade pip wheel setuptools 2>&1 | Out-Null
Write-Ok "Done."

# --------------------------------------------------------------------------
#  Step 4: Auto-detect CUDA and install PyTorch
# --------------------------------------------------------------------------
Write-Step "4/10" "Detecting NVIDIA GPU and installing PyTorch..."

try {
    nvidia-smi 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $CudaDetected = $true
    }
} catch {
    $CudaDetected = $false
}

if ($CudaDetected) {
    Write-Ok "NVIDIA GPU detected! Installing PyTorch with CUDA 12.1 support..."
    pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Ok "No NVIDIA GPU detected. Installing CPU-only PyTorch..."
    pip install torch==2.3.0 torchvision==0.18.0
}

if ($LASTEXITCODE -ne 0) {
    Write-Err "PyTorch installation failed."
    exit 1
}
Write-Ok "PyTorch installed successfully."

# --------------------------------------------------------------------------
#  Step 5: Install project dependencies
# --------------------------------------------------------------------------
Write-Step "5/10" "Installing project dependencies..."

$packages = @(
    "monai==1.3.2",
    "timm==1.0.3",
    "pydicom==2.4.4",
    "SimpleITK==2.3.1",
    "nibabel==5.2.1",
    "pylidc==0.2.2",
    "pyradiomics==3.1.0",
    "scikit-learn==1.5.0",
    "xgboost==2.0.3",
    "lightgbm==4.3.0",
    "pandas==2.2.2",
    "pyarrow==16.1.0",
    "numpy==1.26.4",
    "scipy==1.13.0",
    "shap==0.45.1",
    "grad-cam==1.5.3",
    "captum==0.7.0",
    "matplotlib==3.9.0",
    "seaborn==0.13.2",
    "opencv-python==4.9.0.80",
    "scikit-plot==0.3.7",
    "lungmask==0.2.19",
    "pyyaml==6.0.1",
    "tqdm==4.66.4",
    "statsmodels==0.14.2",
    "mlflow==2.13.0",
    "pytest==8.2.2",
    "pytest-cov==5.0.0"
)

pip install @packages
if ($LASTEXITCODE -ne 0) {
    Write-Err "Dependency installation failed."
    exit 1
}
Write-Ok "Core dependencies installed."

# Try installing pymrmr separately (often fails on Windows without C++ compiler)
Write-Host "       Attempting to install pymrmr..." -ForegroundColor Gray
try {
    $ErrorActionPreference = 'Continue'
    pip install pymrmr==0.1.3 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "pip returned non-zero" }
    $ErrorActionPreference = 'Stop'
    Write-Ok "pymrmr installed successfully."
} catch {
    $ErrorActionPreference = 'Stop'
    $PymrmrFailed = $true
    Write-Host ""
    Write-Warn "pymrmr failed to install. This package requires C++ compilation."
    Write-Host "       Install Microsoft Visual C++ Build Tools from:" -ForegroundColor Yellow
    Write-Host "       https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor White
    Write-Host "       Then re-run: pip install pymrmr==0.1.3" -ForegroundColor Yellow
    Write-Host "       Continuing without pymrmr..." -ForegroundColor Yellow
}

# --------------------------------------------------------------------------
#  Step 6: Auto-download dcm2niix
# --------------------------------------------------------------------------
Write-Step "6/10" "Downloading dcm2niix..."

$dcm2niixUrl = "https://github.com/rordenlab/dcm2niix/releases/latest/download/dcm2niix_win.zip"
$dcm2niixZip = "dcm2niix_win.zip"

try {
    # Download using .NET WebClient (works on all PowerShell versions)
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $dcm2niixUrl -OutFile $dcm2niixZip -UseBasicParsing

    # Extract to .venv\Scripts so dcm2niix is on PATH when venv is active
    Expand-Archive -Path $dcm2niixZip -DestinationPath ".venv\Scripts" -Force
    Remove-Item $dcm2niixZip -Force

    Write-Ok "dcm2niix installed to .venv\Scripts\"
} catch {
    Write-Warn "Failed to download or extract dcm2niix. You can install it manually later."
    Write-Host "       URL: $dcm2niixUrl" -ForegroundColor Gray
    if (Test-Path $dcm2niixZip) { Remove-Item $dcm2niixZip -Force -ErrorAction SilentlyContinue }
}

# --------------------------------------------------------------------------
#  Step 7: Install project as editable
# --------------------------------------------------------------------------
Write-Step "7/10" "Installing project as editable package..."

try {
    pip install -e . 2>&1 | Out-Null
    Write-Ok "Done."
} catch {
    Write-Warn "Editable install failed. Check setup.py / pyproject.toml."
}

# --------------------------------------------------------------------------
#  Step 8: Create data directories
# --------------------------------------------------------------------------
Write-Step "8/10" "Creating data directories..."

$dirs = @("data\raw", "data\interim", "data\processed", "results")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}
Write-Ok "Created: data\raw, data\interim, data\processed, results"

# --------------------------------------------------------------------------
#  Step 9: Freeze requirements
# --------------------------------------------------------------------------
Write-Step "9/10" "Freezing installed packages..."

pip freeze > requirements.lock
Write-Ok "Saved to requirements.lock"

# --------------------------------------------------------------------------
#  Step 10: Summary
# --------------------------------------------------------------------------
Write-Step "10/10" "Setup complete!"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  SETUP SUMMARY" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Activate environment: " -NoNewline; Write-Host ".venv\Scripts\activate" -ForegroundColor White
Write-Host "  Run tests:            " -NoNewline; Write-Host "pytest tests/ -v" -ForegroundColor White
Write-Host ""

if ($CudaDetected) {
    Write-Host "  GPU:  " -NoNewline
    Write-Host "NVIDIA GPU detected — PyTorch installed with CUDA 12.1" -ForegroundColor Green
} else {
    Write-Host "  GPU:  " -NoNewline
    Write-Host "No NVIDIA GPU — PyTorch installed for CPU only" -ForegroundColor Yellow
}

if ($PymrmrFailed) {
    Write-Host ""
    Write-Host "  ⚠ pymrmr is NOT installed." -ForegroundColor Yellow
    Write-Host "    Install Visual C++ Build Tools first:" -ForegroundColor Yellow
    Write-Host "    https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor White
    Write-Host "    Then run: pip install pymrmr==0.1.3" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
