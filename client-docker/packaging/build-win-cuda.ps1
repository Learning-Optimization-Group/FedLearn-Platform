# =============================================================================
# FedLearn Native Client — Windows x64 CUDA 12.4 Build
# =============================================================================
# Produces dist/fedlearn-client-cuda/fedlearn-client.exe with the CUDA 12.4
# torch wheel. Requires:
#   - Windows 10/11 x64 host
#   - Python 3.11+ on PATH
#   - NVIDIA GPU driver >= 550 installed (for CUDA 12.4 runtime compatibility)
#
# The driver itself is NOT bundled — drivers are system-level. The resulting
# exe runs on any Windows machine that has a compatible driver + NVIDIA GPU.
#
# Usage (PowerShell):
#   cd client-docker\packaging
#   .\build-win-cuda.ps1
# =============================================================================

$ErrorActionPreference = 'Stop'

$ScriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot     = Resolve-Path (Join-Path $ScriptDir '..\..')
$FrameworkDir = Join-Path $RepoRoot 'framework'
$VenvDir      = Join-Path $ScriptDir '.venv-win-cuda'

Write-Host "[build-win-cuda] Repo root:    $RepoRoot"
Write-Host "[build-win-cuda] Framework:    $FrameworkDir"
Write-Host "[build-win-cuda] Venv:         $VenvDir"

if (-not (Test-Path $VenvDir)) {
    Write-Host "[build-win-cuda] Creating fresh venv..."
    python -m venv $VenvDir
}

$PyExe  = Join-Path $VenvDir 'Scripts\python.exe'
$PipExe = Join-Path $VenvDir 'Scripts\pip.exe'

& $PyExe -m pip install --upgrade pip wheel setuptools

Write-Host "[build-win-cuda] Installing torch 2.5.1 (CUDA 12.4 wheel — ~2.5GB)..."
& $PipExe install torch==2.5.1 torchvision==0.20.1 `
    --index-url https://download.pytorch.org/whl/cu124

Write-Host "[build-win-cuda] Installing pinned runtime deps..."
& $PipExe install -r (Join-Path $ScriptDir 'requirements-client.txt')

Write-Host "[build-win-cuda] Installing fedlearn framework (editable, --no-deps)..."
# --no-deps preserves pinned versions from requirements-client.txt; the
# framework's requirements.txt pulls in flwr which would otherwise
# downgrade protobuf/numpy/transformers.
& $PipExe install -e $FrameworkDir --no-deps

Write-Host "[build-win-cuda] Running PyInstaller..."
Push-Location $ScriptDir
try {
    & $PyExe -m PyInstaller --clean --noconfirm fedlearn-client.spec
} finally {
    Pop-Location
}

$Out = Join-Path $ScriptDir 'dist\fedlearn-client\fedlearn-client.exe'
if (-not (Test-Path $Out)) {
    Write-Error "[build-win-cuda] Expected $Out to exist"
    exit 1
}

Write-Host "[build-win-cuda] Smoke test: $Out --help"
& $Out --help
if ($LASTEXITCODE -ne 0) {
    Write-Error "[build-win-cuda] Smoke test failed"
    exit 1
}

Write-Host "[build-win-cuda] ✓ Built fedlearn-client (CUDA)"
Write-Host "[build-win-cuda] ✓ Output: $(Join-Path $ScriptDir 'dist\fedlearn-client\fedlearn-client.exe')"
Write-Host ""
Write-Host "[build-win-cuda] Next: cd ..\..\fedlearn-desktop && npm run package:win:cuda"
