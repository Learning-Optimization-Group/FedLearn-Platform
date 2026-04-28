# =============================================================================
# FedLearn Native Client — Windows x64 CPU Build
# =============================================================================
# Produces dist/fedlearn-client/fedlearn-client.exe with the CPU-only torch
# wheel. Run on a Windows 10/11 x64 host with Python 3.11+ installed and on
# PATH. PyInstaller does NOT cross-compile — this must run on Windows.
#
# Usage (PowerShell):
#   cd client-docker\packaging
#   .\build-win-cpu.ps1
# =============================================================================

$ErrorActionPreference = 'Stop'

$ScriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot     = Resolve-Path (Join-Path $ScriptDir '..\..')
$FrameworkDir = Join-Path $RepoRoot 'framework'
$VenvDir      = Join-Path $ScriptDir '.venv-win-cpu'

Write-Host "[build-win-cpu] Repo root:    $RepoRoot"
Write-Host "[build-win-cpu] Framework:    $FrameworkDir"
Write-Host "[build-win-cpu] Venv:         $VenvDir"

if (-not (Test-Path $VenvDir)) {
    Write-Host "[build-win-cpu] Creating fresh venv..."
    python -m venv $VenvDir
}

$PyExe = Join-Path $VenvDir 'Scripts\python.exe'
$PipExe = Join-Path $VenvDir 'Scripts\pip.exe'

& $PyExe -m pip install --upgrade pip wheel setuptools

Write-Host "[build-win-cpu] Installing torch 2.5.1 (CPU wheel)..."
& $PipExe install torch==2.5.1 torchvision==0.20.1 `
    --index-url https://download.pytorch.org/whl/cpu

Write-Host "[build-win-cpu] Installing pinned runtime deps..."
& $PipExe install -r (Join-Path $ScriptDir 'requirements-client.txt')

Write-Host "[build-win-cpu] Installing fedlearn framework (editable, --no-deps)..."
# --no-deps preserves pinned versions from requirements-client.txt; the
# framework's requirements.txt pulls in flwr which would otherwise
# downgrade protobuf/numpy/transformers.
& $PipExe install -e $FrameworkDir --no-deps

Write-Host "[build-win-cpu] Running PyInstaller..."
Push-Location $ScriptDir
try {
    & $PyExe -m PyInstaller --clean --noconfirm fedlearn-client.spec
} finally {
    Pop-Location
}

$Out = Join-Path $ScriptDir 'dist\fedlearn-client\fedlearn-client.exe'
if (-not (Test-Path $Out)) {
    Write-Error "[build-win-cpu] Expected $Out to exist"
    exit 1
}

Write-Host "[build-win-cpu] Smoke test: $Out --help"
& $Out --help
if ($LASTEXITCODE -ne 0) {
    Write-Error "[build-win-cpu] Smoke test failed"
    exit 1
}

Write-Host "[build-win-cpu] ✓ Built fedlearn-client (CPU)"
Write-Host "[build-win-cpu] ✓ Output: $(Join-Path $ScriptDir 'dist\fedlearn-client\fedlearn-client.exe')"
Write-Host ""
Write-Host "[build-win-cpu] Next: cd ..\..\fedlearn-desktop && npm run package:win:cpu"
