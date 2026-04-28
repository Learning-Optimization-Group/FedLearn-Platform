#!/usr/bin/env bash
# =============================================================================
# FedLearn Native Client — macOS Build
# =============================================================================
# Produces dist/fedlearn-client/ with torch 2.5.1. On Apple Silicon the wheel
# includes MPS acceleration; on x64 Macs it's CPU-only (Apple dropped CUDA
# support years ago). PyPI serves the correct wheel automatically based on
# `uname -m`, so the pip command is arch-agnostic.
#
# Run on a Mac (arm64 or x86_64) with Python 3.11+ installed.
#
# Output: client-docker/packaging/dist/fedlearn-client/fedlearn-client
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FRAMEWORK_DIR="$REPO_ROOT/framework"

ARCH="$(uname -m)"
# Arch-suffixed venv avoids x64/arm64 cross-contamination if a developer
# (rare) builds both variants on the same machine via Rosetta.
VENV_DIR="$SCRIPT_DIR/.venv-mac-$ARCH"

echo "[build-mac] Arch:         $ARCH"
echo "[build-mac] Repo root:    $REPO_ROOT"
echo "[build-mac] Framework:    $FRAMEWORK_DIR"
echo "[build-mac] Venv:         $VENV_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[build-mac] Creating fresh venv..."
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

echo "[build-mac] Installing torch 2.5.1 (MPS-enabled, default index)..."
pip install "torch==2.5.1" "torchvision==0.20.1"

echo "[build-mac] Installing pinned runtime deps..."
pip install -r "$SCRIPT_DIR/requirements-client.txt"

echo "[build-mac] Installing fedlearn framework (editable, --no-deps to preserve pins)..."
# --no-deps keeps our explicit pins from being clobbered by the framework's
# requirements.txt (which pulls in flwr + friends that downgrade protobuf,
# numpy, and transformers).
pip install -e "$FRAMEWORK_DIR" --no-deps

echo "[build-mac] Running PyInstaller..."
cd "$SCRIPT_DIR"
pyinstaller --clean --noconfirm fedlearn-client.spec

OUT="$SCRIPT_DIR/dist/fedlearn-client/fedlearn-client"
if [[ ! -x "$OUT" ]]; then
  echo "[build-mac] ERROR: expected $OUT to exist and be executable" >&2
  exit 1
fi

echo "[build-mac] Smoke test: $OUT --help"
"$OUT" --help || {
  echo "[build-mac] ERROR: smoke test failed" >&2
  exit 1
}

SIZE=$(du -sh "$SCRIPT_DIR/dist/fedlearn-client" | cut -f1)
echo "[build-mac] ✓ Built fedlearn-client (bundle size: $SIZE)"
echo "[build-mac] ✓ Output: $SCRIPT_DIR/dist/fedlearn-client/"
