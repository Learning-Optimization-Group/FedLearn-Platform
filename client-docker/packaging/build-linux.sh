#!/usr/bin/env bash
# =============================================================================
# FedLearn Native Client — Linux Build (CPU)
# =============================================================================
# Produces dist/fedlearn-client/ with the CPU torch wheel. Works on x86_64 and
# aarch64 hosts — PyPI serves the matching manylinux wheel automatically.
#
# CUDA on Linux is intentionally out of scope: x86_64+NVIDIA users can use the
# Docker path (fedlearn-client:latest image), and aarch64+NVIDIA means Jetson,
# which requires the L4T-pinned torch wheel (also Docker-only).
#
# Run on a Linux host with Python 3.11+ installed.
#
# Output: client-docker/packaging/dist/fedlearn-client/fedlearn-client
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FRAMEWORK_DIR="$REPO_ROOT/framework"

ARCH="$(uname -m)"
VENV_DIR="$SCRIPT_DIR/.venv-linux-$ARCH"

echo "[build-linux] Arch:         $ARCH"
echo "[build-linux] Repo root:    $REPO_ROOT"
echo "[build-linux] Framework:    $FRAMEWORK_DIR"
echo "[build-linux] Venv:         $VENV_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[build-linux] Creating fresh venv..."
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

echo "[build-linux] Installing torch 2.5.1 (CPU wheel)..."
# pytorch.org/whl/cpu only serves x86_64 wheels. For aarch64 we use the
# default PyPI, which publishes manylinux_2_17_aarch64 wheels for torch 2.5.1
# (CPU-only — CUDA on aarch64 linux is Jetson-only and goes via Docker).
if [[ "$ARCH" == "x86_64" ]]; then
  pip install "torch==2.5.1" "torchvision==0.20.1" \
      --index-url https://download.pytorch.org/whl/cpu
else
  pip install "torch==2.5.1" "torchvision==0.20.1"
fi

echo "[build-linux] Installing pinned runtime deps..."
pip install -r "$SCRIPT_DIR/requirements-client.txt"

echo "[build-linux] Installing fedlearn framework (editable, --no-deps to preserve pins)..."
pip install -e "$FRAMEWORK_DIR" --no-deps

echo "[build-linux] Running PyInstaller..."
cd "$SCRIPT_DIR"
pyinstaller --clean --noconfirm fedlearn-client.spec

OUT="$SCRIPT_DIR/dist/fedlearn-client/fedlearn-client"
if [[ ! -x "$OUT" ]]; then
  echo "[build-linux] ERROR: expected $OUT to exist and be executable" >&2
  exit 1
fi

echo "[build-linux] Smoke test: $OUT --help"
"$OUT" --help || {
  echo "[build-linux] ERROR: smoke test failed" >&2
  exit 1
}

SIZE=$(du -sh "$SCRIPT_DIR/dist/fedlearn-client" | cut -f1)
echo "[build-linux] ✓ Built fedlearn-client (bundle size: $SIZE)"
echo "[build-linux] ✓ Output: $SCRIPT_DIR/dist/fedlearn-client/"
