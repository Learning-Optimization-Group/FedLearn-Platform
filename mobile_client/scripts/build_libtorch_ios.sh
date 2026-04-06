#!/bin/bash
# Build libtorch with training support for iOS arm64
#
# Prerequisites:
#   - Xcode with command line tools
#   - CMake 3.18+
#   - Python 3.9+
#   - PyTorch source cloned
#
# Usage:
#   export PYTORCH_SRC=/path/to/pytorch
#   ./build_libtorch_ios.sh
#
# Output: mobile_client/third_party/libtorch-ios-arm64/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/third_party/libtorch-ios-arm64"

: "${PYTORCH_SRC:?Please set PYTORCH_SRC to your PyTorch source checkout}"

echo "=== Building libtorch for iOS arm64 with training support ==="
echo "PyTorch source: $PYTORCH_SRC"
echo "Output: $OUTPUT_DIR"

cd "$PYTORCH_SRC"

python setup.py clean 2>/dev/null || true

# Build for iOS arm64 (device)
BUILD_PYTORCH_MOBILE=1 \
IOS_PLATFORM=OS \
IOS_ARCH=arm64 \
BUILD_MOBILE_AUTOGRAD=ON \
NO_API=OFF \
python scripts/build_ios.py \
  --build-type Release \
  2>&1 | tee "$PROJECT_DIR/scripts/build_libtorch_ios.log"

mkdir -p "$OUTPUT_DIR"
cp -r build_ios/install/* "$OUTPUT_DIR/"

echo ""
echo "=== Building for iOS simulator (arm64, for Apple Silicon Macs) ==="

BUILD_PYTORCH_MOBILE=1 \
IOS_PLATFORM=SIMULATOR \
IOS_ARCH=arm64 \
BUILD_MOBILE_AUTOGRAD=ON \
NO_API=OFF \
python scripts/build_ios.py \
  --build-type Release \
  2>&1 | tee -a "$PROJECT_DIR/scripts/build_libtorch_ios.log"

SIM_DIR="$PROJECT_DIR/third_party/libtorch-ios-sim-arm64"
mkdir -p "$SIM_DIR"
cp -r build_ios/install/* "$SIM_DIR/"

echo "=== libtorch iOS build complete ==="
echo "Device: $OUTPUT_DIR"
echo "Simulator: $SIM_DIR"
