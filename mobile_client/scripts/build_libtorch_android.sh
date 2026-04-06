#!/bin/bash
# Build libtorch with training support for Android ARM64
#
# Prerequisites:
#   - Android NDK (r21+) installed, ANDROID_NDK set
#   - CMake 3.18+
#   - Python 3.9+
#   - PyTorch source cloned: git clone --recursive https://github.com/pytorch/pytorch.git
#
# Usage:
#   export ANDROID_NDK=/path/to/ndk
#   export PYTORCH_SRC=/path/to/pytorch
#   ./build_libtorch_android.sh
#
# Output: mobile_client/third_party/libtorch-android-arm64/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/third_party/libtorch-android-arm64"
BUILD_DIR="$PROJECT_DIR/build/libtorch-android"

: "${ANDROID_NDK:?Please set ANDROID_NDK to your NDK installation path}"
: "${PYTORCH_SRC:?Please set PYTORCH_SRC to your PyTorch source checkout}"

TOOLCHAIN="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
if [ ! -f "$TOOLCHAIN" ]; then
  echo "ERROR: NDK toolchain not found at $TOOLCHAIN"
  exit 1
fi

if [ ! -f "$PYTORCH_SRC/CMakeLists.txt" ]; then
  echo "ERROR: PyTorch CMakeLists.txt not found at $PYTORCH_SRC/CMakeLists.txt"
  echo "Did you clone with --recursive? Try: cd $PYTORCH_SRC && git submodule update --init --recursive"
  exit 1
fi

NCPU=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

echo "=== Building libtorch for Android ARM64 with training support ==="
echo "NDK: $ANDROID_NDK"
echo "PyTorch source: $PYTORCH_SRC"
echo "Build dir: $BUILD_DIR"
echo "Output: $OUTPUT_DIR"
echo "Parallel jobs: $NCPU"

mkdir -p "$BUILD_DIR"

if [ -x /opt/anaconda3/bin/python3 ]; then
  PYTHON_BIN=/opt/anaconda3/bin/python3
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python3" ]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python3"
else
  PYTHON_BIN=$(command -v python3)
fi
echo "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"

cmake -S "$PYTORCH_SRC" -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
  -DPython_EXECUTABLE="$PYTHON_BIN" \
  -DPYTHON_EXECUTABLE="$PYTHON_BIN" \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_PYTHON=OFF \
  -DBUILD_TEST=OFF \
  -DBUILD_BINARY=OFF \
  -DBUILD_CAFFE2_OPS=OFF \
  -DUSE_CUDA=OFF \
  -DUSE_ROCM=OFF \
  -DUSE_VULKAN=OFF \
  -DUSE_METAL=OFF \
  -DUSE_BLAS=OFF \
  -DUSE_NNPACK=OFF \
  -DUSE_QNNPACK=ON \
  -DUSE_XNNPACK=ON \
  -DUSE_DISTRIBUTED=OFF \
  -DUSE_OPENMP=OFF \
  -DUSE_OBSERVERS=OFF \
  -DUSE_NUMPY=OFF \
  -DINTERN_BUILD_MOBILE=ON \
  -DBUILD_MOBILE_AUTOGRAD=ON \
  -DCAFFE2_CMAKE_BUILDING_WITH_MAIN_REPO=ON \
  2>&1 | tee "$SCRIPT_DIR/build_libtorch_android.log"

cmake --build "$BUILD_DIR" -j"$NCPU" 2>&1 | tee -a "$SCRIPT_DIR/build_libtorch_android.log"

mkdir -p "$OUTPUT_DIR"
cmake --install "$BUILD_DIR" 2>&1 | tee -a "$SCRIPT_DIR/build_libtorch_android.log"

echo "=== libtorch Android ARM64 build complete ==="
echo "Output: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR/lib/" 2>/dev/null || ls -la "$OUTPUT_DIR/" || echo "(check $OUTPUT_DIR for output)"
