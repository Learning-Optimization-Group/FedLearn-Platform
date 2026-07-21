#!/usr/bin/env bash
# build_executorch_arm64.sh — cross-compile the ExecuTorch runtime for Android arm64-v8a, PINNED
# to the ET version the host parity gate uses (v1.3.1). Replaces build_libtorch_arm64.sh for the
# Android native path: ExecuTorch is a LEAN runtime, so this is minutes (not the hours libtorch
# took). CI caches the OUTPUT_DIR by (ET_VERSION, NDK). 15-LLD §11.2 / §13 task 18.
#
# VERIFY-BEFORE-USE: ExecuTorch's Android build flags evolve across releases. Confirm the
# EXECUTORCH_BUILD_* options below against the pinned tag's CMakeLists / docs before first use.
# These mirror the host parity build (the unified suite links these static libs:
#   executorch executorch_core portable_kernels extension_data_loader extension_flat_tensor
#   extension_tensor + whole-archive portable_ops_lib), so the arm64 cross-build MUST produce the
# same library set or libfedlearn_jni.so will fail to link.
#
# Layout: the artifact dir holds the "executorch"-named source tree (ET_SRC, for headers) and its
# cmake-out build output (ET_BUILD, for static libs). shared/CMakeLists derives the include parent
# from ET_SRC, so the source dir MUST keep the name "executorch".
set -euo pipefail

ET_VERSION="${ET_VERSION:-1.3.1}"              # MUST match the host parity gate (mobile.yml ET_VERSION)
ET_TAG="${ET_TAG:-v${ET_VERSION}}"
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"        # the only shipped ABI (size budget)
ANDROID_NDK="${ANDROID_NDK:?set ANDROID_NDK to your NDK path}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-26}"
OUTPUT_DIR="${OUTPUT_DIR:-${PWD}/.artifacts/executorch-android-v${ET_VERSION}-${ANDROID_ABI}}"
ET_SRC_DIR="${OUTPUT_DIR}/executorch"          # dir MUST be named "executorch" (see header)
ET_BUILD_DIR="${ET_SRC_DIR}/cmake-out"

# Cache-hit check: the core static lib is the canonical marker (mirrors the libtorch script).
if [[ -f "${ET_BUILD_DIR}/libexecutorch.a" || -f "${ET_BUILD_DIR}/lib/libexecutorch.a" ]]; then
  echo "ExecuTorch already built at ${OUTPUT_DIR} (cache hit)"
  echo "ET_SRC=${ET_SRC_DIR}"
  echo "ET_BUILD=${ET_BUILD_DIR}"
  exit 0
fi

if [[ ! -d "${ET_SRC_DIR}/.git" ]]; then
  mkdir -p "${OUTPUT_DIR}"
  git clone --depth 1 --branch "${ET_TAG}" https://github.com/pytorch/executorch "${ET_SRC_DIR}"
fi
cd "${ET_SRC_DIR}"
git submodule sync
git submodule update --init --recursive --depth 1

# Cross-compile the lean runtime for arm64 via the NDK toolchain. The EXECUTORCH_BUILD_* flags
# mirror the host parity build (see this script's header).
cmake -S . -B "${ET_BUILD_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DANDROID_PLATFORM="${ANDROID_PLATFORM}" \
  -DCMAKE_BUILD_TYPE=Release \
  ${PYTHON_EXECUTABLE:+-DPYTHON_EXECUTABLE="${PYTHON_EXECUTABLE}"} \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TRAINING=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON

# JOBS caps parallelism on memory-constrained hosts (set JOBS=4 etc.); defaults to nproc.
cmake --build "${ET_BUILD_DIR}" -j"${JOBS:-$(nproc)}"

echo "Built ExecuTorch ${ET_VERSION} (${ANDROID_ABI}) -> ${OUTPUT_DIR}"
echo "ET_SRC=${ET_SRC_DIR}"
echo "ET_BUILD=${ET_BUILD_DIR}"
