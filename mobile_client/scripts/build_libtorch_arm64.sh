#!/usr/bin/env bash
# build_libtorch_arm64.sh — cross-compile libtorch for Android arm64-v8a, PINNED to the torch
# version the golden fixture uses (fixes A6 §M-H1: v1 used an unpinned ${PYTORCH_SRC}). CI caches
# the OUTPUT_DIR by (PYTORCH_TAG, NDK). 15-LLD §11.2 / §13 task 18.
#
# VERIFY-BEFORE-USE: PyTorch's Android build flags evolve; confirm against the pinned tag's
# scripts/build_android.sh. We build FULL JIT (not the lite interpreter) because the mobile core
# uses torch::jit::load + full ops.
set -euo pipefail

PYTORCH_TAG="${PYTORCH_TAG:-v2.12.0}"          # MUST match framework golden manifest torch_version
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"        # the only shipped ABI (size budget)
ANDROID_NDK="${ANDROID_NDK:?set ANDROID_NDK to your NDK path}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-24}"
WORK="${WORK:-${PWD}/.build/pytorch}"
OUTPUT_DIR="${OUTPUT_DIR:-${PWD}/.artifacts/libtorch-android-${PYTORCH_TAG}-${ANDROID_ABI}}"

if [[ -f "${OUTPUT_DIR}/lib/libtorch.a" || -f "${OUTPUT_DIR}/lib/libtorch.so" ]]; then
  echo "libtorch already built at ${OUTPUT_DIR} (cache hit)"
  echo "LIBTORCH_DIR=${OUTPUT_DIR}"
  exit 0
fi

if [[ ! -d "${WORK}/.git" ]]; then
  git clone --depth 1 --branch "${PYTORCH_TAG}" https://github.com/pytorch/pytorch "${WORK}"
fi
cd "${WORK}"
git submodule sync
git submodule update --init --recursive --depth 1

export ANDROID_NDK ANDROID_ABI ANDROID_PLATFORM
export BUILD_LITE_INTERPRETER=0   # need full JIT (torch::jit::load) on device

# PyTorch's official Android builder. Produces build_android/install (headers + static libs).
scripts/build_android.sh \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DANDROID_PLATFORM="${ANDROID_PLATFORM}" \
  -DBUILD_LITE_INTERPRETER=OFF

mkdir -p "${OUTPUT_DIR}"
cp -R build_android/install/. "${OUTPUT_DIR}/"

echo "Built libtorch ${PYTORCH_TAG} (${ANDROID_ABI}) -> ${OUTPUT_DIR}"
echo "LIBTORCH_DIR=${OUTPUT_DIR}"
