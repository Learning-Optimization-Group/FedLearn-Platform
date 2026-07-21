#!/usr/bin/env bash
# build_grpc_arm64.sh — cross-compile the gRPC C++ runtime for Android arm64-v8a, PINNED
# (replaces v1's hardcoded v1.62.0 — 02-TECH-STACK §3.2 / 15-LLD §13 task 18). buf generates the
# C++ *stubs*; this builds the runtime they link against. CI caches OUTPUT_DIR by (version, NDK).
#
# VERIFY-BEFORE-USE: pin GRPC_CPP_VERSION to a current gRPC release and confirm its CMake options.
set -euo pipefail

GRPC_CPP_VERSION="${GRPC_CPP_VERSION:?set GRPC_CPP_VERSION, e.g. v1.67.1}"
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
ANDROID_NDK="${ANDROID_NDK:?set ANDROID_NDK to your NDK path}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-24}"
WORK="${WORK:-${PWD}/.build/grpc}"
OUTPUT_DIR="${OUTPUT_DIR:-${PWD}/.artifacts/grpc-android-${GRPC_CPP_VERSION}-${ANDROID_ABI}}"

if [[ -f "${OUTPUT_DIR}/lib/libgrpc++.a" ]]; then
  echo "gRPC already built at ${OUTPUT_DIR} (cache hit)"
  echo "GRPC_DIR=${OUTPUT_DIR}"
  exit 0
fi

if [[ ! -d "${WORK}/.git" ]]; then
  git clone --depth 1 --branch "${GRPC_CPP_VERSION}" https://github.com/grpc/grpc "${WORK}"
fi
cd "${WORK}"
git submodule sync
git submodule update --init --recursive --depth 1

# NB: build dir is "cmake-out", NOT "build" — gRPC's repo root has a Bazel `BUILD` file, and on a
# case-insensitive filesystem (macOS APFS) `cmake -B build` collides with it ("Unable to (re)create
# ... pkgRedirects"). Linux CI is case-sensitive so it is unaffected; a distinct name fixes both.
cmake -S . -B cmake-out \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DANDROID_PLATFORM="${ANDROID_PLATFORM}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${OUTPUT_DIR}" \
  -DgRPC_INSTALL=ON \
  -DgRPC_BUILD_TESTS=OFF \
  -DgRPC_BUILD_CODEGEN=OFF \
  -DgRPC_PROTOBUF_PROVIDER=module \
  -DgRPC_SSL_PROVIDER=module \
  -DgRPC_ZLIB_PROVIDER=module \
  -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--undefined-version" \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--undefined-version"

# --undefined-version: gRPC's bundled zlib links libz.so with a version script that references
# gz_intmax, a symbol not defined for the Android build. NDK r27's lld is strict about undefined
# version-script symbols (older ld silently allowed them), so the link fails with "version script
# assignment of 'local' to symbol 'gz_intmax' failed". The flag restores the permissive behaviour.
cmake --build cmake-out -j"$(nproc)"
cmake --install cmake-out

echo "Built gRPC ${GRPC_CPP_VERSION} (${ANDROID_ABI}) -> ${OUTPUT_DIR}"
echo "GRPC_DIR=${OUTPUT_DIR}"
