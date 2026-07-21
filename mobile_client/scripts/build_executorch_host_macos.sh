#!/usr/bin/env bash
# build_executorch_host_macos.sh — build the ExecuTorch runtime for the LOCAL macOS host so the C++
# core gtests (fedlearn_core_tests) can be built + run on a developer Mac. CI builds ET host on Linux
# (mobile.yml, the "host parity gate"); this reproduces it on Apple Silicon, which needs two fixes the
# Linux recipe does not.
#
# It also enables EXECUTORCH_BUILD_EXTENSION_TRAINING=ON (additive; the gtest suite doesn't link it)
# so the on-device forward+backward de-risk in scripts/et_training_smoke/ can build against this same
# host build. That extension (training_module + optimizer/sgd) is what a native first-order
# FedAvg/FedProx round needs — the precondition for lifting the mobile client's zeroth-order-only
# (MO-4) restriction. The two macOS-only fixes the Linux recipe does not need:
#
#   1. -DPYTHON_EXECUTABLE=<the python that has torch>. ET's cmake resolves torch's path via
#      `find_spec('torch')`; if cmake auto-picks a python WITHOUT torch it fails with
#      "AttributeError: 'NoneType' object has no attribute ...". Pin the interpreter explicitly.
#   2. -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=OFF. On macOS size_t (unsigned long) and uint64_t
#      (unsigned long long) are DISTINCT types, so ET's optimized_portable_kernels fail to compile
#      (runtime/core/array_ref.h: cannot init uint64_t* from size_t*). On Linux they are identical, so
#      CI is unaffected. The FedLearn tests link portable_ops_lib (portable kernels), NOT the optimized
#      set, so turning the optimized kernels off is harmless for the parity/roundtrip suite.
#
# Pinned to the same ET version as the host parity gate (mobile.yml ET_VERSION). After this succeeds:
#   TORCH_INCLUDE=$(python3 -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'include'))")
#   cmake -S mobile_client -B mobile_client/build-host -G Ninja -DFEDLEARN_BUILD_TESTS=ON \
#     -DET_SRC="$ET_SRC" -DET_BUILD="$ET_SRC/cmake-out" -DTORCH_INCLUDE="$TORCH_INCLUDE" -DCMAKE_BUILD_TYPE=Release
#   cmake --build mobile_client/build-host --target fedlearn_core_tests && \
#     mobile_client/build-host/shared/tests/fedlearn_core_tests
# (The gRPC-gated tests — grpc_marshal_test — additionally need FEDLEARN_BUILD_GRPC=ON + a gRPC/protobuf
#  install + the buf-generated C++ stubs; the gRPC-free parity/roundtrip suite runs without them.)
set -euo pipefail

ET_VERSION="${ET_VERSION:-1.3.1}"
ET_SRC="${ET_SRC:-$HOME/executorch-host/executorch}"
PYEXE="$(python3 -c 'import sys; print(sys.executable)')"
"$PYEXE" -c 'import importlib.util as u; assert u.find_spec("torch"), "torch not importable by this python"'

if [[ -f "$ET_SRC/cmake-out/libexecutorch.a" ]]; then
  echo "ET host already built at $ET_SRC/cmake-out (cache hit)"; echo "ET_BUILD=$ET_SRC/cmake-out"; exit 0
fi

echo "[1/4] clone executorch v$ET_VERSION"
rm -rf "$ET_SRC"; mkdir -p "$(dirname "$ET_SRC")"
git clone --depth 1 --branch "v${ET_VERSION}" https://github.com/pytorch/executorch "$ET_SRC"
cd "$ET_SRC"
echo "[2/4] submodules"
git submodule sync
git submodule update --init --recursive --depth 1
echo "[3/4] cmake configure (explicit python; optimized kernels OFF for macOS)"
cmake -S . -B cmake-out -G Ninja \
  -DPYTHON_EXECUTABLE="$PYEXE" \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TRAINING=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=OFF
echo "[4/4] build"
cmake --build cmake-out -j"$(sysctl -n hw.ncpu)"
echo "ET_BUILD=$ET_SRC/cmake-out"
echo "OK: ExecuTorch host build complete."
