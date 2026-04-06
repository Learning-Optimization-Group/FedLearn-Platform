#!/bin/bash
# Cross-compile gRPC C++ for Android ARM64
#
# Prerequisites:
#   - Android NDK (r21+), ANDROID_NDK set
#   - CMake 3.18+
#
# Usage:
#   export ANDROID_NDK=/path/to/ndk
#   ./build_grpc_android.sh
#
# Output: mobile_client/third_party/grpc-android-arm64/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/third_party/grpc-android-arm64"
BUILD_DIR="$PROJECT_DIR/build/grpc-android"

: "${ANDROID_NDK:?Please set ANDROID_NDK to your NDK installation path}"

GRPC_VERSION="v1.62.0"
NCPU=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

echo "=== Building gRPC $GRPC_VERSION for Android ARM64 ==="

# Clone gRPC if not present
GRPC_SRC="$PROJECT_DIR/build/grpc-src"
if [ ! -d "$GRPC_SRC" ]; then
  git clone --depth 1 --branch $GRPC_VERSION \
    https://github.com/grpc/grpc.git "$GRPC_SRC"
  cd "$GRPC_SRC" && git submodule update --init --depth 1
fi

# First build host tools (protoc, grpc_cpp_plugin)
HOST_BUILD="$PROJECT_DIR/build/grpc-host"
mkdir -p "$HOST_BUILD"
cmake -S "$GRPC_SRC" -B "$HOST_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DBUILD_SHARED_LIBS=OFF \
  -DgRPC_BUILD_TESTS=OFF \
  -DgRPC_BUILD_GRPC_CSHARP_PLUGIN=OFF \
  -DgRPC_BUILD_GRPC_NODE_PLUGIN=OFF \
  -DgRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN=OFF \
  -DgRPC_BUILD_GRPC_PHP_PLUGIN=OFF \
  -DgRPC_BUILD_GRPC_PYTHON_PLUGIN=OFF \
  -DgRPC_BUILD_GRPC_RUBY_PLUGIN=OFF

cmake --build "$HOST_BUILD" --target grpc_cpp_plugin protoc -j"$NCPU"

# Cross-compile for Android
TOOLCHAIN="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
mkdir -p "$BUILD_DIR"
cmake -S "$GRPC_SRC" -B "$BUILD_DIR" \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DgRPC_BUILD_TESTS=OFF \
  -DgRPC_BUILD_CODEGEN=OFF \
  -DgRPC_BUILD_GRPC_CPP_PLUGIN=OFF \
  -DgRPC_BUILD_CSHARP_EXT=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -D_gRPC_PROTOBUF_PROTOC_EXECUTABLE="$HOST_BUILD/third_party/protobuf/protoc" \
  -D_gRPC_CPP_PLUGIN="$HOST_BUILD/grpc_cpp_plugin"

cmake --build "$BUILD_DIR" --target grpc++ grpc gpr address_sorting -j"$NCPU"

mkdir -p "$OUTPUT_DIR/lib" "$OUTPUT_DIR/include"

find "$BUILD_DIR" -name '*.a' | while read -r lib; do
  cp "$lib" "$OUTPUT_DIR/lib/"
done

for dir in grpc grpc++ grpcpp google absl re2 ; do
  if [ -d "$GRPC_SRC/include/$dir" ]; then
    cp -R "$GRPC_SRC/include/$dir" "$OUTPUT_DIR/include/"
  fi
done
cp -R "$GRPC_SRC/third_party/protobuf/src/google" "$OUTPUT_DIR/include/" 2>/dev/null || true
cp -R "$GRPC_SRC/third_party/abseil-cpp/absl" "$OUTPUT_DIR/include/" 2>/dev/null || true

echo "=== gRPC Android ARM64 build complete ==="
echo "Output: $OUTPUT_DIR"
echo "Host protoc: $HOST_BUILD/third_party/protobuf/protoc"
echo "Host grpc_cpp_plugin: $HOST_BUILD/grpc_cpp_plugin"
