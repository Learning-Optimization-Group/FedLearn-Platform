#!/bin/bash
# Cross-compile gRPC C++ for iOS arm64
#
# Prerequisites:
#   - Xcode with command line tools
#   - CMake 3.18+
#
# Usage:
#   ./build_grpc_ios.sh
#
# Output: mobile_client/third_party/grpc-ios-arm64/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/third_party/grpc-ios-arm64"
BUILD_DIR="$PROJECT_DIR/build/grpc-ios"

GRPC_VERSION="v1.62.0"

echo "=== Building gRPC $GRPC_VERSION for iOS arm64 ==="

# Clone gRPC if not present (reuse from Android build if available)
GRPC_SRC="$PROJECT_DIR/build/grpc-src"
if [ ! -d "$GRPC_SRC" ]; then
  git clone --depth 1 --branch $GRPC_VERSION \
    https://github.com/grpc/grpc.git "$GRPC_SRC"
  cd "$GRPC_SRC" && git submodule update --init --depth 1
fi

# Build host tools if not already built
HOST_BUILD="$PROJECT_DIR/build/grpc-host"
if [ ! -f "$HOST_BUILD/grpc_cpp_plugin" ]; then
  mkdir -p "$HOST_BUILD"
  cmake -S "$GRPC_SRC" -B "$HOST_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DgRPC_BUILD_TESTS=OFF
  cmake --build "$HOST_BUILD" --target grpc_cpp_plugin protoc -j$(sysctl -n hw.ncpu)
fi

# iOS toolchain
IOS_TOOLCHAIN="$GRPC_SRC/third_party/abseil-cpp/CMake/ios.toolchain.cmake"

# If the abseil toolchain doesn't exist, use a simple approach
mkdir -p "$BUILD_DIR"
cmake -S "$GRPC_SRC" -B "$BUILD_DIR" \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$OUTPUT_DIR" \
  -DgRPC_BUILD_TESTS=OFF \
  -DgRPC_BUILD_CODEGEN=OFF \
  -DgRPC_BUILD_GRPC_CPP_PLUGIN=OFF \
  -DgRPC_BUILD_CSHARP_EXT=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -D_gRPC_PROTOBUF_PROTOC_EXECUTABLE="$HOST_BUILD/third_party/protobuf/protoc" \
  -D_gRPC_CPP_PLUGIN="$HOST_BUILD/grpc_cpp_plugin"

cmake --build "$BUILD_DIR" -j$(sysctl -n hw.ncpu)
cmake --install "$BUILD_DIR"

echo "=== gRPC iOS arm64 build complete ==="
echo "Output: $OUTPUT_DIR"
