cmake_minimum_required(VERSION 3.13)
project(appmodules)

# ---------- Paths (must be set before ReactNative-application.cmake) ----------
set(SHARED_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../shared")
# Use symlink to avoid spaces in path (spaces break the linker for .a files)
set(THIRD_PARTY_DIR "/tmp/fl-third-party")

set(LIBTORCH_DIR "${THIRD_PARTY_DIR}/libtorch-android-arm64")
set(GRPC_DIR "${THIRD_PARTY_DIR}/grpc-android-arm64")

# Allow CMake to find libraries in our third-party install prefixes
# (Android NDK toolchain restricts find_library to CMAKE_FIND_ROOT_PATH)
list(APPEND CMAKE_FIND_ROOT_PATH "${LIBTORCH_DIR}" "${GRPC_DIR}")

# Standard React Native new-architecture setup (creates ${CMAKE_PROJECT_NAME}
# shared library with default RN sources, links fbjni/jsi/reactnative).
include(${REACT_ANDROID_DIR}/cmake-utils/ReactNative-application.cmake)

# Replace the default OnLoad.cpp with our custom one that registers NativeFedLearnCore
get_target_property(_app_sources ${CMAKE_PROJECT_NAME} SOURCES)
list(FILTER _app_sources EXCLUDE REGEX ".*default-app-setup/OnLoad\\.cpp$")
list(APPEND _app_sources "${CMAKE_CURRENT_SOURCE_DIR}/OnLoad.cpp")
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES SOURCES "${_app_sources}")

# ---------- libtorch (pre-built for Android ARM64) ----------
set(Torch_DIR "${LIBTORCH_DIR}/share/cmake/Torch")
find_package(Torch REQUIRED CONFIG)

# ---------- gRPC / Protobuf (pre-built static libs, no cmake config) ----------
file(GLOB GRPC_STATIC_LIBS "${GRPC_DIR}/lib/*.a")

add_library(grpc_bundle INTERFACE)
target_include_directories(grpc_bundle INTERFACE "${GRPC_DIR}/include")

# ---------- nlohmann/json (header-only) ----------
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

# ---------- Proto generation (host protoc + grpc_cpp_plugin) ----------
set(PROTO_SRC_DIR "${SHARED_DIR}/proto")
set(PROTO_GEN_DIR "${SHARED_DIR}/generated")
file(MAKE_DIRECTORY "${PROTO_GEN_DIR}")

set(HOST_TOOLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../build/grpc-host")
set(PROTOC "${HOST_TOOLS_DIR}/third_party/protobuf/protoc")
set(GRPC_CPP_PLUGIN "${HOST_TOOLS_DIR}/grpc_cpp_plugin")

set(FEDLEARN_PROTO "${PROTO_SRC_DIR}/fedlearn.proto")
set(FEDLEARN_PB_CC "${PROTO_GEN_DIR}/fedlearn.pb.cc")
set(FEDLEARN_PB_H  "${PROTO_GEN_DIR}/fedlearn.pb.h")
set(FEDLEARN_GRPC_CC "${PROTO_GEN_DIR}/fedlearn.grpc.pb.cc")
set(FEDLEARN_GRPC_H  "${PROTO_GEN_DIR}/fedlearn.grpc.pb.h")

add_custom_command(
  OUTPUT ${FEDLEARN_PB_CC} ${FEDLEARN_PB_H} ${FEDLEARN_GRPC_CC} ${FEDLEARN_GRPC_H}
  COMMAND ${PROTOC}
    --proto_path=${PROTO_SRC_DIR}
    --cpp_out=${PROTO_GEN_DIR}
    --grpc_out=${PROTO_GEN_DIR}
    --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
    ${FEDLEARN_PROTO}
  DEPENDS ${FEDLEARN_PROTO}
  COMMENT "Generating gRPC C++ stubs from fedlearn.proto"
)

# ---------- FedLearn core static library ----------
add_library(fedlearn_core STATIC
  ${SHARED_DIR}/src/ModelManager.cpp
  ${SHARED_DIR}/src/ZerothOrderEstimator.cpp
  ${SHARED_DIR}/src/DeComFLClient.cpp
  ${SHARED_DIR}/src/DataLoader.cpp
  ${SHARED_DIR}/src/FedLearnClient.cpp
  ${SHARED_DIR}/src/FederatedLoop.cpp
  ${SHARED_DIR}/src/NativeFedLearnCore.cpp
  ${FEDLEARN_PB_CC}
  ${FEDLEARN_GRPC_CC}
)

target_include_directories(fedlearn_core PUBLIC
  ${SHARED_DIR}/src
  ${SHARED_DIR}
  ${PROTO_GEN_DIR}
  ${GRPC_DIR}/include
  ${TORCH_INCLUDE_DIRS}
)

target_compile_options(fedlearn_core PRIVATE
  -Wno-error
  -Wno-unused-parameter
  -Wno-sign-compare
  -Wno-macro-redefined
  -Wno-error=macro-redefined
)

target_link_libraries(fedlearn_core PUBLIC
  nlohmann_json::nlohmann_json
)

# ---------- Wire into the RN shared library ----------
# OnLoad.cpp is already added by ReactNative-application.cmake from this directory.

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
  ${SHARED_DIR}/src
  ${SHARED_DIR}
  ${PROTO_GEN_DIR}
)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
  -Wno-error=macro-redefined
  -Wno-macro-redefined
)

# Collect all libtorch .a files for the linker group
file(GLOB TORCH_STATIC_LIBS "${LIBTORCH_DIR}/lib/*.a")
list(APPEND TORCH_STATIC_LIBS "${LIBTORCH_DIR}/lib/mimalloc-2.2/libmimalloc.a")

# Link everything in --start-group / --end-group to resolve circular deps
# among fedlearn_core, libtorch, and gRPC static archives
target_link_libraries(${CMAKE_PROJECT_NAME}
  -Wl,--start-group
  fedlearn_core
  -Wl,--whole-archive
  ${TORCH_STATIC_LIBS}
  -Wl,--no-whole-archive
  ${GRPC_STATIC_LIBS}
  -Wl,--end-group
  log
)
