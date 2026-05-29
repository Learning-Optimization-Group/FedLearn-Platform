# Android JNI native build for the FedLearn C++ TurboModule (15-LLD-mobile.md §4 / §11.4 / §13 task 12).
# Referenced by mobile_client/android/app/build.gradle  externalNativeBuild { cmake { path ... } }.
#
# Builds ONE arm64-v8a shared lib that links: the FL core (fedlearn_core), the gRPC layer
# (fedlearn_grpc, which pulls in the buf-generated stubs + gRPC runtime), and the React Native
# JSI/TurboModule runtime (provided by the ReactAndroid prefab).
#
# VERIFY-BEFORE-BUILD: ReactAndroid prefab target names + libtorch/gRPC ARM64 artifact paths are
# environment-specific. LIBTORCH_DIR / GENERATED_PROTO_DIR / the prebuilt gRPC must point at the
# CI-cached ARM64 artifacts (scripts/build_libtorch_arm64.sh, scripts/build_grpc_arm64.sh).
# A6 §M-H1: DO NOT force-link all of libtorch (no --whole-archive); link only referenced kernels
# and gate the resulting .so size in CI.
cmake_minimum_required(VERSION 3.22)
project(fedlearn_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# arm64-v8a is the only shipped ABI (size budget, A6 §M-H1). Set via the Gradle abiFilters too.
if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
  message(WARNING "fedlearn_jni targets arm64-v8a only; ANDROID_ABI=${ANDROID_ABI}")
endif()

# The shared C++ core + gRPC layer live two directories up (mobile_client/).
set(FEDLEARN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
# Build the core (+grpc) as part of this native build. GENERATED_PROTO_DIR must be set by Gradle
# (the buf-generated C++ stubs); LIBTORCH_DIR points at the ARM64 libtorch.
set(FEDLEARN_BUILD_TESTS OFF CACHE BOOL "" FORCE)   # no gtest in the app build
set(FEDLEARN_BUILD_GRPC ON CACHE BOOL "" FORCE)
add_subdirectory("${FEDLEARN_ROOT}/shared" "${CMAKE_BINARY_DIR}/fedlearn_shared")

find_package(ReactAndroid REQUIRED CONFIG)   # provides jsi + reactnative (New Architecture) prefab
find_library(LOG_LIB log)

add_library(fedlearn_jni SHARED
  OnLoad.cpp
  DeviceStateJni.cpp
  ${FEDLEARN_ROOT}/bridge/common/FedLearnCoreModule.cpp
  ${FEDLEARN_ROOT}/bridge/common/DeviceState.cpp
)
target_include_directories(fedlearn_jni PRIVATE
  ${FEDLEARN_ROOT}/bridge/common
  ${FEDLEARN_ROOT}/shared/include
)
target_link_libraries(fedlearn_jni
  fedlearn_core
  fedlearn_grpc
  ReactAndroid::jsi
  ReactAndroid::reactnative   # JSI/TurboModule runtime (target name is RN-version-specific)
  ${LOG_LIB}
)
