# App native build — RN 0.80 New Architecture (standard location: android/app/src/main/jni).
#
# Uses React Native's own ReactNative-application.cmake, which builds libappmodules.so and handles ALL
# of the New-Architecture wiring: autolinking (the react_codegen_* Fabric/TurboModule libs for screens,
# svg, safe-area-context, reanimated, worklets), the jsi/reactnative/fbjni ALIAS targets, the
# `common_flags` include/def carrier, folly flags, C++20, and the rncore core modules. Our custom C++ FL
# core (ExecuTorch + gRPC + the DeComFL C++ TurboModule) is grafted INTO libappmodules.so below. The app
# entry point OnLoad.cpp lives in THIS directory (RN globs *.cpp here) and registers NativeFedLearnCore
# alongside the autolinked providers.
cmake_minimum_required(VERSION 3.13)
project(appmodules)

include(${REACT_ANDROID_DIR}/cmake-utils/ReactNative-application.cmake)

# ---- FedLearn C++ FL core linked into libappmodules.so ----------------------------------------------
# The out-of-sysroot cross-compiled ExecuTorch + gRPC static libs need relaxed find-root modes and the
# gRPC install on the prefix path. ET_SRC / ET_BUILD / TORCH_INCLUDE / GENERATED_PROTO_DIR / GRPC_DIR are
# passed by Gradle's externalNativeBuild cmake { arguments ... }.
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
if(GRPC_DIR)
  list(APPEND CMAKE_PREFIX_PATH "${GRPC_DIR}")
endif()

set(FEDLEARN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../..")   # app/src/main/jni -> mobile_client
set(FEDLEARN_BUILD_TESTS OFF CACHE BOOL "" FORCE)                 # no gtest in the app build
set(FEDLEARN_BUILD_GRPC ON CACHE BOOL "" FORCE)
add_subdirectory("${FEDLEARN_ROOT}/shared" "${CMAKE_BINARY_DIR}/fedlearn_shared")

# OnLoad.cpp is globbed from this dir by ReactNative-application.cmake; add the remaining bridge sources
# (the JNI device-state shim + the shared TurboModule impl) directly into libappmodules.so.
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
  ${FEDLEARN_ROOT}/bridge/android/jni/DeviceStateJni.cpp
  ${FEDLEARN_ROOT}/bridge/common/FedLearnCoreModule.cpp
  ${FEDLEARN_ROOT}/bridge/common/DeviceState.cpp)
# These 3 files are grafted onto the `appmodules` target and so inherit RN's strict -Werror flag set,
# which is RN's policy for RN's own sources — not our contract. Keep warnings visible but non-fatal on
# just our glue (this appends after RN's -Werror, so -Wno-error wins) so an incidental RN-pedantic
# warning can't fail the whole app build. Real correctness lives in the shared/ audit + our own review.
set_source_files_properties(
  ${FEDLEARN_ROOT}/bridge/android/jni/DeviceStateJni.cpp
  ${FEDLEARN_ROOT}/bridge/common/FedLearnCoreModule.cpp
  ${FEDLEARN_ROOT}/bridge/common/DeviceState.cpp
  PROPERTIES COMPILE_OPTIONS "-Wno-error")
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
  ${FEDLEARN_ROOT}/bridge/common
  ${FEDLEARN_ROOT}/shared/include)
target_link_libraries(${CMAKE_PROJECT_NAME} fedlearn_core fedlearn_grpc)
