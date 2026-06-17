# ExecuTorch C++ forward test — standalone (Phase 3a).
#
# Kept SEPARATE from the libtorch test suite (shared/tests/CMakeLists.txt): this target links
# the ExecuTorch runtime, and mixing it with libtorch in one binary clashes on vendored c10/aten
# symbols. Phase 3c removes libtorch and unifies the suites.
#
# Prerequisite: a host ExecuTorch build (analogous to LIBTORCH_DIR for the libtorch suite).
# Build ExecuTorch v1.3.1 from source (dir MUST be named "executorch"); configure with the venv
# python so it finds torch:
#   git clone -b v1.3.1 https://github.com/pytorch/executorch.git /tmp/executorch
#   cd /tmp/executorch && git submodule update --init --recursive --depth 1
#   cmake -B cmake-out -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=<venv>/bin/python \
#     -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
#     -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
#     -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON
#   cmake --build cmake-out -j
# Then:
#   cmake -S . -B build -DET_SRC=/tmp/executorch -DET_BUILD=/tmp/executorch/cmake-out \
#         -DTORCH_INCLUDE=<venv>/lib/pythonX/site-packages/torch/include
#   cmake --build build -j && ctest --test-dir build --output-on-failure

cmake_minimum_required(VERSION 3.24)
project(fedlearn_et_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ET_SRC        "/tmp/executorch"            CACHE PATH "ExecuTorch source tree (dir named 'executorch')")
set(ET_BUILD      "/tmp/executorch/cmake-out"  CACHE PATH "ExecuTorch host build output")
set(TORCH_INCLUDE ""                            CACHE PATH "venv torch include dir (supplies torch/headeronly/macros/cmake_macros.h)")

if(NOT EXISTS "${ET_BUILD}")
  message(FATAL_ERROR "ExecuTorch host build not found at ${ET_BUILD} — build it first (see header).")
endif()

# Golden fixture dir (sibling framework/ tree): .../mobile_client/shared/tests/executorch -> ROOT.
set(GOLDEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../framework/tests/fixtures/decomfl_golden")
set(SHARED "${CMAKE_CURRENT_SOURCE_DIR}/../..")

include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.15.2)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(fedlearn_et_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/../executorch_forward_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../et_g_scalar_test.cpp
  ${SHARED}/src/ExecutorchModel.cpp
  ${SHARED}/src/Sha256.cpp
)

get_filename_component(ET_SRC_PARENT "${ET_SRC}" DIRECTORY)
target_include_directories(fedlearn_et_tests PRIVATE
  ${SHARED}/include
  "${ET_SRC_PARENT}"
  "${ET_SRC}/runtime/core/portable_type/c10"
  "${TORCH_INCLUDE}"            # ordered last: only the missing cmake_macros.h falls through to torch
)
target_compile_definitions(fedlearn_et_tests PRIVATE GOLDEN_DIR="${GOLDEN_DIR}")

# Resolve the ExecuTorch static libs out of the host build tree.
set(_et_libs
  executorch
  executorch_core            # runtime_init / register_kernels / dim-order utils live here
  portable_kernels
  extension_data_loader
  extension_flat_tensor
  extension_tensor
)
set(_resolved "")
foreach(_lib IN LISTS _et_libs)
  find_library(ET_LIB_${_lib} NAMES ${_lib} lib${_lib}
    PATHS "${ET_BUILD}"
    PATH_SUFFIXES . lib kernels/portable extension/data_loader extension/flat_tensor extension/tensor
    NO_DEFAULT_PATH REQUIRED)
  list(APPEND _resolved "${ET_LIB_${_lib}}")
endforeach()

# portable_ops_lib registers kernels via static initializers — whole-archive so they survive.
find_library(ET_PORTABLE_OPS_LIB NAMES portable_ops_lib libportable_ops_lib
  PATHS "${ET_BUILD}" PATH_SUFFIXES . lib kernels/portable NO_DEFAULT_PATH REQUIRED)
if(APPLE)
  set(_force_ops "-Wl,-force_load,${ET_PORTABLE_OPS_LIB}")
else()
  set(_force_ops "-Wl,--whole-archive" "${ET_PORTABLE_OPS_LIB}" "-Wl,--no-whole-archive")
endif()

target_link_libraries(fedlearn_et_tests PRIVATE
  ${_force_ops}
  ${_resolved}
  GTest::gtest_main
)

# Plain registration: gtest_discover_tests' build-time discovery run is unreliable here (the
# ExecuTorch runtime emits static-init stderr noise); one ctest entry running the whole binary
# is sufficient and deterministic.
enable_testing()
add_test(NAME fedlearn_et_tests COMMAND fedlearn_et_tests)
