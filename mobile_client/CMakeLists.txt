# Top-level C++ build for the FedLearn mobile core (15-LLD-mobile.md §4).
#
# Host (parity gate / CI on an ARM runner):
#   cmake -S mobile_client -B mobile_client/build -DFEDLEARN_BUILD_TESTS=ON \
#     -DET_SRC=/tmp/executorch -DET_BUILD=/tmp/executorch/cmake-out \
#     -DTORCH_INCLUDE=<venv>/lib/pythonX/site-packages/torch/include
#   cmake --build mobile_client/build -j
#   ctest --test-dir mobile_client/build --output-on-failure
#
# The core links the ExecuTorch runtime (no libtorch/ATen). Build inputs:
#   ET_SRC        — ExecuTorch v1.3.1 source tree (the dir MUST be named "executorch").
#   ET_BUILD      — its host build output (cmake-out); static libs are resolved out of it.
#   TORCH_INCLUDE — a venv torch include dir, used ONLY to supply the missing
#                   torch/headeronly/macros/cmake_macros.h header (ordered last on the
#                   include path, so nothing else falls through to it).
# Build ExecuTorch from source first (see mobile_client/README.md / shared/tests for the
# exact configure flags). For Android device builds the C++ core is consumed via
# bridge/android/jni (later tasks); this top build is the isolated host parity/roundtrip
# gate (15-LLD §11.3).
cmake_minimum_required(VERSION 3.24)
project(fedlearn_mobile_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ET_SRC        "/tmp/executorch"            CACHE PATH "ExecuTorch source tree (dir named 'executorch')")
set(ET_BUILD      "/tmp/executorch/cmake-out"  CACHE PATH "ExecuTorch host build output")
set(TORCH_INCLUDE ""                            CACHE PATH "venv torch include dir (supplies torch/headeronly/macros/cmake_macros.h)")

if(NOT EXISTS "${ET_BUILD}")
  message(FATAL_ERROR "ExecuTorch host build not found at ${ET_BUILD} — build it first (see header).")
endif()

option(FEDLEARN_BUILD_TESTS "Build the C++ core gtest parity/roundtrip tests" ON)

# enable_testing() must run in the top-level project so `ctest --test-dir <build>` generates a
# root CTestTestfile.cmake that recurses into shared/tests (which registers fedlearn_core_tests
# via plain add_test).
if(FEDLEARN_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory(shared)
