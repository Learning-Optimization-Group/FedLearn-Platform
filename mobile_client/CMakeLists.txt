# Top-level C++ build for the FedLearn mobile core (15-LLD-mobile.md §4).
#
# Host (parity gate / CI on an ARM runner):
#   cmake -S mobile_client -B mobile_client/build -DLIBTORCH_DIR=/path/to/libtorch
#   cmake --build mobile_client/build -j
#   ctest --test-dir mobile_client/build --output-on-failure
#
# LIBTORCH_DIR must point at an unzipped libtorch matching the torch version pinned in
# framework/tests/fixtures/decomfl_golden/manifest.json (currently 2.12.0). For Android
# device builds the C++ core is consumed via bridge/android/jni (later tasks); this top
# build is for the isolated host parity/roundtrip gate (15-LLD §11.3).
cmake_minimum_required(VERSION 3.22)
project(fedlearn_mobile_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED LIBTORCH_DIR)
  message(FATAL_ERROR
    "Set -DLIBTORCH_DIR=/path/to/libtorch (see mobile_client/README.md). "
    "It must match the torch version in framework/tests/fixtures/decomfl_golden/manifest.json.")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

option(FEDLEARN_BUILD_TESTS "Build the C++ core gtest parity/roundtrip tests" ON)

add_subdirectory(shared)
