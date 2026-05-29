# gtest suite — the parity & roundtrip contract for the C++ FL core (15-LLD §13).
# Increment 1: rng_parity_test (the release gate) + dtype_map_test.

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# The Python golden fixture lives in the sibling framework/ tree (source of truth).
# CMAKE_SOURCE_DIR is the mobile_client/ top project dir; framework/ is its sibling.
set(FEDLEARN_GOLDEN_DIR "${CMAKE_SOURCE_DIR}/../framework/tests/fixtures/decomfl_golden")

add_executable(fedlearn_core_tests
  rng_parity_test.cpp
  dtype_map_test.cpp
)
target_link_libraries(fedlearn_core_tests PRIVATE
  fedlearn_core
  GTest::gtest_main
  ${TORCH_LIBRARIES}
)
target_compile_definitions(fedlearn_core_tests PRIVATE
  GOLDEN_DIR="${FEDLEARN_GOLDEN_DIR}"
)

include(GoogleTest)
gtest_discover_tests(fedlearn_core_tests)
