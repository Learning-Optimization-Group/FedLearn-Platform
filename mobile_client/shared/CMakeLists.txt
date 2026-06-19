# shared/ — the cross-platform C++ FL core (libfedlearn_core).
# Increment 1 builds the determinism slice (Perturbation, DtypeMap). Later tasks add
# ModelManager, ZerothOrderEstimator, DeComFLClient, FederatedLoop, FedLearnClient,
# DataLoader (15-LLD §13 tasks 6-10).

add_library(fedlearn_core STATIC
  src/DtypeMap.cpp
  src/Sha256.cpp
  src/ModelManager.cpp
  src/DeComFLClient.cpp
)

target_include_directories(fedlearn_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(fedlearn_core PUBLIC ${TORCH_LIBRARIES})
target_compile_features(fedlearn_core PUBLIC cxx_std_17)

if(FEDLEARN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# --- Optional gRPC layer (15-LLD §13 tasks 9-10): FedLearnClient + FederatedLoop + DataLoader ---
# Requires the buf-generated C++ stubs (run `buf generate` from proto/ -> gen/cpp) plus a
# cross-compiled gRPC C++ runtime + protobuf. NOT needed for the host parity gate above, so it
# is OFF by default — `fedlearn_core` stays buildable with only libtorch.
option(FEDLEARN_BUILD_GRPC "Build the gRPC client + federated loop (needs buf stubs + gRPC runtime)" OFF)
if(FEDLEARN_BUILD_GRPC)
  if(NOT DEFINED GENERATED_PROTO_DIR)
    message(FATAL_ERROR "FEDLEARN_BUILD_GRPC=ON requires -DGENERATED_PROTO_DIR=/path/to/gen/cpp "
                        "(run `buf generate` from the repo proto/ directory first).")
  endif()
  find_package(Protobuf REQUIRED)
  find_package(gRPC CONFIG REQUIRED)
  file(GLOB FEDLEARN_GENERATED_SRCS "${GENERATED_PROTO_DIR}/fedlearn/v2/*.cc")
  add_library(fedlearn_grpc STATIC
    src/FedLearnClient.cpp
    src/FederatedLoop.cpp
    src/DataLoader.cpp
    ${FEDLEARN_GENERATED_SRCS}
  )
  target_include_directories(fedlearn_grpc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${GENERATED_PROTO_DIR}
  )
  target_link_libraries(fedlearn_grpc PUBLIC
    fedlearn_core
    gRPC::grpc++
    protobuf::libprotobuf
    ${TORCH_LIBRARIES}
  )
  target_compile_features(fedlearn_grpc PUBLIC cxx_std_17)

  if(FEDLEARN_BUILD_TESTS)
    add_executable(fedlearn_grpc_tests tests/grpc_marshal_test.cpp)
    target_link_libraries(fedlearn_grpc_tests PRIVATE fedlearn_grpc GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(fedlearn_grpc_tests)
  endif()
endif()
