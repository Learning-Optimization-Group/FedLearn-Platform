# shared/ — the cross-platform C++ FL core (libfedlearn_core).
# The full torch-free core: deterministic RNG/dtype mapping, SHA-256, safetensors + model
# management, the ExecuTorch forward wrapper, the DeComFL client, and the gRPC-free
# FederatedLoop + DataLoader (15-LLD §13 tasks 6-10). It links the ExecuTorch runtime
# (no libtorch/ATen); the gRPC transport is an optional layer on top (OFF by default).

add_library(fedlearn_core STATIC
  src/DtypeMap.cpp
  src/Sha256.cpp
  src/ModelManager.cpp
  src/DeComFLClient.cpp
  src/ExecutorchModel.cpp
  src/Safetensors.cpp
  src/DataLoader.cpp
  src/FederatedLoop.cpp
)

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

# portable_ops_lib registers kernels via static initializers — whole-archive so they survive
# into the final executable (else the runtime reports "operator not found"). PUBLIC so test
# executables linking fedlearn_core inherit the force-load.
find_library(ET_PORTABLE_OPS_LIB NAMES portable_ops_lib libportable_ops_lib
  PATHS "${ET_BUILD}" PATH_SUFFIXES . lib kernels/portable NO_DEFAULT_PATH REQUIRED)
if(APPLE)
  set(_force_ops "-Wl,-force_load,${ET_PORTABLE_OPS_LIB}")
else()
  set(_force_ops "-Wl,--whole-archive" "${ET_PORTABLE_OPS_LIB}" "-Wl,--no-whole-archive")
endif()

get_filename_component(ET_SRC_PARENT "${ET_SRC}" DIRECTORY)
target_include_directories(fedlearn_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  "${ET_SRC_PARENT}"
  "${ET_SRC}/runtime/core/portable_type/c10"
  "${TORCH_INCLUDE}"            # ordered last: only the missing cmake_macros.h falls through to torch
)

target_link_libraries(fedlearn_core PUBLIC
  ${_force_ops}
  ${_resolved}
)
target_compile_features(fedlearn_core PUBLIC cxx_std_17)

if(FEDLEARN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# --- Optional gRPC transport layer (15-LLD §13 tasks 9-10): FedLearnClient over gRPC ---
# Requires the buf-generated C++ stubs (run `buf generate` from proto/ -> gen/cpp) plus a
# cross-compiled gRPC C++ runtime + protobuf. NOT needed for the host parity gate above, so it
# is OFF by default. FederatedLoop + DataLoader now live in fedlearn_core (FederatedLoop is
# gRPC-free via IFedLearnClient); this layer only adds the concrete gRPC client.
option(FEDLEARN_BUILD_GRPC "Build the gRPC client transport (needs buf stubs + gRPC runtime)" OFF)
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
  )
  target_compile_features(fedlearn_grpc PUBLIC cxx_std_17)

  if(FEDLEARN_BUILD_TESTS)
    add_executable(fedlearn_grpc_tests tests/grpc_marshal_test.cpp)
    target_link_libraries(fedlearn_grpc_tests PRIVATE fedlearn_grpc GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(fedlearn_grpc_tests)
  endif()
endif()
