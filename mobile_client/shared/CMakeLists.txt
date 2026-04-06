cmake_minimum_required(VERSION 3.18)
project(fedlearn_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Dependencies ----

# libtorch: set LIBTORCH_DIR to the pre-built libtorch installation
if(NOT DEFINED LIBTORCH_DIR)
  message(FATAL_ERROR "LIBTORCH_DIR must be set to the libtorch installation path")
endif()
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
find_package(Torch REQUIRED)

# gRPC and Protobuf
find_package(protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

# ---- Proto generation ----

set(PROTO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/proto")
set(PROTO_GEN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/generated")
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

# Find protoc and grpc_cpp_plugin
find_program(PROTOC protoc)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)

set(FEDLEARN_PROTO "${PROTO_SRC_DIR}/fedlearn.proto")
set(FEDLEARN_PB_CC "${PROTO_GEN_DIR}/fedlearn.pb.cc")
set(FEDLEARN_PB_H "${PROTO_GEN_DIR}/fedlearn.pb.h")
set(FEDLEARN_GRPC_CC "${PROTO_GEN_DIR}/fedlearn.grpc.pb.cc")
set(FEDLEARN_GRPC_H "${PROTO_GEN_DIR}/fedlearn.grpc.pb.h")

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

# ---- Core library ----

add_library(fedlearn_core STATIC
  src/ModelManager.cpp
  src/ZerothOrderEstimator.cpp
  src/DeComFLClient.cpp
  src/DataLoader.cpp
  src/FedLearnClient.cpp
  src/FederatedLoop.cpp
  src/NativeFedLearnCore.cpp
  ${FEDLEARN_PB_CC}
  ${FEDLEARN_GRPC_CC}
)

target_include_directories(fedlearn_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${PROTO_GEN_DIR}
)

target_link_libraries(fedlearn_core PUBLIC
  ${TORCH_LIBRARIES}
  gRPC::grpc++
  protobuf::libprotobuf
)

# nlohmann/json header-only library for DataLoader JSON parsing
# If available via system/vcpkg:
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
  target_link_libraries(fedlearn_core PRIVATE nlohmann_json::nlohmann_json)
else()
  # Fallback: download header-only
  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  )
  FetchContent_MakeAvailable(nlohmann_json)
  target_link_libraries(fedlearn_core PRIVATE nlohmann_json::nlohmann_json)
endif()
