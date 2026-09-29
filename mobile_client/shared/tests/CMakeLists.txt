# gtest suite — the unified parity & roundtrip contract for the C++ FL core (15-LLD §13).
# One ExecuTorch-linked binary: the RNG/dtype/hash parity tests, the safetensors + model-manager
# roundtrip tests, the DeComFL-equivalence tests, the ExecuTorch forward/infer/g-scalar tests, and
# the gRPC-free FederatedLoop test. All ET runtime + core sources are compiled into fedlearn_core,
# so this target only lists the test translation units and links fedlearn_core (which carries ET).

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
# Execution contract v1 fixtures, shared with the Python, Java and TypeScript readers.
set(FEDLEARN_CONTRACT_DIR "${CMAKE_SOURCE_DIR}/../framework/tests/fixtures/execution_contract_v1")

add_executable(fedlearn_core_tests
  randn_parity_test.cpp
  dtype_map_test.cpp
  sha256_test.cpp
  model_manager_test.cpp
  flatparam_filter_test.cpp
  serialize_roundtrip_test.cpp
  decomfl_equivalence_test.cpp
  decomfl_base_loss_cache_test.cpp
  executorch_forward_test.cpp
  et_g_scalar_test.cpp
  et_multiround_test.cpp
  safetensors_test.cpp
  data_loader_test.cpp
  et_infer_test.cpp
  federated_loop_test.cpp
  auth_metadata_test.cpp
  eval_metrics_test.cpp
  tls_trust_test.cpp
  round_flight_test.cpp
  decomfl_server_config_test.cpp
  model_execution_error_test.cpp
  batch_order_test.cpp
  dataset_evaluation_test.cpp
  dropout_masks_test.cpp
)
# First-order FedAvg parity gtest — only when the training extension is built in (it links
# TrainableExecutorchModel, which needs the ET training libs).
if(FEDLEARN_BUILD_TRAINING)
  target_sources(fedlearn_core_tests PRIVATE
    fedavg_parity_test.cpp
    execution_contract_projection_test.cpp
    fedavg_firstorder_round_test.cpp
    dynamic_batch_training_test.cpp
    minibatch_round_test.cpp
    qualification_test.cpp
    adam_parity_test.cpp
    mlp_masked_round_test.cpp
  )
endif()

target_link_libraries(fedlearn_core_tests PRIVATE
  fedlearn_core
  GTest::gtest_main
)
target_compile_definitions(fedlearn_core_tests PRIVATE
  GOLDEN_DIR="${FEDLEARN_GOLDEN_DIR}"
  CONTRACT_DIR="${FEDLEARN_CONTRACT_DIR}"
  MLP_DIR="${CMAKE_SOURCE_DIR}/../framework/tests/fixtures/mlp_golden"
)

# Plain registration: gtest_discover_tests' build-time discovery run is unreliable here (the
# ExecuTorch runtime emits static-init stderr noise); one ctest entry running the whole binary
# is sufficient and deterministic.
enable_testing()
add_test(NAME fedlearn_core_tests COMMAND fedlearn_core_tests)
