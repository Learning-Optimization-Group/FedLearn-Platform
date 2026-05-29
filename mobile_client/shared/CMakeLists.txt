# shared/ — the cross-platform C++ FL core (libfedlearn_core).
# Increment 1 builds the determinism slice (Perturbation, DtypeMap). Later tasks add
# ModelManager, ZerothOrderEstimator, DeComFLClient, FederatedLoop, FedLearnClient,
# DataLoader (15-LLD §13 tasks 6-10).

add_library(fedlearn_core STATIC
  src/Perturbation.cpp
  src/DtypeMap.cpp
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
