#pragma once
// Shared helpers for loading the Python-frozen golden fixtures
// (framework/tests/fixtures/decomfl_golden/). GOLDEN_DIR is injected by CMake.
// Torch-free (Phase 3c): returns plain vectors + a pre-loaded ExecuTorch ModelManager.
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "fedlearn/ModelManager.h"

#ifndef GOLDEN_DIR
#define GOLDEN_DIR "."
#endif

namespace fedtest {

inline std::string goldenPath(const std::string& file) {
  return std::string(GOLDEN_DIR) + "/" + file;
}

inline std::vector<float> readF32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<float> v;
  float x = 0.0f;
  while (f.read(reinterpret_cast<char*>(&x), sizeof(float))) v.push_back(x);
  return v;
}

inline std::vector<int64_t> readI64(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<int64_t> v;
  int64_t x = 0;
  while (f.read(reinterpret_cast<char*>(&x), sizeof(int64_t))) v.push_back(x);
  return v;
}

// The fixed batch frozen by generate_zo.py: 8 samples of dim 4 (shape {8,4}); int64 targets (len 8).
inline std::vector<float> zoInputs() { return readF32(goldenPath("zo_inputs.f32")); }
inline std::vector<int64_t> zoTargets() { return readI64(goldenPath("zo_targets.i64")); }

// TinyNet trainable layout + the .pte sha, frozen in zo_manifest.json (param_layout / pte_sha256).
inline std::vector<fedlearn::ParamSpec> tinyLayout() {
  return {{"fc1.weight", {5, 4}}, {"fc1.bias", {5}}};
}
inline const char* kTinyPteSha = "2eca3c02e2084383f038494d6ecf7c20a1e7e0a1dcc6d7ce2b6e11e7d82f1c56";

// A ModelManager pre-loaded from the tiny .pte (params_ zero-initialised to flatDim=25).
inline fedlearn::ModelManager makeManager(fedlearn::ModelInfo* info = nullptr) {
  fedlearn::ModelManager mm;
  mm.loadModel(goldenPath("zo_model_tiny.pte"), kTinyPteSha, tinyLayout(), /*totalParamCount=*/43, info);
  return mm;
}

}  // namespace fedtest
