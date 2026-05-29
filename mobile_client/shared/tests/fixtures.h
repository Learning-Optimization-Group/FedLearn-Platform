#pragma once
// Shared helpers for loading the Python-frozen golden fixtures
// (framework/tests/fixtures/decomfl_golden/). GOLDEN_DIR is injected by CMake.
#include <torch/script.h>
#include <torch/torch.h>

#include <fstream>
#include <string>
#include <vector>

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

// The fixed batch frozen by generate_zo.py (8 samples of dim 4; int64 class targets).
inline torch::Tensor zoInputs() {
  std::vector<float> v = readF32(goldenPath("zo_inputs.f32"));
  return torch::from_blob(v.data(), {8, 4}, torch::kFloat32).clone();
}
inline torch::Tensor zoTargets() {
  std::vector<int64_t> v = readI64(goldenPath("zo_targets.i64"));
  return torch::from_blob(v.data(), {8}, torch::kLong).clone();
}
inline torch::jit::Module loadTinyModel() {
  return torch::jit::load(goldenPath("zo_model_tiny.pt"));
}

}  // namespace fedtest
