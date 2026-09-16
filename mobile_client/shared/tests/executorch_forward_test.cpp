// executorch_forward_test.cpp — the ExecuTorch C++ forward release gate.
//
// Asserts ExecutorchModel (the .pte runtime wrapper that replaces torch::jit forward) reproduces
// the frozen golden_loss for the DeComFL ZO fixture, and that sha256 verification rejects a
// tampered .pte. Runs WITHOUT libtorch (separate target from fedlearn_core_tests, which links
// libtorch — mixing the two would clash on vendored c10/aten symbols).
//
// GOLDEN_DIR is injected by CMake and points at framework/tests/fixtures/decomfl_golden.
#include "fedlearn/ExecutorchModel.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef GOLDEN_DIR
#define GOLDEN_DIR "."
#endif

namespace {

template <typename T>
std::vector<T> ReadBin(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  EXPECT_TRUE(f.good()) << "cannot open fixture: " << path;
  std::vector<T> out;
  T x;
  while (f.read(reinterpret_cast<char*>(&x), sizeof(T))) out.push_back(x);
  return out;
}

// Frozen in framework/tests/fixtures/decomfl_golden/zo_manifest.json (torch 2.12.0).
const char* kPteSha256 = "2eca3c02e2084383f038494d6ecf7c20a1e7e0a1dcc6d7ce2b6e11e7d82f1c56";
constexpr double kGoldenLoss = 1.0973092317581177;

std::string ptePath() { return std::string(GOLDEN_DIR) + "/zo_model_tiny.pte"; }

}  // namespace

TEST(ExecutorchForward, MatchesGoldenLoss) {
  fedlearn::ExecutorchModel model(ptePath(), kPteSha256);
  EXPECT_EQ(model.flatDim(), 25);

  const auto flat = ReadBin<float>(std::string(GOLDEN_DIR) + "/zo_flat.f32");
  const auto x = ReadBin<float>(std::string(GOLDEN_DIR) + "/zo_inputs.f32");
  const auto y = ReadBin<int64_t>(std::string(GOLDEN_DIR) + "/zo_targets.i64");
  ASSERT_EQ(flat.size(), 25u);
  ASSERT_EQ(x.size(), 32u);  // 8 x 4
  ASSERT_EQ(y.size(), 8u);

  const float loss = model.loss(flat, x.data(), {8, 4}, y.data(), 8);
  EXPECT_NEAR(loss, static_cast<float>(kGoldenLoss), 1e-4f)
      << "ExecuTorch forward diverged from the golden_loss reference";
}

TEST(ExecutorchForward, RejectsShaMismatch) {
  EXPECT_THROW(fedlearn::ExecutorchModel(ptePath(), "deadbeefdeadbeef"), std::runtime_error);
}
