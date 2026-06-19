// et_infer_test.cpp — ExecuTorch infer-path release gate (libtorch-free).
//
// Verifies ExecutorchModel::infer() on the infer .pte (forward(flat,x)->logits), confirming
// that argmax accuracy matches the Python-frozen golden from zo_manifest.json (T11).
//
// GOLDEN_DIR is injected by CMake and points at framework/tests/fixtures/decomfl_golden.
#include "fedlearn/ExecutorchModel.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
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

// Frozen infer golden (framework/tests/fixtures/decomfl_golden/zo_manifest.json).
const char* kInferPteSha256 = "cf8744b9579d78f14bbb82e2d4ce98dcaffc8d2c6ed2253349c39342de546746";
const int golden_argmax[8] = {0, 0, 2, 0, 0, 0, 2, 2};
constexpr double kGoldenAccuracy = 0.375;

std::string inferPtePath() { return std::string(GOLDEN_DIR) + "/zo_model_tiny_infer.pte"; }

}  // namespace

TEST(EtInfer, LogitsSizeAndArgmaxMatchGolden) {
  fedlearn::ExecutorchModel model(inferPtePath(), kInferPteSha256);

  const auto flat = ReadBin<float>(std::string(GOLDEN_DIR) + "/zo_flat.f32");
  const auto x = ReadBin<float>(std::string(GOLDEN_DIR) + "/zo_inputs.f32");
  const auto y = ReadBin<int64_t>(std::string(GOLDEN_DIR) + "/zo_targets.i64");
  ASSERT_EQ(flat.size(), 25u);
  ASSERT_EQ(x.size(), 32u);  // 8 samples × 4 features
  ASSERT_EQ(y.size(), 8u);

  // Infer graph: forward(flat, x) -> logits (shape [8, 3]).
  auto logits = model.infer(flat, x.data(), {8, 4});
  ASSERT_EQ(logits.size(), 24u);  // 8 batch × 3 classes

  // Compute per-row argmax and compare against the Python-frozen golden.
  std::vector<int> pred;
  pred.reserve(8);
  for (int row = 0; row < 8; ++row) {
    int best = 0;
    float best_val = logits[static_cast<size_t>(row) * 3];
    for (int col = 1; col < 3; ++col) {
      float v = logits[static_cast<size_t>(row) * 3 + col];
      if (v > best_val) { best_val = v; best = col; }
    }
    pred.push_back(best);
  }
  for (int i = 0; i < 8; ++i) {
    EXPECT_EQ(pred[i], golden_argmax[i]) << "argmax mismatch at row " << i;
  }

  // Accuracy: fraction of rows where argmax matches target.
  int matches = 0;
  for (int i = 0; i < 8; ++i) {
    if (pred[i] == static_cast<int>(y[i])) ++matches;
  }
  double acc = matches / 8.0;
  EXPECT_NEAR(acc, kGoldenAccuracy, 1e-6);

  // All logits must be finite.
  for (size_t i = 0; i < logits.size(); ++i) {
    EXPECT_TRUE(std::isfinite(logits[i])) << "non-finite logit at index " << i;
  }
}
