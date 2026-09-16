// flatparam_filter_test.cpp — the trainable flat dim must equal Python's 25 (frozen fc2 excluded
// at export time), and get/set must be lossless (15-LLD §13 task 6). ExecuTorch, torch-free.
#include "fedlearn/ModelManager.h"

#include <gtest/gtest.h>

#include <vector>

#include "fixtures.h"

TEST(FlatParamFilter, TrainableFlatDimMatchesManifest) {
  fedlearn::ModelInfo info;
  fedlearn::ModelManager mm = fedtest::makeManager(&info);
  // TinyNet: fc1 trainable (25), fc2 FROZEN (18) — frozen params are baked into the .pte, so the
  // trainable flat vector is exactly 25; total 43 comes from the manifest.
  EXPECT_EQ(mm.trainableParamCount(), 25);
  EXPECT_EQ(mm.getFlatParams().size(), 25u);
  EXPECT_EQ(info.paramCount, 43);
}

TEST(FlatParamFilter, GetSetIsLossless) {
  fedlearn::ModelManager mm = fedtest::makeManager();
  const std::vector<float> before = mm.getFlatParams();
  ASSERT_EQ(before.size(), 25u);

  std::vector<float> modified = before;
  for (auto& v : modified) v += 1.0f;
  mm.setFlatParams(modified);
  EXPECT_EQ(mm.getFlatParams(), modified);

  mm.setFlatParams(before);  // restore
  EXPECT_EQ(mm.getFlatParams(), before);
}

TEST(FlatParamFilter, SetFlatParamsRejectsWrongSize) {
  fedlearn::ModelManager mm = fedtest::makeManager();
  EXPECT_THROW(mm.setFlatParams(std::vector<float>(24, 0.0f)), std::runtime_error);
  EXPECT_THROW(mm.setFlatParams(std::vector<float>(26, 0.0f)), std::runtime_error);
}
