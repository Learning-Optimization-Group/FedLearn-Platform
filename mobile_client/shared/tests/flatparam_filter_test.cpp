// flatparam_filter_test.cpp — the requires_grad filter must exclude frozen layers, and the
// trainable count must equal Python's (15-LLD §13 task 6; fixes A6 M-C2 #3).
#include "fedlearn/ModelManager.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "fixtures.h"

TEST(FlatParamFilter, ExcludesFrozenLayer) {
  fedlearn::ModelManager mm;
  torch::jit::Module model = fedtest::loadTinyModel();

  // TinyNet: fc1 trainable (25), fc2 FROZEN (18). Total 43, trainable 25.
  EXPECT_EQ(mm.trainableParamCount(model), 25);
  EXPECT_EQ(mm.getFlatParams(model).numel(), 25);

  int64_t total = 0;
  for (const auto& p : model.parameters()) total += p.numel();
  EXPECT_EQ(total, 43);
}

TEST(FlatParamFilter, GetSetIsLossless) {
  fedlearn::ModelManager mm;
  torch::jit::Module model = fedtest::loadTinyModel();

  torch::Tensor before = mm.getFlatParams(model).clone();
  torch::Tensor modified = before + 1.0;
  mm.setFlatParams(model, modified);
  EXPECT_TRUE(torch::allclose(mm.getFlatParams(model), modified, 1e-6, 1e-6));

  mm.setFlatParams(model, before);  // restore
  EXPECT_TRUE(torch::allclose(mm.getFlatParams(model), before, 1e-6, 1e-6));
}
