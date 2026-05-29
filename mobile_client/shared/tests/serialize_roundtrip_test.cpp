// serialize_roundtrip_test.cpp — state-dict save/load symmetry (15-LLD §13 task 6, FedAvg path).
#include "fedlearn/ModelManager.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <string>

#include "fixtures.h"

TEST(SerializeRoundtrip, SaveThenLoadRestoresParams) {
  fedlearn::ModelManager mm;
  torch::jit::Module model = fedtest::loadTinyModel();

  const torch::Tensor original = mm.getFlatParams(model).clone();
  const std::string blob = mm.serializeStateDict(model, /*numExamples=*/7);
  ASSERT_FALSE(blob.empty());

  // Perturb the live params, then restore from the blob.
  mm.setFlatParams(model, original + 3.14);
  ASSERT_FALSE(torch::allclose(mm.getFlatParams(model), original, 1e-6, 1e-6));

  mm.loadStateDict(model, blob);
  EXPECT_TRUE(torch::allclose(mm.getFlatParams(model), original, 1e-6, 1e-6));
}
