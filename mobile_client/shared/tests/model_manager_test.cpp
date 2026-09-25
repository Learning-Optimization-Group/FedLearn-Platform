// model_manager_test.cpp — verify-before-load (E8) + param counts (ExecuTorch, torch-free).
#include "fedlearn/ModelManager.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "fixtures.h"

TEST(ModelManager, LoadsWithCorrectShaAndReportsCounts) {
  fedlearn::ModelManager mm;
  fedlearn::ModelInfo info;
  mm.loadModel(fedtest::goldenPath("zo_model_tiny.pte"), fedtest::kTinyPteSha,
               fedtest::tinyLayout(), /*totalParamCount=*/43, &info);
  EXPECT_EQ(info.paramCount, 43);
  EXPECT_EQ(info.trainableParamCount, 25);
  EXPECT_EQ(info.tier, "");  // 43 params is below the 1M tier
  EXPECT_EQ(mm.getFlatParams().size(), 25u);
}

TEST(ModelManager, RejectsTamperedModel) {
  fedlearn::ModelManager mm;
  EXPECT_THROW(
      mm.loadModel(fedtest::goldenPath("zo_model_tiny.pte"), std::string(64, '0'),
                   fedtest::tinyLayout(), 43, nullptr),
      std::runtime_error);
}

TEST(ModelManager, RejectsLayoutDisagreeingWithModel) {
  fedlearn::ModelManager mm;
  // A layout summing to 24 (not the model's flat dim 25) must error.
  std::vector<fedlearn::ParamSpec> bad = {{"fc1.weight", {5, 4}}, {"fc1.bias", {4}}};
  EXPECT_THROW(mm.loadModel(fedtest::goldenPath("zo_model_tiny.pte"), fedtest::kTinyPteSha, bad, 43, nullptr),
               std::runtime_error);
}

// The canonical digest of a trainable state (execution_plan.state_sha256): SHA-256 of its float32 safetensors
// encoding with no metadata, tensors in layout order. The committed golden TinyNet initial state must hash to the
// digest the golden execution contract binds, so the phone can prove which model it starts from.
TEST(ModelManager, CanonicalStateDigestIsTheContractsDigestForTheGoldenInitialState) {
  fedlearn::ModelManager mm = fedtest::makeManager();
  mm.setFlatParams(fedtest::readF32(fedtest::goldenPath("zo_flat.f32")));
  EXPECT_EQ(mm.canonicalStateSha256(), "1122ba73e49f6df981861bb76d3dcff46666abb5f41e3a6a4d510db9fddd965c");
}
