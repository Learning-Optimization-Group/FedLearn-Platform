// mlp_masked_round_test.cpp — Stage 4 S3b. The MLP trains on a device as its contract states: Adam (the laptop's
// optimizer), seeded minibatches, and each dropout layer's mask drawn from DROPOUT_MASKS_SEEDED_V1 and fed to the
// program as an input. The uploaded update must land on torch training the same masked graph eagerly
// (framework/tests/fixtures/mlp_golden/mlp_manifest.json), within a tolerance far below every wrong-mask control.
#include "fedlearn/FederatedLoop.h"
#include "fedlearn/IFedLearnClient.h"
#include "fedlearn/ModelManager.h"
#include "fedlearn/TrainableExecutorchModel.h"
#include "fixtures.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace {

std::string mlpPath(const std::string& file) { return std::string(MLP_DIR) + "/" + file; }

constexpr const char* kPteSha = "237ce3dae80fa8374e34fde0bbc328c265df2818c6de10e1a98b62d25f22c297";
constexpr const char* kLossSha = "302c82310d1feae3e806dc1663186e7ebc889cdf477c63c3d3dff0496dcee830";
const std::vector<std::string> kNames = {"base.fc1.weight", "base.fc1.bias", "base.fc2.weight",
                                         "base.fc2.bias", "base.fc3.weight", "base.fc3.bias"};
const std::vector<fedlearn::ParamSpec> kLayout = {{"fc1.weight", {64, 140}}, {"fc1.bias", {64}},
                                                  {"fc2.weight", {64, 64}}, {"fc2.bias", {64}},
                                                  {"fc3.weight", {2, 64}}, {"fc3.bias", {2}}};
constexpr int64_t kExamples = 20;
constexpr int kEpochs = 2;
constexpr double kLr = 1e-3;
constexpr uint64_t kSeed = 42;
constexpr float kAtol = 1e-5f;  // mlp_manifest.endpoint_atol; the nearest control is 3.9e-3 away

class RoundMock : public fedlearn::IFedLearnClient {
 public:
  std::string globalBlob;
  std::string lastModelBlob;
  bool shouldStop() const override { return false; }
  fedlearn::DeComFLConfig getDeComFLConfig(const std::string&, const std::string&) override { return {}; }
  void submitGradientScalars(const std::string&, const std::string&, int, const fedlearn::Seeds2D&,
                             const fedlearn::GradientScalars2D&, int64_t) override {}
  std::string getGlobalModelStream(const std::string&, const std::string&, int* outRound,
                                   std::map<std::string, std::string>* = nullptr) override {
    if (outRound) *outRound = 1;
    return globalBlob;
  }
  void submitModelUpdate(const std::string&, const std::string&, int, const std::string& blob, int64_t) override {
    lastModelBlob = blob;
  }
};

fedlearn::ModelManager mlpManager() {
  fedlearn::ModelManager mm;
  mm.loadModel(mlpPath("mlp_loss_dynbatch.pte"), kLossSha, kLayout, /*totalParamCount=*/13314);
  return mm;
}

std::vector<float> runRound(const fedlearn::DropoutSpec& dropout) {
  fedlearn::ModelManager seed = mlpManager();
  seed.setFlatParams(fedtest::readF32(mlpPath("mlp_init.f32")));
  RoundMock mock;
  mock.globalBlob = seed.serializeStateDict(kExamples);
  const auto x = fedtest::readF32(mlpPath("mlp_inputs.f32"));
  const auto y = fedtest::readI64(mlpPath("mlp_targets.i64"));
  fedlearn::TrainableExecutorchModel model(mlpPath("mlp_trainable_masked_dynbatch.pte"), kPteSha, kNames);
  fedlearn::ModelManager mm = mlpManager();
  fedlearn::FederatedLoop loop(mock, mm);
  fedlearn::DataBatch data{x.data(), {kExamples, 140}, y.data(), kExamples};
  fedlearn::LocalOptimizer adam;
  adam.adam = true;
  const auto out = loop.firstOrderRound(model, "run", "client", data, kEpochs, kLr, false, 0.0,
                                        fedlearn::LocalBatching{8, true, kSeed}, adam, dropout);
  EXPECT_TRUE(out.ranTraining);
  fedlearn::ModelManager decode = mlpManager();
  decode.loadStateDict(mock.lastModelBlob);
  return decode.getFlatParams();
}

float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

}  // namespace

TEST(MlpMaskedRound, SeededMasksAndAdamLandOnTheTorchEndpoint) {
  const auto got = runRound(fedlearn::DropoutSpec{{0.3, 0.3}, kSeed});
  const auto golden = fedtest::readF32(mlpPath("mlp_masked_adam_final.f32"));
  ASSERT_EQ(got.size(), golden.size());
  const float diff = maxAbsDiff(got, golden);
  std::printf("[mlp] max |native - torch| = %.3g\n", diff);
  EXPECT_LT(diff, kAtol);
}

TEST(MlpMaskedRound, AnotherSeedDrawsOtherMasks) {
  const auto got = runRound(fedlearn::DropoutSpec{{0.3, 0.3}, kSeed + 1});
  EXPECT_GT(maxAbsDiff(got, fedtest::readF32(mlpPath("mlp_masked_adam_final.f32"))), kAtol * 10);
}

TEST(MlpMaskedRound, ADropoutListThatDisagreesWithTheProgramIsRefused) {
  EXPECT_THROW(runRound(fedlearn::DropoutSpec{{0.3}, kSeed}), std::runtime_error);
}
