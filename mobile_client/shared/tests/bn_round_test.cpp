// bn_round_test.cpp — Stage 4 S7. A device trains BatchNorm in training mode as a laptop does: its program normalises
// with each batch's statistics (BatchNorm rebuilt from primitive ops, returning them), and the round updates the running
// statistics by BATCH_NORM_RUNNING_STATS_V1 and federates them with the parameters. The uploaded state must land on
// torch's own nn.BatchNorm2d in training mode (framework/tests/fixtures/bn_golden/bn_manifest.json), parameters and
// running statistics alike, within a tolerance far below every wrong-training control.
#include "fedlearn/FederatedLoop.h"
#include "fedlearn/IFedLearnClient.h"
#include "fedlearn/ModelManager.h"
#include "fedlearn/TrainableExecutorchModel.h"
#include "fixtures.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace {

std::string bnPath(const std::string& file) { return std::string(BN_DIR) + "/" + file; }

constexpr const char* kPteSha = "aa4ff48cf7d38d43d812c1b73cb45c232a14b15b7d8ca5ec1f276167f6db6aa9";
constexpr const char* kLossSha = "7e9c5e9dd8082b197285d8d32b4371ad540bfee160c3b5e3c895973ad5e1a84e";
const std::vector<std::string> kNames = {"base.conv1.weight", "base.bn1.weight", "base.bn1.bias",
                                         "base.conv2.weight", "base.bn2.weight", "base.bn2.bias",
                                         "base.fc.weight",    "base.fc.bias"};
// The federated state: the trainable parameters, then each BatchNorm layer's running mean and variance. The convolutions
// have no bias, as in ResNet (a bias before BatchNorm has a true gradient of zero, which Adam cannot replay).
const std::vector<fedlearn::ParamSpec> kState = {
    {"conv1.weight", {8, 3, 3, 3}}, {"bn1.weight", {8}},       {"bn1.bias", {8}},         {"conv2.weight", {8, 8, 3, 3}},
    {"bn2.weight", {8}},            {"bn2.bias", {8}},         {"fc.weight", {10, 8}},    {"fc.bias", {10}},
    {"bn1.running_mean", {8}},      {"bn1.running_var", {8}},  {"bn2.running_mean", {8}}, {"bn2.running_var", {8}}};
constexpr int64_t kExamples = 20;
constexpr int64_t kParams = 914;
constexpr int kEpochs = 2;
constexpr double kLr = 1e-3;
constexpr uint64_t kSeed = 42;
constexpr float kAtol = 1e-4f;  // bn_manifest.endpoint_atol; the nearest control is 3.8e-3 away

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

fedlearn::ModelManager stateManager() {
  fedlearn::ModelManager mm;
  mm.loadModel(bnPath("bn_loss_dynbatch.pte"), kLossSha, kState, /*totalParamCount=*/kParams);
  return mm;
}

fedlearn::BatchNormSpec contractBatchNorm(double momentum = 0.1) {
  fedlearn::BatchNormSpec spec;
  spec.layers = {{8, momentum, 36}, {8, momentum, 4}};
  return spec;
}

std::vector<float> runRound(const fedlearn::BatchNormSpec& batchNorm) {
  fedlearn::ModelManager seed = stateManager();
  seed.setFlatParams(fedtest::readF32(bnPath("bn_init.f32")));
  RoundMock mock;
  mock.globalBlob = seed.serializeStateDict(kExamples);
  const auto x = fedtest::readF32(bnPath("bn_inputs.f32"));
  const auto y = fedtest::readI64(bnPath("bn_targets.i64"));
  fedlearn::TrainableExecutorchModel model(bnPath("bn_trainable_dynbatch.pte"), kPteSha, kNames);
  fedlearn::ModelManager mm = stateManager();
  fedlearn::FederatedLoop loop(mock, mm);
  fedlearn::DataBatch data{x.data(), {kExamples, 3, 8, 8}, y.data(), kExamples};
  fedlearn::LocalOptimizer adam;
  adam.adam = true;
  const auto out = loop.firstOrderRound(model, "run", "client", data, kEpochs, kLr, false, 0.0,
                                        fedlearn::LocalBatching{8, true, kSeed}, adam, fedlearn::DropoutSpec{},
                                        batchNorm);
  EXPECT_TRUE(out.ranTraining);
  fedlearn::ModelManager decode = stateManager();
  decode.loadStateDict(mock.lastModelBlob);
  return decode.getFlatParams();
}

float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b, size_t from = 0, size_t to = SIZE_MAX) {
  float m = 0;
  for (size_t i = from; i < std::min(a.size(), to); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

}  // namespace

TEST(BatchNormRound, ParametersAndRunningStatisticsLandOnTorchsTrainingModeBatchNorm) {
  const auto got = runRound(contractBatchNorm());
  const auto golden = fedtest::readF32(bnPath("bn_adam_final.f32"));
  ASSERT_EQ(got.size(), golden.size());
  const float params = maxAbsDiff(got, golden, 0, kParams);
  const float stats = maxAbsDiff(got, golden, kParams);
  std::printf("[bn] max |native - torch|: parameters %.3g, running statistics %.3g\n", params, stats);
  EXPECT_LT(params, kAtol);
  EXPECT_LT(stats, kAtol);
}

TEST(BatchNormRound, AnotherMomentumLandsElsewhere) {
  const auto got = runRound(contractBatchNorm(0.01));
  EXPECT_GT(maxAbsDiff(got, fedtest::readF32(bnPath("bn_adam_final.f32"))), kAtol * 10);
}

TEST(BatchNormRound, AStateCarryingStatisticsWithoutTheContractsBatchNormIsRefused) {
  EXPECT_THROW(runRound(fedlearn::BatchNormSpec{}), std::runtime_error);
}

TEST(BatchNormRound, BatchNormLayersTheProgramDoesNotReportAreRefused) {
  fedlearn::BatchNormSpec three = contractBatchNorm();
  three.layers.push_back({8, 0.1, 4});
  EXPECT_THROW(runRound(three), std::runtime_error);
}
