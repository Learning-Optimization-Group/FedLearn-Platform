// cnn_round_test.cpp — Stage 4 S6. The CNN (CIFAR-10 LeNet: conv-pool-conv-pool-fc-fc-fc) trains on a device as its
// contract states: images prepared by ImageToUnitTensor and NormalizeChannels, Adam (the laptop's optimizer) and seeded
// minibatches with the final partial one kept. The uploaded update must land on torch training the same model eagerly
// (framework/tests/fixtures/cnn_golden/cnn_manifest.json), within a tolerance far below every wrong-training control.
// This is also the first test of a Conv/MaxPool backward pass through ExecuTorch's training module.
#include "fedlearn/DatasetEvaluation.h"
#include "fedlearn/ExecutorchModel.h"
#include "fedlearn/FederatedLoop.h"
#include "fedlearn/IFedLearnClient.h"
#include "fedlearn/ModelManager.h"
#include "fedlearn/Qualification.h"
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

std::string cnnPath(const std::string& file) { return std::string(CNN_DIR) + "/" + file; }

constexpr const char* kPteSha = "471f222d1c2d530da58cb5ee3801ca5ef3453ad52c3c272649dcb07bb4c0a190";
constexpr const char* kLossSha = "5c945589ae5a39cb46035a646719dbd2a002e91348c3d9ae20c31a09039e4ba8";
const std::vector<std::string> kNames = {"base.conv1.weight", "base.conv1.bias", "base.conv2.weight",
                                         "base.conv2.bias",   "base.fc1.weight",  "base.fc1.bias",
                                         "base.fc2.weight",   "base.fc2.bias",    "base.fc3.weight",
                                         "base.fc3.bias"};
const std::vector<fedlearn::ParamSpec> kLayout = {
    {"conv1.weight", {6, 3, 5, 5}}, {"conv1.bias", {6}},  {"conv2.weight", {16, 6, 5, 5}}, {"conv2.bias", {16}},
    {"fc1.weight", {120, 400}},     {"fc1.bias", {120}},  {"fc2.weight", {84, 120}},      {"fc2.bias", {84}},
    {"fc3.weight", {10, 84}},       {"fc3.bias", {10}}};
constexpr int64_t kExamples = 20;
constexpr int kEpochs = 2;
constexpr double kLr = 1e-3;
constexpr uint64_t kSeed = 42;
// cnn_manifest.endpoint_atol. Not the MLP's 1e-5: torch's own float32 training of this golden differs from float64 by
// 1.26e-5, because Adam normalises near-zero gradients; native measured 1.1e-5. The nearest control is 6.0e-3 away.
constexpr float kAtol = 1e-4f;

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

fedlearn::ModelManager cnnManager() {
  fedlearn::ModelManager mm;
  mm.loadModel(cnnPath("cnn_loss_dynbatch.pte"), kLossSha, kLayout, /*totalParamCount=*/62006);
  return mm;
}

std::vector<float> runRound(int64_t batchSize, uint64_t batchSeed = kSeed) {
  fedlearn::ModelManager seed = cnnManager();
  seed.setFlatParams(fedtest::readF32(cnnPath("cnn_init.f32")));
  RoundMock mock;
  mock.globalBlob = seed.serializeStateDict(kExamples);
  const auto x = fedtest::readF32(cnnPath("cnn_inputs.f32"));
  const auto y = fedtest::readI64(cnnPath("cnn_targets.i64"));
  fedlearn::TrainableExecutorchModel model(cnnPath("cnn_trainable_dynbatch.pte"), kPteSha, kNames);
  fedlearn::ModelManager mm = cnnManager();
  fedlearn::FederatedLoop loop(mock, mm);
  fedlearn::DataBatch data{x.data(), {kExamples, 3, 32, 32}, y.data(), kExamples};
  fedlearn::LocalOptimizer adam;
  adam.adam = true;
  const auto out = loop.firstOrderRound(model, "run", "client", data, kEpochs, kLr, false, 0.0,
                                        fedlearn::LocalBatching{batchSize, true, batchSeed}, adam);
  EXPECT_TRUE(out.ranTraining);
  fedlearn::ModelManager decode = cnnManager();
  decode.loadStateDict(mock.lastModelBlob);
  return decode.getFlatParams();
}

float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

}  // namespace

TEST(CnnRound, ImagesAdamAndSeededMinibatchesLandOnTheTorchEndpoint) {
  const auto got = runRound(8);
  const auto golden = fedtest::readF32(cnnPath("cnn_adam_final.f32"));
  ASSERT_EQ(got.size(), golden.size());
  const float diff = maxAbsDiff(got, golden);
  std::printf("[cnn] max |native - torch| = %.3g\n", diff);
  EXPECT_LT(diff, kAtol);
}

TEST(CnnRound, AnotherBatchOrderLandsElsewhere) {
  // The contract's batches, visited in the order another seed draws: the golden must reject it. (A whole-dataset batch
  // cannot be the control here: the program takes at most 8 examples per call, and refuses 20.)
  const auto got = runRound(8, kSeed + 1);
  EXPECT_GT(maxAbsDiff(got, fedtest::readF32(cnnPath("cnn_adam_final.f32"))), kAtol * 10);
}

TEST(CnnRound, MoreExamplesThanTheProgramTakesAreRefused) {
  EXPECT_THROW(runRound(kExamples), std::runtime_error);
}

TEST(CnnRound, TheProgramQualifiesWithAnImageShapedProbe) {
  // cnn_manifest.json probe: the flat probe pattern viewed as [rows, 3, 32, 32].
  fedlearn::ProbeSpec s;
  s.rows = 8;
  s.width = 3072;
  s.classes = 10;
  s.inputShape = {3, 32, 32};
  s.learningRate = 0.1f;
  s.expectedLossStep1 = 2.2907896041870117;
  s.expectedLossStep2 = 2.2854325771331787;
  s.lossTolerance = 1e-4;
  s.maxProbeMs = 5000;
  const auto r = fedlearn::qualifyTrainable(cnnPath("cnn_trainable_dynbatch.pte"), kPteSha, kNames, s);
  EXPECT_TRUE(r.passed) << r.failedCheck << ": " << r.detail;
  std::printf("[cnn] probe %.7f %.7f in %lld ms\n", r.lossStep1, r.lossStep2, static_cast<long long>(r.wallMs));
}

TEST(CnnRound, AProbeShapeThatDoesNotHoldItsWidthIsRefused) {
  fedlearn::ProbeSpec s;
  s.rows = 8;
  s.width = 3000;
  s.classes = 10;
  s.inputShape = {3, 32, 32};
  s.learningRate = 0.1f;
  s.expectedLossStep1 = 2.29;
  s.expectedLossStep2 = 2.28;
  s.lossTolerance = 1e-4;
  s.maxProbeMs = 5000;
  const auto r = fedlearn::qualifyTrainable(cnnPath("cnn_trainable_dynbatch.pte"), kPteSha, kNames, s);
  EXPECT_FALSE(r.passed);
  EXPECT_EQ(r.failedCheck, "SPEC");
}

TEST(CnnRound, EvaluationRespectsTheBoundExecuTorchPlannedTheProgramsFor) {
  // The live phone run's failure. The run's loss and infer programs were exported for 32 examples, but ExecuTorch planned
  // them for 15, so evaluating a dataset in chunks of the training batch (32) failed every round with
  // set_input(x) error 16. Evaluation asks each program for its bound instead.
  fedlearn::ExecutorchModel loss(cnnPath("cnn_loss_bound32.pte"),
                                 "dc64e8c0328c4934b0f88eab2258aa138b5143657bafe5275183b50cfa1a2ab0");
  fedlearn::ExecutorchModel infer(cnnPath("cnn_infer_bound32.pte"),
                                  "aedbeb084fe11f0452ff41137c54404b59bf963ce107a2d9783837b764d58f8a");
  EXPECT_EQ(loss.maxExamplesPerCall(), 15);
  EXPECT_EQ(infer.maxExamplesPerCall(), 15);
  const auto flat = fedtest::readF32(cnnPath("cnn_init.f32"));
  const auto x = fedtest::readF32(cnnPath("cnn_inputs.f32"));
  const auto y = fedtest::readI64(cnnPath("cnn_targets.i64"));
  const fedlearn::DataBatch data{x.data(), {kExamples, 3, 32, 32}, y.data(), kExamples};
  const auto asked32 = fedlearn::evaluateDataset(loss, infer, flat, data, 32);
  const auto by10 = fedlearn::evaluateDataset(loss, infer, flat, data, 10);
  EXPECT_NEAR(asked32.loss, by10.loss, 1e-6);
  EXPECT_EQ(asked32.accuracy, by10.accuracy);
  std::printf("[cnn] dataset loss %.6f accuracy %.3f with programs bounded at 15\n", asked32.loss, asked32.accuracy);
}
