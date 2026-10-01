// minibatch_round_test.cpp — Stage 3 slice C2. A device dataset larger than one batch is trained as the contract
// states: every epoch visits the examples in BATCH_ORDER_SEEDED_PERMUTATION_V1 order, one SGD step per batch_size
// minibatch with the final partial one kept, and the uploaded update must land on the framework's LocalTrainer run
// in that same order (fedavg_minibatch_manifest.json). The manifest records how far each wrong order lands; the
// tolerance sits far below the nearest of them.
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

constexpr float kLr = 0.1f;                // fedavg_minibatch_manifest.learning_rate
constexpr int kEpochs = 2;                 // local_epochs
constexpr int64_t kExamples = 20;          // examples
constexpr int64_t kBatchSize = 8;          // batch_size
constexpr uint64_t kSeed = 42;             // seed
constexpr int kRound = 1;                  // round
constexpr float kNearestWrongOrder = 5.5e-3f;  // control_separation.epoch_zero_order_every_epoch (0.00554)
constexpr float kAtol = 1e-5f;             // well under a tenth of the nearest wrong order

constexpr const char* kDynamicPte = "tinynet_trainable_dynbatch.pte";
constexpr const char* kDynamicPteSha = "8df99d447a8883a7a65ace595493a15f40cc6339c1e23553fbf922ef9851f124";
const std::vector<std::string> kParamNames = {"base.fc1.weight", "base.fc1.bias"};

class RoundMock : public fedlearn::IFedLearnClient {
 public:
  std::string globalBlob;
  int globalRound = kRound;
  std::string lastModelBlob;
  int64_t lastNumExamples = -1;

  bool shouldStop() const override { return false; }
  fedlearn::DeComFLConfig getDeComFLConfig(const std::string&, const std::string&) override { return {}; }
  void submitGradientScalars(const std::string&, const std::string&, int, const fedlearn::Seeds2D&,
                             const fedlearn::GradientScalars2D&, int64_t) override {}
  std::string getGlobalModelStream(const std::string&, const std::string&, int* outRound,
                                   std::map<std::string, std::string>* = nullptr) override {
    if (outRound) *outRound = globalRound;
    return globalBlob;
  }
  void submitModelUpdate(const std::string&, const std::string&, int, const std::string& blob,
                         int64_t numExamples) override {
    lastModelBlob = blob;
    lastNumExamples = numExamples;
  }
};

std::vector<float> uploadedFlat(const RoundMock& mock) {
  fedlearn::ModelManager decode = fedtest::makeManager();
  decode.loadStateDict(mock.lastModelBlob);
  return decode.getFlatParams();
}

std::vector<float> runRound(RoundMock& mock, const fedlearn::LocalBatching& batching) {
  const auto init = fedtest::readF32(fedtest::goldenPath("zo_flat.f32"));
  fedlearn::ModelManager seed = fedtest::makeManager();
  seed.setFlatParams(init);
  mock.globalBlob = seed.serializeStateDict(8);

  const auto x = fedtest::readF32(fedtest::goldenPath("minibatch_inputs.f32"));
  const auto y = fedtest::readI64(fedtest::goldenPath("minibatch_targets.i64"));
  fedlearn::TrainableExecutorchModel model(fedtest::goldenPath(kDynamicPte), kDynamicPteSha, kParamNames);
  fedlearn::ModelManager mm = fedtest::makeManager();
  fedlearn::FederatedLoop loop(mock, mm);
  fedlearn::DataBatch data{x.data(), {kExamples, 4}, y.data(), kExamples};
  const auto out = loop.firstOrderRound(model, "run", "client", data, kEpochs, kLr, false, 0.0, batching);
  EXPECT_TRUE(out.ranTraining);
  return uploadedFlat(mock);
}

float maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

}  // namespace

TEST(MinibatchRound, SeededMinibatchesLandOnTheFrameworkEndpoint) {
  ASSERT_LT(kAtol * 10, kNearestWrongOrder);
  RoundMock mock;
  const auto got = runRound(mock, fedlearn::LocalBatching{kBatchSize, true, kSeed});
  const auto golden = fedtest::readF32(fedtest::goldenPath("fedavg_minibatch_final.f32"));
  ASSERT_EQ(got.size(), golden.size());
  for (size_t i = 0; i < golden.size(); ++i)
    EXPECT_NEAR(got[i], golden[i], kAtol) << "minibatch endpoint diverged at param " << i;
  EXPECT_EQ(mock.lastNumExamples, kExamples);  // the update is weighted by every example, not one batch
  std::printf("[minibatch] max |native - framework| = %.3g\n", maxAbsDiff(got, golden));
}

// The permutation depends on the run seed: a different seed trains another order and must not land on the golden.
TEST(MinibatchRound, AnotherSeedTrainsAnotherOrder) {
  RoundMock mock;
  const auto got = runRound(mock, fedlearn::LocalBatching{kBatchSize, true, kSeed + 1});
  EXPECT_GT(maxAbsDiff(got, fedtest::readF32(fedtest::goldenPath("fedavg_minibatch_final.f32"))), kAtol * 10);
}

// So does the round: the server's round number, not a local counter, seeds the permutation.
TEST(MinibatchRound, TheServerRoundSeedsThePermutation) {
  RoundMock mock;
  mock.globalRound = kRound + 1;
  const auto got = runRound(mock, fedlearn::LocalBatching{kBatchSize, true, kSeed});
  EXPECT_GT(maxAbsDiff(got, fedtest::readF32(fedtest::goldenPath("fedavg_minibatch_final.f32"))), kAtol * 10);
}

TEST(MinibatchRound, AnInvalidBatchSizeIsRefusedBeforeUpload) {
  RoundMock mock;
  EXPECT_THROW(runRound(mock, fedlearn::LocalBatching{-1, true, kSeed}), std::runtime_error);
  EXPECT_TRUE(mock.lastModelBlob.empty());
}
