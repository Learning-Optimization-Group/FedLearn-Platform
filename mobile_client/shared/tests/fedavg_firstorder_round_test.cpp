// fedavg_firstorder_round_test.cpp — Phase B M2. The TRUE first-order (FedAvg) ROUND must produce
// the framework's LocalTrainer.fit endpoint AND upload it as a WEIGHT BLOB (submitModelUpdate), not
// ZO scalars. M1c proved TrainableExecutorchModel's compute parity; this pins the whole round
// orchestration end-to-end with a mock seam: GetGlobalModelStream -> K SGD steps -> serializeStateDict
// -> submitModelUpdate, and the uploaded blob decodes to the framework golden within tolerance.
#include "fedlearn/FederatedLoop.h"
#include "fedlearn/IFedLearnClient.h"
#include "fedlearn/ModelManager.h"
#include "fedlearn/TrainableExecutorchModel.h"
#include "fixtures.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

constexpr float kLr = 0.1f;             // fedavg_local_manifest.learning_rate
constexpr int kLocalEpochs = 5;         // fedavg_local_manifest.local_epochs (== full-batch SGD steps)
constexpr float kEndpointAtol = 2e-3f;  // fedavg_local_manifest.endpoint_atol
constexpr const char* kTrainablePte = "tinynet_trainable.pte";
constexpr const char* kTrainablePteSha =
    "ff398410f7339172295386dfc6220c5f46f21eddfb8ea145daf54e6a15dae412";
const std::vector<std::string> kParamNames = {"base.fc1.weight", "base.fc1.bias"};

// Minimal seam mock: hands back a canned global-model blob, captures the uploaded weight blob.
class RoundMock : public fedlearn::IFedLearnClient {
 public:
  std::string globalBlob;
  int globalRound = 0;
  bool submitCalled = false;
  std::string lastModelBlob;
  int64_t lastNumExamples = -1;

  bool shouldStop() const override { return false; }
  fedlearn::DeComFLConfig getDeComFLConfig(const std::string&, const std::string&) override { return {}; }
  void submitGradientScalars(const std::string&, const std::string&, int, const fedlearn::Seeds2D&,
                             const fedlearn::GradientScalars2D&, int64_t) override {}
  std::string getGlobalModelStream(const std::string&, const std::string&, int* outCurrentRound) override {
    if (outCurrentRound) *outCurrentRound = globalRound;
    return globalBlob;
  }
  void submitModelUpdate(const std::string&, const std::string&, int, const std::string& modelBlob,
                         int64_t numExamples) override {
    submitCalled = true;
    lastModelBlob = modelBlob;
    lastNumExamples = numExamples;
  }
};

// Serialize a trainable flat into the global-model blob the server would stream down.
std::string blobFromFlat(const std::vector<float>& flat) {
  fedlearn::ModelManager mm = fedtest::makeManager();
  mm.setFlatParams(flat);
  return mm.serializeStateDict(/*numExamples=*/8);
}

}  // namespace

TEST(FedAvgFirstOrderRound, EndpointMatchesFrameworkGoldenAndUploadsWeightBlob) {
  const auto init = fedtest::readF32(fedtest::goldenPath("zo_flat.f32"));  // 25 — the committed init
  ASSERT_EQ(init.size(), 25u);

  RoundMock mock;
  mock.globalBlob = blobFromFlat(init);  // server streams the global weights (== the committed init)
  mock.globalRound = 3;

  fedlearn::TrainableExecutorchModel model(
      fedtest::goldenPath(kTrainablePte), kTrainablePteSha, kParamNames);
  fedlearn::ModelManager mm = fedtest::makeManager();
  fedlearn::FederatedLoop loop(mock, mm);

  const auto x = fedtest::zoInputs();
  const auto y = fedtest::zoTargets();
  fedlearn::DataBatch batch{x.data(), {8, 4}, y.data(), 8};

  fedlearn::RoundOutcome out = loop.firstOrderRound(model, "run", "client", batch, kLocalEpochs, kLr);

  EXPECT_TRUE(out.ranTraining);
  EXPECT_FALSE(out.shouldStop);
  EXPECT_EQ(out.round, 3);
  ASSERT_TRUE(mock.submitCalled);       // uploaded a WEIGHT blob (not ZO scalars)
  EXPECT_EQ(mock.lastNumExamples, 8);
  EXPECT_FALSE(mock.lastModelBlob.empty());

  // The UPLOADED weight blob must decode to the framework FedAvg endpoint within tolerance — the
  // whole round, not just the compute primitive, matches LocalTrainer.fit(mu=0).
  fedlearn::ModelManager decode = fedtest::makeManager();
  decode.loadStateDict(mock.lastModelBlob);
  const std::vector<float>& got = decode.getFlatParams();
  const auto golden = fedtest::readF32(fedtest::goldenPath("fedavg_local_final.f32"));
  ASSERT_EQ(got.size(), golden.size());
  for (size_t i = 0; i < golden.size(); ++i)
    EXPECT_NEAR(got[i], golden[i], kEndpointAtol)
        << "first-order round endpoint (uploaded blob) diverged from the framework golden at " << i;
}
