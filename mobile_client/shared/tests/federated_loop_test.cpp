// federated_loop_test.cpp — unit-tests FederatedLoop against a mock IFedLearnClient (no gRPC).
//
// This is the proof that FederatedLoop is module-free: it builds + runs in the libtorch-free,
// gRPC-free ET suite, driven by a hand-rolled MockFedLearnClient. It exercises the DeComFL path
// (abort guards, happy path, shape of the uploaded scalars), the FedAvg ZO-SGD path (download +
// load + K*P scalar/seed upload), and the absence of the torch_version gate.
//
// GOLDEN_DIR is injected by CMake and points at framework/tests/fixtures/decomfl_golden.
#include "fedlearn/ExecutorchModel.h"
#include "fedlearn/FederatedLoop.h"
#include "fedlearn/IFedLearnClient.h"
#include "fedlearn/ModelManager.h"
#include "fedlearn/Types.h"

#include "fixtures.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

// A fully controllable IFedLearnClient seam. `stopAfterCalls` trips shouldStop() true once
// shouldStop() has been polled that many times (so a test can let the pre-round / pre-rebuild
// checks pass and fail the post-fit one). Recorders capture the last submitGradientScalars args.
class MockFedLearnClient : public fedlearn::IFedLearnClient {
 public:
  // shouldStop control
  mutable int stopPollCount = 0;
  int stopAfterCalls = -1;        // -1 => never stop; N => true once polled > N times
  bool forceStop = false;         // hard override: always true

  // canned DeComFL config
  fedlearn::DeComFLConfig cfg;

  // canned FedAvg global-model blob
  std::string globalBlob;
  int globalRound = 0;

  // recorders
  bool submitCalled = false;
  fedlearn::Seeds2D lastSeeds;
  fedlearn::GradientScalars2D lastGradients;
  int64_t lastNumExamples = -1;
  int lastTrainedOnRound = -1;

  bool shouldStop() const override {
    int n = ++stopPollCount;
    if (forceStop) return true;
    if (stopAfterCalls >= 0 && n > stopAfterCalls) return true;
    return false;
  }

  fedlearn::DeComFLConfig getDeComFLConfig(const std::string&, const std::string&) override {
    return cfg;
  }

  void submitGradientScalars(const std::string&, const std::string&, int trainedOnRound,
                             const fedlearn::Seeds2D& seeds,
                             const fedlearn::GradientScalars2D& gradients,
                             int64_t numExamples) override {
    submitCalled = true;
    lastTrainedOnRound = trainedOnRound;
    lastSeeds = seeds;
    lastGradients = gradients;
    lastNumExamples = numExamples;
  }

  // recorder for the first-order weight-blob upload (M2)
  std::string lastModelBlob;

  void submitModelUpdate(const std::string&, const std::string&, int trainedOnRound,
                         const std::string& modelBlob, int64_t numExamples) override {
    submitCalled = true;
    lastTrainedOnRound = trainedOnRound;
    lastModelBlob = modelBlob;
    lastNumExamples = numExamples;
  }

  std::string getGlobalModelStream(const std::string&, const std::string&,
                                   int* outCurrentRound,
                                   std::map<std::string, std::string>* outConfig = nullptr) override {
    if (outCurrentRound) *outCurrentRound = globalRound;
    if (outConfig) outConfig->clear();
    return globalBlob;
  }
};

// A small DeComFL config: K=2 local steps, P=2 perturbations (golden seeds), lr/mu like the gate.
fedlearn::DeComFLConfig makeDeComFLConfig() {
  fedlearn::DeComFLConfig c;
  c.shouldStop = false;
  c.currentRound = 3;
  c.config.learningRate = 0.01;
  c.config.mu = 0.001;
  c.seeds = {{11, 22}, {33, 4242}};  // [K=2][P=2], reusing the golden seeds
  return c;
}

struct LoopFixture {
  fedlearn::ModelManager mm = fedtest::makeManager();
  fedlearn::ExecutorchModel model{fedtest::goldenPath("zo_model_tiny.pte"), fedtest::kTinyPteSha};
  std::vector<float> in = fedtest::zoInputs();
  std::vector<int64_t> tg = fedtest::zoTargets();
  fedlearn::DataBatch batch{in.data(), {8, 4}, tg.data(), 8};
};

}  // namespace

TEST(FederatedLoop, AbortBeforeRound) {
  LoopFixture f;
  MockFedLearnClient mock;
  mock.forceStop = true;  // shouldStop() true before the round even starts
  fedlearn::FederatedLoop loop(mock, f.mm);

  fedlearn::RoundOutcome out = loop.deComFLRound(f.model, "run", "client", f.batch);
  EXPECT_TRUE(out.shouldStop);
  EXPECT_FALSE(out.ranTraining);
  EXPECT_FALSE(mock.submitCalled);
}

TEST(FederatedLoop, AbortAfterFit) {
  LoopFixture f;
  MockFedLearnClient mock;
  mock.cfg = makeDeComFLConfig();  // no rebuild history => 2 shouldStop polls before the post-fit one
  // Polls: (1) pre-round, (2) post-fit. Let pre-round pass, trip post-fit.
  mock.stopAfterCalls = 1;
  fedlearn::FederatedLoop loop(mock, f.mm);

  fedlearn::RoundOutcome out = loop.deComFLRound(f.model, "run", "client", f.batch);
  EXPECT_TRUE(out.shouldStop);
  EXPECT_FALSE(out.ranTraining);
  EXPECT_FALSE(mock.submitCalled);
}

TEST(FederatedLoop, DeComFLHappyPath) {
  LoopFixture f;
  MockFedLearnClient mock;
  mock.cfg = makeDeComFLConfig();
  fedlearn::FederatedLoop loop(mock, f.mm);

  fedlearn::RoundOutcome out = loop.deComFLRound(f.model, "run", "client", f.batch);
  EXPECT_TRUE(out.ranTraining);
  EXPECT_FALSE(out.shouldStop);
  EXPECT_EQ(out.round, 3);
  ASSERT_TRUE(mock.submitCalled);
  // gradients shape [K][P] = [2][2]
  ASSERT_EQ(mock.lastGradients.size(), 2u);
  EXPECT_EQ(mock.lastGradients[0].size(), 2u);
  EXPECT_EQ(mock.lastGradients[1].size(), 2u);
  // seeds echoed back at the same shape
  ASSERT_EQ(mock.lastSeeds.size(), 2u);
  EXPECT_EQ(mock.lastSeeds[0].size(), 2u);
  EXPECT_EQ(mock.lastNumExamples, 8);
  EXPECT_EQ(mock.lastTrainedOnRound, 3);
}

TEST(FederatedLoop, FedAvgZOStepCount) {
  LoopFixture f;
  MockFedLearnClient mock;
  // Produce a valid global blob via the same layout the loop's mm_ uses (so loadStateDict accepts).
  mock.globalBlob = fedtest::makeManager().serializeStateDict(/*numExamples=*/8);
  mock.globalRound = 5;
  fedlearn::FederatedLoop loop(mock, f.mm);

  const int K = 2, P = 3;
  fedlearn::RoundOutcome out =
      loop.fedAvgRound(f.model, "run", "client", f.batch, /*numLocalSteps=*/K,
                       /*learningRate=*/0.01, /*mu=*/0.001, /*numPerturbations=*/P);
  EXPECT_TRUE(out.ranTraining);
  EXPECT_FALSE(out.shouldStop);
  EXPECT_EQ(out.round, 5);
  ASSERT_TRUE(mock.submitCalled);
  // scalars / seeds shape [K][P] (ZO-SGD scalar upload, not a weight blob)
  ASSERT_EQ(mock.lastGradients.size(), static_cast<size_t>(K));
  EXPECT_EQ(mock.lastGradients[0].size(), static_cast<size_t>(P));
  EXPECT_EQ(mock.lastGradients[1].size(), static_cast<size_t>(P));
  ASSERT_EQ(mock.lastSeeds.size(), static_cast<size_t>(K));
  EXPECT_EQ(mock.lastSeeds[0].size(), static_cast<size_t>(P));
  EXPECT_EQ(mock.lastNumExamples, 8);
  EXPECT_EQ(mock.lastTrainedOnRound, 5);
}

TEST(FederatedLoop, NoTorchVersionGate) {
  LoopFixture f;
  MockFedLearnClient mock;
  mock.cfg = makeDeComFLConfig();
  mock.cfg.config.torchVersion = "9.9.9-mismatched";  // the loop must not even look at this
  fedlearn::FederatedLoop loop(mock, f.mm);

  fedlearn::RoundOutcome out = loop.deComFLRound(f.model, "run", "client", f.batch);
  EXPECT_TRUE(out.ranTraining);
  EXPECT_FALSE(out.shouldStop);
}

// --- a DeComFL round held to the run's execution contract ------------------------------------------------------
// The contract states the zeroth-order training; the server's round config is a cross-check. Anything it asks for
// that the contract does not state -- or leaves out -- refuses the round before any upload.

namespace {

fedlearn::DeComFLConfig contractedConfig() {
  fedlearn::DeComFLConfig c = makeDeComFLConfig();   // lr 0.01, mu 0.001, seeds 2 x 2
  c.learningRateSent = true;
  c.muSent = true;
  return c;
}

fedlearn::ZerothOrderContract contractFor(const fedlearn::DeComFLConfig& c) {
  return {c.config.learningRate, c.config.mu, 2, 2, fedlearn::GradEstimateMethod::Forward};
}

}  // namespace

TEST(FederatedLoop, DeComFLTrainsWhenTheServerConfirmsTheContract) {
  LoopFixture f;
  MockFedLearnClient mock;
  mock.cfg = contractedConfig();
  const fedlearn::ZerothOrderContract contract = contractFor(mock.cfg);
  fedlearn::FederatedLoop loop(mock, f.mm);

  fedlearn::RoundOutcome out = loop.deComFLRound(f.model, "run", "client", f.batch, &contract);
  EXPECT_TRUE(out.ranTraining);
  EXPECT_TRUE(mock.submitCalled);
}

struct ServerDeviation {
  const char* name;
  void (*apply)(fedlearn::DeComFLConfig&);
};

class DeComFLServerDisagreesWithContract : public ::testing::TestWithParam<ServerDeviation> {};

TEST_P(DeComFLServerDisagreesWithContract, RefusesTheRoundBeforeUpload) {
  LoopFixture f;
  MockFedLearnClient mock;
  mock.cfg = contractedConfig();
  const fedlearn::ZerothOrderContract contract = contractFor(mock.cfg);
  GetParam().apply(mock.cfg);
  fedlearn::FederatedLoop loop(mock, f.mm);

  EXPECT_THROW(loop.deComFLRound(f.model, "run", "client", f.batch, &contract), std::runtime_error);
  EXPECT_FALSE(mock.submitCalled);
}

INSTANTIATE_TEST_SUITE_P(
    FederatedLoop, DeComFLServerDisagreesWithContract,
    ::testing::Values(
        ServerDeviation{"OtherRate", [](fedlearn::DeComFLConfig& c) { c.config.learningRate = 0.02; }},
        ServerDeviation{"OtherSmoothing", [](fedlearn::DeComFLConfig& c) { c.config.mu = 0.01; }},
        ServerDeviation{"RateNotSent", [](fedlearn::DeComFLConfig& c) { c.learningRateSent = false; }},
        ServerDeviation{"SmoothingNotSent", [](fedlearn::DeComFLConfig& c) { c.muSent = false; }},
        ServerDeviation{"MoreLocalSteps", [](fedlearn::DeComFLConfig& c) { c.seeds.push_back({5, 6}); }},
        ServerDeviation{"FewerPerturbations", [](fedlearn::DeComFLConfig& c) { c.seeds[1].pop_back(); }},
        ServerDeviation{"CentralEstimator",
                        [](fedlearn::DeComFLConfig& c) { c.config.method = fedlearn::GradEstimateMethod::Central; }}),
    [](const ::testing::TestParamInfo<ServerDeviation>& info) { return std::string(info.param.name); });
