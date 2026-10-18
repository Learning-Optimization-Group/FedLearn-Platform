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
#include "fedlearn/ModelExecutionError.h"
#include "fedlearn/Safetensors.h"
#include "fedlearn/Types.h"

#include "fixtures.h"
#include "progress_test_support.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
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
  std::vector<fedtest::Progress> downloadProgress;
  int globalRound = 1;

  // recorders
  bool submitCalled = false;
  bool failScalarSubmission = false;
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
    if (failScalarSubmission) throw std::runtime_error("SubmitGradientScalars unavailable");
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
    downloadProgress.push_back(fedtest::progressOf(*this));
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

// The golden execution contract's initialStateSha256: the committed TinyNet initial state (zo_flat.f32).
const char* kInitialStateSha256 = "1122ba73e49f6df981861bb76d3dcff46666abb5f41e3a6a4d510db9fddd965c";

fedlearn::DeComFLConfig contractedConfig() {
  fedlearn::DeComFLConfig c = makeDeComFLConfig();   // lr 0.01, mu 0.001, seeds 2 x 2
  c.currentRound = 1;
  c.learningRateSent = true;
  c.muSent = true;
  return c;
}

fedlearn::ZerothOrderContract contractFor(const fedlearn::DeComFLConfig& c) {
  return {c.config.learningRate, c.config.mu, 2, 2, fedlearn::GradEstimateMethod::Forward, kInitialStateSha256};
}

// The global model the server streams down, from a trainable flat.
std::string globalBlobFrom(const std::vector<float>& flat) {
  fedlearn::ModelManager mm = fedtest::makeManager();
  mm.setFlatParams(flat);
  return mm.serializeStateDict(/*numExamples=*/0);
}

std::vector<float> initialState() { return fedtest::readF32(fedtest::goldenPath("zo_flat.f32")); }

}  // namespace

TEST(FederatedLoop, DeComFLTrainsWhenTheServerConfirmsTheContract) {
  LoopFixture f;
  MockFedLearnClient mock;
  mock.cfg = contractedConfig();
  mock.globalBlob = globalBlobFrom(initialState());
  const fedlearn::ZerothOrderContract contract = contractFor(mock.cfg);
  fedlearn::FederatedLoop loop(mock, f.mm);

  fedlearn::RoundOutcome out = loop.deComFLRound(f.model, "run", "client", f.batch, &contract);
  EXPECT_TRUE(out.ranTraining);
  EXPECT_TRUE(mock.submitCalled);
}

// The phone used to start DeComFL from an all-zero model -- the ModelManager's initial params -- because the DeComFL
// path never downloads the global model: every scalar it uploaded was a derivative at the wrong point. A contracted
// round starts from the server's initial model and proves it is the one the contract binds.
TEST(FederatedLoop, DeComFLStartsFromTheRunsInitialModel) {
  LoopFixture f;
  MockFedLearnClient mock;
  mock.cfg = contractedConfig();
  mock.globalBlob = globalBlobFrom(initialState());
  const fedlearn::ZerothOrderContract contract = contractFor(mock.cfg);
  fedlearn::FederatedLoop loop(mock, f.mm);

  loop.deComFLRound(f.model, "run", "client", f.batch, &contract);

  EXPECT_EQ(f.mm.getFlatParams(), initialState());   // the model the round trained from, not zeros
}

TEST(FederatedLoop, DeComFLRefusesAnInitialModelTheContractDoesNotBind) {
  LoopFixture f;
  MockFedLearnClient mock;
  mock.cfg = contractedConfig();
  mock.globalBlob = globalBlobFrom(std::vector<float>(25, 0.0f));
  const fedlearn::ZerothOrderContract contract = contractFor(mock.cfg);
  fedlearn::FederatedLoop loop(mock, f.mm);

  EXPECT_THROW(loop.deComFLRound(f.model, "run", "client", f.batch, &contract), std::runtime_error);
  EXPECT_FALSE(mock.submitCalled);
}

TEST(FederatedLoop, DeComFLRefusesASnapshotWithoutRoundMetadata) {
  // A snapshot without a served-round marker cannot establish a safe replay position.
  LoopFixture f;
  MockFedLearnClient mock;
  mock.cfg = contractedConfig();
  mock.cfg.currentRound = 2;
  mock.globalRound = 0;
  mock.globalBlob = globalBlobFrom(initialState());
  const fedlearn::ZerothOrderContract contract = contractFor(mock.cfg);
  fedlearn::FederatedLoop loop(mock, f.mm);

  EXPECT_THROW(loop.deComFLRound(f.model, "run", "client", f.batch, &contract), std::runtime_error);
  EXPECT_FALSE(mock.submitCalled);
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
  mock.globalBlob = globalBlobFrom(initialState());
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

TEST(FederatedLoop, DeComFLProgressUsesLocalStepsRatherThanPerturbations) {
  LoopFixture f;
  MockFedLearnClient net;
  net.cfg = makeDeComFLConfig();
  net.cfg.seeds = {{11, 22, 33}, {44, 55, 66}};  // K=2, P=3
  fedlearn::FederatedLoop loop(net, f.mm);
  loop.deComFLRound(f.model, "run", "client", f.batch);
  EXPECT_EQ(fedtest::progressOf(net), (fedtest::Progress{"WAITING", 3, 2, 2}));
}

TEST(FederatedLoop, DeComFLInitialDownloadDoesNotClaimTrainingHasStarted) {
  LoopFixture f;
  MockFedLearnClient net;
  net.cfg = contractedConfig();
  net.globalBlob = globalBlobFrom(initialState());
  const auto contract = contractFor(net.cfg);
  fedlearn::FederatedLoop loop(net, f.mm);
  loop.deComFLRound(f.model, "run", "client", f.batch, &contract);
  ASSERT_EQ(net.downloadProgress.size(), 1u);
  EXPECT_EQ(net.downloadProgress[0], (fedtest::Progress{"DOWNLOADING", 1, 0, 2}));
}

namespace {
fedlearn::RebuildRound rebuildRound(int round) {
  return {round, {{11, 22}, {33, 4242}}, {{0.1 * round, -0.2}, {0.3, 0.05 * round}}, 0.01};
}
}  // namespace

TEST(FederatedLoop, DeComFLColdRestartUsesTheCurrentSnapshotWithoutReplayingItsHistory) {
  LoopFixture f;
  MockFedLearnClient net;
  net.cfg = contractedConfig();
  net.cfg.currentRound = 2;
  net.cfg.rebuildHistory = {rebuildRound(1)};
  net.globalRound = 2;
  const std::vector<float> snapshot(25, 0.25f);
  net.globalBlob = globalBlobFrom(snapshot);
  const auto contract = contractFor(net.cfg);
  fedlearn::DeComFLState freshState;
  fedlearn::FederatedLoop loop(net, f.mm, freshState);

  EXPECT_TRUE(loop.deComFLRound(f.model, "run", "client", f.batch, &contract).ranTraining);
  EXPECT_EQ(f.mm.getFlatParams(), snapshot);
  EXPECT_EQ(net.lastTrainedOnRound, 2);
  EXPECT_EQ(net.lastSeeds, net.cfg.seeds);
}

TEST(FederatedLoop, DeComFLRetriesARoundRaceWithoutBindingTheWrongSnapshot) {
  LoopFixture f;
  MockFedLearnClient net;
  net.cfg = contractedConfig();
  net.cfg.currentRound = 2;
  net.globalRound = 3;
  const std::vector<float> snapshot(25, 0.5f);
  net.globalBlob = globalBlobFrom(snapshot);
  const auto before = f.mm.getFlatParams();
  const auto contract = contractFor(net.cfg);
  fedlearn::DeComFLState state;
  fedlearn::FederatedLoop loop(net, f.mm, state);

  EXPECT_THROW(loop.deComFLRound(f.model, "run", "client", f.batch, &contract), std::runtime_error);
  EXPECT_FALSE(net.submitCalled);
  EXPECT_EQ(f.mm.getFlatParams(), before);
  net.cfg.currentRound = 3;
  net.cfg.rebuildHistory = {rebuildRound(1), rebuildRound(2)};
  EXPECT_TRUE(loop.deComFLRound(f.model, "run", "client", f.batch, &contract).ranTraining);
  EXPECT_EQ(f.mm.getFlatParams(), snapshot);
  EXPECT_EQ(net.lastTrainedOnRound, 3);
}

TEST(FederatedLoop, DeComFLInvalidInitialSnapshotLeavesTheModelUnchanged) {
  LoopFixture f;
  MockFedLearnClient net;
  net.cfg = contractedConfig();
  net.globalBlob = globalBlobFrom(std::vector<float>(25, 0.5f));
  const auto contract = contractFor(net.cfg);
  const auto before = f.mm.getFlatParams();
  fedlearn::FederatedLoop loop(net, f.mm);

  EXPECT_THROW(loop.deComFLRound(f.model, "run", "client", f.batch, &contract), std::runtime_error);
  EXPECT_EQ(f.mm.getFlatParams(), before);
  EXPECT_FALSE(net.submitCalled);
}

TEST(FederatedLoop, DeComFLInvalidSnapshotDoesNotPoisonAColdRetry) {
  LoopFixture f;
  MockFedLearnClient net;
  net.cfg = contractedConfig();
  net.cfg.currentRound = 2;
  net.globalRound = 2;
  auto invalid = std::vector<float>(25, 0.25f);
  invalid[0] = std::numeric_limits<float>::infinity();
  net.globalBlob = globalBlobFrom(invalid);
  const auto contract = contractFor(net.cfg);
  const auto before = f.mm.getFlatParams();
  fedlearn::DeComFLState state;
  fedlearn::FederatedLoop loop(net, f.mm, state);
  try {
    loop.deComFLRound(f.model, "run", "client", f.batch, &contract);
    FAIL() << "Non-finite snapshot was trained";
  } catch (const std::runtime_error& e) {
    EXPECT_EQ(fedlearn::rejectionMessage(e).find("TRAINING_STATE: "), 0u);
  }
  EXPECT_FALSE(net.submitCalled);
  EXPECT_EQ(f.mm.getFlatParams(), before);
  const std::vector<float> valid(25, 0.25f);
  net.globalBlob = globalBlobFrom(valid);
  EXPECT_TRUE(loop.deComFLRound(f.model, "run", "client", f.batch, &contract).ranTraining);
  EXPECT_EQ(f.mm.getFlatParams(), valid);
}

TEST(FederatedLoop, DeComFLRejectsASnapshotWithTheWrongShapeEvenWhenItsSizeMatches) {
  LoopFixture f;
  MockFedLearnClient net;
  net.cfg = contractedConfig();
  net.cfg.currentRound = 2;
  net.globalRound = 2;
  auto tensors = fedlearn::loadSafetensors(globalBlobFrom(initialState()));
  ASSERT_GT(tensors[0].shape.size(), 1u);
  tensors[0].shape = {static_cast<int64_t>(tensors[0].data.size())};
  net.globalBlob = fedlearn::saveSafetensors(tensors, {});
  const auto contract = contractFor(net.cfg);
  const auto before = f.mm.getFlatParams();
  fedlearn::FederatedLoop loop(net, f.mm);

  try {
    loop.deComFLRound(f.model, "run", "client", f.batch, &contract);
    FAIL() << "Wrong-shaped snapshot was trained";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("shape"), std::string::npos);
    EXPECT_EQ(fedlearn::rejectionMessage(e).find("TRAINING_STATE: "), 0u);
  }
  EXPECT_EQ(f.mm.getFlatParams(), before);
  EXPECT_FALSE(net.submitCalled);
}

TEST(FederatedLoop, DeComFLRetryDoesNotApplyTheSameHistoryTwice) {
  LoopFixture f;
  MockFedLearnClient net;
  net.cfg = contractedConfig();
  net.globalBlob = globalBlobFrom(initialState());
  const auto contract = contractFor(net.cfg);
  fedlearn::FederatedLoop loop(net, f.mm);
  loop.deComFLRound(f.model, "run", "client", f.batch, &contract);

  net.cfg.currentRound = 2;
  net.cfg.rebuildHistory = {rebuildRound(1)};
  net.failScalarSubmission = true;
  EXPECT_THROW(loop.deComFLRound(f.model, "run", "client", f.batch, &contract), std::runtime_error);
  const auto beforeRetry = f.mm.getFlatParams();
  const auto scalars = net.lastGradients;
  net.failScalarSubmission = false;
  loop.deComFLRound(f.model, "run", "client", f.batch, &contract);
  EXPECT_EQ(f.mm.getFlatParams(), beforeRetry);
  EXPECT_EQ(net.lastGradients, scalars);
}

TEST(FederatedLoop, DeComFLRefusesMissingRebuildRoundsBeforeChangingTheModel) {
  LoopFixture f;
  MockFedLearnClient net;
  net.cfg = contractedConfig();
  net.globalBlob = globalBlobFrom(initialState());
  const auto contract = contractFor(net.cfg);
  fedlearn::FederatedLoop loop(net, f.mm);
  loop.deComFLRound(f.model, "run", "client", f.batch, &contract);

  net.cfg.currentRound = 3;
  net.cfg.rebuildHistory = {rebuildRound(2)};  // round one is missing
  net.submitCalled = false;
  EXPECT_THROW(loop.deComFLRound(f.model, "run", "client", f.batch, &contract), std::runtime_error);
  EXPECT_FALSE(net.submitCalled);
  EXPECT_EQ(f.mm.getFlatParams(), initialState());
}

TEST(FederatedLoop, DeComFLRefusesNonFiniteHistoryWithoutCommittingItsValidPrefix) {
  LoopFixture f;
  MockFedLearnClient net;
  net.cfg = contractedConfig();
  net.globalBlob = globalBlobFrom(initialState());
  const auto contract = contractFor(net.cfg);
  fedlearn::DeComFLState state;
  fedlearn::FederatedLoop loop(net, f.mm, state);
  loop.deComFLRound(f.model, "run", "client", f.batch, &contract);

  net.cfg.currentRound = 3;
  net.cfg.rebuildHistory = {rebuildRound(1), rebuildRound(2)};
  net.cfg.rebuildHistory[1].gradients[0][0] = std::numeric_limits<double>::infinity();
  net.submitCalled = false;
  EXPECT_THROW(loop.deComFLRound(f.model, "run", "client", f.batch, &contract), std::runtime_error);
  EXPECT_FALSE(net.submitCalled);
  EXPECT_EQ(f.mm.getFlatParams(), initialState());

  net.cfg.rebuildHistory = {rebuildRound(1), rebuildRound(2)};
  fedlearn::FederatedLoop replacement(net, f.mm, state);
  replacement.deComFLRound(f.model, "run", "client", f.batch, &contract);
  LoopFixture clean;
  MockFedLearnClient reference;
  reference.cfg = contractedConfig();
  reference.globalBlob = globalBlobFrom(initialState());
  fedlearn::FederatedLoop cleanLoop(reference, clean.mm);
  cleanLoop.deComFLRound(clean.model, "run", "client", clean.batch, &contract);
  reference.cfg = net.cfg;
  cleanLoop.deComFLRound(clean.model, "run", "client", clean.batch, &contract);
  EXPECT_EQ(f.mm.getFlatParams(), clean.mm.getFlatParams());
  EXPECT_EQ(net.lastGradients, reference.lastGradients);
}

TEST(FederatedLoop, DeComFLReconnectKeepsTheAppliedHistoryAfterAFailedSubmission) {
  LoopFixture f;
  fedlearn::DeComFLState state;
  MockFedLearnClient first;
  first.cfg = contractedConfig();
  first.globalBlob = globalBlobFrom(initialState());
  const auto contract = contractFor(first.cfg);
  {
    fedlearn::FederatedLoop loop(first, f.mm, state);
    loop.deComFLRound(f.model, "run", "client", f.batch, &contract);
    first.cfg.currentRound = 2;
    first.cfg.rebuildHistory = {rebuildRound(1)};
    first.failScalarSubmission = true;
    EXPECT_THROW(loop.deComFLRound(f.model, "run", "client", f.batch, &contract), std::runtime_error);
  }
  const auto expected = f.mm.getFlatParams();
  MockFedLearnClient reconnected;
  reconnected.cfg = first.cfg;
  fedlearn::FederatedLoop resumed(reconnected, f.mm, state);
  EXPECT_TRUE(resumed.deComFLRound(f.model, "run", "client", f.batch, &contract).ranTraining);
  EXPECT_EQ(f.mm.getFlatParams(), expected);
  EXPECT_EQ(reconnected.lastGradients, first.lastGradients);
  EXPECT_TRUE(reconnected.downloadProgress.empty());
}

TEST(FederatedLoop, DeComFLOverlappingHistoryMatchesACleanSingleApplication) {
  LoopFixture resumed;
  MockFedLearnClient net;
  net.cfg = contractedConfig();
  net.globalBlob = globalBlobFrom(initialState());
  const auto contract = contractFor(net.cfg);
  fedlearn::DeComFLState state;
  {
    fedlearn::FederatedLoop loop(net, resumed.mm, state);
    loop.deComFLRound(resumed.model, "run", "client", resumed.batch, &contract);
    net.cfg.currentRound = 2;
    net.cfg.rebuildHistory = {rebuildRound(1)};
    net.failScalarSubmission = true;
    EXPECT_THROW(loop.deComFLRound(resumed.model, "run", "client", resumed.batch, &contract), std::runtime_error);
  }
  net.failScalarSubmission = false;
  net.cfg.currentRound = 3;
  net.cfg.rebuildHistory = {rebuildRound(1), rebuildRound(2)};
  fedlearn::FederatedLoop reconnected(net, resumed.mm, state);
  reconnected.deComFLRound(resumed.model, "run", "client", resumed.batch, &contract);

  LoopFixture clean;
  MockFedLearnClient reference;
  reference.cfg = contractedConfig();
  reference.globalBlob = globalBlobFrom(initialState());
  fedlearn::FederatedLoop cleanLoop(reference, clean.mm);
  cleanLoop.deComFLRound(clean.model, "run", "client", clean.batch, &contract);
  reference.cfg = net.cfg;
  cleanLoop.deComFLRound(clean.model, "run", "client", clean.batch, &contract);
  EXPECT_EQ(resumed.mm.getFlatParams(), clean.mm.getFlatParams());
  EXPECT_EQ(net.lastGradients, reference.lastGradients);
}

TEST(FederatedLoop, DeComFLModelReloadDiscardsTheReconnectState) {
  LoopFixture f;
  MockFedLearnClient net;
  net.cfg = contractedConfig();
  net.globalBlob = globalBlobFrom(initialState());
  const auto contract = contractFor(net.cfg);
  fedlearn::DeComFLState state;
  fedlearn::FederatedLoop loop(net, f.mm, state);
  loop.deComFLRound(f.model, "run", "client", f.batch, &contract);
  state.reset();
  net.cfg.currentRound = 2;
  net.cfg.rebuildHistory = {rebuildRound(1)};
  net.globalRound = 2;
  const std::vector<float> snapshot(25, 0.25f);
  net.globalBlob = globalBlobFrom(snapshot);
  EXPECT_TRUE(loop.deComFLRound(f.model, "run", "client", f.batch, &contract).ranTraining);
  EXPECT_EQ(f.mm.getFlatParams(), snapshot);
  EXPECT_EQ(net.lastTrainedOnRound, 2);
  EXPECT_EQ(net.downloadProgress.size(), 2u);
}

TEST(FederatedLoop, DeComFLDoesNotReuseStateAcrossRunsClientsOrContracts) {
  for (int change = 0; change < 4; ++change) {
    SCOPED_TRACE(change);
    LoopFixture f;
    MockFedLearnClient net;
    net.cfg = contractedConfig();
    net.globalBlob = globalBlobFrom(initialState());
    auto contract = contractFor(net.cfg);
    fedlearn::DeComFLState state;
    fedlearn::FederatedLoop loop(net, f.mm, state);
    loop.deComFLRound(f.model, "run", "client", f.batch, &contract);
    net.cfg.currentRound = 2;
    net.cfg.rebuildHistory = {rebuildRound(1)};
    if (change == 2) { contract.learningRate = 0.02; net.cfg.config.learningRate = 0.02; }
    if (change == 3) contract.initialStateSha256 = std::string(64, '0');
    net.globalRound = 2;
    const std::vector<float> snapshot(25, 0.5f);
    net.globalBlob = globalBlobFrom(snapshot);
    EXPECT_TRUE(loop.deComFLRound(f.model, change == 0 ? "other-run" : "run",
                                change == 1 ? "other-client" : "client", f.batch, &contract).ranTraining);
    EXPECT_EQ(f.mm.getFlatParams(), snapshot);
    EXPECT_EQ(net.lastTrainedOnRound, 2);
    EXPECT_EQ(net.downloadProgress.size(), 2u);
  }
}

TEST(FederatedLoop, DeComFLRejectsMalformedHistoryBeforeAnyModelChange) {
  for (int change = 0; change < 8; ++change) {
    SCOPED_TRACE(change);
    LoopFixture f;
    MockFedLearnClient net;
    net.cfg = contractedConfig();
    net.globalBlob = globalBlobFrom(initialState());
    const auto contract = contractFor(net.cfg);
    fedlearn::FederatedLoop loop(net, f.mm);
    loop.deComFLRound(f.model, "run", "client", f.batch, &contract);
    net.cfg.currentRound = 3;
    net.cfg.rebuildHistory = {rebuildRound(1), rebuildRound(2)};
    auto& history = net.cfg.rebuildHistory;
    switch (change) {
      case 0: history.pop_back(); break;
      case 1: history[1].roundNumber = 1; break;
      case 2: history[0].roundNumber = 2; history[1].roundNumber = 1; break;
      case 3: history[1].roundNumber = 3; break;
      case 4: history[1].seeds[0].clear(); break;
      case 5: history[1].gradients.pop_back(); break;
      case 6: history[1].gradients[0].pop_back(); break;
      case 7: history[0].roundNumber = 0; break;
    }
    net.submitCalled = false;
    EXPECT_THROW(loop.deComFLRound(f.model, "run", "client", f.batch, &contract), std::runtime_error);
    EXPECT_FALSE(net.submitCalled);
    EXPECT_EQ(f.mm.getFlatParams(), initialState());
  }
}

TEST(FederatedLoop, DeComFLReconnectRebuildsTheFrozenPythonTrajectory) {
  LoopFixture f;
  fedlearn::DeComFLState state;
  MockFedLearnClient net;
  net.cfg = contractedConfig();
  net.cfg.config.learningRate = 0.02;
  net.cfg.seeds = {{191664963, 1662057957, 1405681631, 942484272}};
  net.globalBlob = globalBlobFrom(initialState());
  fedlearn::ZerothOrderContract contract{0.02, 0.001, 1, 4,
      fedlearn::GradEstimateMethod::Forward, kInitialStateSha256};
  {
    fedlearn::FederatedLoop loop(net, f.mm, state);
    loop.deComFLRound(f.model, "run", "client", f.batch, &contract);
  }
  const auto seeds = fedtest::readI64(fedtest::goldenPath("zo_multiround_seeds.i64"));
  const auto gradients = fedtest::readF64(fedtest::goldenPath("zo_multiround_g.f64"));
  ASSERT_EQ(seeds.size(), 12u);
  ASSERT_EQ(gradients.size(), 12u);
  net.cfg.currentRound = 4;
  for (int r = 0; r < 3; ++r) {
    net.cfg.rebuildHistory.push_back({r + 1,
        {{seeds[r * 4], seeds[r * 4 + 1], seeds[r * 4 + 2], seeds[r * 4 + 3]}},
        {{gradients[r * 4], gradients[r * 4 + 1], gradients[r * 4 + 2], gradients[r * 4 + 3]}}, 0.02});
  }
  fedlearn::FederatedLoop reconnected(net, f.mm, state);
  reconnected.deComFLRound(f.model, "run", "client", f.batch, &contract);
  const auto expected = fedtest::readF32(fedtest::goldenPath("zo_multiround_final.f32"));
  const auto actual = f.mm.getFlatParams();
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) EXPECT_NEAR(actual[i], expected[i], 2e-6f) << i;
}
