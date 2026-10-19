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
#include "progress_test_support.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <thread>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
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
  std::map<std::string, std::string> globalConfig;

  mutable std::vector<fedtest::Progress> observations;
  bool failUpload = false;
  bool failDownload = false;
  bool shouldStop() const override {
    observations.push_back(fedtest::progressOf(*this));
    return false;
  }
  fedlearn::DeComFLConfig getDeComFLConfig(const std::string&, const std::string&) override { return {}; }
  void submitGradientScalars(const std::string&, const std::string&, int, const fedlearn::Seeds2D&,
                             const fedlearn::GradientScalars2D&, int64_t) override {}
  std::string getGlobalModelStream(const std::string&, const std::string&, int* outCurrentRound,
                                   std::map<std::string, std::string>* outConfig = nullptr) override {
    observations.push_back(fedtest::progressOf(*this));
    if (failDownload) throw std::runtime_error("download unavailable");
    if (outCurrentRound) *outCurrentRound = globalRound;
    if (outConfig) *outConfig = globalConfig;
    return globalBlob;
  }
  void submitModelUpdate(const std::string&, const std::string&, int, const std::string& modelBlob,
                         int64_t numExamples) override {
    observations.push_back(fedtest::progressOf(*this));
    if (failUpload) throw std::runtime_error("upload confirmation lost");
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
  EXPECT_EQ(out.downlinkBytes, static_cast<int64_t>(mock.globalBlob.size()));
  EXPECT_EQ(out.uplinkBytes, static_cast<int64_t>(mock.lastModelBlob.size()));
  EXPECT_GT(out.uplinkBytes, 25 * static_cast<int64_t>(sizeof(float)));  // includes safetensors metadata

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

// The contract states the round's training; a FedOpt server confirms it by sending the same values. (These used
// to override the phone's own values, so the server rather than the run's contract decided what it trained.)
TEST(FedAvgFirstOrderRound, FedOptTrainsTheContractsSettingsWhenTheServerConfirmsThem) {
  const auto init = fedtest::readF32(fedtest::goldenPath("zo_flat.f32"));
  RoundMock mock;
  mock.globalBlob = blobFromFlat(init);
  mock.globalConfig = {{"learning_rate", "0.1"}, {"local_epochs", "5"}, {"proximal_mu", "0.0"}};
  fedlearn::TrainableExecutorchModel model(
      fedtest::goldenPath(kTrainablePte), kTrainablePteSha, kParamNames);
  fedlearn::ModelManager mm = fedtest::makeManager();
  fedlearn::FederatedLoop loop(mock, mm);
  const auto x = fedtest::zoInputs();
  const auto y = fedtest::zoTargets();
  fedlearn::DataBatch batch{x.data(), {8, 4}, y.data(), 8};

  const auto out = loop.firstOrderRound(model, "run", "client", batch,
                                        /*numLocalSteps=*/kLocalEpochs, /*learningRate=*/0.1,
                                        /*requireServerConfig=*/true);

  ASSERT_TRUE(out.ranTraining);
  ASSERT_TRUE(mock.submitCalled);
  fedlearn::ModelManager uploaded = fedtest::makeManager();
  uploaded.loadStateDict(mock.lastModelBlob);
  const auto golden = fedtest::readF32(fedtest::goldenPath("fedavg_local_final.f32"));
  for (size_t i = 0; i < golden.size(); ++i) {
    EXPECT_NEAR(uploaded.getFlatParams()[i], golden[i], kEndpointAtol) << "parameter " << i;
  }
}

struct Disagreement {
  const char* name;
  const char* rate;
  const char* epochs;
  bool requireServerConfig;
};

class ServerDisagreesWithContract : public ::testing::TestWithParam<Disagreement> {};

// The phone was given the contract's 0.001 x 1. A server asking for anything else refuses the round before any
// upload -- on a FedOpt round that requires the server's values, and on a FedAvg or Robust round that does not.
TEST_P(ServerDisagreesWithContract, RefusesTheRoundBeforeUpload) {
  const auto& p = GetParam();
  RoundMock mock;
  mock.globalBlob = blobFromFlat(fedtest::readF32(fedtest::goldenPath("zo_flat.f32")));
  mock.globalConfig = {{"learning_rate", p.rate}, {"local_epochs", p.epochs}, {"proximal_mu", "0.0"}};
  fedlearn::TrainableExecutorchModel model(
      fedtest::goldenPath(kTrainablePte), kTrainablePteSha, kParamNames);
  fedlearn::ModelManager mm = fedtest::makeManager();
  fedlearn::FederatedLoop loop(mock, mm);
  const auto x = fedtest::zoInputs();
  const auto y = fedtest::zoTargets();
  fedlearn::DataBatch batch{x.data(), {8, 4}, y.data(), 8};

  EXPECT_THROW(loop.firstOrderRound(model, "run", "client", batch, 1, 0.001, p.requireServerConfig),
               std::runtime_error);
  EXPECT_FALSE(mock.submitCalled);
}

INSTANTIATE_TEST_SUITE_P(
    FedAvgFirstOrderRound, ServerDisagreesWithContract,
    ::testing::Values(Disagreement{"FedOptOtherRate", "0.1", "1", true},
                      Disagreement{"FedOptOtherEpochs", "0.001", "5", true},
                      Disagreement{"FedOptBoth", "0.1", "5", true},
                      Disagreement{"FedAvgServerSendsARate", "0.1", "1", false}),
    [](const ::testing::TestParamInfo<Disagreement>& info) { return std::string(info.param.name); });

TEST(FedAvgFirstOrderRound, FedOptRejectsMissingServerSettingsBeforeUpload) {
  RoundMock mock;
  mock.globalBlob = blobFromFlat(fedtest::readF32(fedtest::goldenPath("zo_flat.f32")));
  fedlearn::TrainableExecutorchModel model(
      fedtest::goldenPath(kTrainablePte), kTrainablePteSha, kParamNames);
  fedlearn::ModelManager mm = fedtest::makeManager();
  fedlearn::FederatedLoop loop(mock, mm);
  const auto x = fedtest::zoInputs();
  const auto y = fedtest::zoTargets();
  fedlearn::DataBatch batch{x.data(), {8, 4}, y.data(), 8};

  EXPECT_THROW(loop.firstOrderRound(model, "run", "client", batch, 1, 0.001, true), std::runtime_error);
  EXPECT_FALSE(mock.submitCalled);
}

TEST(FedAvgFirstOrderRound, FedOptRejectsMalformedServerSettingsBeforeUpload) {
  RoundMock mock;
  mock.globalBlob = blobFromFlat(fedtest::readF32(fedtest::goldenPath("zo_flat.f32")));
  mock.globalConfig = {{"learning_rate", " 0.1"}, {"local_epochs", "5"}};
  fedlearn::TrainableExecutorchModel model(
      fedtest::goldenPath(kTrainablePte), kTrainablePteSha, kParamNames);
  fedlearn::ModelManager mm = fedtest::makeManager();
  fedlearn::FederatedLoop loop(mock, mm);
  const auto x = fedtest::zoInputs();
  const auto y = fedtest::zoTargets();
  fedlearn::DataBatch batch{x.data(), {8, 4}, y.data(), 8};

  EXPECT_THROW(loop.firstOrderRound(model, "run", "client", batch, 1, 0.001, true), std::runtime_error);
  EXPECT_FALSE(mock.submitCalled);
}

// --- FedProx: the same round, plus the proximal gradient mu * (w - w_global) on every step ----------------------

namespace {

constexpr double kFedProxMu = 0.1;                // fedprox_local_manifest.proximal_mu
constexpr float kFedProxEndpointAtol = 1e-4f;     // fedprox_local_manifest.endpoint_atol

}  // namespace

// The proximal term is zero on a round's first step and moves this endpoint by only ~1.4e-3 over five, inside the
// FedAvg golden's 2e-3. So this test uses the FedProx golden's own tolerance, and first proves that tolerance can
// tell the two apart: a trainer with no proximal term would land on the FedAvg golden and fail here.
TEST(FedAvgFirstOrderRound, FedProxEndpointMatchesTheFrameworkGoldenAndNotFedAvgs) {
  const auto prox = fedtest::readF32(fedtest::goldenPath("fedprox_local_final.f32"));
  const auto plain = fedtest::readF32(fedtest::goldenPath("fedavg_local_final.f32"));
  ASSERT_EQ(prox.size(), plain.size());
  float separation = 0.0f;
  for (size_t i = 0; i < prox.size(); ++i) separation = std::max(separation, std::fabs(prox[i] - plain[i]));
  ASSERT_GE(separation, 10 * kFedProxEndpointAtol) << "the FedProx golden cannot discriminate at its tolerance";

  RoundMock mock;
  mock.globalBlob = blobFromFlat(fedtest::readF32(fedtest::goldenPath("zo_flat.f32")));
  mock.globalConfig = {{"learning_rate", "0.1"}, {"local_epochs", "5"}, {"proximal_mu", "0.1"}};
  fedlearn::TrainableExecutorchModel model(
      fedtest::goldenPath(kTrainablePte), kTrainablePteSha, kParamNames);
  fedlearn::ModelManager mm = fedtest::makeManager();
  fedlearn::FederatedLoop loop(mock, mm);
  const auto x = fedtest::zoInputs();
  const auto y = fedtest::zoTargets();
  fedlearn::DataBatch batch{x.data(), {8, 4}, y.data(), 8};

  const auto out = loop.firstOrderRound(model, "run", "client", batch, kLocalEpochs, 0.1,
                                        /*requireServerConfig=*/true, /*proximalMu=*/kFedProxMu);

  ASSERT_TRUE(out.ranTraining);
  ASSERT_TRUE(mock.submitCalled);
  fedlearn::ModelManager uploaded = fedtest::makeManager();
  uploaded.loadStateDict(mock.lastModelBlob);
  float worst = 0.0f;
  for (size_t i = 0; i < prox.size(); ++i) {
    worst = std::max(worst, std::fabs(uploaded.getFlatParams()[i] - prox[i]));
    EXPECT_NEAR(uploaded.getFlatParams()[i], prox[i], kFedProxEndpointAtol) << "parameter " << i;
  }
  std::printf("[fedprox] max|endpoint diff| = %g (atol %g, separation from FedAvg %g)\n", worst,
              kFedProxEndpointAtol, separation);
}

struct ProximalDisagreement {
  const char* name;
  double contractMu;
  const char* serverMu;  // nullptr: the server sends none
};

class ServerProximalTermDisagreesWithContract : public ::testing::TestWithParam<ProximalDisagreement> {};

// The proximal coefficient is a cross-check like the rate: absent is zero on both sides.
TEST_P(ServerProximalTermDisagreesWithContract, RefusesTheRoundBeforeUpload) {
  const auto& p = GetParam();
  RoundMock mock;
  mock.globalBlob = blobFromFlat(fedtest::readF32(fedtest::goldenPath("zo_flat.f32")));
  mock.globalConfig = {{"learning_rate", "0.01"}, {"local_epochs", "1"}};
  if (p.serverMu != nullptr) mock.globalConfig["proximal_mu"] = p.serverMu;
  fedlearn::TrainableExecutorchModel model(
      fedtest::goldenPath(kTrainablePte), kTrainablePteSha, kParamNames);
  fedlearn::ModelManager mm = fedtest::makeManager();
  fedlearn::FederatedLoop loop(mock, mm);
  const auto x = fedtest::zoInputs();
  const auto y = fedtest::zoTargets();
  fedlearn::DataBatch batch{x.data(), {8, 4}, y.data(), 8};

  EXPECT_THROW(loop.firstOrderRound(model, "run", "client", batch, 1, 0.01, true, p.contractMu),
               std::runtime_error);
  EXPECT_FALSE(mock.submitCalled);
}

INSTANTIATE_TEST_SUITE_P(
    FedAvgFirstOrderRound, ServerProximalTermDisagreesWithContract,
    ::testing::Values(ProximalDisagreement{"FedProxOtherMu", 0.1, "0.2"},
                      ProximalDisagreement{"FedProxServerSendsNoMu", 0.1, nullptr},
                      ProximalDisagreement{"FedProxServerSendsZero", 0.1, "0.0"},
                      ProximalDisagreement{"FedOptServerSendsAMu", 0.0, "0.1"}),
    [](const ::testing::TestParamInfo<ProximalDisagreement>& info) { return std::string(info.param.name); });

TEST(FedAvgFirstOrderRound, ProgressTracksCompletedStepsAndServerRound) {
  RoundMock net;
  net.globalBlob = blobFromFlat(fedtest::readF32(fedtest::goldenPath("zo_flat.f32")));
  net.globalRound = 3;
  fedlearn::TrainableExecutorchModel model(fedtest::goldenPath(kTrainablePte), kTrainablePteSha, kParamNames);
  auto manager = fedtest::makeManager();
  fedlearn::FederatedLoop loop(net, manager);
  const auto x = fedtest::zoInputs();
  const auto y = fedtest::zoTargets();
  const fedlearn::DataBatch data{x.data(), {8, 4}, y.data(), 8};
  loop.firstOrderRound(model, "run", "client", data, 5, 0.1);
  std::vector<fedtest::Progress> training;
  for (const auto& p : net.observations) if (std::get<0>(p) == "TRAINING") training.push_back(p);
  const std::vector<fedtest::Progress> expected = {
      {"TRAINING", 3, 0, 5}, {"TRAINING", 3, 1, 5}, {"TRAINING", 3, 2, 5},
      {"TRAINING", 3, 3, 5}, {"TRAINING", 3, 4, 5}, {"TRAINING", 3, 5, 5}};
  EXPECT_EQ(training, expected);
  EXPECT_EQ(net.observations.back(), (fedtest::Progress{"UPLOADING", 3, 5, 5}));
  EXPECT_EQ(fedtest::progressOf(net), (fedtest::Progress{"WAITING", 3, 5, 5}));
}

TEST(FedAvgFirstOrderRound, FailedUploadIsNotReportedAsSubmittedAndRetryResetsSteps) {
  RoundMock net;
  net.globalBlob = blobFromFlat(fedtest::readF32(fedtest::goldenPath("zo_flat.f32")));
  net.globalRound = 4;
  net.failUpload = true;
  fedlearn::TrainableExecutorchModel model(fedtest::goldenPath(kTrainablePte), kTrainablePteSha, kParamNames);
  auto manager = fedtest::makeManager();
  fedlearn::FederatedLoop loop(net, manager);
  const auto x = fedtest::zoInputs();
  const auto y = fedtest::zoTargets();
  const fedlearn::DataBatch data{x.data(), {8, 4}, y.data(), 8};
  EXPECT_THROW(loop.firstOrderRound(model, "run", "client", data, 2, 0.1), std::runtime_error);
  EXPECT_EQ(fedtest::progressOf(net), (fedtest::Progress{"ERROR", 4, 2, 2}));
  net.failUpload = false;
  net.observations.clear();
  loop.firstOrderRound(model, "run", "client", data, 2, 0.1);
  const auto firstTraining = std::find_if(net.observations.begin(), net.observations.end(),
      [](const auto& p) { return std::get<0>(p) == "TRAINING"; });
  ASSERT_NE(firstTraining, net.observations.end());
  EXPECT_EQ(*firstTraining, (fedtest::Progress{"TRAINING", 4, 0, 2}));
  EXPECT_EQ(fedtest::progressOf(net), (fedtest::Progress{"WAITING", 4, 2, 2}));
}

TEST(FedAvgFirstOrderRound, ProgressSnapshotCannotMixRoundPhaseAndCounters) {
  RoundMock net;
  const fedtest::Progress a{"TRAINING", 3, 1, 5};
  const fedtest::Progress b{"UPLOADING", 8, 6, 6};
  fedtest::publishTo(net, a);
  std::atomic<bool> done{false};
  std::thread writer([&] {
    for (int i = 0; i < 10000; ++i) {
      fedtest::publishTo(net, b);
      fedtest::publishTo(net, a);
    }
    done = true;
  });
  bool invalid = false;
  do {
    const auto p = fedtest::progressOf(net);
    if (p != a && p != b) invalid = true;
  } while (!done.load());
  writer.join();
  EXPECT_FALSE(invalid);
}

TEST(FedAvgFirstOrderRound, DownloadFailureRetainsLastConfirmedRoundWithFreshCounters) {
  RoundMock net;
  net.failDownload = true;
  net.updateProgress("WAITING", 4, 5, 5);
  fedlearn::TrainableExecutorchModel model(fedtest::goldenPath(kTrainablePte), kTrainablePteSha, kParamNames);
  auto manager = fedtest::makeManager();
  fedlearn::FederatedLoop loop(net, manager);
  const auto x = fedtest::zoInputs();
  const auto y = fedtest::zoTargets();
  const fedlearn::DataBatch data{x.data(), {8, 4}, y.data(), 8};
  EXPECT_THROW(loop.firstOrderRound(model, "run", "client", data, 5, 0.1), std::runtime_error);
  EXPECT_EQ(net.observations.back(), (fedtest::Progress{"DOWNLOADING", 4, 0, 0}));
  EXPECT_EQ(fedtest::progressOf(net), (fedtest::Progress{"ERROR", 4, 0, 0}));
}
