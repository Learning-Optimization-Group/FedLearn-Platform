// qualification_test.cpp — Stage 3 D2. Before its first round a device runs the run's trainable program for two
// SGD steps on a synthetic batch and checks it against the exporter's reference (fedavg_pte_manifest.json
// dynbatch.probe): the program loads, both step losses match, the steps change the parameters, and the probe
// finishes within its time budget. A device that cannot run the program correctly never trains with it.
#include "fedlearn/Qualification.h"
#include "fixtures.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

constexpr const char* kPte = "tinynet_trainable_dynbatch.pte";
constexpr const char* kSha = "8df99d447a8883a7a65ace595493a15f40cc6339c1e23553fbf922ef9851f124";
const std::vector<std::string> kNames = {"base.fc1.weight", "base.fc1.bias"};

// fedavg_pte_manifest.json dynbatch.probe
fedlearn::ProbeSpec spec() {
  fedlearn::ProbeSpec s;
  s.rows = 8;
  s.width = 4;
  s.classes = 3;
  s.learningRate = 0.1f;
  s.expectedLossStep1 = 1.1053780317306519;
  s.expectedLossStep2 = 1.0776357650756836;
  s.lossTolerance = 1e-4;
  s.maxProbeMs = 2000;
  return s;
}

}  // namespace

TEST(Qualification, TheStagedProgramQualifies) {
  const auto r = fedlearn::qualifyTrainable(fedtest::goldenPath(kPte), kSha, kNames, spec());
  EXPECT_TRUE(r.passed) << r.failedCheck << ": " << r.detail;
  EXPECT_NEAR(r.lossStep1, spec().expectedLossStep1, 1e-6);
  EXPECT_NEAR(r.lossStep2, spec().expectedLossStep2, 1e-6);
}

TEST(Qualification, AProgramThatComputesAnotherLossDoesNotQualify) {
  auto s = spec();
  s.expectedLossStep1 += 1e-3;
  const auto r = fedlearn::qualifyTrainable(fedtest::goldenPath(kPte), kSha, kNames, s);
  EXPECT_FALSE(r.passed);
  EXPECT_EQ(r.failedCheck, "LOSS_MISMATCH");
}

TEST(Qualification, AWrongUpdateIsCaughtAtTheSecondStep) {
  auto s = spec();
  s.learningRate = 0.2f;  // a device that stepped at another rate lands on another step-2 loss
  const auto r = fedlearn::qualifyTrainable(fedtest::goldenPath(kPte), kSha, kNames, s);
  EXPECT_FALSE(r.passed);
  EXPECT_EQ(r.failedCheck, "LOSS_MISMATCH");
}

TEST(Qualification, AStepThatLeavesTheParametersUnchangedDoesNotQualify) {
  auto s = spec();
  s.learningRate = 1e-30f;  // every update underflows: the step runs but trains nothing
  const auto r = fedlearn::qualifyTrainable(fedtest::goldenPath(kPte), kSha, kNames, s);
  EXPECT_FALSE(r.passed);
  EXPECT_EQ(r.failedCheck, "STALE_WEIGHTS");
}

TEST(Qualification, AProgramThatDoesNotLoadDoesNotQualify) {
  const auto r = fedlearn::qualifyTrainable(fedtest::goldenPath(kPte), std::string(64, '0'), kNames, spec());
  EXPECT_FALSE(r.passed);
  EXPECT_EQ(r.failedCheck, "LOAD");
}

TEST(Qualification, AProbeOverItsTimeBudgetDoesNotQualify) {
  auto s = spec();
  s.maxProbeMs = -1;  // no time at all
  const auto r = fedlearn::qualifyTrainable(fedtest::goldenPath(kPte), kSha, kNames, s);
  EXPECT_FALSE(r.passed);
  EXPECT_EQ(r.failedCheck, "TOO_SLOW");
}

TEST(Qualification, AMalformedSpecDoesNotQualify) {
  auto s = spec();
  s.rows = 0;
  const auto r = fedlearn::qualifyTrainable(fedtest::goldenPath(kPte), kSha, kNames, s);
  EXPECT_FALSE(r.passed);
  EXPECT_EQ(r.failedCheck, "SPEC");
}
