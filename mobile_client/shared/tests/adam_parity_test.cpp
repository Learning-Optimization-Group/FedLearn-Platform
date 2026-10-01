// adam_parity_test.cpp — Stage 4 S2. The laptop's non-TinyNet clients train with torch.optim.Adam, fresh each round;
// the device's native Adam must land on the same endpoint (adam_local_manifest.json). Two goldens pin it: eps 1e-8
// (the laptop's) and eps 0.1, where eps's placement moves the endpoint visibly. The manifest records how far each
// plausible bug lands (no bias correction, eps inside the square root, betas swapped); the tolerance sits at least
// ten times below the nearest control that each golden can see.
#include "fedlearn/TrainableExecutorchModel.h"
#include "fixtures.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace {

constexpr const char* kPte = "tinynet_trainable_dynbatch.pte";
constexpr const char* kSha = "8df99d447a8883a7a65ace595493a15f40cc6339c1e23553fbf922ef9851f124";
const std::vector<std::string> kNames = {"base.fc1.weight", "base.fc1.bias"};
constexpr int kSteps = 5;         // adam_local_manifest.steps
constexpr double kLr = 1e-2;      // learning_rate
constexpr float kAtol = 1e-5f;    // controls land 2.3e-3 .. 0.185 away (see the manifest)

std::vector<float> train(fedlearn::TrainableExecutorchModel& m, double eps) {
  m.setFlatParams(fedtest::readF32(fedtest::goldenPath("zo_flat.f32")));
  m.resetOptimizerState();
  const auto x = fedtest::zoInputs();
  const auto y = fedtest::zoTargets();
  const fedlearn::AdamSettings adam{kLr, 0.9, 0.999, eps};
  for (int t = 0; t < kSteps; ++t) m.trainStepAdam(x.data(), {8, 4}, y.data(), 8, adam);
  return m.getFlatParams();
}

void expectNear(const std::vector<float>& got, const char* goldenFile) {
  const auto golden = fedtest::readF32(fedtest::goldenPath(goldenFile));
  ASSERT_EQ(got.size(), golden.size());
  for (size_t i = 0; i < golden.size(); ++i) EXPECT_NEAR(got[i], golden[i], kAtol) << goldenFile << " at " << i;
}

}  // namespace

TEST(AdamParity, MatchesTorchAdamAtTheLaptopsEpsilon) {
  fedlearn::TrainableExecutorchModel m(fedtest::goldenPath(kPte), kSha, kNames);
  expectNear(train(m, 1e-8), "adam_local_final.f32");
}

TEST(AdamParity, MatchesTorchAdamWhereEpsilonsPlacementShows) {
  fedlearn::TrainableExecutorchModel m(fedtest::goldenPath(kPte), kSha, kNames);
  expectNear(train(m, 0.1), "adam_eps_local_final.f32");
}

// The laptop creates a fresh Adam every round: a round must not inherit the previous round's moments.
TEST(AdamParity, ResettingTheStateStartsAFreshOptimizer) {
  fedlearn::TrainableExecutorchModel m(fedtest::goldenPath(kPte), kSha, kNames);
  train(m, 1e-8);                                    // a first "round"
  expectNear(train(m, 1e-8), "adam_local_final.f32");  // train() resets: the second lands on the same endpoint
}

TEST(AdamParity, WithoutAResetTheMomentsCarryOver) {
  fedlearn::TrainableExecutorchModel m(fedtest::goldenPath(kPte), kSha, kNames);
  train(m, 1e-8);
  m.setFlatParams(fedtest::readF32(fedtest::goldenPath("zo_flat.f32")));
  const fedlearn::AdamSettings adam{kLr, 0.9, 0.999, 1e-8};
  const auto x = fedtest::zoInputs();
  const auto y = fedtest::zoTargets();
  for (int t = 0; t < kSteps; ++t) m.trainStepAdam(x.data(), {8, 4}, y.data(), 8, adam);
  const auto golden = fedtest::readF32(fedtest::goldenPath("adam_local_final.f32"));
  float maxDiff = 0;
  const auto got = m.getFlatParams();
  for (size_t i = 0; i < got.size(); ++i) maxDiff = std::max(maxDiff, std::fabs(got[i] - golden[i]));
  EXPECT_GT(maxDiff, kAtol * 10);
}

TEST(AdamParity, InvalidSettingsAreRefused) {
  fedlearn::TrainableExecutorchModel m(fedtest::goldenPath(kPte), kSha, kNames);
  const auto x = fedtest::zoInputs();
  const auto y = fedtest::zoTargets();
  EXPECT_THROW(m.trainStepAdam(x.data(), {8, 4}, y.data(), 8, fedlearn::AdamSettings{0.0, 0.9, 0.999, 1e-8}),
               std::runtime_error);
  EXPECT_THROW(m.trainStepAdam(x.data(), {8, 4}, y.data(), 8, fedlearn::AdamSettings{kLr, 1.0, 0.999, 1e-8}),
               std::runtime_error);
  EXPECT_THROW(m.trainStepAdam(x.data(), {8, 4}, y.data(), 8, fedlearn::AdamSettings{kLr, 0.9, 0.999, 0.0}),
               std::runtime_error);
}
