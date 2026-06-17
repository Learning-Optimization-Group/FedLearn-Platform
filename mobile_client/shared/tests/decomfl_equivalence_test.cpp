// decomfl_equivalence_test.cpp — the participate-vs-rebuild invariant (mirror of DeComFL
// spec T1, 15-LLD §13 task 8): replaying a round's scalars via rebuildModel must reproduce
// the (single-client) server update built from the same scalars + seeds. ExecuTorch, torch-free.
#include "fedlearn/DeComFLClient.h"
#include "fedlearn/ExecutorchModel.h"
#include "fedlearn/RandnEngine.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "fixtures.h"

TEST(DeComFLEquivalence, RebuildMatchesServerUpdate) {
  using namespace fedlearn;

  ExecutorchModel model(fedtest::goldenPath("zo_model_tiny.pte"), fedtest::kTinyPteSha);

  const double eta = 0.05;
  const int P = 3;
  const int K = 2;
  const double mu = 0.001;
  DeComFLClient client(eta, P, K);

  const std::vector<float> xin = fedtest::zoInputs();
  const std::vector<int64_t> y = fedtest::zoTargets();
  DataBatch batch{xin.data(), {8, 4}, y.data(), 8};

  const std::vector<float> x0 = fedtest::readF32(fedtest::goldenPath("zo_flat.f32"));
  const int64_t d = static_cast<int64_t>(x0.size());
  ASSERT_EQ(d, 25);

  const Seeds2D seeds = {{101, 102, 103}, {201, 202, 203}};  // [K][P]
  const GradientScalars2D scalars = client.fit(model, x0, seeds, batch, mu);
  ASSERT_EQ(scalars.size(), static_cast<size_t>(K));
  ASSERT_EQ(scalars[0].size(), static_cast<size_t>(P));

  // Single-client server update: per local step k, x -= (eta/P) * sum_p g*z.
  std::vector<float> xs = x0;
  for (int k = 0; k < K; ++k) {
    std::vector<float> delta(static_cast<size_t>(d), 0.0f);
    for (int p = 0; p < P; ++p) {
      const std::vector<float> z = flat_randn(seeds[k][p], d);
      for (int64_t i = 0; i < d; ++i) delta[i] += static_cast<float>(scalars[k][p]) * z[i];
    }
    const float step = static_cast<float>(eta / P);
    for (int64_t i = 0; i < d; ++i) xs[i] -= step * delta[i];
  }

  // Algorithm-2 replay from x0 using the same scalars + seeds.
  std::vector<float> xr = x0;
  RebuildHistory history = {RebuildRound{/*roundNumber=*/1, seeds, scalars, /*learningRate=*/eta}};
  client.rebuildModel(xr, history);

  ASSERT_EQ(xr.size(), xs.size());
  for (int64_t i = 0; i < d; ++i) {
    EXPECT_NEAR(xr[i], xs[i], 1e-5) << "rebuild trajectory != server update at " << i;
  }
}
