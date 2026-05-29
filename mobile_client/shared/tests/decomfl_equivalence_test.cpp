// decomfl_equivalence_test.cpp — the participate-vs-rebuild invariant (mirror of DeComFL
// spec T1, 15-LLD §13 task 8): replaying a round's scalars via rebuildModel must reproduce
// the (single-client) server update built from the same scalars + seeds. Also asserts fit()
// reverts the model exactly (snapshot-restore, B1-M1).
#include "fedlearn/DeComFLClient.h"
#include "fedlearn/ModelManager.h"
#include "fedlearn/Perturbation.h"
#include "fedlearn/ZerothOrderEstimator.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "fixtures.h"

TEST(DeComFLEquivalence, RebuildMatchesServerUpdate) {
  using namespace fedlearn;

  ModelManager mm;
  torch::jit::Module model = fedtest::loadTinyModel();
  ZerothOrderEstimator zo(mm, /*mu=*/0.001, GradEstimateMethod::Forward);

  const double eta = 0.05;
  const int P = 3;
  const int K = 2;
  DeComFLClient client(mm, zo, eta, P, K);
  DataBatch batch{fedtest::zoInputs(), fedtest::zoTargets()};

  const torch::Tensor x0 = mm.getFlatParams(model).clone();
  const int64_t d = x0.numel();

  const Seeds2D seeds = {{101, 102, 103}, {201, 202, 203}};  // [K][P]
  const GradientScalars2D scalars = client.fit(model, seeds, batch);

  // fit() must revert the model to its pre-round state (snapshot-restore).
  ASSERT_TRUE(torch::allclose(mm.getFlatParams(model), x0, 1e-6, 1e-6));

  // Single-client server update: per local step k, x -= (eta/P) * sum_p g*z.
  torch::Tensor xs = x0.clone();
  for (int k = 0; k < K; ++k) {
    torch::Tensor delta = torch::zeros_like(xs);
    for (int p = 0; p < P; ++p) {
      delta += scalars[k][p] * canonical_perturbation(seeds[k][p], d, torch::kFloat32);
    }
    xs = xs - (eta / P) * delta;
  }

  // Algorithm-2 replay from x0 using the same scalars + seeds.
  mm.setFlatParams(model, x0);
  RebuildHistory history = {RebuildRound{/*roundNumber=*/1, seeds, scalars, /*learningRate=*/eta}};
  client.rebuildModel(model, history);
  const torch::Tensor xr = mm.getFlatParams(model);

  EXPECT_TRUE(torch::allclose(xs, xr, 1e-5, 1e-5)) << "rebuild trajectory != server update";
}
