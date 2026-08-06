// decomfl_base_loss_cache_test.cpp — the forward-difference base loss must be evaluated ONCE
// per local step, not once per perturbation.
//
// Within a DeComFL local step k the base point x and the batch are both fixed; only z varies
// across the P perturbations. So L(x) is the same number every time, and the algorithm needs
// P+1 forward passes per step, not 2P. The authors' reference implementation hoists the
// unperturbed loss above the perturbation loop for the forward-difference method.
//
// This is the native mirror of framework/tests/test_decomfl_base_loss_cache.py. It matters most
// here: this is the path the on-device latency numbers are measured through, so a 2x forward
// overcount lands directly in published round-cost figures.
#include "fedlearn/DeComFLClient.h"
#include "fedlearn/EtZeroOrder.h"
#include "fedlearn/ExecutorchModel.h"
#include "fedlearn/RandnEngine.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "fixtures.h"

namespace {

// Counting stand-in with ExecutorchModel's loss() signature. Deterministic and cheap: the loss
// is a fixed function of the parameters, which is all the forward-difference needs.
class CountingModel {
 public:
  int lossCalls = 0;

  float loss(const std::vector<float>& flat, const float* /*x*/,
             const std::vector<int64_t>& /*xShape*/, const int64_t* /*y*/, int64_t /*n*/) {
    ++lossCalls;
    double acc = 0.0;
    for (size_t i = 0; i < flat.size(); ++i) acc += static_cast<double>(flat[i]) * (i + 1);
    return static_cast<float>(acc);
  }
};

}  // namespace

// The cost contract: K local steps x (P perturbed evaluations + 1 base evaluation).
TEST(DeComFLBaseLossCache, FitCostsPPlusOneForwardPassesPerLocalStep) {
  using namespace fedlearn;

  struct Case {
    int K;
    int P;
  };
  const Case kCases[] = {{1, 1}, {1, 10}, {1, 20}, {5, 10}};

  const std::vector<float> xin = fedtest::zoInputs();
  const std::vector<int64_t> y = fedtest::zoTargets();
  const DataBatch batch{xin.data(), {8, 4}, y.data(), 8};
  const std::vector<float> x0 = fedtest::readF32(fedtest::goldenPath("zo_flat.f32"));

  for (const auto& c : kCases) {
    Seeds2D seeds(c.K, std::vector<int64_t>(c.P));
    for (int k = 0; k < c.K; ++k) {
      for (int p = 0; p < c.P; ++p) seeds[k][p] = 5000 + k * 100 + p;
    }

    CountingModel model;
    DeComFLClient client(0.05, c.P, c.K);
    client.fit(model, x0, seeds, batch, 0.001);

    EXPECT_EQ(model.lossCalls, c.K * (c.P + 1))
        << "K=" << c.K << " P=" << c.P << ": forward-difference needs one base loss per local "
        << "step plus one perturbed loss per perturbation; " << c.K * 2 * c.P
        << " means L(x) is being recomputed inside the perturbation loop";
  }
}

// The safety contract: passing a precomputed base loss must not change the scalar.
TEST(DeComFLBaseLossCache, CachedBaseLossMatchesRecomputed) {
  using namespace fedlearn;

  ExecutorchModel model(fedtest::goldenPath("zo_model_tiny.pte"), fedtest::kTinyPteSha);
  const auto flat = fedtest::readF32(fedtest::goldenPath("zo_flat.f32"));
  const auto x = fedtest::zoInputs();
  const auto y = fedtest::zoTargets();
  constexpr double kMu = 0.001;

  const double base = etBaseLoss(model, flat, x.data(), {8, 4}, y.data(), 8);

  for (int64_t seed : {11, 22, 33, 4242}) {
    const std::vector<float> z = flat_randn(seed, static_cast<int64_t>(flat.size()));
    const double recomputed =
        etGScalarForward(model, flat, z, kMu, x.data(), {8, 4}, y.data(), 8);
    const double cached =
        etGScalarForward(model, flat, z, kMu, x.data(), {8, 4}, y.data(), 8, base);
    EXPECT_DOUBLE_EQ(cached, recomputed)
        << "seed " << seed << ": a cached base loss must be bit-identical, not an approximation";
  }
}

// fit()'s uploaded scalars must be unchanged by the caching — pinned against the same
// etGScalarForward the golden test already gates.
TEST(DeComFLBaseLossCache, FitScalarsUnchangedByCaching) {
  using namespace fedlearn;

  ExecutorchModel model(fedtest::goldenPath("zo_model_tiny.pte"), fedtest::kTinyPteSha);
  const auto x = fedtest::zoInputs();
  const auto y = fedtest::zoTargets();
  const DataBatch batch{x.data(), {8, 4}, y.data(), 8};
  const auto x0 = fedtest::readF32(fedtest::goldenPath("zo_flat.f32"));
  constexpr double kMu = 0.001;

  const Seeds2D seeds = {{101, 102, 103}};  // K=1, P=3 — one step, so x is the fit input
  DeComFLClient client(0.05, 3, 1);
  const GradientScalars2D got = client.fit(model, x0, seeds, batch, kMu);

  ASSERT_EQ(got.size(), 1u);
  ASSERT_EQ(got[0].size(), 3u);
  for (int p = 0; p < 3; ++p) {
    const std::vector<float> z = flat_randn(seeds[0][p], static_cast<int64_t>(x0.size()));
    const double want = etGScalarForward(model, x0, z, kMu, x.data(), {8, 4}, y.data(), 8);
    EXPECT_DOUBLE_EQ(got[0][p], want) << "fit scalar diverged from the gated g-scalar at p=" << p;
  }
}
