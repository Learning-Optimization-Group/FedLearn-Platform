// g_scalar_parity_test.cpp — the C++ ZerothOrderEstimator must reproduce the Python
// reference g scalars frozen by generate_zo.py (15-LLD §13 task 7).
#include "fedlearn/ModelManager.h"
#include "fedlearn/ZerothOrderEstimator.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>

#include "fixtures.h"

TEST(GScalarParity, MatchesPythonReference) {
  fedlearn::ModelManager mm;
  torch::jit::Module model = fedtest::loadTinyModel();
  fedlearn::ZerothOrderEstimator zo(mm, /*mu=*/0.001, fedlearn::GradEstimateMethod::Forward);

  const torch::Tensor inputs = fedtest::zoInputs();
  const torch::Tensor targets = fedtest::zoTargets();
  const torch::Tensor flat = mm.getFlatParams(model);
  ASSERT_EQ(flat.numel(), 25);

  struct Case {
    int64_t seed;
    double golden_g;
  };
  // Frozen by framework/tests/fixtures/decomfl_golden/generate_zo.py (manifest golden_g).
  const Case cases[] = {
      {11, 0.36466118693351746},
      {22, -0.04482268914580345},
      {33, -0.6026029586791992},
      {4242, 0.5087852478027344},
  };

  for (const auto& c : cases) {
    const double g = zo.computeGradientScalar(model, flat, c.seed, inputs, targets);
    // The Python reference subtracts two float32 losses then divides; this C++ subtracts in
    // DOUBLE (the M-C2 #1 fix), so the two agree only to ~1e-3 in this catastrophic-
    // cancellation regime (mu=1e-3). Gross errors — a wrong z order, a missing frozen-layer
    // filter, a float-not-double diff — diverge far more and are caught here.
    EXPECT_NEAR(g, c.golden_g, 2e-3) << "g-scalar parity broke at seed " << c.seed;
  }
}
