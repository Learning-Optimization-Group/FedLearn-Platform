// et_multiround_test.cpp — DA-5 / D2b: multi-round DeComFL trajectory parity.
//
// The single-step goldens (et_g_scalar, executorch_forward, ...) prove every DeComFL
// KERNEL agrees Python<->C++. This test proves their COMPOSITION over N rounds: the C++
// mobile core replays the SAME N-round trajectory the Python framework froze
// (framework/tests/fixtures/decomfl_golden/generate_zo_multiround.py) and must land on the
// frozen endpoint within tolerance.
//
// Update rule (single client, N=1 so eta/(N*P) = eta/P), per round r, K local steps:
//     x <- x - (eta/P) * sum_{p=0..P-1} g_{k,p} * z_{k,p}
//   g = one-sided forward diff (etGScalarForward); z = flat_randn(seed, d); all float32.
// Matches DeComFLClient.cpp:15-27 / decomfl_equivalence_test.cpp:39-49 and the Python
// decomfl_client.py:266-290. Constants mirror zo_multiround_manifest.json.
//
// Tolerance-based (never bit-exact): forward-backend + float32-vs-double g + z arch-ULP
// drift accumulate over N*K*P evals; absorbed by *Atol below (see the conformance note in
// research/notes/2026-07-15-cross-language-conformance-contract.md).
#include "fedlearn/EtZeroOrder.h"
#include "fedlearn/ExecutorchModel.h"
#include "fedlearn/RandnEngine.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "fixtures.h"

namespace {

// Reads a raw little-endian binary of POD T (for the .f64 g-scalar golden; fixtures.h
// only ships readF32/readI64).
template <typename T>
std::vector<T> readBin(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<T> out;
  T v;
  while (f.read(reinterpret_cast<char*>(&v), sizeof(T))) out.push_back(v);
  return out;
}

// Mirror framework/tests/fixtures/decomfl_golden/zo_multiround_manifest.json.
constexpr int kNRounds = 3;
constexpr int kK = 1;
constexpr int kP = 4;
constexpr double kEta = 0.02;
constexpr double kMu = 0.001;
constexpr float kEndpointAtol = 2e-3f;  // manifest.endpoint_atol
constexpr double kGAtol = 2e-3;         // manifest.g_atol

}  // namespace

TEST(DeComFLMultiRound, EndpointMatchesFrozenPythonTrajectory) {
  using namespace fedlearn;

  ExecutorchModel model(fedtest::goldenPath("zo_model_tiny.pte"), fedtest::kTinyPteSha);
  const int64_t d = model.flatDim();
  ASSERT_EQ(d, 25);

  // start from the byte-identical initial flat the Python generator used
  std::vector<float> x = fedtest::readF32(fedtest::goldenPath("zo_flat.f32"));
  ASSERT_EQ(x.size(), static_cast<size_t>(d));

  const auto xin = fedtest::zoInputs();   // {8,4} -> 32 floats
  const auto yin = fedtest::zoTargets();  // 8 int64
  ASSERT_EQ(xin.size(), 32u);
  ASSERT_EQ(yin.size(), 8u);

  // frozen references (row-major [r][k][p] for seeds/g)
  const auto seeds = fedtest::readI64(fedtest::goldenPath("zo_multiround_seeds.i64"));
  const auto goldenG = readBin<double>(fedtest::goldenPath("zo_multiround_g.f64"));
  const auto finalFlat = fedtest::readF32(fedtest::goldenPath("zo_multiround_final.f32"));
  ASSERT_EQ(seeds.size(), static_cast<size_t>(kNRounds * kK * kP));
  ASSERT_EQ(goldenG.size(), static_cast<size_t>(kNRounds * kK * kP));
  ASSERT_EQ(finalFlat.size(), static_cast<size_t>(d));

  const std::vector<int64_t> xShape = {8, 4};
  const float stepCoeff = static_cast<float>(kEta / kP);
  int idx = 0;  // flattened [r][k][p] index

  for (int r = 0; r < kNRounds; ++r) {
    for (int k = 0; k < kK; ++k) {
      // all P perturbations evaluated at the SAME pre-step x, summed, then x advances once
      std::vector<float> delta(static_cast<size_t>(d), 0.0f);
      for (int p = 0; p < kP; ++p) {
        const int64_t seed = seeds[static_cast<size_t>(idx)];
        const std::vector<float> z = flat_randn(seed, d);
        const double g = etGScalarForward(model, x, z, kMu, xin.data(), xShape, yin.data(), 8);
        EXPECT_NEAR(g, goldenG[static_cast<size_t>(idx)], kGAtol)
            << "g diverged from Python at round " << r << " step " << k << " perturbation " << p;
        for (int64_t i = 0; i < d; ++i) delta[static_cast<size_t>(i)] += static_cast<float>(g) * z[static_cast<size_t>(i)];
        ++idx;
      }
      for (int64_t i = 0; i < d; ++i) x[static_cast<size_t>(i)] -= stepCoeff * delta[static_cast<size_t>(i)];
    }
  }

  // primary conformance claim: the N-round endpoint matches the Python-frozen trajectory
  float maxAbs = 0.0f;
  for (int64_t i = 0; i < d; ++i) {
    EXPECT_NEAR(x[static_cast<size_t>(i)], finalFlat[static_cast<size_t>(i)], kEndpointAtol)
        << "endpoint diverged at param " << i;
    maxAbs = std::max(maxAbs, std::fabs(x[static_cast<size_t>(i)] - finalFlat[static_cast<size_t>(i)]));
  }
  std::cout << "[multiround] max|endpoint diff| = " << maxAbs << " (atol " << kEndpointAtol << ")\n";
}
