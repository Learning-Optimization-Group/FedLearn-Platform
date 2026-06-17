// et_g_scalar_test.cpp — the ExecuTorch ZO g-scalar release gate (libtorch-free).
//
// Asserts the forward-difference g-scalar computed via ExecutorchModel + flat_randn reproduces
// the frozen golden_g (the same reference the libtorch g_scalar_parity_test validates), within
// the same 2e-3 tolerance. This proves the DeComFL zeroth-order path works on ExecuTorch.
//
// GOLDEN_DIR is injected by CMake and points at framework/tests/fixtures/decomfl_golden.
#include "fedlearn/EtZeroOrder.h"
#include "fedlearn/ExecutorchModel.h"
#include "fedlearn/RandnEngine.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#ifndef GOLDEN_DIR
#define GOLDEN_DIR "."
#endif

namespace {

template <typename T>
std::vector<T> ReadBin(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  EXPECT_TRUE(f.good()) << "cannot open fixture: " << path;
  std::vector<T> out;
  T x;
  while (f.read(reinterpret_cast<char*>(&x), sizeof(T))) out.push_back(x);
  return out;
}

const char* kPteSha256 = "2eca3c02e2084383f038494d6ecf7c20a1e7e0a1dcc6d7ce2b6e11e7d82f1c56";
constexpr double kMu = 0.001;  // matches generate_zo.py MU

// Frozen golden_g for these seeds (framework/tests/fixtures/decomfl_golden/zo_manifest.json).
struct Case {
  int64_t seed;
  double golden_g;
};
const Case kCases[] = {
    {11, 0.36466118693351746},
    {22, -0.04482268914580345},
    {33, -0.6026029586791992},
    {4242, 0.5087852478027344},
};

std::string ptePath() { return std::string(GOLDEN_DIR) + "/zo_model_tiny.pte"; }

}  // namespace

TEST(EtGScalar, MatchesGoldenG) {
  fedlearn::ExecutorchModel model(ptePath(), kPteSha256);

  const auto flat = ReadBin<float>(std::string(GOLDEN_DIR) + "/zo_flat.f32");
  const auto x = ReadBin<float>(std::string(GOLDEN_DIR) + "/zo_inputs.f32");
  const auto y = ReadBin<int64_t>(std::string(GOLDEN_DIR) + "/zo_targets.i64");
  ASSERT_EQ(flat.size(), 25u);
  ASSERT_EQ(x.size(), 32u);
  ASSERT_EQ(y.size(), 8u);

  for (const auto& c : kCases) {
    const std::vector<float> z = fedlearn::flat_randn(c.seed, static_cast<int64_t>(flat.size()));
    const double g =
        fedlearn::etGScalarForward(model, flat, z, kMu, x.data(), {8, 4}, y.data(), 8);
    // Same tolerance as the libtorch g_scalar_parity_test: the forward-difference at mu=1e-3 is
    // a cancellation regime, but a wrong z order / frozen-layer slip would blow well past 2e-3.
    EXPECT_NEAR(g, c.golden_g, 2e-3) << "ExecuTorch g-scalar parity broke at seed " << c.seed;
  }
}

TEST(EtGScalar, CentralMatchesForwardIdentity) {
  fedlearn::ExecutorchModel model(ptePath(), kPteSha256);
  const auto flat = ReadBin<float>(std::string(GOLDEN_DIR) + "/zo_flat.f32");
  const auto x = ReadBin<float>(std::string(GOLDEN_DIR) + "/zo_inputs.f32");
  const auto y = ReadBin<int64_t>(std::string(GOLDEN_DIR) + "/zo_targets.i64");
  ASSERT_EQ(flat.size(), 25u);

  for (const auto& c : kCases) {
    const std::vector<float> z = fedlearn::flat_randn(c.seed, static_cast<int64_t>(flat.size()));
    std::vector<float> negZ(z.size());
    for (size_t i = 0; i < z.size(); ++i) negZ[i] = -z[i];

    const double central =
        fedlearn::etGScalarCentral(model, flat, z, kMu, x.data(), {8, 4}, y.data(), 8);
    const double fwdPos =
        fedlearn::etGScalarForward(model, flat, z, kMu, x.data(), {8, 4}, y.data(), 8);
    const double fwdNeg =
        fedlearn::etGScalarForward(model, flat, negZ, kMu, x.data(), {8, 4}, y.data(), 8);

    // Exact identity (up to float rounding): central(z) = (forward(z) - forward(-z)) / 2.
    // forward(z) is already golden-gated, so this transitively validates etGScalarCentral.
    EXPECT_NEAR(central, 0.5 * (fwdPos - fwdNeg), 1e-5)
        << "central-difference identity broke at seed " << c.seed;
  }
}
