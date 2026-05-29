// rng_parity_test.cpp — THE RELEASE GATE for cross-language RNG parity (15-LLD §13 task 4).
//
// Asserts the C++ canonical_perturbation reproduces the Python golden vectors frozen in
// framework/tests/fixtures/decomfl_golden/ (the source of truth). If this fails, the
// mobile build MUST NOT ship: a divergent perturbation silently corrupts DeComFL
// aggregation between a Python server and a C++ mobile client.
//
// GOLDEN_DIR is injected by CMake (target_compile_definitions) and points at the
// framework fixture directory. The .f32 files are raw little-endian float32.

#include "fedlearn/Perturbation.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#ifndef GOLDEN_DIR
#define GOLDEN_DIR "."
#endif

namespace {

std::vector<float> ReadF32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  EXPECT_TRUE(f.good()) << "cannot open golden fixture: " << path;
  std::vector<float> out;
  float x = 0.0f;
  while (f.read(reinterpret_cast<char*>(&x), sizeof(float))) {
    out.push_back(x);
  }
  return out;
}

struct GoldenCase {
  int64_t seed;
  int64_t n;
  const char* file;
};

// Must match the cases frozen by
// framework/tests/fixtures/decomfl_golden/generate.py (manifest.json).
const GoldenCase kCases[] = {
    {0, 16, "z_0_16.f32"},
    {1, 100, "z_1_100.f32"},
    {1234567, 1000, "z_1234567_1000.f32"},
    {2147483646, 4096, "z_2147483646_4096.f32"},
};

}  // namespace

TEST(RngParity, MatchesPythonGoldenVectors) {
  for (const auto& c : kCases) {
    const std::string path = std::string(GOLDEN_DIR) + "/" + c.file;
    const std::vector<float> golden = ReadF32(path);
    ASSERT_EQ(static_cast<int64_t>(golden.size()), c.n) << "size mismatch for " << c.file;

    at::Tensor z = fedlearn::canonical_perturbation(c.seed, c.n, at::kFloat);
    ASSERT_EQ(z.numel(), c.n);
    ASSERT_EQ(z.scalar_type(), at::kFloat);
    ASSERT_TRUE(z.device().is_cpu());

    const auto za = z.accessor<float, 1>();
    for (int64_t i = 0; i < c.n; ++i) {
      // ULP-tolerance parity. Tighten toward bit-exact (EXPECT_FLOAT_EQ) once the pinned
      // libtorch is confirmed to share Python's normal kernel exactly.
      ASSERT_NEAR(za[i], golden[i], 1e-6f)
          << "RNG parity broken at " << c.file << "[" << i << "]";
    }
  }
}
