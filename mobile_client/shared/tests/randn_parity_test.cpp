// randn_parity_test.cpp — release gate over the standalone, ATen-free RNG (RandnEngine.h).
// Proves flat_randn reproduces the Python golden vectors WITHOUT libtorch. The legacy
// rng_parity_test.cpp covers the at::Tensor wrapper; this one covers the runtime-independent
// core that survives the migration off libtorch.
//
// GOLDEN_DIR is injected by CMake; the .f32 files are raw little-endian float32.
#include "fedlearn/RandnEngine.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <stdexcept>
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
  while (f.read(reinterpret_cast<char*>(&x), sizeof(float))) out.push_back(x);
  return out;
}

struct GoldenCase {
  int64_t seed;
  int64_t n;
  const char* file;
};

// Must match framework/tests/fixtures/decomfl_golden/manifest.json.
const GoldenCase kCases[] = {
    {0, 16, "z_0_16.f32"},
    {1, 100, "z_1_100.f32"},
    {1234567, 1000, "z_1234567_1000.f32"},
    {2147483646, 4096, "z_2147483646_4096.f32"},
};

}  // namespace

TEST(RandnParity, MatchesPythonGoldenVectors) {
  for (const auto& c : kCases) {
    const std::string path = std::string(GOLDEN_DIR) + "/" + c.file;
    const std::vector<float> golden = ReadF32(path);
    ASSERT_EQ(static_cast<int64_t>(golden.size()), c.n) << "size mismatch for " << c.file;

    const std::vector<float> z = fedlearn::flat_randn(c.seed, c.n);
    ASSERT_EQ(static_cast<int64_t>(z.size()), c.n);
    for (int64_t i = 0; i < c.n; ++i) {
      ASSERT_NEAR(z[static_cast<size_t>(i)], golden[static_cast<size_t>(i)], 1e-6f)
          << "RNG parity broken at " << c.file << "[" << i << "]";
    }
  }
}

TEST(RandnParity, RejectsNonPositiveN) {
  EXPECT_THROW(fedlearn::flat_randn(0, 0), std::invalid_argument);
  EXPECT_THROW(fedlearn::flat_randn(0, -1), std::invalid_argument);
}
