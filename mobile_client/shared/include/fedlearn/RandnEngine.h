#pragma once
//
// RandnEngine.h — ATen-free reproduction of
//   torch.randn(n, generator=torch.Generator("cpu").manual_seed(seed), dtype=float32)
//
// This is the runtime-independent core of the DeComFL cross-language RNG contract.
// It reproduces PyTorch's CPU randn byte-for-byte (within float rounding) WITHOUT any
// dependency on libtorch/ATen, so the perturbation parity gate survives the migration
// off libtorch to ExecuTorch.
//
// Source of truth : framework/src/fedlearn/estimators/perturbation.py::canonical_perturbation
// Release gate     : mobile_client/shared/tests/randn_parity_test.cpp (golden vectors, 1e-6).
// Golden version   : torch 2.12.0 (framework/tests/fixtures/decomfl_golden/manifest.json).
//
// Algorithm (mirrors aten/src/ATen/native/cpu/DistributionTemplates.h):
//   * MT19937 engine, seeded as torch.Generator.manual_seed (state[0] = seed & 0xffffffff).
//   * n < 16  -> scalar normal_distribution<double>: per-element Box-Muller (cos returned,
//                sin cached in a double cache), uniforms from random64()'s low 53 bits.
//   * n >= 16 -> vectorized normal_fill (float): fill the buffer with float uniforms (each a
//                single 32-bit draw, low 24 bits), then Box-Muller in blocks of 16
//                (u1 = 1 - data[j], u2 = data[j+8]); the non-16-multiple tail is recomputed
//                from a fresh block of 16 uniforms over the final 16 slots.
//
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace fedlearn {
namespace detail {

// PyTorch at::mt19937 (aten/src/ATen/core/MT19937RNGEngine.h).
class MT19937 {
 public:
  explicit MT19937(uint64_t seed) {
    state_[0] = static_cast<uint32_t>(seed & 0xffffffffULL);
    for (int i = 1; i < kN; ++i)
      state_[i] = 1812433253u * (state_[i - 1] ^ (state_[i - 1] >> 30)) + static_cast<uint32_t>(i);
    left_ = 1;
    next_ = 0;
  }
  uint32_t operator()() {
    if (--left_ <= 0) next_state();
    uint32_t y = state_[next_++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
  }

 private:
  static constexpr int kN = 624, kM = 397;
  uint32_t state_[kN];
  int left_ = 1;
  int next_ = 0;
  static uint32_t twist(uint32_t u, uint32_t v) {
    return (((u & 0x80000000u) | (v & 0x7fffffffu)) >> 1) ^ ((v & 1u) ? 0x9908b0dfu : 0u);
  }
  void next_state() {
    uint32_t* p = state_;
    left_ = kN;
    next_ = 0;
    int j;
    for (j = kN - kM + 1; --j; ++p) *p = p[kM] ^ twist(p[0], p[1]);
    for (j = kM; --j; ++p)          *p = p[kM - kN] ^ twist(p[0], p[1]);
    *p = p[kM - kN] ^ twist(p[0], state_[0]);
  }
};

// uniform_real_distribution<double>: low 53 bits of random64() (two 32-bit draws, hi first).
inline double uniform_real_double(MT19937& g) {
  uint64_t r1 = g(), r2 = g();
  uint64_t val = (r1 << 32) | r2;
  constexpr uint64_t kMask = (static_cast<uint64_t>(1) << 53) - 1;
  constexpr double kDiv = 1.0 / static_cast<double>(static_cast<uint64_t>(1) << 53);
  return static_cast<double>(val & kMask) * kDiv;
}

// uniform_real_distribution<float>: low 24 bits of a single 32-bit draw.
inline float uniform_real_float(MT19937& g) {
  uint32_t val = g();
  constexpr uint32_t kMask = (1u << 24) - 1;
  constexpr float kDiv = 1.0f / static_cast<float>(1u << 24);
  return static_cast<float>(val & kMask) * kDiv;
}

// PyTorch normal_fill_16: in-place block-of-16 Box-Muller over pre-filled float uniforms.
inline void normal_fill_16(float* data) {
  constexpr float kPi = 3.14159265358979323846f;
  for (int j = 0; j < 8; ++j) {
    float u1 = 1.0f - data[j];  // map [0,1) -> (0,1] so log() is finite
    float u2 = data[j + 8];
    float r = std::sqrt(-2.0f * std::log(u1));
    float theta = 2.0f * kPi * u2;
    data[j] = r * std::cos(theta);
    data[j + 8] = r * std::sin(theta);
  }
}

}  // namespace detail

// Returns a device-independent N(0, I_d) sample of length n, reproducing
// torch.randn(n, generator=Generator("cpu").manual_seed(seed), float32).
// Throws std::invalid_argument if n <= 0.
inline std::vector<float> flat_randn(int64_t seed, int64_t n) {
  if (n <= 0) throw std::invalid_argument("num_params must be positive");
  detail::MT19937 g(static_cast<uint64_t>(seed));
  std::vector<float> data(static_cast<size_t>(n));

  if (n < 16) {
    constexpr double kPi = 3.14159265358979323846;
    bool have_cache = false;
    double cache = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      double sample;
      if (have_cache) {
        have_cache = false;
        sample = cache;
      } else {
        double u1 = detail::uniform_real_double(g);
        double u2 = detail::uniform_real_double(g);
        double r = std::sqrt(-2.0 * std::log(1.0 - u2));
        double theta = 2.0 * kPi * u1;
        cache = r * std::sin(theta);
        have_cache = true;
        sample = r * std::cos(theta);
      }
      data[static_cast<size_t>(i)] = static_cast<float>(sample);
    }
    return data;
  }

  for (int64_t i = 0; i < n; ++i) data[static_cast<size_t>(i)] = detail::uniform_real_float(g);
  for (int64_t i = 0; i < n - 15; i += 16) detail::normal_fill_16(&data[static_cast<size_t>(i)]);
  if (n % 16 != 0) {
    float* tail = &data[static_cast<size_t>(n - 16)];
    for (int i = 0; i < 16; ++i) tail[i] = detail::uniform_real_float(g);
    detail::normal_fill_16(tail);
  }
  return data;
}

}  // namespace fedlearn
