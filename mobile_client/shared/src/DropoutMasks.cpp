#include "fedlearn/DropoutMasks.h"

#include <cmath>
#include <stdexcept>

#include "fedlearn/BatchOrder.h"

namespace fedlearn {

namespace {
void checkRate(double rate) {
  if (!(std::isfinite(rate) && rate >= 0 && rate < 1)) {
    throw std::invalid_argument("a dropout rate must be in [0, 1)");
  }
}
}  // namespace

std::vector<bool> dropoutKeep(uint64_t n, double rate, uint64_t seed, uint64_t round, uint64_t step, uint64_t layer) {
  checkRate(rate);
  SplitMix64 rng(splitMix(splitMix(splitMix(splitMix(seed) ^ round) ^ step) ^ layer));
  std::vector<bool> keep(n);
  for (uint64_t i = 0; i < n; ++i) {
    const double u = static_cast<double>(rng.next() >> 11) * 0x1.0p-53;
    keep[i] = u >= rate;
  }
  return keep;
}

float dropoutScale(double rate) {
  checkRate(rate);
  return 1.0f / static_cast<float>(1.0 - rate);
}

std::vector<float> dropoutMask(uint64_t n, double rate, uint64_t seed, uint64_t round, uint64_t step, uint64_t layer) {
  const float scale = dropoutScale(rate);
  const auto keep = dropoutKeep(n, rate, seed, round, step, layer);
  std::vector<float> mask(n);
  for (uint64_t i = 0; i < n; ++i) mask[i] = keep[i] ? scale : 0.0f;
  return mask;
}

}  // namespace fedlearn
