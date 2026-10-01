#include "fedlearn/BatchOrder.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace fedlearn {

namespace {
constexpr uint64_t kGamma = 0x9E3779B97F4A7C15ULL;
}  // namespace

uint64_t splitMix(uint64_t z) {
  z += kGamma;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

uint64_t SplitMix64::next() {
  const uint64_t r = splitMix(state_);
  state_ += kGamma;
  return r;
}

uint64_t SplitMix64::below(uint64_t bound) {
  if (bound == 0) throw std::invalid_argument("SplitMix64::below: bound must be positive");
  // Accept r < 2^64 - (2^64 mod bound). 2^64 mod bound == (UINT64_MAX mod bound + 1) mod bound, and when it is 0
  // every draw is accepted.
  const uint64_t rem = (UINT64_MAX % bound + 1) % bound;
  for (;;) {
    const uint64_t r = next();
    if (rem == 0 || r <= UINT64_MAX - rem) return r % bound;
  }
}

std::vector<uint64_t> seededPermutation(uint64_t n, uint64_t seed, uint64_t round, uint64_t epoch) {
  SplitMix64 rng(splitMix(splitMix(splitMix(seed) ^ round) ^ epoch));
  std::vector<uint64_t> order(n);
  for (uint64_t i = 0; i < n; ++i) order[i] = i;
  for (uint64_t i = n; i-- > 1;) std::swap(order[i], order[rng.below(i + 1)]);
  return order;
}

uint64_t parseBatchSeed(const std::string& decimal) {
  if (decimal.empty() || decimal.size() > 20) throw std::invalid_argument("invalid batch seed");
  uint64_t value = 0;
  for (const char c : decimal) {
    if (c < '0' || c > '9') throw std::invalid_argument("invalid batch seed");
    const uint64_t digit = static_cast<uint64_t>(c - '0');
    if (value > (UINT64_MAX - digit) / 10) throw std::invalid_argument("invalid batch seed");
    value = value * 10 + digit;
  }
  return value;
}

std::vector<std::vector<uint64_t>> batches(const std::vector<uint64_t>& order, uint64_t batchSize, bool dropLast) {
  if (batchSize == 0) throw std::invalid_argument("batches: batchSize must be positive");
  std::vector<std::vector<uint64_t>> out;
  for (uint64_t start = 0; start < order.size(); start += batchSize) {
    const uint64_t end = std::min<uint64_t>(start + batchSize, order.size());
    if (dropLast && end - start < batchSize) break;
    out.emplace_back(order.begin() + static_cast<std::ptrdiff_t>(start), order.begin() + static_cast<std::ptrdiff_t>(end));
  }
  return out;
}

}  // namespace fedlearn
