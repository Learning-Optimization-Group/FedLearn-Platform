// BatchOrder.h — the execution contract's reproducible batch order (BATCH_ORDER_SEEDED_PERMUTATION_V1).
//
// Every runtime draws the same permutation of a participant's examples for a given run seed, round and local epoch,
// so an update can be replayed exactly from its data. The algorithm is specified in execution_contract.proto
// (SplitMix64 stream, Fisher-Yates with rejection sampling) and pinned, with the framework's implementation, by
// framework/tests/fixtures/execution_contract_v1/batch_permutation_v1.golden.
#pragma once

#include <cstdint>
#include <vector>

namespace fedlearn {

/** SplitMix64: each draw is the output function of the state, which then advances by the golden gamma. */
class SplitMix64 {
 public:
  explicit SplitMix64(uint64_t state) : state_(state) {}
  uint64_t next();
  /** Uniform in [0, bound), rejecting the draws that would bias r mod bound. */
  uint64_t below(uint64_t bound);

 private:
  uint64_t state_;
};

/** SplitMix64's output function applied to z + gamma. */
uint64_t splitMix(uint64_t z);

/** The order in which a participant visits its n examples in this round's epoch. */
std::vector<uint64_t> seededPermutation(uint64_t n, uint64_t seed, uint64_t round, uint64_t epoch);

/** Consecutive batchSize slices of order; the last, shorter one is kept unless dropLast. */
std::vector<std::vector<uint64_t>> batches(const std::vector<uint64_t>& order, uint64_t batchSize, bool dropLast);

}  // namespace fedlearn
