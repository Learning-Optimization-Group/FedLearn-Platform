// DropoutMasks.h — the execution contract's reproducible dropout masks (DROPOUT_MASKS_SEEDED_V1).
//
// For a run seed, round, local step and dropout layer, every runtime draws the same mask: the SplitMix64 stream of
// BatchOrder.h seeded with mix(mix(mix(mix(seed) ^ round) ^ step) ^ layer), one draw per activation element in
// row-major order, u = (r >> 11) * 2^-53, dropped iff u < rate, kept elements scaled by float32(1) / float32(1 - rate).
// Pinned with the framework's implementation by framework/tests/fixtures/execution_contract_v1/dropout_masks_v1.golden.
#pragma once

#include <cstdint>
#include <vector>

namespace fedlearn {

/** Which of a layer's n activation elements (row-major) this step keeps. Throws for a rate outside [0, 1). */
std::vector<bool> dropoutKeep(uint64_t n, double rate, uint64_t seed, uint64_t round, uint64_t step, uint64_t layer);

/** The kept elements' multiplier: float32(1) / float32(1 - rate). */
float dropoutScale(double rate);

/** The float32 mask the layer's activations are multiplied by: the scale where kept, 0 where dropped. */
std::vector<float> dropoutMask(uint64_t n, double rate, uint64_t seed, uint64_t round, uint64_t step, uint64_t layer);

}  // namespace fedlearn
