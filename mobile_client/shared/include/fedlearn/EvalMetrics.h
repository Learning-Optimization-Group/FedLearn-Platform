#pragma once
//
// EvalMetrics.h — bounds-safe top-1 accuracy counting for on-device eval (MO-6).
//
// Extracted out of FedLearnCoreModule::evalBatch so the argmax/bounds logic is unit-testable on the
// host (the RN bridge only compiles in the Android build; the shared core has a gtest gate). The naive
// inline loop it replaces read logits[row * classes] unconditionally — when numSamples > 0 but the
// infer graph returned an empty/short logits vector, `classes` was 0 and logits[0] indexed an empty
// buffer (OOB / UB). This guards that and every ragged-output case.
//
// Torch-free, header-only, no ExecuTorch dependency: pure arithmetic over the caller's buffers.
//
#include <cstdint>
#include <vector>

namespace fedlearn {

struct AccuracyCount {
  int64_t scored = 0;   // rows actually compared (0 when nothing was evaluable)
  int64_t correct = 0;  // of those, how many argmax(logits) == target
};

// Top-1 accuracy count of row-major logits [numSamples x C] against int64 targets. `targets` is trusted
// to hold `numSamples` entries (its DataBatch contract). Returns {0,0} — "not evaluable" — for an empty
// batch, a null targets pointer, or an empty/short infer output (the classes<=0 case that used to OOB).
// Rows are additionally capped by the logits length so a ragged infer result never indexes past logits.
// On a tie the lowest class index wins (strict '>' keeps the first maximum), matching the prior loop.
inline AccuracyCount argmaxCorrect(const std::vector<float>& logits, const int64_t* targets,
                                   int64_t numSamples) {
  if (numSamples <= 0 || targets == nullptr || logits.empty()) return {};
  const int64_t classes = static_cast<int64_t>(logits.size()) / numSamples;
  if (classes <= 0) return {};  // fewer logits than samples -> unusable; never read logits[0] blindly.
  const int64_t byLogits = static_cast<int64_t>(logits.size()) / classes;  // rows the logits actually cover
  const int64_t rows = numSamples < byLogits ? numSamples : byLogits;
  int64_t correct = 0;
  for (int64_t row = 0; row < rows; ++row) {
    const size_t base = static_cast<size_t>(row) * static_cast<size_t>(classes);
    int64_t best = 0;
    float bestVal = logits[base];
    for (int64_t col = 1; col < classes; ++col) {
      const float v = logits[base + static_cast<size_t>(col)];
      if (v > bestVal) {
        bestVal = v;
        best = col;
      }
    }
    if (best == targets[row]) ++correct;
  }
  return {rows, correct};
}

}  // namespace fedlearn
