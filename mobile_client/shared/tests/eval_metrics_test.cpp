// eval_metrics_test.cpp — argmaxCorrect bounds + correctness (MO-6). Torch-free; pure arithmetic.
// The headline regression: numSamples > 0 with an EMPTY logits vector must NOT read logits[0].

#include "fedlearn/EvalMetrics.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using fedlearn::argmaxCorrect;

TEST(EvalMetrics, CountsTopOneCorrectly) {
  // 3 samples x 3 classes; argmax rows -> {2, 0, 1}. targets {2, 0, 0} -> 2 correct.
  const std::vector<float> logits = {
      0.1f, 0.2f, 0.9f,  // -> class 2
      0.8f, 0.1f, 0.1f,  // -> class 0
      0.3f, 0.7f, 0.0f,  // -> class 1
  };
  const std::vector<int64_t> targets = {2, 0, 0};
  const auto r = argmaxCorrect(logits, targets.data(), 3);
  EXPECT_EQ(r.scored, 3);
  EXPECT_EQ(r.correct, 2);
}

TEST(EvalMetrics, EmptyLogitsWithPositiveSamplesIsNotEvaluable_NoOOB) {
  // The exact prior-bug shape: n>0 but infer returned nothing. Must be {0,0}, not an OOB read.
  const std::vector<int64_t> targets = {0, 1, 2};
  const auto r = argmaxCorrect(std::vector<float>{}, targets.data(), 3);
  EXPECT_EQ(r.scored, 0);
  EXPECT_EQ(r.correct, 0);
}

TEST(EvalMetrics, FewerLogitsThanSamplesIsNotEvaluable) {
  // 2 floats but 3 samples -> classes = 0 -> not evaluable (guarded before any logits[base] read).
  const std::vector<float> logits = {0.5f, 0.5f};
  const std::vector<int64_t> targets = {0, 0, 0};
  const auto r = argmaxCorrect(logits, targets.data(), 3);
  EXPECT_EQ(r.scored, 0);
}

TEST(EvalMetrics, ZeroOrNegativeSamplesIsNotEvaluable) {
  const std::vector<float> logits = {0.1f, 0.9f};
  const std::vector<int64_t> targets = {1};
  EXPECT_EQ(argmaxCorrect(logits, targets.data(), 0).scored, 0);
  EXPECT_EQ(argmaxCorrect(logits, targets.data(), -1).scored, 0);
}

TEST(EvalMetrics, NullTargetsIsNotEvaluable) {
  const std::vector<float> logits = {0.1f, 0.9f};
  EXPECT_EQ(argmaxCorrect(logits, nullptr, 1).scored, 0);
}

TEST(EvalMetrics, RaggedLogitsNeverIndexPastTheBuffer) {
  // 7 floats, 3 samples -> classes = 2 (floor); rows capped so max index (2*2 + 1 = 5) stays < 7.
  // Rows: {0.1,0.9}->1, {0.8,0.2}->0, {0.3,0.4}->1. targets {1,0,1} -> 3 correct over 3 scored.
  const std::vector<float> logits = {0.1f, 0.9f, 0.8f, 0.2f, 0.3f, 0.4f, 0.0f};
  const std::vector<int64_t> targets = {1, 0, 1};
  const auto r = argmaxCorrect(logits, targets.data(), 3);
  EXPECT_EQ(r.scored, 3);
  EXPECT_EQ(r.correct, 3);
}

TEST(EvalMetrics, TieGoesToLowestClassIndex) {
  // Both classes equal -> argmax stays at index 0 (strict '>'), matching the prior loop.
  const std::vector<float> logits = {0.5f, 0.5f};
  const std::vector<int64_t> hitLow = {0};
  const std::vector<int64_t> hitHigh = {1};
  EXPECT_EQ(argmaxCorrect(logits, hitLow.data(), 1).correct, 1);
  EXPECT_EQ(argmaxCorrect(logits, hitHigh.data(), 1).correct, 0);
}

TEST(EvalMetrics, AllWrongIsZeroCorrectNotUnknown) {
  // Distinguish a real 0.0 accuracy (scored>0, correct==0) from "not evaluable" (scored==0).
  const std::vector<float> logits = {0.9f, 0.1f, 0.9f, 0.1f};  // both -> class 0
  const std::vector<int64_t> targets = {1, 1};
  const auto r = argmaxCorrect(logits, targets.data(), 2);
  EXPECT_EQ(r.scored, 2);
  EXPECT_EQ(r.correct, 0);
}
