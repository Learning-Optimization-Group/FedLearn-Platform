// Qualification.h — the portable-CPU qualification probe (Stage 3 D2).
//
// Before a device trains a run it proves it can run the run's trainable program correctly: two SGD steps on a
// synthetic batch from the program's embedded weights (at step s, x[i][j] = ((31 i + 7 j + 3 s) mod 17 - 8) / 8; labels
// i mod classes), each step's loss matching the exporter's reference,
// the parameters changing, and the whole probe finishing within its budget. The probe is a property of the program,
// so a device caches its result by program digest.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace fedlearn {

/** The probe the exporter recorded for a trainable program (its manifest's probe block). */
struct ProbeSpec {
  int64_t rows = 0;
  int64_t width = 0;
  int64_t classes = 0;
  float learningRate = 0.0f;
  double expectedLossStep1 = 0.0;
  double expectedLossStep2 = 0.0;
  double lossTolerance = 0.0;
  int64_t maxProbeMs = 0;
};

/** passed, or the first failed check: SPEC, LOAD, LOSS_MISMATCH, STALE_WEIGHTS or TOO_SLOW, with its detail. */
struct QualificationReport {
  bool passed = false;
  std::string failedCheck;
  std::string detail;
  double lossStep1 = 0.0;
  double lossStep2 = 0.0;
  int64_t wallMs = 0;
};

#ifdef FEDLEARN_HAS_TRAINING
QualificationReport qualifyTrainable(const std::string& ptePath, const std::string& sha256,
                                     const std::vector<std::string>& paramNames, const ProbeSpec& spec);
#endif

}  // namespace fedlearn
