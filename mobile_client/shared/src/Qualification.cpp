#include "fedlearn/Qualification.h"

#include <chrono>
#include <cmath>
#include <exception>
#include <memory>
#include <utility>

#include "fedlearn/TrainableExecutorchModel.h"

namespace fedlearn {

namespace {

QualificationReport fail(QualificationReport r, const char* check, std::string detail) {
  r.passed = false;
  r.failedCheck = check;
  r.detail = std::move(detail);
  return r;
}

void probeBatch(int step, const ProbeSpec& s, std::vector<float>& x, std::vector<int64_t>& y) {
  x.assign(static_cast<size_t>(s.rows * s.width), 0.0f);
  y.resize(static_cast<size_t>(s.rows));
  for (int64_t i = 0; i < s.rows; ++i) {
    y[static_cast<size_t>(i)] = i % s.classes;
    if (step == 1) continue;
    for (int64_t j = 0; j < s.width; ++j) {
      x[static_cast<size_t>(i * s.width + j)] = static_cast<float>((i * s.width + j) % 7 - 3) / 4.0f;
    }
  }
}

}  // namespace

QualificationReport qualifyTrainable(const std::string& ptePath, const std::string& sha256,
                                     const std::vector<std::string>& paramNames, const ProbeSpec& spec) {
  QualificationReport r;
  if (spec.rows < 1 || spec.width < 1 || spec.classes < 1 || !(spec.learningRate > 0) ||
      !std::isfinite(spec.expectedLossStep1) || !std::isfinite(spec.expectedLossStep2) || !(spec.lossTolerance > 0)) {
    return fail(r, "SPEC", "the probe specification is incomplete");
  }
  const auto t0 = std::chrono::steady_clock::now();
  std::unique_ptr<TrainableExecutorchModel> model;
  try {
    model = std::make_unique<TrainableExecutorchModel>(ptePath, sha256, paramNames);
  } catch (const std::exception& e) {
    return fail(r, "LOAD", e.what());
  }
  std::vector<float> x;
  std::vector<int64_t> y;
  std::vector<float> before;
  std::vector<float> afterStep1;
  try {
    before = model->getFlatParams();
    probeBatch(1, spec, x, y);
    r.lossStep1 = model->trainStep(x.data(), {spec.rows, spec.width}, y.data(), spec.rows, spec.learningRate);
    afterStep1 = model->getFlatParams();
    probeBatch(2, spec, x, y);
    r.lossStep2 = model->trainStep(x.data(), {spec.rows, spec.width}, y.data(), spec.rows, spec.learningRate);
  } catch (const std::exception& e) {
    return fail(r, "LOAD", e.what());
  }
  const std::vector<float> afterStep2 = model->getFlatParams();
  r.wallMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();

  const auto lossMismatch = [&](int step, double got, double want) {
    return fail(r, "LOSS_MISMATCH", "step " + std::to_string(step) + " loss " + std::to_string(got) +
                                        ", expected " + std::to_string(want));
  };
  if (!std::isfinite(r.lossStep1) || std::fabs(r.lossStep1 - spec.expectedLossStep1) > spec.lossTolerance) {
    return lossMismatch(1, r.lossStep1, spec.expectedLossStep1);
  }
  // Checked before the step-2 loss, which unchanged weights would also throw off: the more specific diagnosis.
  if (afterStep1 == before || afterStep2 == afterStep1) {
    return fail(r, "STALE_WEIGHTS", "a probe step left the parameters unchanged");
  }
  if (!std::isfinite(r.lossStep2) || std::fabs(r.lossStep2 - spec.expectedLossStep2) > spec.lossTolerance) {
    return lossMismatch(2, r.lossStep2, spec.expectedLossStep2);
  }
  if (r.wallMs > spec.maxProbeMs) {
    return fail(r, "TOO_SLOW", "the probe took " + std::to_string(r.wallMs) + " ms, over its " +
                                   std::to_string(spec.maxProbeMs) + " ms budget");
  }
  r.passed = true;
  return r;
}

}  // namespace fedlearn
