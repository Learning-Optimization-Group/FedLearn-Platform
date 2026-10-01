#include "fedlearn/Qualification.h"

#include <chrono>
#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
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
  x.resize(static_cast<size_t>(s.rows * s.width));
  y.resize(static_cast<size_t>(s.rows));
  for (int64_t i = 0; i < s.rows; ++i) {
    y[static_cast<size_t>(i)] = i % s.classes;
    for (int64_t j = 0; j < s.width; ++j) {
      x[static_cast<size_t>(i * s.width + j)] = static_cast<float>((31 * i + 7 * j + 3 * step) % 17 - 8) / 8.0f;
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
  std::vector<int64_t> xShape{spec.rows};
  if (spec.inputShape.empty()) {
    xShape.push_back(spec.width);
  } else {
    int64_t numel = 1;
    for (int64_t d : spec.inputShape) {
      if (d < 1) return fail(r, "SPEC", "the probe's input shape has an empty dimension");
      numel *= d;
      xShape.push_back(d);
    }
    if (numel != spec.width) return fail(r, "SPEC", "the probe's input shape does not hold its width");
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
  // A masked-dropout program takes one mask per dropout layer after (x, y); every mask is all ones, so dropout passes
  // activations through unchanged, as in the exporter's reference.
  std::vector<std::vector<float>> maskData;
  std::vector<InputTensor> masks;
  try {
    for (auto shape : model->extraInputShapes()) {
      if (shape.empty()) throw std::runtime_error("a mask input has no batch dimension");
      shape[0] = spec.rows;
      int64_t numel = 1;
      for (int64_t d : shape) numel *= d;
      maskData.emplace_back(static_cast<size_t>(numel), 1.0f);
      masks.push_back({nullptr, shape});
    }
    for (size_t k = 0; k < masks.size(); ++k) masks[k].data = maskData[k].data();
    const std::vector<InputTensor>* extra = masks.empty() ? nullptr : &masks;
    before = model->getFlatParams();
    probeBatch(1, spec, x, y);
    r.lossStep1 = model->trainStep(x.data(), xShape, y.data(), spec.rows, spec.learningRate,
                                   nullptr, 0.0f, extra);
    afterStep1 = model->getFlatParams();
    probeBatch(2, spec, x, y);
    r.lossStep2 = model->trainStep(x.data(), xShape, y.data(), spec.rows, spec.learningRate,
                                   nullptr, 0.0f, extra);
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
