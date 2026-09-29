#include "fedlearn/DatasetEvaluation.h"

#include <algorithm>

#include "fedlearn/EvalMetrics.h"

namespace fedlearn {

DatasetMetrics evaluateDataset(ExecutorchModel& lossModel, ExecutorchModel& inferModel,
                               const std::vector<float>& flat, const DataBatch& data, int64_t chunk) {
  DatasetMetrics out;
  const int64_t n = data.numSamples;
  if (n <= 0 || data.inputs == nullptr || data.targets == nullptr || data.inputShape.empty()) return out;
  int64_t rowWidth = 1;
  for (size_t d = 1; d < data.inputShape.size(); ++d) rowWidth *= data.inputShape[d];
  const int64_t step = chunk > 0 ? chunk : n;

  double weightedLoss = 0.0;
  int64_t scored = 0, correct = 0;
  for (int64_t start = 0; start < n; start += step) {
    const int64_t count = std::min(step, n - start);
    std::vector<int64_t> shape = data.inputShape;
    shape[0] = count;
    const float* x = data.inputs + start * rowWidth;  // consecutive rows are contiguous: no copy
    const int64_t* y = data.targets + start;
    weightedLoss += static_cast<double>(lossModel.loss(flat, x, shape, y, count)) * static_cast<double>(count);
    const AccuracyCount acc = argmaxCorrect(inferModel.infer(flat, x, shape), y, count);
    scored += acc.scored;
    correct += acc.correct;
  }
  out.loss = weightedLoss / static_cast<double>(n);
  out.accuracy = scored > 0 ? static_cast<double>(correct) / static_cast<double>(scored) : 0.0;
  return out;
}

}  // namespace fedlearn
