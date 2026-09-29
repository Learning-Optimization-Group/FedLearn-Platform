// DatasetEvaluation.h — evaluate a device's whole dataset with the weights-as-inputs loss and infer programs.
//
// The programs take at most a batch of examples per call, so the dataset is evaluated in consecutive chunks of at
// most `chunk` examples. The loss is the example-weighted mean of the chunk losses (the whole dataset's mean
// cross-entropy) and the accuracy counts every example once.
#pragma once

#include <cstdint>
#include <vector>

#include "fedlearn/ExecutorchModel.h"
#include "fedlearn/Types.h"

namespace fedlearn {

struct DatasetMetrics {
  double loss = 0.0;
  double accuracy = 0.0;
};

/** chunk <= 0 evaluates the whole dataset in one call. An empty dataset is not evaluable and reports (0, 0). */
DatasetMetrics evaluateDataset(ExecutorchModel& lossModel, ExecutorchModel& inferModel,
                               const std::vector<float>& flat, const DataBatch& data, int64_t chunk);

}  // namespace fedlearn
