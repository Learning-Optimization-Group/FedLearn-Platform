#pragma once
//
// DataLoader.h — on-device data loading (client-private; validated input). 15-LLD §4 / §13 task 10.
// Raw features/labels live ONLY on the device and never enter any server table (FL invariant,
// 03-DATA-MODEL §1). This loader validates shapes/dtypes before training touches the data.
//
#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "fedlearn/Types.h"

namespace fedlearn {

class DataLoader {
 public:
  // Load one batch from raw little-endian files written by the device (inputs: float32 in the
  // given shape; targets: int64 class indices). Validates and throws std::runtime_error on any
  // malformed input. inputShape[0] is the batch dimension and must equal the target count.
  static DataBatch fromRawFiles(const std::string& inputsF32Path,
                                const std::vector<int64_t>& inputShape,
                                const std::string& targetsI64Path);

  // Validate an in-memory batch: float inputs, int64 1-D targets, matching batch dimension.
  // Throws std::invalid_argument otherwise. Call before handing a batch to fit().
  static void validate(const DataBatch& batch);
};

}  // namespace fedlearn
