#pragma once
//
// DataLoader.h — on-device data loading (client-private; validated input). 15-LLD §4 / §13 task 10.
// Raw features/labels live ONLY on the device and never enter any server table (FL invariant,
// 03-DATA-MODEL §1). This loader validates shapes before training touches the data.
//
// Torch-free (Phase 3c): DataBatch is a non-owning view, so loading returns an OwnedBatch that
// holds the backing std::vectors; call .view() for a DataBatch valid while the owner lives.
//
#include <cstdint>
#include <string>
#include <vector>

#include "fedlearn/Types.h"

namespace fedlearn {

// Owns a batch's backing storage. The DataBatch from view() points into these vectors and is
// valid only while this OwnedBatch is alive and unmoved.
struct OwnedBatch {
  std::vector<float> inputs;
  std::vector<int64_t> inputShape;
  std::vector<int64_t> targets;

  DataBatch view() const {
    return DataBatch{inputs.data(), inputShape, targets.data(),
                     static_cast<int64_t>(targets.size())};
  }
};

class DataLoader {
 public:
  // Load one batch from raw little-endian files written by the device (inputs: float32 in the
  // given shape; targets: int64 class indices). Validates and throws std::runtime_error on any
  // malformed input. inputShape[0] is the batch dimension and must equal the target count.
  static OwnedBatch fromRawFiles(const std::string& inputsF32Path,
                                 const std::vector<int64_t>& inputShape,
                                 const std::string& targetsI64Path);

  // Validate a batch view: non-null buffers, positive sample count, non-empty shape whose first
  // dim equals numSamples. Throws std::invalid_argument otherwise. Call before handing to fit().
  static void validate(const DataBatch& batch);
};

}  // namespace fedlearn
