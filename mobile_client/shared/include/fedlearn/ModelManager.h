#pragma once
//
// ModelManager.h — ExecuTorch model load (verify-before-load) + owned flat trainable params +
// safetensors state-dict (de)serialization (libtorch-free; Phase 3c).
//
// The .pte is a weight-free functional graph (params are an input), so ModelManager OWNS the
// trainable parameter vector (params_) alongside the ExecutorchModel and the name->shape layout.
// getFlatParams/setFlatParams operate on that owned vector; loss() forwards through the model.
//
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fedlearn/ExecutorchModel.h"
#include "fedlearn/Types.h"

namespace fedlearn {

// One trainable parameter's name + shape, in the flat order pte_export.py wrote (== the order
// the model's `flat` input expects). Comes from the model's sidecar manifest (param_layout).
struct ParamSpec {
  std::string name;
  std::vector<int64_t> shape;  // numel = product(shape)
};

class ModelManager {
 public:
  ModelManager() = default;

  // sha256-verifies the .pte (via ExecutorchModel), loads the "forward" method, and sets up the
  // owned param vector (zero-initialised to the trainable flat dim) + the name->shape layout.
  // `totalParamCount` is the model's full parameter count (incl. frozen) for ModelInfo, from the
  // sidecar manifest. Throws std::runtime_error on hash mismatch, load failure, or a layout whose
  // element count disagrees with the model's flat input dim.
  void loadModel(const std::string& ptePath, const std::string& expectedSha256,
                 const std::vector<ParamSpec>& layout, int64_t totalParamCount,
                 ModelInfo* info = nullptr);

  // The owned trainable flat vector (length == trainable param count).
  const std::vector<float>& getFlatParams() const;

  // Overwrite the owned flat vector (size-checked against the trainable param count).
  void setFlatParams(const std::vector<float>& flat);

  int64_t trainableParamCount() const;

  // safetensors state-dict over the owned params, split into named tensors per the layout, with
  // "num_examples" in the metadata. Inverse loads a blob back into the owned params.
  // (FedAvg download / checkpoint path; byte-compatible with the Python serializer.)
  std::string serializeStateDict(int64_t numExamples) const;
  void loadStateDict(const std::string& blob);

  // Forward through the loaded model: loss(flat, x, y).
  float loss(const std::vector<float>& flat, const float* x,
             const std::vector<int64_t>& xShape, const int64_t* y, int64_t n) const;

 private:
  static std::string tierForParamCount(int64_t totalParams);

  std::unique_ptr<ExecutorchModel> model_;
  std::vector<float> params_;       // owned trainable flat vector
  std::vector<ParamSpec> layout_;   // name->shape, flat order
};

}  // namespace fedlearn
