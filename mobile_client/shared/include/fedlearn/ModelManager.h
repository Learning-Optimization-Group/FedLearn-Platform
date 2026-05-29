#pragma once
//
// ModelManager.h — TorchScript load (verify-before-load), requires_grad-filtered flat
// params, and symmetric state-dict (de)serialization (15-LLD-mobile.md §5.2, §13 task 6).
//
#include <string>

#include <torch/script.h>

#include "fedlearn/Types.h"

namespace fedlearn {

class ModelManager {
 public:
  // Verifies sha256(file) == expectedSha256 BEFORE torch::jit::load (untrusted-input rule,
  // M-C4 / E8), then loads and eval()s the module. Returns the loaded module; if `info` is
  // non-null it is filled with param counts + tier.
  //
  // NOTE (reconciles §5.2): the LLD signature returns only ModelInfo for brevity; the caller
  // needs the module, so we return the Module and surface ModelInfo via an out-param.
  // Throws std::runtime_error on hash mismatch or load failure.
  torch::jit::Module loadScriptModel(const std::string& path,
                                     const std::string& expectedSha256,
                                     ModelInfo* info = nullptr);

  // Flatten ONLY parameters with requires_grad == true, in module-iteration order (matches
  // Python ZerothOrderEstimator._get_flat_params; fixes the frozen-layer divergence A6 M-C2 #3).
  torch::Tensor getFlatParams(const torch::jit::Module& model) const;

  // Inverse of getFlatParams: write `flat` back into the requires_grad params, same order.
  void setFlatParams(torch::jit::Module& model, const torch::Tensor& flat) const;

  // Symmetric state-dict blob wrapping {"parameters": named_params, "num_examples": n}
  // (mirrors the framework serializer's wrapped torch.save — the Bug-3 fix; safetensors is
  // the documented target codec to migrate to, 04 §10.3). FedAvg-path only.
  std::string serializeStateDict(const torch::jit::Module& model, int64_t numExamples) const;
  void loadStateDict(torch::jit::Module& model, const std::string& blob) const;

  int64_t trainableParamCount(const torch::jit::Module& model) const;

 private:
  static std::string tierForParamCount(int64_t totalParams);
};

}  // namespace fedlearn
