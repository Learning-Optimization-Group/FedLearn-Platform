#pragma once
//
// TrainableExecutorchModel.h — RAII wrapper around an ExecuTorch TRAINING-extension .pte
// (first-order, real backprop). Where ExecutorchModel runs a weights-as-inputs FORWARD graph for
// the zeroth-order path, this loads a joint forward+backward graph (exported by
// pte_export.export_trainable_pte via _export_forward_backward) and drives ET's TrainingModule:
// execute_forward_backward -> named_gradients -> an in-place SGD step. This is the compute core of
// first-order FedAvg/FedProx on device (Phase B), which lifts the mobile client past zeroth-order.
//
// The model's trainable parameters live INSIDE the module (not passed as a flat input). Global
// weights are written in via setFlatParams and the trained result read back via getFlatParams — the
// flat vector uses the framework's CANONICAL order (named_parameters(), trainable-only). ET's
// TrainingModule keys named_parameters ALPHABETICALLY, so this wrapper is constructed with the
// canonical ordered names and projects ET's map onto them; getting that projection wrong silently
// transposes weight/bias blocks (the M1 ordering gotcha).
//
// sha256-verifies the .pte BEFORE load (untrusted-input rule, mirrors ExecutorchModel/ModelManager).
// PIMPL: no ExecuTorch headers leak to consumers.
//
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fedlearn {

class TrainableExecutorchModel {
 public:
  // Verifies sha256(ptePath) == expectedSha256, then loads the joint "forward" graph as an ET
  // TrainingModule. paramNamesInFlatOrder are the fully-qualified ET parameter names (e.g.
  // "base.fc1.weight") in the framework's canonical flat order — the order setFlatParams /
  // getFlatParams read and write. Throws std::runtime_error on hash mismatch, load failure, or if
  // any named parameter is missing from the loaded module.
  TrainableExecutorchModel(const std::string& ptePath, const std::string& expectedSha256,
                           std::vector<std::string> paramNamesInFlatOrder);
  ~TrainableExecutorchModel();

  // Owns ET runtime objects with internal self-pointers; hold by reference or unique_ptr.
  TrainableExecutorchModel(TrainableExecutorchModel&&) = delete;
  TrainableExecutorchModel& operator=(TrainableExecutorchModel&&) = delete;
  TrainableExecutorchModel(const TrainableExecutorchModel&) = delete;
  TrainableExecutorchModel& operator=(const TrainableExecutorchModel&) = delete;

  // Write the flat trainable vector into the module's parameters, in canonical order. Throws if
  // flat.size() != flatDim().
  void setFlatParams(const std::vector<float>& flat);

  // Read the module's trainable parameters as a flat vector, in canonical order (length flatDim()).
  std::vector<float> getFlatParams() const;

  // One full-batch SGD step on (x, y): execute_forward_backward, then for every trainable param
  //   p <- p - lr * grad(p)
  // exactly matching torch.optim.SGD(params, lr) with no momentum/weight-decay (the FedAvg client,
  // local_trainer.py:84). Returns the loss at the params BEFORE the update. NOT const, NOT
  // concurrency-safe on one instance. Throws std::runtime_error on any execution failure.
  float trainStep(const float* x, const std::vector<int64_t>& xShape,
                  const int64_t* y, int64_t n, float lr);

  // Total trainable parameter count (sum of the canonical params' numels).
  int64_t flatDim() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fedlearn
