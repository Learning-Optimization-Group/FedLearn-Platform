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

/**
 * torch.optim.Adam's settings (Stage 4 S2). Doubles, as torch holds them: each is cast to float only where the tensor
 * arithmetic uses it, which keeps the device's steps on torch's.
 */
/** An extra float input a training program takes after (x, y), such as one dropout layer's mask for this step. */
struct InputTensor {
  const float* data = nullptr;
  std::vector<int64_t> shape;
};

struct AdamSettings {
  double learningRate = 0.0;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon = 1e-8;
};

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
  //
  // FedProx: with a proximal anchor (the round's global weights, canonical flat order, flatDim()
  // long) and mu > 0, each gradient first gets the proximal gradient added, exactly as the laptop
  // client's _apply_proximal_gradient does between backward and the SGD step:
  //   p <- p - lr * (grad(p) + mu * (p - anchor))
  //
  // `extra` are the program's inputs after (x, y), in order (a masked-dropout program's per-layer masks).
  float trainStep(const float* x, const std::vector<int64_t>& xShape,
                  const int64_t* y, int64_t n, float lr,
                  const std::vector<float>* proximalAnchor = nullptr, float proximalMu = 0.0f,
                  const std::vector<InputTensor>* extra = nullptr);

  // One full-batch Adam step on (x, y), exactly torch.optim.Adam's single-tensor step without weight decay or amsgrad:
  //   m <- lerp(m, g, 1 - beta1);  v <- beta2 * v + (1 - beta2) * g^2
  //   p <- p - (lr / (1 - beta1^t)) * m / (sqrt(v) / sqrt(1 - beta2^t) + eps)
  // The moments and the step count t live in the model and persist across calls until resetOptimizerState(), which
  // a round calls first (the laptop creates a fresh Adam every round). Returns the loss before the update.
  float trainStepAdam(const float* x, const std::vector<int64_t>& xShape, const int64_t* y, int64_t n,
                      const AdamSettings& adam, const std::vector<InputTensor>* extra = nullptr);

  // The shapes of the program's inputs after (x, y), as its method metadata declares them (a dynamic batch dimension
  // at its upper bound): for a masked-dropout program, one per dropout layer, [batch, activation dims...].
  std::vector<std::vector<int64_t>> extraInputShapes() const;

  // Forget Adam's moments and step count: the next trainStepAdam starts a fresh optimizer.
  void resetOptimizerState();

  // Total trainable parameter count (sum of the canonical params' numels).
  int64_t flatDim() const;

 private:
  // execute_forward_backward on (x, y); returns the loss at the current parameters, leaving the gradients in place.
  float forwardBackward(const float* x, const std::vector<int64_t>& xShape, const int64_t* y, int64_t n,
                        const std::vector<InputTensor>* extra);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fedlearn
