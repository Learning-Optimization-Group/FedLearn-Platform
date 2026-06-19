#pragma once
//
// ExecutorchModel.h — RAII wrapper around ExecuTorch .pte forward graphs.
//
// Supports two functional graph signatures (weights-as-inputs; see pte_export.py):
//   Loss graph:  forward(flat_trainable, x, y) -> cross_entropy   [loss()]
//   Infer graph: forward(flat_trainable, x)    -> logits           [infer()]
//
// sha256-verifies the .pte BEFORE load (untrusted-input rule, mirrors ModelManager). This is
// the ExecuTorch replacement for the libtorch torch::jit forward path; the C++ FL core owns
// the parameter vector and passes perturbed params (θ ± μz) as the `flat` input.
//
// PIMPL: no ExecuTorch headers leak here, so consumers (ZerothOrderEstimator, tests) need no
// ET include paths.
//
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fedlearn {

class ExecutorchModel {
 public:
  // Verifies sha256(ptePath) == expectedSha256, then loads the program + "forward" method.
  // Throws std::runtime_error on hash mismatch or any ExecuTorch load failure.
  ExecutorchModel(const std::string& ptePath, const std::string& expectedSha256);
  ~ExecutorchModel();

  // Non-movable/non-copyable: it owns ExecuTorch runtime objects whose internal pointers must
  // not be invalidated. Hold it by reference or std::unique_ptr<ExecutorchModel>.
  ExecutorchModel(ExecutorchModel&&) = delete;
  ExecutorchModel& operator=(ExecutorchModel&&) = delete;
  ExecutorchModel(const ExecutorchModel&) = delete;
  ExecutorchModel& operator=(const ExecutorchModel&) = delete;

  // Run forward(flat, x, y) and return the scalar loss.
  //   flat   : trainable parameter vector (length flatDim()).
  //   x      : row-major input features with shape xShape.
  //   xShape : shape of x (e.g. {batch, features}).
  //   y      : int64 targets, length n.
  // NOT const: each call mutates the underlying ExecuTorch Method's execution state, so it is
  // NOT safe to call concurrently on the same instance. Throws std::runtime_error on any
  // execution failure, dimension overflow, or unexpected output shape/dtype.
  float loss(const std::vector<float>& flat,
             const float* x, const std::vector<int64_t>& xShape,
             const int64_t* y, int64_t n);

  // Run forward(flat, x) and return the full logits tensor (flattened, row-major [batch, classes]).
  //   flat   : trainable parameter vector (length flatDim()).
  //   x      : row-major input features with shape xShape.
  //   xShape : shape of x (e.g. {batch, features}).
  // Returns a vector of size batch*num_classes. NOT const (same ExecutorchModel method-state
  // mutation caveat as loss()). Throws std::runtime_error on any failure.
  std::vector<float> infer(const std::vector<float>& flat,
                           const float* x, const std::vector<int64_t>& xShape);

  // Expected length of the `flat` input (input 0's element count, from the .pte metadata).
  int64_t flatDim() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fedlearn
