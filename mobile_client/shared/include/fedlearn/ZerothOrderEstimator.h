#pragma once
//
// ZerothOrderEstimator.h — DeComFL zeroth-order gradient-scalar estimator (15-LLD §5.2, §6.2).
//
#include <torch/script.h>
#include <torch/torch.h>

#include "fedlearn/ModelManager.h"
#include "fedlearn/Types.h"

namespace fedlearn {

class ZerothOrderEstimator {
 public:
  // mu is DOUBLE (Python smoothing_param is float64) — fixes A6 M-C2 #2.
  //
  // NOTE (reconciles §5.2): the LLD ctor lists (mu, method); the §6.2 body uses a
  // ModelManager to set flat params, so the estimator holds one. Same `mm` the
  // DeComFLClient is constructed with.
  ZerothOrderEstimator(ModelManager& mm, double mu, GradEstimateMethod method);

  // Returns the scalar g for one perturbation seed (the value uploaded; z never leaves):
  //   forward: g = (f(x + mu*z) - f(x))      / mu
  //   central: g = (f(x + mu*z) - f(x - mu*z)) / (2*mu)
  // CRITICAL (M-C2 #1): each loss is extracted to DOUBLE before subtracting (mu=1e-3 is a
  // catastrophic-cancellation regime); do NOT subtract two float32 scalars.
  // Leaves the model's flat params restored to `flatParams` on return.
  double computeGradientScalar(torch::jit::Module& model,
                               const torch::Tensor& flatParams,
                               int64_t perturbationSeed,
                               const torch::Tensor& batchInputs,
                               const torch::Tensor& batchTargets);

 private:
  // Cross-entropy loss of model(inputs) vs targets (CNN/MLP path). Kept as a tensor;
  // the caller extracts to double.
  torch::Tensor lossTensor(torch::jit::Module& model,
                           const torch::Tensor& inputs,
                           const torch::Tensor& targets) const;

  ModelManager& mm_;
  double mu_;
  GradEstimateMethod method_;
};

}  // namespace fedlearn
