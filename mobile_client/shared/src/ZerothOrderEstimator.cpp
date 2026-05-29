#include "fedlearn/ZerothOrderEstimator.h"

#include <vector>

#include "fedlearn/Perturbation.h"

namespace fedlearn {

ZerothOrderEstimator::ZerothOrderEstimator(ModelManager& mm, double mu, GradEstimateMethod method)
    : mm_(mm), mu_(mu), method_(method) {}

torch::Tensor ZerothOrderEstimator::lossTensor(torch::jit::Module& model,
                                               const torch::Tensor& inputs,
                                               const torch::Tensor& targets) const {
  torch::NoGradGuard no_grad;
  std::vector<torch::jit::IValue> in{inputs};
  torch::Tensor logits = model.forward(in).toTensor();
  // cross_entropy(logits[N,C], targets[N]) — matches Python nn.CrossEntropyLoss default (mean).
  return torch::nn::functional::cross_entropy(logits, targets);
}

double ZerothOrderEstimator::computeGradientScalar(torch::jit::Module& model,
                                                   const torch::Tensor& flatParams,
                                                   int64_t perturbationSeed,
                                                   const torch::Tensor& batchInputs,
                                                   const torch::Tensor& batchTargets) {
  // z on CPU (canonical) so server and client agree; move to the params' device for the add.
  torch::Tensor z =
      canonical_perturbation(perturbationSeed, flatParams.numel(), torch::kFloat32)
          .to(flatParams.device());

  // f(x + mu*z) — extract to DOUBLE late (M-C2 #1).
  mm_.setFlatParams(model, flatParams + mu_ * z);
  const double loss_plus = lossTensor(model, batchInputs, batchTargets).item<double>();

  double g;
  if (method_ == GradEstimateMethod::Central) {
    mm_.setFlatParams(model, flatParams - mu_ * z);
    const double loss_minus = lossTensor(model, batchInputs, batchTargets).item<double>();
    g = (loss_plus - loss_minus) / (2.0 * mu_);  // O(mu^2) bias
  } else {
    mm_.setFlatParams(model, flatParams);  // restore for f(x)
    const double loss_ref = lossTensor(model, batchInputs, batchTargets).item<double>();
    g = (loss_plus - loss_ref) / mu_;  // O(mu) bias
  }

  mm_.setFlatParams(model, flatParams);  // leave the model exactly as we found it
  return g;
}

}  // namespace fedlearn
