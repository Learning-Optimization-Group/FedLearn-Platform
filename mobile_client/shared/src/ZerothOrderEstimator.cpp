#include "ZerothOrderEstimator.h"

namespace fedlearn {

ZerothOrderEstimator::ZerothOrderEstimator(float mu) : mu_(mu) {}

torch::Tensor ZerothOrderEstimator::generatePerturbation(int64_t seed,
                                                          int64_t num_params) {
  // Matches Python: torch.Generator(device='cpu').manual_seed(seed) -> torch.randn(num_params, generator=gen)
  // C++ torch::Generator uses the same Mersenne Twister as Python, producing identical outputs.
  auto gen = torch::Generator();
  gen.set_current_seed(seed);
  return torch::randn({num_params}, gen);
}

double ZerothOrderEstimator::computeGradientScalar(
    FedLearnModule& model,
    const torch::Tensor& flat_params,
    const torch::Tensor& perturbation,
    const torch::Tensor& inputs,
    const torch::Tensor& targets) {
  torch::NoGradGuard no_grad;

  // f(x; xi)
  setFlatParams(model, flat_params);
  std::vector<torch::jit::IValue> model_inputs;
  model_inputs.push_back(inputs);
  auto output_x = model.forward(model_inputs).toTensor();
  auto loss_x = torch::nn::functional::detail::cross_entropy(
      output_x,
      targets,
      torch::Tensor(),
      -100,
      torch::nn::functional::CrossEntropyFuncOptions::reduction_t(
          torch::enumtype::kMean{}),
      0.0);

  // f(x + mu*z; xi)
  auto perturbed = flat_params + mu_ * perturbation;
  setFlatParams(model, perturbed);
  model_inputs.clear();
  model_inputs.push_back(inputs);
  auto output_perturbed = model.forward(model_inputs).toTensor();
  auto loss_perturbed = torch::nn::functional::detail::cross_entropy(
      output_perturbed,
      targets,
      torch::Tensor(),
      -100,
      torch::nn::functional::CrossEntropyFuncOptions::reduction_t(
          torch::enumtype::kMean{}),
      0.0);

  // g = (f(x + mu*z) - f(x)) / mu
  double g = (loss_perturbed.item<float>() - loss_x.item<float>()) / mu_;
  return g;
}

}  // namespace fedlearn
