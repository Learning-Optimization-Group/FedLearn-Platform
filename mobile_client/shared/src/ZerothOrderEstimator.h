#pragma once

#include "Utils.h"

namespace fedlearn {

/**
 * Port of framework/src/fedlearn/estimators/zeroth_order.py
 *
 * Computes gradient scalars: g = (f(x + mu*z) - f(x)) / mu
 * Uses torch::Generator with manual_seed for RNG parity with Python server.
 */
class ZerothOrderEstimator {
 public:
  explicit ZerothOrderEstimator(float mu = 0.001f);

  /**
   * Generate perturbation z ~ N(0, I_d) from a seed.
   * Uses torch::Generator::manual_seed(seed) + torch::randn to produce
   * bit-identical vectors to Python's torch.randn(n, generator=gen).
   */
  torch::Tensor generatePerturbation(int64_t seed, int64_t num_params);

  /**
   * Compute ZO gradient scalar: g = (f(x + mu*z) - f(x)) / mu
   */
  double computeGradientScalar(
      FedLearnModule& model,
      const torch::Tensor& flat_params,
      const torch::Tensor& perturbation,
      const torch::Tensor& inputs,
      const torch::Tensor& targets);

  float mu() const { return mu_; }

 private:
  float mu_;
};

}  // namespace fedlearn
