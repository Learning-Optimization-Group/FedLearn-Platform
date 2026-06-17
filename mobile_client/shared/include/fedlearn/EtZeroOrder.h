#pragma once
//
// EtZeroOrder.h — zeroth-order gradient scalar on the ExecuTorch forward path.
//
// The libtorch-free replacement for the ZerothOrderEstimator forward used by the DeComFL client:
// parameters live in plain std::vector<float>, the perturbation comes from fedlearn::flat_randn
// (byte-exact with the Python canonical_perturbation), and the loss comes from ExecutorchModel.
//
// Matches Python ZerothOrderEstimator.compute_gradient_scalar (GradEstimateMethod::Forward,
// Algorithm 4 line 18):
//     g = (L(flat + mu*z) - L(flat)) / mu
// The perturbed params are formed in float32 (flat[i] + (float)mu * z[i]) to mirror the Python
// float32 update exactly; the small loss difference is amplified by 1/mu, so the forward must
// track the reference closely (it does — ExecuTorch reproduces the .pte loss to ~1e-10).
//
#include "fedlearn/ExecutorchModel.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace fedlearn {

// Forward-difference g-scalar. `z` must be flat_randn(seed, flat.size()) (the canonical z).
inline double etGScalarForward(ExecutorchModel& model,
                               const std::vector<float>& flat,
                               const std::vector<float>& z, double mu,
                               const float* x, const std::vector<int64_t>& xShape,
                               const int64_t* y, int64_t n) {
  if (z.size() != flat.size()) {
    throw std::invalid_argument("etGScalarForward: perturbation size != parameter size");
  }
  const double lossRef = static_cast<double>(model.loss(flat, x, xShape, y, n));

  std::vector<float> perturbed(flat.size());
  const float muf = static_cast<float>(mu);
  for (size_t i = 0; i < flat.size(); ++i) perturbed[i] = flat[i] + muf * z[i];

  const double lossPerturbed = static_cast<double>(model.loss(perturbed, x, xShape, y, n));
  return (lossPerturbed - lossRef) / mu;
}

// Central-difference g-scalar: g = (L(flat + mu*z) - L(flat - mu*z)) / (2*mu). Lower O(mu^2)
// bias than the forward difference; same float32 perturbation discipline. `z` must be
// flat_randn(seed, flat.size()). Satisfies the identity central(z) = (forward(z) - forward(-z))/2.
inline double etGScalarCentral(ExecutorchModel& model,
                               const std::vector<float>& flat,
                               const std::vector<float>& z, double mu,
                               const float* x, const std::vector<int64_t>& xShape,
                               const int64_t* y, int64_t n) {
  if (z.size() != flat.size()) {
    throw std::invalid_argument("etGScalarCentral: perturbation size != parameter size");
  }
  const float muf = static_cast<float>(mu);
  std::vector<float> plus(flat.size()), minus(flat.size());
  for (size_t i = 0; i < flat.size(); ++i) {
    plus[i] = flat[i] + muf * z[i];
    minus[i] = flat[i] - muf * z[i];
  }
  const double lossPlus = static_cast<double>(model.loss(plus, x, xShape, y, n));
  const double lossMinus = static_cast<double>(model.loss(minus, x, xShape, y, n));
  return (lossPlus - lossMinus) / (2.0 * mu);
}

}  // namespace fedlearn
