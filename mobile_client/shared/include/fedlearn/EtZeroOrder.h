#pragma once
//
// EtZeroOrder.h — zeroth-order gradient scalar on the ExecuTorch forward path.
//
// The libtorch-free DeComFL forward-difference g-scalar used by the DeComFL client: parameters
// live in plain std::vector<float>, the perturbation comes from fedlearn::flat_randn (byte-exact
// with the Python forward-difference reference), and the loss comes from ExecutorchModel.
//
// Matches the Python DeComFL g-scalar (GradEstimateMethod::Forward, Algorithm 4 line 18):
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

// The unperturbed loss L(flat). Within one DeComFL local step this is the SAME number for every
// perturbation — flat and the batch are both fixed there, only z varies — so evaluate it once
// per step and pass it to the cached etGScalarForward overload below. That makes a local step
// cost P+1 forward passes instead of 2P, matching the reference implementation, which hoists the
// unperturbed loss above the perturbation loop for the forward-difference method.
// Only valid for THIS (flat, x, y) triple: flat advances between local steps.
//
// `Model` is any type exposing ExecutorchModel's loss() signature; templated so the ZO math is
// decoupled from ExecuTorch and the forward-pass count is observable in tests.
template <class Model>
inline double etBaseLoss(Model& model, const std::vector<float>& flat,
                         const float* x, const std::vector<int64_t>& xShape,
                         const int64_t* y, int64_t n) {
  return static_cast<double>(model.loss(flat, x, xShape, y, n));
}

// Forward-difference g-scalar with a PRE-COMPUTED unperturbed loss (see etBaseLoss).
// `z` must be flat_randn(seed, flat.size()) (the canonical z). Bit-identical to the overload
// that recomputes lossRef — caching is a pure cost reduction, not an approximation.
template <class Model>
inline double etGScalarForward(Model& model,
                               const std::vector<float>& flat,
                               const std::vector<float>& z, double mu,
                               const float* x, const std::vector<int64_t>& xShape,
                               const int64_t* y, int64_t n, double lossRef) {
  if (z.size() != flat.size()) {
    throw std::invalid_argument("etGScalarForward: perturbation size != parameter size");
  }
  std::vector<float> perturbed(flat.size());
  const float muf = static_cast<float>(mu);
  for (size_t i = 0; i < flat.size(); ++i) perturbed[i] = flat[i] + muf * z[i];

  const double lossPerturbed = static_cast<double>(model.loss(perturbed, x, xShape, y, n));
  return (lossPerturbed - lossRef) / mu;
}

// Forward-difference g-scalar. `z` must be flat_randn(seed, flat.size()) (the canonical z).
// Evaluates the unperturbed loss itself — prefer the cached overload inside a perturbation loop.
template <class Model>
inline double etGScalarForward(Model& model,
                               const std::vector<float>& flat,
                               const std::vector<float>& z, double mu,
                               const float* x, const std::vector<int64_t>& xShape,
                               const int64_t* y, int64_t n) {
  if (z.size() != flat.size()) {
    throw std::invalid_argument("etGScalarForward: perturbation size != parameter size");
  }
  return etGScalarForward(model, flat, z, mu, x, xShape, y, n,
                          etBaseLoss(model, flat, x, xShape, y, n));
}

// Central-difference g-scalar: g = (L(flat + mu*z) - L(flat - mu*z)) / (2*mu). Lower O(mu^2)
// bias than the forward difference; same float32 perturbation discipline. `z` must be
// flat_randn(seed, flat.size()). Satisfies the identity central(z) = (forward(z) - forward(-z))/2.
template <class Model>
inline double etGScalarCentral(Model& model,
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
