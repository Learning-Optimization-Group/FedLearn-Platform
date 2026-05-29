#pragma once
//
// Perturbation.h — canonical, device-independent perturbation generation for DeComFL.
//
// This is the C++ (libtorch) side of the cross-language RNG (Random Number Generator)
// contract. It MUST reproduce the Python source of truth
//   framework/src/fedlearn/estimators/perturbation.py :: canonical_perturbation
// for the pinned torch version. The contract is frozen as golden vectors in
//   framework/tests/fixtures/decomfl_golden/
// and the gtest rng_parity_test.cpp is the RELEASE GATE (15-LLD-mobile.md §13 task 4):
// if parity fails, the mobile build must not ship.
//
#include <cstdint>
#include <torch/torch.h>

namespace fedlearn {

// Returns a device-independent N(0, I_d) sample of length num_params.
//
// Always generated on the CPU with a *local* generator (never the global RNG), so the
// output is stable across compute devices. Callers move it to their device at the use
// site, e.g.  auto z = canonical_perturbation(seed, d).to(device);
//
// seed       : non-negative shared seed for one (local_step, perturbation) index.
// num_params : dimension d (number of trainable parameters); must be > 0.
// dtype      : output floating dtype; FIXED to kFloat (float32) for parity. Do not pass
//              the model's dtype — that would break the golden-vector contract.
//
// Throws std::invalid_argument if num_params <= 0.
at::Tensor canonical_perturbation(int64_t seed, int64_t num_params,
                                  at::ScalarType dtype = at::kFloat);

}  // namespace fedlearn
