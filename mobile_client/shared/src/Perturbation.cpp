#include "fedlearn/Perturbation.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/Generator.h>

#include <stdexcept>

namespace fedlearn {

at::Tensor canonical_perturbation(int64_t seed, int64_t num_params, at::ScalarType dtype) {
  if (num_params <= 0) {
    throw std::invalid_argument("canonical_perturbation: num_params must be > 0");
  }

  // A LOCAL CPU generator seeded identically to Python's
  //   torch.Generator(device="cpu").manual_seed(seed)
  // This never touches the process-global RNG (mirrors the Python contract, which uses a
  // local torch.Generator to avoid the global-state mutation flagged as Bug B-2).
  at::Generator generator = at::detail::createCPUGenerator(static_cast<uint64_t>(seed));

  auto options = at::TensorOptions().dtype(dtype).device(at::kCPU);

  // Same ATen normal-distribution kernel as Python torch.randn(..., generator=gen) for a
  // pinned libtorch version. rng_parity_test.cpp is the gate that proves this holds; if a
  // future libtorch changes the kernel, that test fails and the fixture must be re-frozen
  // from the Python source of truth (do not "fix" it by editing the golden vectors).
  return at::randn({num_params}, generator, options);
}

}  // namespace fedlearn
