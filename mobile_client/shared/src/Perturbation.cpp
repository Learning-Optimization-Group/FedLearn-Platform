#include "fedlearn/Perturbation.h"
#include "fedlearn/RandnEngine.h"

#include <stdexcept>
#include <vector>

namespace fedlearn {

at::Tensor canonical_perturbation(int64_t seed, int64_t num_params, at::ScalarType dtype) {
  if (num_params <= 0) {
    throw std::invalid_argument("canonical_perturbation: num_params must be > 0");
  }

  // Single source of RNG truth: the ATen-free engine gated by randn_parity_test.cpp. It
  // reproduces torch.randn(..., generator=Generator("cpu").manual_seed(seed), float32)
  // byte-for-byte (PyTorch's CPU MT19937 + size-dependent normal kernel), so the perturbation
  // contract no longer depends on libtorch's RNG — it survives the move to ExecuTorch.
  std::vector<float> z = flat_randn(seed, num_params);

  // Wrap as an owning CPU float32 tensor (clone so the tensor owns its storage, not z's),
  // then cast to the requested dtype. dtype is fixed to kFloat for the parity contract.
  at::Tensor t = at::from_blob(z.data(), {num_params}, at::kFloat).clone();
  return dtype == at::kFloat ? t : t.to(dtype);
}

}  // namespace fedlearn
