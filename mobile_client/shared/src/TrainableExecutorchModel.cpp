#include "fedlearn/TrainableExecutorchModel.h"
#include "fedlearn/Sha256.h"

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/training/module/training_module.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/runtime.h>

#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fedlearn {
namespace {

using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::aten::Tensor;
using executorch::extension::FileDataLoader;
using executorch::extension::make_tensor_ptr;
using executorch::extension::training::TrainingModule;
using executorch::runtime::Error;
using executorch::runtime::EValue;

void ensureRuntimeInit() {
  static std::once_flag once;
  std::call_once(once, [] { executorch::runtime::runtime_init(); });
}

[[noreturn]] void fail(const std::string& what, Error e) {
  throw std::runtime_error("TrainableExecutorchModel: " + what + " (error " +
                           std::to_string(static_cast<int>(e)) + ")");
}

inline SizesType toSize(int64_t d) {
  if (d < 0 ||
      static_cast<uint64_t>(d) > static_cast<uint64_t>(std::numeric_limits<SizesType>::max())) {
    throw std::runtime_error("TrainableExecutorchModel: dimension " + std::to_string(d) +
                             " exceeds the model index type range");
  }
  return static_cast<SizesType>(d);
}

}  // namespace

struct TrainableExecutorchModel::Impl {
  std::unique_ptr<TrainingModule> mod;  // owns the moved-in FileDataLoader
  std::vector<std::string> names;       // canonical (framework named_parameters) flat order
  std::vector<int64_t> numels;          // per-name element count, cached from load
  int64_t flat_dim = 0;

  // Look up a canonical param name in an ET named_* map, failing loudly if absent. Returns a COPY of
  // the Tensor handle (a non-owning view of the same storage), so mutable_data_ptr can write params
  // even though the map yields a const Tensor.
  static Tensor lookup(const std::map<std::string_view, Tensor>& m, const std::string& name,
                       const char* which) {
    auto it = m.find(name);
    if (it == m.end()) {
      throw std::runtime_error(std::string("TrainableExecutorchModel: ") + which +
                               " has no parameter '" + name + "'");
    }
    return it->second;
  }
};

TrainableExecutorchModel::TrainableExecutorchModel(const std::string& ptePath,
                                                   const std::string& expectedSha256,
                                                   std::vector<std::string> paramNamesInFlatOrder)
    : impl_(std::make_unique<Impl>()) {
  // Untrusted-input rule: verify the hash BEFORE handing bytes to ExecuTorch.
  const std::string actual = Sha256::hexDigestFile(ptePath);
  if (actual != expectedSha256) {
    throw std::runtime_error("TrainableExecutorchModel: sha256 mismatch for '" + ptePath +
                             "' (expected " + expectedSha256 + ", got " + actual + ")");
  }
  if (paramNamesInFlatOrder.empty()) {
    throw std::runtime_error("TrainableExecutorchModel: paramNamesInFlatOrder is empty");
  }
  ensureRuntimeInit();

  auto loaderRes = FileDataLoader::from(ptePath.c_str());
  if (!loaderRes.ok()) fail("FileDataLoader::from failed", loaderRes.error());
  auto loader = std::make_unique<FileDataLoader>(std::move(*loaderRes));
  impl_->mod = std::make_unique<TrainingModule>(std::move(loader));
  impl_->names = std::move(paramNamesInFlatOrder);

  // Cache each canonical param's numel + the total flat dim, and validate every requested name is a
  // trainable parameter of the joint "forward" graph (a missing/misspelled name is a load error, not
  // a silent wrong-shape flat vector).
  auto paramsRes = impl_->mod->named_parameters("forward");
  if (!paramsRes.ok()) fail("named_parameters(forward) failed", paramsRes.error());
  const auto& params = paramsRes.get();
  impl_->numels.reserve(impl_->names.size());
  for (const auto& name : impl_->names) {
    const Tensor t = Impl::lookup(params, name, "named_parameters");
    const int64_t k = t.numel();
    impl_->numels.push_back(k);
    impl_->flat_dim += k;
  }
}

TrainableExecutorchModel::~TrainableExecutorchModel() = default;

int64_t TrainableExecutorchModel::flatDim() const { return impl_->flat_dim; }

void TrainableExecutorchModel::setFlatParams(const std::vector<float>& flat) {
  if (static_cast<int64_t>(flat.size()) != impl_->flat_dim) {
    throw std::runtime_error("TrainableExecutorchModel::setFlatParams: size " +
                             std::to_string(flat.size()) + " != flatDim " +
                             std::to_string(impl_->flat_dim));
  }
  auto paramsRes = impl_->mod->named_parameters("forward");
  if (!paramsRes.ok()) fail("named_parameters(forward) failed", paramsRes.error());
  const auto& params = paramsRes.get();
  size_t off = 0;
  for (size_t i = 0; i < impl_->names.size(); ++i) {
    Tensor t = Impl::lookup(params, impl_->names[i], "named_parameters");
    const auto k = static_cast<size_t>(impl_->numels[i]);
    std::memcpy(t.mutable_data_ptr<float>(), flat.data() + off, k * sizeof(float));
    off += k;
  }
}

std::vector<float> TrainableExecutorchModel::getFlatParams() const {
  auto paramsRes = impl_->mod->named_parameters("forward");
  if (!paramsRes.ok()) fail("named_parameters(forward) failed", paramsRes.error());
  const auto& params = paramsRes.get();
  std::vector<float> out(static_cast<size_t>(impl_->flat_dim));
  size_t off = 0;
  for (size_t i = 0; i < impl_->names.size(); ++i) {
    const Tensor t = Impl::lookup(params, impl_->names[i], "named_parameters");
    const auto k = static_cast<size_t>(impl_->numels[i]);
    std::memcpy(out.data() + off, t.const_data_ptr<float>(), k * sizeof(float));
    off += k;
  }
  return out;
}

float TrainableExecutorchModel::trainStep(const float* x, const std::vector<int64_t>& xShape,
                                          const int64_t* y, int64_t n, float lr) {
  std::vector<SizesType> xSizes;
  xSizes.reserve(xShape.size());
  for (int64_t d : xShape) xSizes.push_back(toSize(d));
  std::vector<SizesType> ySizes{toSize(n)};

  // Alias the caller-owned buffers (no copy); they stay valid across the forward+backward call.
  auto tX = make_tensor_ptr(xSizes, const_cast<float*>(x), ScalarType::Float);
  auto tY = make_tensor_ptr(ySizes, const_cast<int64_t*>(y), ScalarType::Long);

  auto res = impl_->mod->execute_forward_backward("forward", {*tX, *tY});
  if (!res.ok()) fail("execute_forward_backward failed", res.error());
  const auto& outs = res.get();
  if (outs.empty() || !outs[0].isTensor()) {
    throw std::runtime_error("TrainableExecutorchModel: forward_backward produced no loss tensor");
  }
  const Tensor lossT = outs[0].toTensor();
  if (lossT.scalar_type() != ScalarType::Float || lossT.numel() < 1) {
    throw std::runtime_error("TrainableExecutorchModel: loss output is not a non-empty Float tensor");
  }
  const float loss = lossT.const_data_ptr<float>()[0];

  // In-place SGD: p <- p - lr * grad(p) for every trainable param — exactly torch.optim.SGD(lr) with
  // no momentum/weight-decay (the FedAvg client). Gradients are fresh from THIS forward_backward.
  auto gradsRes = impl_->mod->named_gradients("forward");
  if (!gradsRes.ok()) fail("named_gradients(forward) failed", gradsRes.error());
  auto paramsRes = impl_->mod->named_parameters("forward");
  if (!paramsRes.ok()) fail("named_parameters(forward) failed", paramsRes.error());
  const auto& grads = gradsRes.get();
  const auto& params = paramsRes.get();
  for (size_t i = 0; i < impl_->names.size(); ++i) {
    Tensor p = Impl::lookup(params, impl_->names[i], "named_parameters");
    const Tensor g = Impl::lookup(grads, impl_->names[i], "named_gradients");
    const auto k = static_cast<int64_t>(impl_->numels[i]);
    if (g.numel() != k) {
      throw std::runtime_error("TrainableExecutorchModel: gradient numel != parameter numel for '" +
                               impl_->names[i] + "'");
    }
    float* pd = p.mutable_data_ptr<float>();
    const float* gd = g.const_data_ptr<float>();
    for (int64_t j = 0; j < k; ++j) pd[j] -= lr * gd[j];
  }
  return loss;
}

}  // namespace fedlearn
