#include "fedlearn/TrainableExecutorchModel.h"
#include "fedlearn/ModelExecutionError.h"
#include "fedlearn/Sha256.h"

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/training/module/training_module.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/runtime.h>

#include <cmath>
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
  throw ModelExecutionError("TrainableExecutorchModel: " + what + " (error " +
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
  // Adam's first and second moments per canonical parameter, and its step count (0 = a fresh optimizer).
  std::vector<std::vector<float>> adamM, adamV;
  int64_t adamStep = 0;

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

float TrainableExecutorchModel::forwardBackward(const float* x, const std::vector<int64_t>& xShape,
                                                const int64_t* y, int64_t n,
                                                const std::vector<InputTensor>* extra) {
  std::vector<SizesType> xSizes;
  xSizes.reserve(xShape.size());
  for (int64_t d : xShape) xSizes.push_back(toSize(d));
  std::vector<SizesType> ySizes{toSize(n)};

  // Alias the caller-owned buffers (no copy); they stay valid across the forward+backward call.
  auto tX = make_tensor_ptr(xSizes, const_cast<float*>(x), ScalarType::Float);
  auto tY = make_tensor_ptr(ySizes, const_cast<int64_t*>(y), ScalarType::Long);

  // The extra inputs (masks) are aliased like x and y; their TensorPtrs must outlive the call.
  std::vector<executorch::extension::TensorPtr> extraTensors;
  std::vector<EValue> inputs{*tX, *tY};
  if (extra != nullptr) {
    for (const auto& in : *extra) {
      std::vector<SizesType> sizes;
      for (int64_t d : in.shape) sizes.push_back(toSize(d));
      extraTensors.push_back(make_tensor_ptr(sizes, const_cast<float*>(in.data), ScalarType::Float));
      inputs.emplace_back(*extraTensors.back());
    }
  }
  auto res = impl_->mod->execute_forward_backward("forward", inputs);
  if (!res.ok()) fail("execute_forward_backward failed", res.error());
  const auto& outs = res.get();
  if (outs.empty() || !outs[0].isTensor()) {
    throw std::runtime_error("TrainableExecutorchModel: forward_backward produced no loss tensor");
  }
  const Tensor lossT = outs[0].toTensor();
  if (lossT.scalar_type() != ScalarType::Float || lossT.numel() < 1) {
    throw std::runtime_error("TrainableExecutorchModel: loss output is not a non-empty Float tensor");
  }
  return lossT.const_data_ptr<float>()[0];
}

float TrainableExecutorchModel::trainStep(const float* x, const std::vector<int64_t>& xShape,
                                          const int64_t* y, int64_t n, float lr,
                                          const std::vector<float>* proximalAnchor, float proximalMu,
                                          const std::vector<InputTensor>* extra) {
  const bool proximal = proximalAnchor != nullptr && proximalMu != 0.0f;
  if (proximal && static_cast<int64_t>(proximalAnchor->size()) != flatDim()) {
    throw std::runtime_error("TrainableExecutorchModel: proximal anchor size != flatDim()");
  }
  const float loss = forwardBackward(x, xShape, y, n, extra);

  // In-place SGD: p <- p - lr * grad(p) for every trainable param — exactly torch.optim.SGD(lr) with
  // no momentum/weight-decay (the FedAvg client). Gradients are fresh from THIS forward_backward.
  // Under FedProx the gradient first gets mu * (p - anchor) added (the laptop's grad.add_(p - w0, alpha=mu)).
  auto gradsRes = impl_->mod->named_gradients("forward");
  if (!gradsRes.ok()) fail("named_gradients(forward) failed", gradsRes.error());
  auto paramsRes = impl_->mod->named_parameters("forward");
  if (!paramsRes.ok()) fail("named_parameters(forward) failed", paramsRes.error());
  const auto& grads = gradsRes.get();
  const auto& params = paramsRes.get();
  int64_t offset = 0;
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
    if (proximal) {
      const float* anchor = proximalAnchor->data() + offset;
      for (int64_t j = 0; j < k; ++j) {
        const float grad = gd[j] + proximalMu * (pd[j] - anchor[j]);
        pd[j] -= lr * grad;
      }
    } else {
      for (int64_t j = 0; j < k; ++j) pd[j] -= lr * gd[j];
    }
    offset += k;
  }
  return loss;
}

void TrainableExecutorchModel::resetOptimizerState() {
  impl_->adamM.clear();
  impl_->adamV.clear();
  impl_->adamStep = 0;
}

float TrainableExecutorchModel::trainStepAdam(const float* x, const std::vector<int64_t>& xShape, const int64_t* y,
                                              int64_t n, const AdamSettings& adam,
                                              const std::vector<InputTensor>* extra) {
  if (!(adam.learningRate > 0) || !(adam.beta1 >= 0 && adam.beta1 < 1) || !(adam.beta2 >= 0 && adam.beta2 < 1) ||
      !(adam.epsilon > 0)) {
    throw std::runtime_error("TrainableExecutorchModel: invalid Adam settings");
  }
  const float loss = forwardBackward(x, xShape, y, n, extra);
  if (impl_->adamStep == 0) {
    impl_->adamM.assign(impl_->names.size(), {});
    impl_->adamV.assign(impl_->names.size(), {});
    for (size_t i = 0; i < impl_->names.size(); ++i) {
      impl_->adamM[i].assign(static_cast<size_t>(impl_->numels[i]), 0.0f);
      impl_->adamV[i].assign(static_cast<size_t>(impl_->numels[i]), 0.0f);
    }
  }
  const int64_t t = ++impl_->adamStep;

  // torch's _single_tensor_adam, in its order and precisions: the bias corrections and the step size are doubles,
  // cast to float where they meet the float tensors.
  const float lerpWeight = static_cast<float>(1.0 - adam.beta1);
  const float beta2 = static_cast<float>(adam.beta2);
  const float oneMinusBeta2 = static_cast<float>(1.0 - adam.beta2);
  const double biasCorrection1 = 1.0 - std::pow(adam.beta1, static_cast<double>(t));
  const double biasCorrection2 = 1.0 - std::pow(adam.beta2, static_cast<double>(t));
  const float stepSize = static_cast<float>(adam.learningRate / biasCorrection1);
  const float biasCorrection2Sqrt = static_cast<float>(std::sqrt(biasCorrection2));
  const float eps = static_cast<float>(adam.epsilon);

  auto gradsRes = impl_->mod->named_gradients("forward");
  if (!gradsRes.ok()) fail("named_gradients(forward) failed", gradsRes.error());
  auto paramsRes = impl_->mod->named_parameters("forward");
  if (!paramsRes.ok()) fail("named_parameters(forward) failed", paramsRes.error());
  for (size_t i = 0; i < impl_->names.size(); ++i) {
    Tensor p = Impl::lookup(paramsRes.get(), impl_->names[i], "named_parameters");
    const Tensor g = Impl::lookup(gradsRes.get(), impl_->names[i], "named_gradients");
    const auto k = static_cast<size_t>(impl_->numels[i]);
    if (static_cast<size_t>(g.numel()) != k) {
      throw std::runtime_error("TrainableExecutorchModel: gradient numel != parameter numel for '" +
                               impl_->names[i] + "'");
    }
    float* pd = p.mutable_data_ptr<float>();
    const float* gd = g.const_data_ptr<float>();
    float* m = impl_->adamM[i].data();
    float* v = impl_->adamV[i].data();
    for (size_t j = 0; j < k; ++j) {
      m[j] = m[j] + lerpWeight * (gd[j] - m[j]);        // exp_avg.lerp_(grad, 1 - beta1), weight < 0.5
      v[j] = v[j] * beta2 + oneMinusBeta2 * gd[j] * gd[j];  // exp_avg_sq.mul_(beta2).addcmul_(g, g, 1 - beta2)
      const float denom = std::sqrt(v[j]) / biasCorrection2Sqrt + eps;
      pd[j] = pd[j] + (-stepSize) * (m[j] / denom);      // param.addcdiv_(exp_avg, denom, value=-step_size)
    }
  }
  return loss;
}

std::vector<std::vector<int64_t>> TrainableExecutorchModel::extraInputShapes() const {
  auto meta = impl_->mod->method_meta("forward");
  if (!meta.ok()) fail("method_meta(forward) failed", meta.error());
  std::vector<std::vector<int64_t>> shapes;
  for (size_t i = 2; i < meta->num_inputs(); ++i) {
    auto tensor = meta->input_tensor_meta(i);
    if (!tensor.ok()) fail("input_tensor_meta failed", tensor.error());
    std::vector<int64_t> dims;
    for (auto d : tensor->sizes()) dims.push_back(static_cast<int64_t>(d));
    shapes.push_back(dims);
  }
  return shapes;
}

}  // namespace fedlearn
