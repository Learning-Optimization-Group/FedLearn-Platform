#include "fedlearn/ModelManager.h"

#include <stdexcept>
#include <unordered_map>

#include "fedlearn/Safetensors.h"
#include "fedlearn/Sha256.h"

namespace fedlearn {

namespace {

int64_t numelOf(const ParamSpec& spec) {
  int64_t n = 1;
  for (int64_t d : spec.shape) n *= d;
  return n;
}

}  // namespace

void ModelManager::loadModel(const std::string& ptePath, const std::string& expectedSha256,
                             const std::vector<ParamSpec>& layout, int64_t totalParamCount,
                             ModelInfo* info) {
  // ExecutorchModel sha256-verifies before load (untrusted-input rule) and throws on mismatch.
  model_ = std::make_unique<ExecutorchModel>(ptePath, expectedSha256);
  layout_ = layout;

  int64_t flatDim = 0;
  for (const auto& spec : layout_) flatDim += numelOf(spec);
  if (flatDim != model_->flatDim()) {
    throw std::runtime_error("ModelManager: param_layout element count (" +
                             std::to_string(flatDim) + ") != model flat input dim (" +
                             std::to_string(model_->flatDim()) + ")");
  }
  params_.assign(static_cast<size_t>(flatDim), 0.0f);

  if (info != nullptr) {
    info->paramCount = totalParamCount;
    info->trainableParamCount = flatDim;
    info->sha256 = expectedSha256;  // verified == actual by ExecutorchModel
    info->tier = tierForParamCount(totalParamCount);
  }
}

const std::vector<float>& ModelManager::getFlatParams() const { return params_; }

void ModelManager::setFlatParams(const std::vector<float>& flat) {
  if (flat.size() != params_.size()) {
    throw std::runtime_error("ModelManager::setFlatParams: size mismatch (got " +
                             std::to_string(flat.size()) + ", expected " +
                             std::to_string(params_.size()) + ")");
  }
  params_ = flat;
}

int64_t ModelManager::trainableParamCount() const { return static_cast<int64_t>(params_.size()); }

namespace {

std::vector<NamedTensor> namedTensors(const std::vector<ParamSpec>& layout, const std::vector<float>& params) {
  std::vector<NamedTensor> tensors;
  tensors.reserve(layout.size());
  size_t off = 0;
  for (const auto& spec : layout) {
    int64_t k = 1;
    for (int64_t d : spec.shape) k *= d;
    if (off + static_cast<size_t>(k) > params.size()) {
      throw std::runtime_error("ModelManager: layout overruns params");
    }
    NamedTensor nt;
    nt.name = spec.name;
    nt.shape = spec.shape;
    nt.data.assign(params.begin() + static_cast<std::ptrdiff_t>(off),
                   params.begin() + static_cast<std::ptrdiff_t>(off + static_cast<size_t>(k)));
    tensors.push_back(std::move(nt));
    off += static_cast<size_t>(k);
  }
  return tensors;
}

}  // namespace

std::string ModelManager::canonicalStateSha256() const {
  return Sha256::hexDigest(saveSafetensors(namedTensors(layout_, params_), {}));
}

std::string ModelManager::serializeStateDict(int64_t numExamples) const {
  std::vector<NamedTensor> tensors;
  tensors.reserve(layout_.size());
  size_t off = 0;
  for (const auto& spec : layout_) {
    const auto k = static_cast<size_t>(numelOf(spec));
    if (off + k > params_.size()) {
      throw std::runtime_error("ModelManager::serializeStateDict: layout overruns params");
    }
    NamedTensor nt;
    nt.name = spec.name;
    nt.shape = spec.shape;
    nt.data.assign(params_.begin() + static_cast<std::ptrdiff_t>(off),
                   params_.begin() + static_cast<std::ptrdiff_t>(off + k));
    tensors.push_back(std::move(nt));
    off += k;
  }
  return saveSafetensors(tensors, {{"num_examples", std::to_string(numExamples)}});
}

void ModelManager::loadStateDict(const std::string& blob) {
  const std::vector<NamedTensor> tensors = loadSafetensors(blob);
  if (tensors.size() != layout_.size()) {
    throw std::runtime_error("ModelManager::loadStateDict: tensor count != layout size");
  }
  std::unordered_map<std::string, const NamedTensor*> byName;
  byName.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    if (!byName.emplace(tensor.name, &tensor).second) {
      throw std::runtime_error("ModelManager::loadStateDict: duplicate tensor '" + tensor.name + "'");
    }
  }
  std::vector<float> next;
  next.reserve(params_.size());
  for (const auto& spec : layout_) {
    const auto it = byName.find(spec.name);
    if (it == byName.end()) {
      throw std::runtime_error("ModelManager::loadStateDict: missing tensor '" + spec.name + "'");
    }
    const auto& tensor = *it->second;
    if (tensor.data.size() != static_cast<size_t>(numelOf(spec))) {
      throw std::runtime_error("ModelManager::loadStateDict: size mismatch for '" +
                               spec.name + "'");
    }
    next.insert(next.end(), tensor.data.begin(), tensor.data.end());
  }
  if (next.size() != params_.size()) {
    throw std::runtime_error("ModelManager::loadStateDict: total size mismatch");
  }
  params_ = std::move(next);
}

float ModelManager::loss(const std::vector<float>& flat, const float* x,
                         const std::vector<int64_t>& xShape, const int64_t* y, int64_t n) const {
  if (!model_) throw std::runtime_error("ModelManager::loss: no model loaded");
  return model_->loss(flat, x, xShape, y, n);
}

std::string ModelManager::tierForParamCount(int64_t totalParams) {
  if (totalParams >= 100'000'000) return "100M";
  if (totalParams >= 10'000'000) return "10M";
  if (totalParams >= 1'000'000) return "1M";
  return "";
}

}  // namespace fedlearn
