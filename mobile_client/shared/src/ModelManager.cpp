#include "fedlearn/ModelManager.h"

#include <stdexcept>

#include "fedlearn/Safetensors.h"

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
  std::vector<float> next;
  next.reserve(params_.size());
  for (size_t i = 0; i < layout_.size(); ++i) {
    if (tensors[i].name != layout_[i].name) {
      throw std::runtime_error("ModelManager::loadStateDict: name mismatch at index " +
                               std::to_string(i) + " ('" + tensors[i].name + "' != '" +
                               layout_[i].name + "')");
    }
    if (tensors[i].data.size() != static_cast<size_t>(numelOf(layout_[i]))) {
      throw std::runtime_error("ModelManager::loadStateDict: size mismatch for '" +
                               layout_[i].name + "'");
    }
    next.insert(next.end(), tensors[i].data.begin(), tensors[i].data.end());
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
