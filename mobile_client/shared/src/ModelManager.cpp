#include "fedlearn/ModelManager.h"

#include <stdexcept>
#include <vector>

#include "fedlearn/Sha256.h"

namespace fedlearn {

torch::jit::Module ModelManager::loadScriptModel(const std::string& path,
                                                 const std::string& expectedSha256,
                                                 ModelInfo* info) {
  // Verify BEFORE load — never hand untrusted bytes to torch::jit::load (arbitrary-code risk).
  const std::string actual = Sha256::hexDigestFile(path);
  if (actual != expectedSha256) {
    throw std::runtime_error("ModelManager: sha256 mismatch for '" + path +
                             "' (expected " + expectedSha256 + ", got " + actual + ")");
  }

  torch::jit::Module model = torch::jit::load(path);
  model.eval();

  if (info != nullptr) {
    int64_t total = 0;
    int64_t trainable = 0;
    for (const auto& p : model.parameters()) {
      total += p.numel();
      if (p.requires_grad()) trainable += p.numel();
    }
    info->paramCount = total;
    info->trainableParamCount = trainable;
    info->sha256 = actual;
    info->tier = tierForParamCount(total);
  }
  return model;
}

torch::Tensor ModelManager::getFlatParams(const torch::jit::Module& model) const {
  std::vector<torch::Tensor> parts;
  for (const auto& p : model.parameters()) {
    if (!p.requires_grad()) continue;  // exclude frozen layers, exactly like Python
    parts.push_back(p.detach().reshape({-1}));
  }
  TORCH_CHECK(!parts.empty(), "ModelManager::getFlatParams: model has no trainable parameters");
  return torch::cat(parts);
}

void ModelManager::setFlatParams(torch::jit::Module& model, const torch::Tensor& flat) const {
  torch::NoGradGuard no_grad;
  const torch::Tensor src = flat.contiguous();
  int64_t offset = 0;
  for (auto p : model.parameters()) {
    if (!p.requires_grad()) continue;
    const int64_t n = p.numel();
    TORCH_CHECK(offset + n <= src.numel(),
                "ModelManager::setFlatParams: flat vector too short for the model");
    // view(-1) is a view into p's storage (params are contiguous) -> copy_ writes through.
    p.view({-1}).copy_(src.narrow(0, offset, n));
    offset += n;
  }
  TORCH_CHECK(offset == src.numel(),
              "ModelManager::setFlatParams: flat vector longer than trainable params");
}

std::string ModelManager::serializeStateDict(const torch::jit::Module& model,
                                             int64_t numExamples) const {
  c10::Dict<std::string, torch::Tensor> params;
  for (const auto& np : model.named_parameters()) {
    params.insert(np.name, np.value.detach().cpu());
  }
  c10::Dict<std::string, c10::IValue> root;
  root.insert("parameters", c10::IValue(params));
  root.insert("num_examples", c10::IValue(numExamples));

  std::vector<char> blob = torch::jit::pickle_save(c10::IValue(root));
  return std::string(blob.begin(), blob.end());
}

void ModelManager::loadStateDict(torch::jit::Module& model, const std::string& blob) const {
  std::vector<char> bytes(blob.begin(), blob.end());
  c10::IValue root = torch::jit::pickle_load(bytes);
  c10::Dict<c10::IValue, c10::IValue> rootDict = root.toGenericDict();
  c10::IValue paramsIv = rootDict.at(c10::IValue(std::string("parameters")));
  c10::Dict<c10::IValue, c10::IValue> params = paramsIv.toGenericDict();

  torch::NoGradGuard no_grad;
  for (auto np : model.named_parameters()) {
    auto it = params.find(c10::IValue(np.name));
    if (it != params.end()) {
      np.value.copy_(it->value().toTensor().to(np.value.device()));
    }
  }
}

int64_t ModelManager::trainableParamCount(const torch::jit::Module& model) const {
  int64_t n = 0;
  for (const auto& p : model.parameters()) {
    if (p.requires_grad()) n += p.numel();
  }
  return n;
}

std::string ModelManager::tierForParamCount(int64_t totalParams) {
  if (totalParams >= 100'000'000) return "100M";
  if (totalParams >= 10'000'000) return "10M";
  if (totalParams >= 1'000'000) return "1M";
  return "";
}

}  // namespace fedlearn
