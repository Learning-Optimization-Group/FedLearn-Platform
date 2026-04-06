#pragma once

#include "Utils.h"
#include <torch/script.h>

namespace fedlearn {

class ModelManager {
 public:
  ModelManager() = default;

  bool loadScriptModel(const std::string& path);
  bool loadStateDict(const std::vector<uint8_t>& data);
  std::vector<uint8_t> serializeStateDict();

  torch::Tensor forward(torch::Tensor input);
  float computeLoss(torch::Tensor output, torch::Tensor target);

  torch::Tensor getFlatParams();
  void setFlatParams(const torch::Tensor& flat);
  int64_t numParams();

  float trainStep(torch::Tensor input, torch::Tensor target, float lr);

  bool isLoaded() const { return loaded_; }

  OrderedDict getStateDict();
  void setStateDict(const OrderedDict& state_dict);

  FedLearnModule& getModule() { return model_; }

 private:
  FedLearnModule model_;
  bool loaded_ = false;
};

}  // namespace fedlearn
