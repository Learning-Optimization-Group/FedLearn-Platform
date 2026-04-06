#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <atomic>
#include <thread>
#include <sstream>
#include <iostream>
#include <functional>
#include <chrono>
#ifdef ANDROID
#include <android/log.h>
#endif

namespace fedlearn {

using OrderedDict = std::map<std::string, torch::Tensor>;
using ConfigMap = std::map<std::string, std::string>;
using Seeds2D = std::vector<std::vector<int32_t>>;
using FedLearnModule = torch::jit::script::Module;

struct RebuildRound {
  int32_t round_number;
  Seeds2D seeds;
  std::vector<std::vector<double>> gradients;
};

struct TrainingConfig {
  float learning_rate = 0.001f;
  float smoothing_param = 0.001f;
  int num_local_steps = 1;
  int num_perturbations = 10;
  Seeds2D seeds;
};

inline void log(const std::string& tag, const std::string& msg) {
#ifdef ANDROID
  __android_log_print(ANDROID_LOG_INFO, tag.c_str(), "%s", msg.c_str());
#endif
  std::cout << "[" << tag << "] " << msg << std::endl;
}

inline torch::Tensor getFlatParams(FedLearnModule& model) {
  std::vector<torch::Tensor> parts;
  for (const auto& param : model.parameters()) {
    parts.push_back(param.data().view(-1));
  }
  return torch::cat(parts);
}

inline void setFlatParams(FedLearnModule& model,
                          const torch::Tensor& flat) {
  int64_t offset = 0;
  for (const auto& param : model.parameters()) {
    int64_t numel = param.numel();
    param.data().copy_(flat.slice(0, offset, offset + numel).view_as(param));
    offset += numel;
  }
}

inline int64_t getNumParams(FedLearnModule& model) {
  int64_t total = 0;
  for (const auto& param : model.parameters()) {
    total += param.numel();
  }
  return total;
}

}  // namespace fedlearn
