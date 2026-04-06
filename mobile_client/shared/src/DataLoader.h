#pragma once

#include "Utils.h"
#include <torch/torch.h>
#include <string>
#include <vector>

namespace fedlearn {

/**
 * Simple data loader for bundled datasets.
 * Loads JSON-formatted MNIST data from app assets.
 */
class DataLoader {
 public:
  DataLoader() = default;

  bool loadFromJson(const std::string& trainPath, const std::string& testPath);
  bool loadFromTensors(torch::Tensor X_train, torch::Tensor y_train,
                       torch::Tensor X_test, torch::Tensor y_test);

  std::vector<std::pair<torch::Tensor, torch::Tensor>> getBatches(
      int batch_size, bool train = true) const;

  int64_t numTrainSamples() const { return X_train_.size(0); }
  int64_t numTestSamples() const { return X_test_.size(0); }
  bool isLoaded() const { return loaded_; }

 private:
  torch::Tensor X_train_, y_train_;
  torch::Tensor X_test_, y_test_;
  bool loaded_ = false;
};

}  // namespace fedlearn
