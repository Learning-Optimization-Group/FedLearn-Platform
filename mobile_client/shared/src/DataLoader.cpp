#include "DataLoader.h"
#include <fstream>
#include <nlohmann/json.hpp>

namespace fedlearn {

bool DataLoader::loadFromJson(const std::string& trainPath,
                               const std::string& testPath) {
  try {
    auto loadFile = [](const std::string& path,
                       torch::Tensor& X, torch::Tensor& y) {
      std::ifstream f(path);
      if (!f.is_open()) {
        log("DataLoader", "Cannot open: " + path);
        return false;
      }

      nlohmann::json data = nlohmann::json::parse(f);
      auto& images = data["images"];
      auto& labels = data["labels"];

      int n = static_cast<int>(images.size());
      // MNIST: 28x28 images, flattened as arrays of 784 floats
      std::vector<float> img_data;
      std::vector<int64_t> label_data;

      for (int i = 0; i < n; ++i) {
        auto& img = images[i];
        for (auto& pixel : img) {
          img_data.push_back(pixel.get<float>());
        }
        label_data.push_back(labels[i].get<int64_t>());
      }

      X = torch::from_blob(img_data.data(), {n, 1, 28, 28},
                            torch::kFloat32)
              .clone();
      y = torch::from_blob(label_data.data(), {n}, torch::kInt64).clone();
      return true;
    };

    bool ok = loadFile(trainPath, X_train_, y_train_) &&
              loadFile(testPath, X_test_, y_test_);
    loaded_ = ok;

    if (ok) {
      log("DataLoader",
          "Loaded " + std::to_string(X_train_.size(0)) + " train, " +
              std::to_string(X_test_.size(0)) + " test samples");
    }
    return ok;
  } catch (const std::exception& e) {
    log("DataLoader", std::string("Failed to load data: ") + e.what());
    loaded_ = false;
    return false;
  }
}

bool DataLoader::loadFromTensors(torch::Tensor X_train, torch::Tensor y_train,
                                  torch::Tensor X_test, torch::Tensor y_test) {
  X_train_ = std::move(X_train);
  y_train_ = std::move(y_train);
  X_test_ = std::move(X_test);
  y_test_ = std::move(y_test);
  loaded_ = true;
  return true;
}

std::vector<std::pair<torch::Tensor, torch::Tensor>> DataLoader::getBatches(
    int batch_size, bool train) const {
  const auto& X = train ? X_train_ : X_test_;
  const auto& y = train ? y_train_ : y_test_;

  std::vector<std::pair<torch::Tensor, torch::Tensor>> batches;
  int64_t n = X.size(0);

  for (int64_t i = 0; i < n; i += batch_size) {
    int64_t end = std::min(i + batch_size, n);
    batches.emplace_back(X.slice(0, i, end), y.slice(0, i, end));
  }

  return batches;
}

}  // namespace fedlearn
