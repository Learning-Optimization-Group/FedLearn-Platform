#include "NativeFedLearnCore.h"
#include <torch/torch.h>
#include <sstream>

// Minimal JSON serialization (avoids pulling in nlohmann/json for the module itself)
namespace {
std::string jsonEscape(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      default: out += c;
    }
  }
  return out;
}
}  // namespace

namespace fedlearn {

bool NativeFedLearnCoreImpl::loadModel(const std::string& modelPath) {
  std::lock_guard<std::mutex> lock(mutex_);
  model_mgr_ = std::make_unique<ModelManager>();
  return model_mgr_->loadScriptModel(modelPath);
}

std::string NativeFedLearnCoreImpl::getModelInfo() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!model_mgr_ || !model_mgr_->isLoaded()) {
    return R"({"numParams":0,"sizeBytes":0})";
  }
  int64_t n = model_mgr_->numParams();
  int64_t bytes = n * 4;  // float32
  return R"({"numParams":)" + std::to_string(n) +
         R"(,"sizeBytes":)" + std::to_string(bytes) + "}";
}

std::string NativeFedLearnCoreImpl::trainStep(const std::string& inputPath,
                                               int numEpochs, float lr) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!model_mgr_ || !model_mgr_->isLoaded()) {
    return R"({"loss":-1,"accuracy":-1,"error":"Model not loaded"})";
  }

  if (!data_loader_ || !data_loader_->isLoaded()) {
    data_loader_ = std::make_unique<DataLoader>();
    // Try loading from the provided path
    // inputPath should point to the directory containing train/test JSON files
    std::string trainFile = inputPath + "/mnist_train_100.json";
    std::string testFile = inputPath + "/mnist_test_20.json";
    if (!data_loader_->loadFromJson(trainFile, testFile)) {
      return R"({"loss":-1,"accuracy":-1,"error":"Failed to load data"})";
    }
  }

  auto batches = data_loader_->getBatches(32, true);
  float total_loss = 0.0f;
  int steps = 0;

  for (int epoch = 0; epoch < numEpochs; ++epoch) {
    for (auto& [input, target] : batches) {
      float loss = model_mgr_->trainStep(input, target, lr);
      total_loss += loss;
      steps++;
    }
  }

  float avg_loss = steps > 0 ? total_loss / steps : 0.0f;

  // Compute accuracy on test set
  float accuracy = 0.0f;
  auto test_batches = data_loader_->getBatches(32, false);
  int correct = 0, total = 0;
  for (auto& [input, target] : test_batches) {
    auto output = model_mgr_->forward(input);
    auto pred = output.argmax(1);
    correct += pred.eq(target).sum().item<int>();
    total += target.size(0);
  }
  accuracy = total > 0 ? static_cast<float>(correct) / total : 0.0f;

  return R"({"loss":)" + std::to_string(avg_loss) +
         R"(,"accuracy":)" + std::to_string(accuracy) + "}";
}

bool NativeFedLearnCoreImpl::connect(const std::string& serverAddress,
                                      const std::string& clientId) {
  std::lock_guard<std::mutex> lock(mutex_);
  try {
    grpc_client_ =
        std::make_unique<FedLearnClient>(serverAddress, clientId);
    return true;
  } catch (const std::exception& e) {
    log("NativeFedLearnCore",
        std::string("Connect failed: ") + e.what());
    return false;
  }
}

void NativeFedLearnCoreImpl::disconnect() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (fed_loop_) {
    fed_loop_->stop();
    fed_loop_.reset();
  }
  if (grpc_client_) {
    grpc_client_->close();
    grpc_client_.reset();
  }
}

void NativeFedLearnCoreImpl::startFedAvgLoop(const std::string& configJson) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!grpc_client_ || !model_mgr_) {
    log("NativeFedLearnCore", "Not ready: model or connection missing");
    return;
  }

  // Auto-initialize data loader with synthetic MNIST data if not already loaded
  if (!data_loader_ || !data_loader_->isLoaded()) {
    log("NativeFedLearnCore", "No data loaded - using synthetic MNIST data for FL round");
    data_loader_ = std::make_unique<DataLoader>();
    // Small synthetic dataset - kept minimal to avoid OOM on emulator with large models
    auto X_train = torch::randn({10, 1, 28, 28});
    auto y_train = torch::randint(0, 10, {10}, torch::kInt64);
    auto X_test  = torch::randn({5, 1, 28, 28});
    auto y_test  = torch::randint(0, 10, {5}, torch::kInt64);
    data_loader_->loadFromTensors(X_train, y_train, X_test, y_test);
  }

  // Parse configJson fields using simple key search (same pattern as setZOConfig)
  FederatedLoop::FedAvgConfig config;
  auto findFloatCfg = [&](const std::string& key) -> float {
    auto pos = configJson.find("\"" + key + "\"");
    if (pos == std::string::npos) return -1.0f;
    pos = configJson.find(':', pos);
    if (pos == std::string::npos) return -1.0f;
    try { return std::stof(configJson.substr(pos + 1)); } catch (...) { return -1.0f; }
  };
  float epochs = findFloatCfg("local_epochs");
  if (epochs > 0) config.local_epochs = static_cast<int>(epochs);
  float lr = findFloatCfg("learning_rate");
  if (lr > 0) config.learning_rate = lr;
  float bs = findFloatCfg("batch_size");
  if (bs > 0) config.batch_size = static_cast<int>(bs);
  log("NativeFedLearnCore",
      "FedAvg config: epochs=" + std::to_string(config.local_epochs) +
      " lr=" + std::to_string(config.learning_rate) +
      " batch_size=" + std::to_string(config.batch_size));

  fed_loop_ = std::make_unique<FederatedLoop>(*grpc_client_, *model_mgr_,
                                               *data_loader_);
  fed_loop_->startFedAvg(config);
}

void NativeFedLearnCoreImpl::startDeComFLLoop(const std::string& configJson) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!grpc_client_ || !model_mgr_) {
    log("NativeFedLearnCore", "Not ready: model or connection missing");
    return;
  }

  if (!data_loader_ || !data_loader_->isLoaded()) {
    log("NativeFedLearnCore", "No data loaded - using synthetic MNIST data for DeComFL round");
    data_loader_ = std::make_unique<DataLoader>();
    auto X_train = torch::randn({10, 1, 28, 28});
    auto y_train = torch::randint(0, 10, {10}, torch::kInt64);
    auto X_test  = torch::randn({5, 1, 28, 28});
    auto y_test  = torch::randint(0, 10, {5}, torch::kInt64);
    data_loader_->loadFromTensors(X_train, y_train, X_test, y_test);
  }

  FederatedLoop::DeComFLConfig config;
  config.smoothing_param = zo_mu_;

  fed_loop_ = std::make_unique<FederatedLoop>(*grpc_client_, *model_mgr_,
                                               *data_loader_);
  fed_loop_->startDeComFL(config);
}

void NativeFedLearnCoreImpl::stopTraining() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (fed_loop_) {
    fed_loop_->stop();
  }
}

std::string NativeFedLearnCoreImpl::getStatus() {
  if (!fed_loop_) {
    return R"({"phase":"idle","round":0,"loss":0,"accuracy":0,"step":0,"totalSteps":0,"error":""})";
  }

  auto status = fed_loop_->getStatus();
  return R"({"phase":")" + jsonEscape(status.phase) +
         R"(","round":)" + std::to_string(status.round) +
         R"(,"loss":)" + std::to_string(status.loss) +
         R"(,"accuracy":)" + std::to_string(status.accuracy) +
         R"(,"step":)" + std::to_string(status.step) +
         R"(,"totalSteps":)" + std::to_string(status.total_steps) +
         R"(,"error":")" + jsonEscape(status.error_message) + R"("})";
}

void NativeFedLearnCoreImpl::setZOConfig(const std::string& configJson) {
  std::lock_guard<std::mutex> lock(mutex_);
  // Simple parsing: look for "mu" and "numPert" values
  // In production, use a proper JSON parser
  auto findFloat = [&](const std::string& key) -> float {
    auto pos = configJson.find("\"" + key + "\"");
    if (pos == std::string::npos) return -1;
    pos = configJson.find(':', pos);
    if (pos == std::string::npos) return -1;
    return std::stof(configJson.substr(pos + 1));
  };

  float mu = findFloat("mu");
  if (mu > 0) zo_mu_ = mu;

  float np = findFloat("numPert");
  if (np > 0) zo_num_pert_ = static_cast<int>(np);

  log("NativeFedLearnCore",
      "ZO config: mu=" + std::to_string(zo_mu_) +
          " numPert=" + std::to_string(zo_num_pert_));
}

}  // namespace fedlearn
