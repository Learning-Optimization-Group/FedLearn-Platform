#pragma once

#include "ModelManager.h"
#include "FedLearnClient.h"
#include "FederatedLoop.h"
#include "DataLoader.h"
#include "ZerothOrderEstimator.h"
#include "DeComFLClient.h"

#include <memory>
#include <string>
#include <mutex>


namespace fedlearn {

/**
 * C++ TurboModule implementation for React Native.
 *
 * This is the bridge between JavaScript and the native C++ core.
 * All methods are thread-safe and dispatch to the core classes.
 *
 * After codegen runs on specs/NativeFedLearnCore.ts, this class
 * should extend NativeFedLearnCoreCxxSpec<NativeFedLearnCore>.
 * For now, we define the methods directly so the core logic compiles
 * independently of the RN codegen.
 */
class NativeFedLearnCoreImpl {
 public:
  NativeFedLearnCoreImpl() = default;

  // Model lifecycle
  bool loadModel(const std::string& modelPath);
  std::string getModelInfo();

  // Local training
  std::string trainStep(const std::string& inputPath, int numEpochs, float lr);

  // Federated learning
  bool connect(const std::string& serverAddress, const std::string& clientId);
  void disconnect();
  void startFedAvgLoop(const std::string& configJson);
  void startDeComFLLoop(const std::string& configJson);
  void stopTraining();
  std::string getStatus();

  // ZO config
  void setZOConfig(const std::string& configJson);

  // Log retrieval for JS-side display
  std::string getRecentLogs();

 private:
  std::unique_ptr<ModelManager> model_mgr_;
  std::unique_ptr<FedLearnClient> grpc_client_;
  std::unique_ptr<FederatedLoop> fed_loop_;
  std::unique_ptr<DataLoader> data_loader_;
  std::mutex mutex_;

  // ZO configuration
  float zo_mu_ = 0.001f;
  int zo_num_pert_ = 10;
};

}  // namespace fedlearn
