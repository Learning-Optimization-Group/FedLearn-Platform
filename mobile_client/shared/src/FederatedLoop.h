#pragma once

#include "Utils.h"
#include "FedLearnClient.h"
#include "ModelManager.h"
#include "DataLoader.h"
#include "DeComFLClient.h"

namespace fedlearn {

/**
 * Port of framework/src/fedlearn/client/client.py (start_client)
 * and framework/src/fedlearn/client/decomfl_start.py (start_decomfl_client).
 *
 * Runs federated training loops on a background thread.
 * Supports both FedAvg (standard parameters) and DeComFL (gradient scalars).
 */
class FederatedLoop {
 public:
  FederatedLoop(FedLearnClient& grpc_client, ModelManager& model_mgr,
                DataLoader& data_loader);
  ~FederatedLoop();

  struct FedAvgConfig {
    int local_epochs = 1;
    float learning_rate = 0.01f;
    int batch_size = 1;
  };

  struct DeComFLConfig {
    float learning_rate = 0.001f;
    float smoothing_param = 0.001f;
    int batch_size = 32;
  };

  void startFedAvg(const FedAvgConfig& config);
  void startDeComFL(const DeComFLConfig& config);
  void stop();

  struct Status {
    std::string phase;  // "idle", "registering", "fetching", "training",
                        // "uploading", "waiting", "stopped", "error"
    int32_t round = 0;
    float loss = 0.0f;
    float accuracy = 0.0f;
    int step = 0;
    int total_steps = 0;
    std::string error_message;
  };
  Status getStatus() const;

 private:
  void fedAvgLoop(FedAvgConfig config);
  void deComFLLoop(DeComFLConfig config);
  void setStatus(const std::string& phase, int32_t round = -1,
                 float loss = -1, float accuracy = -1,
                 int step = -1, int total = -1,
                 const std::string& error = "");

  FedLearnClient& grpc_client_;
  ModelManager& model_mgr_;
  DataLoader& data_loader_;

  std::atomic<bool> running_{false};
  std::thread loop_thread_;
  mutable std::mutex status_mutex_;
  Status status_;
};

}  // namespace fedlearn
