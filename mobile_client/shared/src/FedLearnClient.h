#pragma once

#include "Utils.h"
#include <grpcpp/grpcpp.h>
#include "generated/fedlearn.grpc.pb.h"

namespace fedlearn {

/**
 * Port of framework/src/fedlearn/client/grpc_client.py
 *
 * Native gRPC client that communicates directly with the fedlearn server.
 * Uses a dual-channel pattern: one for training RPCs, one for heartbeat.
 */
class FedLearnClient {
 public:
  FedLearnClient(const std::string& server_address,
                 const std::string& client_id);
  ~FedLearnClient();

  // Registration
  bool registerClient();

  // Download: receives ModelParameters proto via unary call, returns tensors
  struct GlobalModelResult {
    OrderedDict tensors;      // parsed parameter tensors from proto
    int32_t current_round = 0;
    ConfigMap config;
    bool success = false;
  };
  GlobalModelResult getGlobalModel();

  // Upload: serialise tensors as ModelParameters proto, call unary SubmitModelUpdate
  bool submitUpdate(const OrderedDict& params,
                    int64_t num_examples, int32_t round_number);

  // Heartbeat management (separate gRPC channel, background thread)
  void startHeartbeat(int interval_seconds = 5);
  void stopHeartbeat();
  void updateStatus(const std::string& status, int step, int total);
  void setCurrentRound(int32_t round) { current_round_ = round; }

  // DeComFL RPCs
  struct DeComFLConfigResult {
    int32_t current_round;
    Seeds2D seeds;
    std::vector<RebuildRound> rebuild_history;
    ConfigMap config;
    bool success;
  };
  DeComFLConfigResult getDeComFLConfig();
  bool submitGradientScalars(
      const std::vector<std::vector<double>>& scalars,
      int64_t num_examples, int32_t round_number);

  // Connection management
  void close();
  bool isConnected() const { return connected_; }

 private:
  void heartbeatLoop();

  std::string client_id_;
  bool connected_ = false;

  // Main channel for training RPCs
  std::shared_ptr<grpc::Channel> channel_;
  std::unique_ptr<fedlearn::v1::FederatedLearningService::Stub> stub_;

  // Separate channel for heartbeat (prevents timeout during long training)
  std::shared_ptr<grpc::Channel> heartbeat_channel_;
  std::unique_ptr<fedlearn::v1::FederatedLearningService::Stub> heartbeat_stub_;

  // Heartbeat state
  std::atomic<bool> heartbeat_active_{false};
  std::thread heartbeat_thread_;
  int heartbeat_interval_ = 5;
  std::mutex status_mutex_;
  std::string current_status_ = "idle";
  int current_step_ = 0;
  int total_steps_ = 0;
  int32_t current_round_ = 0;

  static constexpr int CHUNK_SIZE = 50 * 1024 * 1024;  // 50 MB
};

}  // namespace fedlearn
