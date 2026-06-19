#pragma once
//
// FedLearnClient.h — the C++ gRPC layer (15-LLD-mobile.md §5.2, §6; proto fedlearn.v2).
//
// REQUIRES the buf-generated C++ stubs (proto/ -> gen/cpp via `buf generate`) and a
// cross-compiled gRPC C++ runtime. It is therefore built ONLY under -DFEDLEARN_BUILD_GRPC=ON;
// the host parity gate (fedlearn_core) does not need it. Exact generated symbol names follow
// buf's protocolbuffers/grpc cpp plugins (namespace fedlearn::v2). Channel security defaults to
// TLS + mutual-TLS; insecure is dev-only (E13 / 04 §10.3).
//
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include <grpcpp/grpcpp.h>

#include "fedlearn/v2/fedlearn.grpc.pb.h"  // buf-generated; -I <gen/cpp>

#include "fedlearn/IFedLearnClient.h"
#include "fedlearn/Types.h"

namespace fedlearn {

struct GrpcClientConfig {
  std::string serverAddress;
  bool useTls = true;  // default secure (TLS+mTLS); only a dev RN build sets false (E13)
  std::string clientCertPath;
  std::string clientKeyPath;
  std::string caCertPath;
  int maxMessageBytes = 33554432;  // 32 MB phone cap (E11 / M-M3), NOT 1 GB
  int heartbeatIntervalMs = 5000;
  int heartbeatFailureLimit = 3;   // N consecutive failures -> abortFlag_ (E4 / M-H3)
};

// The concrete gRPC implementation of the core-typed IFedLearnClient seam (proto<->core
// marshaling). FederatedLoop depends only on IFedLearnClient; this class is the production wiring.
class FedLearnClient : public IFedLearnClient {
 public:
  explicit FedLearnClient(const GrpcClientConfig& cfg);
  ~FedLearnClient() override;

  FedLearnClient(const FedLearnClient&) = delete;
  FedLearnClient& operator=(const FedLearnClient&) = delete;

  // --- lifecycle / control (unary) ---
  v2::RegisterClientResponse registerClient(const std::string& runId, const std::string& clientId,
                                            const std::string& enrollmentToken, int protocolVersion);
  v2::GetServerStatusResponse getServerStatus(const std::string& runId);

  // --- IFedLearnClient (core-typed seam consumed by FederatedLoop) ---
  bool shouldStop() const override { return abortFlag_.load(); }
  // GetDeComFLConfig -> core DeComFLConfig (lr/mu/method from the config map, seeds, rebuild
  // history, should_stop, current_round).
  DeComFLConfig getDeComFLConfig(const std::string& runId, const std::string& clientId) override;
  // DeComFL upload: K*P scalars + num_examples (the O(K*P) wedge). The interface also carries the
  // seeds (Constraint 7); see the .cpp for the seeds-on-the-wire TODO (no proto regen this task).
  void submitGradientScalars(const std::string& runId, const std::string& clientId,
                             int trainedOnRound, const Seeds2D& seeds,
                             const GradientScalars2D& gradients, int64_t numExamples) override;
  // FedAvg download: reassembles the ModelChunk stream into a validated blob (codec whitelist +
  // cumulative size cap + sha256), sets *outCurrentRound. Single decode site is ModelManager.
  std::string getGlobalModelStream(const std::string& runId, const std::string& clientId,
                                   int* outCurrentRound) override;

  // --- FedAvg path (streaming, chunked) — proto-typed, kept for the (legacy) weight-blob upload ---
  v2::SubmitModelUpdateResponse submitModelUpdateStream(const std::string& runId,
                                                        const std::string& clientId,
                                                        int trainedOnRound,
                                                        const std::string& modelBlob,
                                                        int64_t numExamples, bool compressed = false);

  // --- telemetry (best-effort; never throws) ---
  void reportClientMetrics(const v2::ReportClientMetricsRequest& metrics);

  // --- dual heartbeat: own thread + own channel; sets abortFlag_ on N failures OR should_stop ---
  void startHeartbeat(const std::string& runId, const std::string& clientId, int currentRound);
  void stopHeartbeat();

  // --- pure proto<->core marshaling (exposed for unit tests; no network) ---
  static v2::GradientScalars toProtoScalars(const GradientScalars2D& g);
  static GradientScalars2D fromProtoScalars(const v2::GradientScalars& g);
  static Seeds2D fromProtoSeeds(const v2::PerturbationSeeds& s);
  static RebuildHistory fromProtoRebuildHistory(const v2::RebuildHistory& h);

  // Whitelisted codecs (04 §10.3). Throws std::invalid_argument otherwise.
  static void validateCodec(const std::string& codec);

 private:
  std::shared_ptr<grpc::Channel> makeChannel() const;
  // The raw GetDeComFLConfig RPC (proto response); getDeComFLConfig() marshals it to core types.
  v2::GetDeComFLConfigResponse fetchDeComFLConfig(const std::string& runId,
                                                  const std::string& clientId);

  GrpcClientConfig cfg_;
  std::shared_ptr<grpc::Channel> trainingChannel_;
  std::shared_ptr<grpc::Channel> heartbeatChannel_;  // SEPARATE channel (dual-heartbeat invariant)
  std::unique_ptr<v2::FederatedLearningService::Stub> trainingStub_;
  std::unique_ptr<v2::FederatedLearningService::Stub> heartbeatStub_;

  std::atomic<bool> abortFlag_{false};
  std::atomic<bool> heartbeatStop_{false};
  std::thread heartbeatThread_;
};

}  // namespace fedlearn
