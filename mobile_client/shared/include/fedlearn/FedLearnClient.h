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
#include <mutex>
#include <string>
#include <thread>

#include <grpcpp/grpcpp.h>

#include "fedlearn/v2/fedlearn.grpc.pb.h"  // buf-generated; -I <gen/cpp>

#include "fedlearn/AuthMetadata.h"
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
  // MO-2: per-RPC deadline for the training-stub calls (register/status/config/submit/download). A
  // bare ClientContext has NO deadline, so a stalled server hangs the call forever and the round loop's
  // between-step shouldStop() poll never gets to run. This bounds every call; requestAbort() also
  // TryCancel()s the in-flight one for a prompt stop. Generous by default (large model downloads).
  int rpcDeadlineMs = 60000;
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
  // Proto-free seam upload: marshals the weight blob onto SubmitModelUpdateStream (below).
  void submitModelUpdate(const std::string& runId, const std::string& clientId,
                         int trainedOnRound, const std::string& modelBlob,
                         int64_t numExamples) override;

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

  // Abort: flips the shared abort flag so an in-progress round's shouldStop() poll breaks out at the
  // next between-step check, AND TryCancel()s the in-flight training RPC (if any) so a call blocked on
  // a slow/hung server returns immediately instead of waiting out the deadline. Safe to call from any
  // thread (e.g. the JS-thread stop()) — it never takes the round mutex; only the brief trainCtx mutex
  // (which the RPC thread never holds across the blocking call). MO-2.
  void requestAbort();

  // --- pure proto<->core marshaling (exposed for unit tests; no network) ---
  static v2::GradientScalars toProtoScalars(const GradientScalars2D& g);
  static v2::PerturbationSeeds toProtoSeeds(const Seeds2D& s);
  static GradientScalars2D fromProtoScalars(const v2::GradientScalars& g);
  static Seeds2D fromProtoSeeds(const v2::PerturbationSeeds& s);
  static RebuildHistory fromProtoRebuildHistory(const v2::RebuildHistory& h);

  // Whitelisted codecs (04 §10.3). Throws std::invalid_argument otherwise.
  static void validateCodec(const std::string& codec);

 private:
  std::shared_ptr<grpc::Channel> makeChannel() const;
  // Attaches the FL connection token (if any) as x-connection-token metadata so a fail-closed
  // server's ConnectionTokenInterceptor admits the call. No-op when connectionToken_ is empty.
  void applyAuth(grpc::ClientContext& ctx) const;
  // MO-2: applyAuth + a bounded per-RPC deadline (cfg_.rpcDeadlineMs). Every training-stub call runs
  // through this so none can hang the round loop indefinitely.
  void prepareRpc(grpc::ClientContext& ctx) const;
  // The raw GetDeComFLConfig RPC (proto response); getDeComFLConfig() marshals it to core types.
  v2::GetDeComFLConfigResponse fetchDeComFLConfig(const std::string& runId,
                                                  const std::string& clientId);

  GrpcClientConfig cfg_;
  // Backend-minted FL connection token, set once at registerClient and attached as
  // x-connection-token metadata on every RPC. Written before the heartbeat thread starts
  // (safe publication), so no lock is needed.
  std::string connectionToken_;
  std::shared_ptr<grpc::Channel> trainingChannel_;
  std::shared_ptr<grpc::Channel> heartbeatChannel_;  // SEPARATE channel (dual-heartbeat invariant)
  std::unique_ptr<v2::FederatedLearningService::Stub> trainingStub_;
  std::unique_ptr<v2::FederatedLearningService::Stub> heartbeatStub_;

  std::atomic<bool> abortFlag_{false};
  std::atomic<bool> heartbeatStop_{false};
  std::thread heartbeatThread_;
  std::mutex hbCtxMutex_;                 // guards hbCtx_ (the heartbeat's in-flight ClientContext)
  grpc::ClientContext* hbCtx_ = nullptr;  // non-null while a Heartbeat RPC is in flight, for TryCancel
  // MO-2: same pattern for the TRAINING stub — non-null while a training RPC is in flight, so
  // requestAbort() can TryCancel() it for a prompt stop (mirrors hbCtx_).
  std::mutex trainCtxMutex_;
  grpc::ClientContext* trainCtx_ = nullptr;
};

}  // namespace fedlearn
