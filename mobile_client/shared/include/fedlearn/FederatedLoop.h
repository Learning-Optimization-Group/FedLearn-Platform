#pragma once
//
// FederatedLoop.h — one-round bodies for the DeComFL (primary) and FedAvg (fallback) paths
// (15-LLD-mobile.md §6.2 / §6.3 / §13 task 10). Orchestrates FedLearnClient (gRPC) +
// DeComFLClient/ZerothOrderEstimator/ModelManager (the C++ core). Built only under
// -DFEDLEARN_BUILD_GRPC=ON (it depends on FedLearnClient).
//
#include <string>

#include <torch/script.h>

#include "fedlearn/FedLearnClient.h"
#include "fedlearn/ModelManager.h"
#include "fedlearn/Types.h"

namespace fedlearn {

struct RoundOutcome {
  bool ranTraining = false;   // true if this client trained + submitted this round
  bool shouldStop = false;    // true if the loop must stop (server should_stop / deadline / terminal)
  int round = 0;
  std::string note;           // human-readable reason when shouldStop / skipped
};

class FederatedLoop {
 public:
  // localTorchVersion is the device's torch build; it MUST match the run's manifest for RNG
  // parity (E2). allowVersionMismatch is the dev override (release passes false -> refuse).
  FederatedLoop(FedLearnClient& net, ModelManager& mm, std::string localTorchVersion,
                bool allowVersionMismatch = false);

  // One DeComFL round (§6.2): GetDeComFLConfig -> (rebuild if missed) -> fit -> SubmitGradientScalars.
  // The K/P/eta/mu/method come from the per-round server config (the server is authoritative).
  RoundOutcome deComFLRound(torch::jit::Module& model, const std::string& runId,
                            const std::string& clientId, const DataBatch& batch);

  // One FedAvg round (§6.3): GetGlobalModelStream -> loadStateDict -> local SGD -> submit.
  // numLocalSteps + learningRate come from the server config carried in the download.
  RoundOutcome fedAvgRound(torch::jit::Module& model, const std::string& runId,
                           const std::string& clientId, const DataBatch& batch,
                           int numLocalSteps, double learningRate);

 private:
  FedLearnClient& net_;
  ModelManager& mm_;
  std::string localTorchVersion_;
  bool allowVersionMismatch_;
};

}  // namespace fedlearn
