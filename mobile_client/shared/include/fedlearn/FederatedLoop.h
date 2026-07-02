#pragma once
//
// FederatedLoop.h — one-round bodies for the DeComFL (primary) and FedAvg (fallback) paths
// (15-LLD-mobile.md §6.2 / §6.3 / §13 task 10). Orchestrates the gRPC seam (IFedLearnClient) +
// DeComFLClient / ExecutorchModel / ModelManager (the libtorch-free C++ core, Phase 3c).
//
// Module-free: it talks to gRPC through the core-typed IFedLearnClient interface (no proto, no
// grpcpp), so it builds + unit-tests in the libtorch-free ET suite with a mock client. No
// torch_version gate (RandnEngine makes the perturbation RNG version-independent).
//
#include <string>
#include <vector>

#include "fedlearn/ExecutorchModel.h"
#include "fedlearn/IFedLearnClient.h"
#include "fedlearn/ModelManager.h"
#include "fedlearn/Types.h"

namespace fedlearn {

struct RoundOutcome {
  bool ranTraining = false;   // true if this client trained + submitted this round
  bool shouldStop = false;    // true if the loop must stop (server should_stop / deadline / terminal)
  int round = 0;
  int scalarsK = 0;           // server-authoritative K (local steps) actually used this round
  int scalarsP = 0;           // server-authoritative P (perturbations) actually used this round
  std::string note;           // human-readable reason when shouldStop / skipped
};

class FederatedLoop {
 public:
  FederatedLoop(IFedLearnClient& net, ModelManager& mm);

  // One DeComFL round (§6.2): GetDeComFLConfig -> (rebuild if missed) -> fit -> SubmitGradientScalars.
  // The K/P/eta/mu come from the per-round server config (the server is authoritative).
  RoundOutcome deComFLRound(ExecutorchModel& model, const std::string& runId,
                            const std::string& clientId, const DataBatch& batch);

  // One FedAvg round (§6.3, ZO-SGD): GetGlobalModelStream -> loadStateDict -> K local ZO-SGD steps
  // -> SubmitGradientScalars (scalar upload, Constraint 7 — not a weight blob). numLocalSteps (K),
  // learningRate, mu, and numPerturbations (P) come from the per-round server config.
  RoundOutcome fedAvgRound(ExecutorchModel& model, const std::string& runId,
                           const std::string& clientId, const DataBatch& batch,
                           int numLocalSteps, double learningRate, double mu,
                           int numPerturbations = 1);

 private:
  IFedLearnClient& net_;
  ModelManager& mm_;
  std::vector<float> flatState_;
};

}  // namespace fedlearn
