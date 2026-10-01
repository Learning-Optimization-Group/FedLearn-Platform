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
#ifdef FEDLEARN_HAS_TRAINING
#include "fedlearn/TrainableExecutorchModel.h"
#endif

namespace fedlearn {

/**
 * How a first-order round batches the local dataset (the contract's local_training). The default, batchSize 0, is one
 * whole-dataset step per epoch: what every single-batch contract states, and what the loop always did. With
 * seededPermutation (BATCH_ORDER_SEEDED_PERMUTATION_V1), each epoch visits the examples in the order drawn from
 * (seed, the server's round, epoch) and takes one step per batchSize minibatch, keeping the final partial one.
 */
struct LocalBatching {
  int64_t batchSize = 0;
  bool seededPermutation = false;
  uint64_t seed = 0;  // ExecutionContract.seed (0 when absent)
};

/** The contract's local optimizer: SGD at the round's learning rate, or Adam (fresh each round) at it. */
struct LocalOptimizer {
  bool adam = false;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon = 1e-8;
};

/**
 * The contract's dropout layers (ModelTraining.dropout): their rates in forward order and the run seed their
 * DROPOUT_MASKS_SEEDED_V1 masks are drawn with. Empty for a model without dropout. The program must take one mask
 * input per layer, after (x, y).
 */
struct DropoutSpec {
  std::vector<double> rates;
  uint64_t seed = 0;
};

}  // namespace fedlearn

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
  RoundOutcome deComFLRound(ExecutorchModel& model, const std::string& runId, const std::string& clientId,
                            const DataBatch& batch, const ZerothOrderContract* contract = nullptr);

  // One FedAvg round (§6.3, ZO-SGD): GetGlobalModelStream -> loadStateDict -> K local ZO-SGD steps
  // -> SubmitGradientScalars (scalar upload, Constraint 7 — not a weight blob). numLocalSteps (K),
  // learningRate, mu, and numPerturbations (P) come from the per-round server config.
  RoundOutcome fedAvgRound(ExecutorchModel& model, const std::string& runId,
                           const std::string& clientId, const DataBatch& batch,
                           int numLocalSteps, double learningRate, double mu,
                           int numPerturbations = 1);

#ifdef FEDLEARN_HAS_TRAINING
  // One TRUE first-order (FedAvg) round via real backprop (Phase B M2): GetGlobalModelStream ->
  // load the global weights into `model` -> K local SGD steps (execute_forward_backward + SGD in
  // TrainableExecutorchModel::trainStep) -> serialize the updated weights -> submitModelUpdate (the
  // weight-blob wire, NOT the ZO scalar wire). numLocalSteps and learningRate are the run's
  // execution contract; per-round server settings must equal them or the round is refused before upload.
  // FedOpt additionally requires the server to send them. proximalMu is the contract's FedProx
  // coefficient (0 for every other strategy): each step adds mu * (w - w_global), with w_global the
  // downloaded model, and the server's proximal_mu (absent = 0) must equal it. batching is the contract's minibatching
  // (LocalBatching); the update is always weighted by the whole dataset's example count.
  RoundOutcome firstOrderRound(TrainableExecutorchModel& model, const std::string& runId,
                               const std::string& clientId, const DataBatch& batch,
                               int numLocalSteps, double learningRate,
                               bool requireServerConfig = false, double proximalMu = 0.0,
                               const LocalBatching& batching = {}, const LocalOptimizer& optimizer = {},
                               const DropoutSpec& dropout = {});
#endif

 private:
  IFedLearnClient& net_;
  ModelManager& mm_;
  std::vector<float> flatState_;
};

}  // namespace fedlearn
