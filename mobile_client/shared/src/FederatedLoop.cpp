#include "fedlearn/FederatedLoop.h"

#include <cstdint>
#include <vector>

#include "fedlearn/DeComFLClient.h"
#include "fedlearn/EtZeroOrder.h"
#include "fedlearn/RandnEngine.h"

namespace fedlearn {

FederatedLoop::FederatedLoop(IFedLearnClient& net, ModelManager& mm) : net_(net), mm_(mm) {}

RoundOutcome FederatedLoop::deComFLRound(ExecutorchModel& model, const std::string& runId,
                                         const std::string& clientId, const DataBatch& batch) {
  RoundOutcome out;
  if (net_.shouldStop()) {
    out.shouldStop = true;
    out.note = "heartbeat/abort flag set before round";
    return out;
  }

  DeComFLConfig cfg = net_.getDeComFLConfig(runId, clientId);
  out.round = cfg.currentRound;
  // Server-authoritative completion: the config path no longer carries should_stop (that lives on the
  // heartbeat), so the server signals "training complete" with current_round == -1 (matches the Python
  // client's decomfl_start sentinel). Terminate cleanly.
  if (cfg.currentRound < 0) {
    out.shouldStop = true;
    out.note = "training complete (server signalled round -1)";
    return out;
  }
  if (cfg.shouldStop) {
    out.shouldStop = true;
    out.note = "server should_stop in DeComFL config";
    return out;
  }

  // NO torch_version gate: RandnEngine makes the perturbation RNG version-independent.

  const Seeds2D& seeds = cfg.seeds;
  if (seeds.empty() || seeds[0].empty()) {
    out.note = "empty seed set from server; skipping round";
    return out;
  }
  const int K = static_cast<int>(seeds.size());
  const int P = static_cast<int>(seeds[0].size());
  out.scalarsK = K;  // server-authoritative K/P actually used (for accurate comm-cost reporting)
  out.scalarsP = P;

  // Lazily snapshot the global params into the loop's owned working state.
  if (flatState_.empty()) flatState_ = mm_.getFlatParams();

  DeComFLClient client(cfg.config.learningRate, P, K);

  // Replay any rounds this client missed (Algorithm 2), using config lr for each round.
  if (!cfg.rebuildHistory.empty()) {
    RebuildHistory hist = cfg.rebuildHistory;
    for (auto& r : hist) r.learningRate = cfg.config.learningRate;
    client.rebuildModel(flatState_, hist);
    if (net_.shouldStop()) {
      out.shouldStop = true;
      out.note = "abort after rebuild";
      return out;
    }
  }

  // fit() works on a copy and reverts flatState_ (the server owns the true global trajectory).
  GradientScalars2D scalars = client.fit(model, flatState_, seeds, batch, cfg.config.mu);

  if (net_.shouldStop()) {  // check between the (blocking) fit and the upload
    out.shouldStop = true;
    out.note = "abort after fit (heartbeat death / server should_stop)";
    return out;
  }

  net_.submitGradientScalars(runId, clientId, cfg.currentRound, seeds, scalars, batch.numSamples);
  out.ranTraining = true;
  return out;
}

// MO-4: this round body is GATED OFF at the JS layer (runTrainingLoop throws MobileFedAvgUnsupportedError
// for a FedAvg-strategy run) and is therefore currently unreachable in production. The reason is the
// submit at the bottom: we upload ZO-SGD seeds + gradient SCALARS via submitGradientScalars (the DeComFL
// wire), but a server running the FedAvg *strategy* aggregates weight updates (SubmitModelUpdateStream)
// and cannot consume scalars — so this path would submit into a void. Kept intact so that wiring
// SubmitModelUpdateStream (+ server-side aggregation of a mobile weight blob) re-enables it by lifting
// the JS guard, not by rewriting the round.
RoundOutcome FederatedLoop::fedAvgRound(ExecutorchModel& model, const std::string& runId,
                                        const std::string& clientId, const DataBatch& batch,
                                        int numLocalSteps, double learningRate, double mu,
                                        int numPerturbations) {
  RoundOutcome out;
  if (net_.shouldStop()) {
    out.shouldStop = true;
    out.note = "abort flag set before round";
    return out;
  }

  int currentRound = 0;
  std::string blob = net_.getGlobalModelStream(runId, clientId, &currentRound);
  out.round = currentRound;
  mm_.loadStateDict(blob);            // codec-validated + sha-checked by the stream layer
  flatState_ = mm_.getFlatParams();   // fresh global params each FedAvg round

  // Local ZO-SGD: K steps, each averaging P forward-difference gradient estimates. We upload the
  // per-(k,p) seeds + g-scalars (Constraint 7), NOT a weight blob — the server reconstructs the
  // local trajectory from (seed -> z, g) exactly as in DeComFL. The per-step seed is derived
  // deterministically from (currentRound, k, p) so it is reproducible AND uploaded with the
  // scalars: seed = currentRound*1'000'003 + k*P + p (1'000'003 is a prime stride that keeps
  // distinct rounds' seed spaces from colliding for any realistic K*P).
  const int P = numPerturbations > 0 ? numPerturbations : 1;
  out.scalarsK = numLocalSteps;  // K/P actually used (for accurate comm-cost reporting)
  out.scalarsP = P;
  const int64_t d = static_cast<int64_t>(flatState_.size());

  Seeds2D seeds(static_cast<size_t>(numLocalSteps));
  GradientScalars2D scalars(static_cast<size_t>(numLocalSteps));

  for (int k = 0; k < numLocalSteps; ++k) {
    if (net_.shouldStop()) {
      out.shouldStop = true;
      out.note = "abort during local SGD";
      return out;
    }
    std::vector<float> delta(static_cast<size_t>(d), 0.0f);
    for (int p = 0; p < P; ++p) {
      const int64_t seed = static_cast<int64_t>(currentRound) * 1'000'003 +
                           static_cast<int64_t>(k) * P + p;
      const std::vector<float> z = flat_randn(seed, d);
      const double g = etGScalarForward(model, flatState_, z, mu, batch.inputs, batch.inputShape,
                                        batch.targets, batch.numSamples);
      for (int64_t i = 0; i < d; ++i) delta[i] += static_cast<float>(g) * z[i];
      seeds[static_cast<size_t>(k)].push_back(seed);
      scalars[static_cast<size_t>(k)].push_back(g);
    }
    const float step = static_cast<float>(learningRate / P);  // 1/P averaging, matches DeComFL
    for (int64_t i = 0; i < d; ++i) flatState_[i] -= step * delta[i];
  }
  // No model.train()/eval(): ExecuTorch kernels are stateless. etGScalarForward reads params from
  // the flatState_ we pass it, so the model state advances purely through flatState_; push the
  // final params back once for any downstream eval.
  mm_.setFlatParams(flatState_);

  if (net_.shouldStop()) {
    out.shouldStop = true;
    out.note = "abort after local SGD";
    return out;
  }

  net_.submitGradientScalars(runId, clientId, currentRound, seeds, scalars, batch.numSamples);
  out.ranTraining = true;
  return out;
}

#ifdef FEDLEARN_HAS_TRAINING
// M2: the TRUE first-order (FedAvg) round. Unlike fedAvgRound (ZO-SGD + scalar upload), this uses
// TrainableExecutorchModel's real backprop and uploads the resulting WEIGHT BLOB via submitModelUpdate
// — which is what a FedAvg-strategy server aggregates (SubmitModelUpdateStream), so this path lifts the
// mismatch that MO-4 gated the ZO fedAvgRound off for. ModelManager owns the (de)serialization; the
// compute is TrainableExecutorchModel. The endpoint is parity-tested against the framework's
// LocalTrainer.fit golden (fedavg_firstorder_round_test.cpp).
RoundOutcome FederatedLoop::firstOrderRound(TrainableExecutorchModel& model, const std::string& runId,
                                            const std::string& clientId, const DataBatch& batch,
                                            int numLocalSteps, double learningRate) {
  RoundOutcome out;
  if (net_.shouldStop()) {
    out.shouldStop = true;
    out.note = "abort flag set before round";
    return out;
  }

  int currentRound = 0;
  const std::string blob = net_.getGlobalModelStream(runId, clientId, &currentRound);
  out.round = currentRound;
  mm_.loadStateDict(blob);                    // codec-validated + sha-checked by the stream layer
  model.setFlatParams(mm_.getFlatParams());   // load the fresh global weights into the trainable model

  const float lr = static_cast<float>(learningRate);
  for (int k = 0; k < numLocalSteps; ++k) {
    if (net_.shouldStop()) {
      out.shouldStop = true;
      out.note = "abort during local SGD";
      return out;
    }
    model.trainStep(batch.inputs, batch.inputShape, batch.targets, batch.numSamples, lr);
  }

  mm_.setFlatParams(model.getFlatParams());   // updated (locally-advanced) weights back into the manager
  const std::string upload = mm_.serializeStateDict(batch.numSamples);

  if (net_.shouldStop()) {
    out.shouldStop = true;
    out.note = "abort after local SGD";
    return out;
  }

  net_.submitModelUpdate(runId, clientId, currentRound, upload, batch.numSamples);
  out.ranTraining = true;
  return out;
}
#endif  // FEDLEARN_HAS_TRAINING

}  // namespace fedlearn
