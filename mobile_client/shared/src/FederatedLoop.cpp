#include "fedlearn/FederatedLoop.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "fedlearn/BatchOrder.h"
#include "fedlearn/DeComFLClient.h"
#include "fedlearn/EtZeroOrder.h"
#include "fedlearn/RandnEngine.h"

namespace fedlearn {
namespace {

bool hasEdgeWhitespace(const std::string& raw) {
  return raw.empty() || std::isspace(static_cast<unsigned char>(raw.front())) ||
         std::isspace(static_cast<unsigned char>(raw.back()));
}

double parseFiniteSetting(const std::string& raw, const char* name, bool allowZero = false) {
  if (hasEdgeWhitespace(raw)) throw std::runtime_error(std::string("invalid first-order ") + name);
  size_t consumed = 0;
  double value;
  try {
    value = std::stod(raw, &consumed);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid first-order ") + name);
  }
  if (consumed != raw.size() || !std::isfinite(value) || value < 0 || (!allowZero && value == 0)) {
    throw std::runtime_error(std::string("invalid first-order ") + name);
  }
  return value;
}

int parseLocalEpochs(const std::string& raw) {
  if (hasEdgeWhitespace(raw)) throw std::runtime_error("invalid first-order local_epochs");
  size_t consumed = 0;
  long long value;
  try {
    value = std::stoll(raw, &consumed);
  } catch (const std::exception&) {
    throw std::runtime_error("invalid first-order local_epochs");
  }
  if (consumed != raw.size() || value <= 0 || value > 100000) {
    throw std::runtime_error("invalid first-order local_epochs");
  }
  return static_cast<int>(value);
}

}  // namespace

FederatedLoop::FederatedLoop(IFedLearnClient& net, ModelManager& mm) : net_(net), mm_(mm) {}

RoundOutcome FederatedLoop::deComFLRound(ExecutorchModel& model, const std::string& runId,
                                         const std::string& clientId, const DataBatch& batch,
                                         const ZerothOrderContract* contract) {
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
  if (contract != nullptr) {
    // The run's execution contract states the zeroth-order training; the server's round config is a cross-check.
    // Anything it asks for that the contract does not state -- or leaves out -- refuses the round before upload.
    if (!cfg.learningRateSent || cfg.config.learningRate != contract->learningRate) {
      throw std::runtime_error("the server's learning_rate disagrees with the execution contract");
    }
    if (!cfg.muSent || cfg.config.mu != contract->smoothing) {
      throw std::runtime_error("the server's smoothing_param disagrees with the execution contract");
    }
    if (K != contract->numLocalSteps) {
      throw std::runtime_error("the server's local steps disagree with the execution contract");
    }
    for (const auto& step : seeds) {
      if (static_cast<int>(step.size()) != contract->numPerturbations) {
        throw std::runtime_error("the server's perturbations disagree with the execution contract");
      }
    }
    if (cfg.config.method != contract->method) {
      throw std::runtime_error("the server's gradient estimator disagrees with the execution contract");
    }
  }
  out.scalarsK = K;  // server-authoritative K/P actually used (for accurate comm-cost reporting)
  out.scalarsP = P;

  if (contract != nullptr && flatState_.empty()) {
    // The DeComFL path never downloads the global model: it starts from a state and follows the aggregates from
    // there. That state used to be the ModelManager's zero-initialised params, so every scalar was a derivative at
    // the wrong point. Start from the server's model instead -- the run's initial model, which only round 1 serves --
    // and only once it is proven to be the one the contract binds.
    if (cfg.currentRound != 1) {
      throw std::runtime_error("this device can join a DeComFL run only at its first round, when the server serves "
                               "the run's initial model");
    }
    int served = 0;
    const std::string initial = net_.getGlobalModelStream(runId, clientId, &served, nullptr);
    mm_.loadStateDict(initial);
    if (mm_.canonicalStateSha256() != contract->initialStateSha256) {
      throw std::runtime_error("the server's initial model is not the one the execution contract binds");
    }
    flatState_ = mm_.getFlatParams();
  }
  // Lazily snapshot the global params into the loop's owned working state (a round held to no contract).
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

  // Evaluation reports the model this round trains from, not a state the DeComFL path never updates.
  mm_.setFlatParams(flatState_);

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
                                            int numLocalSteps, double learningRate,
                                            bool requireServerConfig, double proximalMu,
                                            const LocalBatching& batching) {
  RoundOutcome out;
  if (batching.batchSize < 0 || (batching.seededPermutation && batching.batchSize == 0)) {
    throw std::runtime_error("invalid first-order batch_size");
  }
  if (batch.numSamples < 1 || batch.inputShape.empty() || batch.inputShape[0] != batch.numSamples) {
    throw std::runtime_error("the local dataset's shape disagrees with its example count");
  }
  if (net_.shouldStop()) {
    out.shouldStop = true;
    out.note = "abort flag set before round";
    return out;
  }

  int currentRound = 0;
  std::map<std::string, std::string> serverConfig;
  const std::string blob = net_.getGlobalModelStream(runId, clientId, &currentRound, &serverConfig);
  out.round = currentRound;
  const auto rate = serverConfig.find("learning_rate");
  const auto epochs = serverConfig.find("local_epochs");
  if (requireServerConfig && (rate == serverConfig.end() || epochs == serverConfig.end())) {
    throw std::runtime_error("first-order strategy requires server learning_rate and local_epochs");
  }
  if ((rate == serverConfig.end()) != (epochs == serverConfig.end())) {
    throw std::runtime_error("incomplete first-order server settings");
  }
  // learningRate and numLocalSteps are the run's execution contract. Server-sent values are a cross-check, never an
  // override: a server asking for other training refuses the round before anything is uploaded.
  if (!std::isfinite(learningRate) || learningRate <= 0) {
    throw std::runtime_error("invalid first-order learning_rate");
  }
  if (numLocalSteps <= 0 || numLocalSteps > 100000) {
    throw std::runtime_error("invalid first-order local_epochs");
  }
  if (rate != serverConfig.end()) {
    if (parseFiniteSetting(rate->second, "learning_rate") != learningRate) {
      throw std::runtime_error("the server's learning_rate " + rate->second +
                               " disagrees with the execution contract");
    }
    if (parseLocalEpochs(epochs->second) != numLocalSteps) {
      throw std::runtime_error("the server's local_epochs " + epochs->second +
                               " disagrees with the execution contract");
    }
  }
  // The proximal coefficient is a cross-check too. Absent means zero on both sides: only a FedProx contract states one.
  if (!std::isfinite(proximalMu) || proximalMu < 0) {
    throw std::runtime_error("invalid first-order proximal_mu");
  }
  const auto proximal = serverConfig.find("proximal_mu");
  const double serverMu =
      proximal == serverConfig.end() ? 0.0 : parseFiniteSetting(proximal->second, "proximal_mu", true);
  if (serverMu != proximalMu) {
    throw std::runtime_error("the server's proximal_mu " +
                             (proximal == serverConfig.end() ? std::string("(none)") : proximal->second) +
                             " disagrees with the execution contract");
  }
  mm_.loadStateDict(blob);                    // codec-validated + sha-checked by the stream layer
  model.setFlatParams(mm_.getFlatParams());   // load the fresh global weights into the trainable model
  // FedProx's anchor: the round's downloaded global weights, fixed for the whole round.
  const std::vector<float> anchor = proximalMu > 0 ? mm_.getFlatParams() : std::vector<float>{};

  const float lr = static_cast<float>(learningRate);
  const float mu = static_cast<float>(proximalMu);
  // One example's feature count: the product of the input shape past the example dimension.
  int64_t rowWidth = 1;
  for (size_t d = 1; d < batch.inputShape.size(); ++d) rowWidth *= batch.inputShape[d];
  std::vector<float> xs;
  std::vector<int64_t> ys;
  for (int k = 0; k < numLocalSteps; ++k) {
    if (!batching.seededPermutation) {
      if (net_.shouldStop()) {
        out.shouldStop = true;
        out.note = "abort during local SGD";
        return out;
      }
      model.trainStep(batch.inputs, batch.inputShape, batch.targets, batch.numSamples, lr,
                      proximalMu > 0 ? &anchor : nullptr, mu);
      continue;
    }
    // Epoch k: gather each minibatch's rows, in the contract's order, into one contiguous buffer.
    const auto order = seededPermutation(static_cast<uint64_t>(batch.numSamples), batching.seed,
                                         static_cast<uint64_t>(currentRound), static_cast<uint64_t>(k));
    for (const auto& rows : batches(order, static_cast<uint64_t>(batching.batchSize), /*dropLast=*/false)) {
      if (net_.shouldStop()) {
        out.shouldStop = true;
        out.note = "abort during local SGD";
        return out;
      }
      const auto n = static_cast<int64_t>(rows.size());
      xs.resize(static_cast<size_t>(n * rowWidth));
      ys.resize(static_cast<size_t>(n));
      for (int64_t r = 0; r < n; ++r) {
        const auto src = static_cast<int64_t>(rows[static_cast<size_t>(r)]);
        std::memcpy(xs.data() + r * rowWidth, batch.inputs + src * rowWidth,
                    static_cast<size_t>(rowWidth) * sizeof(float));
        ys[static_cast<size_t>(r)] = batch.targets[src];
      }
      std::vector<int64_t> shape = batch.inputShape;
      shape[0] = n;
      model.trainStep(xs.data(), shape, ys.data(), n, lr, proximalMu > 0 ? &anchor : nullptr, mu);
    }
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
