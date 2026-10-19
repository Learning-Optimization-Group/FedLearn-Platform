#include "fedlearn/FederatedLoop.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "fedlearn/BatchOrder.h"
#include "fedlearn/DeComFLClient.h"
#include "fedlearn/DropoutMasks.h"
#include "fedlearn/EtZeroOrder.h"
#include "fedlearn/RandnEngine.h"
#include "fedlearn/TrainingStateError.h"

namespace fedlearn {
namespace {

// This scope reports a failed attempt on exceptions and never treats completed
// local computation as a successful submission. All numeric training stays outside it.
class RoundProgress {
 public:
  explicit RoundProgress(IFedLearnClient& net) : net_(net), round_(net.progressSnapshot().round) {
    phase("DOWNLOADING");
  }
  ~RoundProgress() {
    if (!finished_) {
      try { phase("ERROR"); } catch (...) { /* preserve the original failure */ }
    }
  }
  void configure(int round, int64_t total, const char* status = "TRAINING") {
    round_ = round;
    total_ = total;
    completed_ = 0;
    phase(status);
  }
  void advance(int64_t completed) { completed_ = completed; phase("TRAINING"); }
  void phase(const char* status) { net_.updateProgress(status, round_, completed_, total_); }
  void finish(const char* status) { phase(status); finished_ = true; }
 private:
  IFedLearnClient& net_;
  int round_ = 0;
  int64_t total_ = 0;
  int64_t completed_ = 0;
  bool finished_ = false;
};

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

FederatedLoop::FederatedLoop(IFedLearnClient& net, ModelManager& mm)
    : net_(net), mm_(mm), deComFLState_(ownedDeComFLState_) {}

FederatedLoop::FederatedLoop(IFedLearnClient& net, ModelManager& mm, DeComFLState& state)
    : net_(net), mm_(mm), deComFLState_(state) {}

RoundOutcome FederatedLoop::deComFLRound(ExecutorchModel& model, const std::string& runId,
                                         const std::string& clientId, const DataBatch& batch,
                                         const ZerothOrderContract* contract) {
  RoundProgress progress(net_);
  RoundOutcome out;
  if (net_.shouldStop()) {
    progress.finish("STOPPED");
    out.shouldStop = true;
    out.note = "heartbeat/abort flag set before round";
    return out;
  }

  DeComFLConfig cfg = net_.getDeComFLConfig(runId, clientId);
  out.round = cfg.currentRound;
  progress.configure(cfg.currentRound, 0, "DOWNLOADING");
  // Server-authoritative completion: the config path no longer carries should_stop (that lives on the
  // heartbeat), so the server signals "training complete" with current_round == -1 (matches the Python
  // client's decomfl_start sentinel). Terminate cleanly.
  if (cfg.currentRound < 0) {
    progress.finish("STOPPED");
    out.shouldStop = true;
    out.note = "training complete (server signalled round -1)";
    return out;
  }
  if (cfg.shouldStop) {
    progress.finish("STOPPED");
    out.shouldStop = true;
    out.note = "server should_stop in DeComFL config";
    return out;
  }

  // NO torch_version gate: RandnEngine makes the perturbation RNG version-independent.

  const Seeds2D& seeds = cfg.seeds;
  if (seeds.empty() || seeds[0].empty()) {
    progress.finish("WAITING");
    out.note = "empty seed set from server; skipping round";
    return out;
  }
  const int K = static_cast<int>(seeds.size());
  const int P = static_cast<int>(seeds[0].size());
  if (contract != nullptr) {
    // The run's execution contract states the zeroth-order training; the server's round config is a cross-check.
    // Anything it asks for that the contract does not state -- or leaves out -- refuses the round before upload.
    if (!cfg.learningRateSent || cfg.config.learningRate != contract->learningRate) {
      throw TrainingStateError("the server's learning_rate disagrees with the execution contract");
    }
    if (!cfg.muSent || cfg.config.mu != contract->smoothing) {
      throw TrainingStateError("the server's smoothing_param disagrees with the execution contract");
    }
    if (K != contract->numLocalSteps) {
      throw TrainingStateError("the server's local steps disagree with the execution contract");
    }
    for (const auto& step : seeds) {
      if (static_cast<int>(step.size()) != contract->numPerturbations) {
        throw TrainingStateError("the server's perturbations disagree with the execution contract");
      }
    }
    if (cfg.config.method != contract->method) {
      throw TrainingStateError("the server's gradient estimator disagrees with the execution contract");
    }
  }
  progress.configure(cfg.currentRound, K, "DOWNLOADING");
  out.scalarsK = K;  // server-authoritative K/P actually used (for accurate comm-cost reporting)
  out.scalarsP = P;

  auto& trajectory = deComFLState_;
  if (contract != nullptr && trajectory.bound) {
    const auto& previous = trajectory.contract;
    if (trajectory.runId != runId || trajectory.clientId != clientId ||
        previous.initialStateSha256 != contract->initialStateSha256 ||
        previous.learningRate != contract->learningRate || previous.smoothing != contract->smoothing ||
        previous.numLocalSteps != contract->numLocalSteps ||
        previous.numPerturbations != contract->numPerturbations || previous.method != contract->method) {
      trajectory.reset();
    }
  }
  auto& flat = contract != nullptr ? trajectory.flat : flatState_;
  if (contract != nullptr && !trajectory.bound) {
    // A cold client adopts the server's current global state, which already
    // contains every update before its served round. Older history must not be
    // replayed on top of that snapshot. Retained in-process trajectories still
    // use the history path below without another model download.
    int served = 0;
    const std::string snapshot = net_.getGlobalModelStream(runId, clientId, &served, nullptr);
    if (served < 1) {
      throw TrainingStateError("DeComFL snapshot is missing a valid served round");
    }
    if (served != cfg.currentRound) {
      // A concurrent aggregation can move the model beyond the config we fetched.
      // Leave both model and trajectory unchanged; the outer loop retries config.
      throw std::runtime_error("DeComFL round changed during snapshot download; retry configuration");
    }
    DeComFLState next;
    try {
      next.flat = mm_.decodeStateDict(snapshot);
      if (served == 1 && mm_.canonicalStateSha256(next.flat) != contract->initialStateSha256) {
        throw TrainingStateError("the server's initial model is not the one the execution contract binds");
      }
    } catch (const std::exception& e) {
      throw TrainingStateError(std::string("invalid DeComFL snapshot: ") + e.what());
    }
    next.syncedThrough = served - 1;
    next.runId = runId;
    next.clientId = clientId;
    next.contract = *contract;
    next.bound = true;
    trajectory = std::move(next);
  }
  // Lazily snapshot the global params into the loop's owned working state (a round held to no contract).
  if (flat.empty()) flat = mm_.getFlatParams();

  DeComFLClient client(cfg.config.learningRate, P, K);

  // Replay any rounds this client missed (Algorithm 2), using config lr for each round.
  if (contract != nullptr) {
    // The server's last-client-round advances on aggregation, whereas this state
    // advances before submission. A retry can therefore repeat an applied prefix.
    // Validate the full chain first and build on a copy, so a torn or malformed
    // response cannot partially commit either the weights or the round cursor.
    if (cfg.currentRound <= trajectory.syncedThrough) {
      throw TrainingStateError("DeComFL server round precedes the retained trajectory");
    }
    RebuildHistory pending;
    int previousRound = 0;
    int next = trajectory.syncedThrough + 1;
    for (const auto& round : cfg.rebuildHistory) {
      if (round.roundNumber <= previousRound || round.roundNumber >= cfg.currentRound) {
        throw TrainingStateError("DeComFL rebuild history has unordered or invalid round numbers");
      }
      previousRound = round.roundNumber;
      if (round.roundNumber <= trajectory.syncedThrough) continue;
      if (round.roundNumber != next || static_cast<int>(round.seeds.size()) != K ||
          static_cast<int>(round.gradients.size()) != K) {
        throw TrainingStateError("DeComFL rebuild history is incomplete or has invalid local steps");
      }
      for (int k = 0; k < K; ++k) {
        if (static_cast<int>(round.seeds[k].size()) != P ||
            static_cast<int>(round.gradients[k].size()) != P) {
          throw TrainingStateError("DeComFL rebuild history has invalid perturbations");
        }
        for (double value : round.gradients[k]) {
          if (!std::isfinite(value) || !std::isfinite(static_cast<float>(value))) {
            throw TrainingStateError("DeComFL rebuild history has non-finite gradients");
          }
        }
      }
      pending.push_back(round);
      pending.back().learningRate = cfg.config.learningRate;
      ++next;
    }
    if (next != cfg.currentRound) {
      throw TrainingStateError("DeComFL rebuild history is missing a completed round");
    }
    if (!pending.empty()) {
      progress.phase("REBUILDING");
      auto rebuilt = flat;
      client.rebuildModel(rebuilt, pending);
      for (float value : rebuilt) {
        if (!std::isfinite(value)) throw TrainingStateError("DeComFL rebuilt model is non-finite");
      }
      flat = std::move(rebuilt);
      trajectory.syncedThrough = cfg.currentRound - 1;
    }
  } else if (!cfg.rebuildHistory.empty()) {
    RebuildHistory hist = cfg.rebuildHistory;
    for (auto& r : hist) r.learningRate = cfg.config.learningRate;
    progress.phase("REBUILDING");
    client.rebuildModel(flat, hist);
  }
  if (!cfg.rebuildHistory.empty()) {
    if (net_.shouldStop()) {
      progress.finish("STOPPED");
      out.shouldStop = true;
      out.note = "abort after rebuild";
      return out;
    }
  }

  // Evaluation reports the model this round trains from, not a state the DeComFL path never updates.
  mm_.setFlatParams(flat);

  progress.phase("TRAINING");
  // fit() works on a copy and reverts flatState_ (the server owns the true global trajectory).
  GradientScalars2D scalars = client.fit(model, flat, seeds, batch, cfg.config.mu,
      [&](int completed) { progress.advance(completed); });

  if (net_.shouldStop()) {  // check between the (blocking) fit and the upload
    progress.finish("STOPPED");
    out.shouldStop = true;
    out.note = "abort after fit (heartbeat death / server should_stop)";
    return out;
  }

  progress.phase("UPLOADING");
  net_.submitGradientScalars(runId, clientId, cfg.currentRound, seeds, scalars, batch.numSamples);
  progress.finish("WAITING");
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
  RoundProgress progress(net_);
  RoundOutcome out;
  if (net_.shouldStop()) {
    progress.finish("STOPPED");
    out.shouldStop = true;
    out.note = "abort flag set before round";
    return out;
  }

  int currentRound = 0;
  std::string blob = net_.getGlobalModelStream(runId, clientId, &currentRound);
  out.round = currentRound;
  progress.configure(currentRound, numLocalSteps);
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
      progress.finish("STOPPED");
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
    progress.advance(k + 1);
  }
  // No model.train()/eval(): ExecuTorch kernels are stateless. etGScalarForward reads params from
  // the flatState_ we pass it, so the model state advances purely through flatState_; push the
  // final params back once for any downstream eval.
  mm_.setFlatParams(flatState_);

  if (net_.shouldStop()) {
    progress.finish("STOPPED");
    out.shouldStop = true;
    out.note = "abort after local SGD";
    return out;
  }

  progress.phase("UPLOADING");
  net_.submitGradientScalars(runId, clientId, currentRound, seeds, scalars, batch.numSamples);
  progress.finish("WAITING");
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
                                            const LocalBatching& batching, const LocalOptimizer& optimizer,
                                            const DropoutSpec& dropout, const BatchNormSpec& batchNorm) {
  RoundProgress progress(net_);
  RoundOutcome out;
  if (batching.batchSize < 0 || (batching.seededPermutation && batching.batchSize == 0)) {
    throw std::runtime_error("invalid first-order batch_size");
  }
  if (batch.numSamples < 1 || batch.inputShape.empty() || batch.inputShape[0] != batch.numSamples) {
    throw std::runtime_error("the local dataset's shape disagrees with its example count");
  }
  if (net_.shouldStop()) {
    progress.finish("STOPPED");
    out.shouldStop = true;
    out.note = "abort flag set before round";
    return out;
  }

  int currentRound = 0;
  std::map<std::string, std::string> serverConfig;
  const std::string blob = net_.getGlobalModelStream(runId, clientId, &currentRound, &serverConfig);
  out.round = currentRound;
  progress.configure(currentRound, 0, "DOWNLOADING");
  out.downlinkBytes = static_cast<int64_t>(blob.size());
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
  // The federated state is the trainable parameters, then (training-mode BatchNorm) each layer's running mean and
  // running variance. The program holds the parameters; the round holds and updates the running statistics.
  const std::vector<float> global = mm_.getFlatParams();
  const auto params = static_cast<size_t>(model.flatDim());
  size_t statCount = 0;
  for (const auto& layer : batchNorm.layers) statCount += 2 * static_cast<size_t>(layer.features);
  if (global.size() != params + statCount) {
    throw std::runtime_error("the federated state holds " + std::to_string(global.size()) + " values, but the program "
                             "trains " + std::to_string(params) + " and the contract's BatchNorm layers carry " +
                             std::to_string(statCount) + " running statistics");
  }
  std::vector<double> runningStats(global.begin() + static_cast<std::ptrdiff_t>(params), global.end());
  model.setFlatParams(std::vector<float>(global.begin(), global.begin() + static_cast<std::ptrdiff_t>(params)));
  // FedProx's anchor: the round's downloaded global weights, fixed for the whole round.
  const std::vector<float> anchor =
      proximalMu > 0 ? std::vector<float>(global.begin(), global.begin() + static_cast<std::ptrdiff_t>(params))
                     : std::vector<float>{};

  const float lr = static_cast<float>(learningRate);
  const float mu = static_cast<float>(proximalMu);
  if (optimizer.adam) model.resetOptimizerState();  // the laptop creates a fresh Adam every round
  const AdamSettings adam{learningRate, optimizer.beta1, optimizer.beta2, optimizer.epsilon};
  // Each dropout layer's mask input shape, from the program: [batch, activation dims...].
  std::vector<std::vector<int64_t>> maskShapes;
  if (!dropout.rates.empty()) {
    maskShapes = model.extraInputShapes();
    if (maskShapes.size() != dropout.rates.size()) {
      throw std::runtime_error("the training program takes " + std::to_string(maskShapes.size()) +
                               " dropout masks, but the contract states " + std::to_string(dropout.rates.size()) +
                               " dropout layers");
    }
  }
  int64_t totalSteps = numLocalSteps;
  if (batching.seededPermutation) {
    const int64_t perEpoch = 1 + (batch.numSamples - 1) / batching.batchSize;
    totalSteps = perEpoch > std::numeric_limits<int64_t>::max() / numLocalSteps
        ? 0 : perEpoch * numLocalSteps;
  }
  progress.configure(currentRound, totalSteps);
  uint64_t step = 0;
  std::vector<std::vector<float>> maskData(maskShapes.size());
  std::vector<InputTensor> masks(maskShapes.size());
  // One local SGD step: the contract's optimizer, with this step's dropout masks when the model has dropout.
  const auto trainOne = [&](const float* stepX, const std::vector<int64_t>& stepShape, const int64_t* stepY,
                            int64_t rows) {
    for (size_t l = 0; l < maskShapes.size(); ++l) {
      int64_t perExample = 1;
      for (size_t d = 1; d < maskShapes[l].size(); ++d) perExample *= maskShapes[l][d];
      maskData[l] = dropoutMask(static_cast<uint64_t>(rows * perExample), dropout.rates[l], dropout.seed,
                                static_cast<uint64_t>(currentRound), step, l);
      masks[l].data = maskData[l].data();
      masks[l].shape = maskShapes[l];
      masks[l].shape[0] = rows;
    }
    const std::vector<InputTensor>* extra = masks.empty() ? nullptr : &masks;
    if (optimizer.adam) {
      model.trainStepAdam(stepX, stepShape, stepY, rows, adam, extra, proximalMu > 0 ? &anchor : nullptr, mu);
    } else {
      model.trainStep(stepX, stepShape, stepY, rows, lr, proximalMu > 0 ? &anchor : nullptr, mu, extra);
    }
    if (!batchNorm.layers.empty()) {
      // BATCH_NORM_RUNNING_STATS_V1: each layer's batch mean and biased variance, in float64, rounded once below.
      const auto& stats = model.lastExtraOutputs();
      if (stats.size() != 2 * batchNorm.layers.size()) {
        throw std::runtime_error("the training program reports " + std::to_string(stats.size() / 2) +
                                 " BatchNorm layers, but the contract states " +
                                 std::to_string(batchNorm.layers.size()));
      }
      size_t off = 0;
      for (size_t l = 0; l < batchNorm.layers.size(); ++l) {
        const auto& layer = batchNorm.layers[l];
        const auto c = static_cast<size_t>(layer.features);
        const double n = static_cast<double>(rows) * static_cast<double>(layer.valuesPerExample);
        if (stats[2 * l].size() != c || stats[2 * l + 1].size() != c || n < 2) {
          throw std::runtime_error("BatchNorm layer " + std::to_string(l) + " reports statistics the contract does "
                                   "not describe");
        }
        const double m = layer.momentum;
        for (size_t k = 0; k < c; ++k) {
          double& mean = runningStats[off + k];
          double& var = runningStats[off + c + k];
          mean = static_cast<double>(static_cast<float>((1.0 - m) * mean + m * static_cast<double>(stats[2 * l][k])));
          var = static_cast<double>(static_cast<float>(
              (1.0 - m) * var + m * static_cast<double>(stats[2 * l + 1][k]) * (n / (n - 1.0))));
        }
        off += 2 * c;
      }
    }
    ++step;
    progress.advance(static_cast<int64_t>(step));
  };
  // One example's feature count: the product of the input shape past the example dimension.
  int64_t rowWidth = 1;
  for (size_t d = 1; d < batch.inputShape.size(); ++d) rowWidth *= batch.inputShape[d];
  std::vector<float> xs;
  std::vector<int64_t> ys;
  for (int k = 0; k < numLocalSteps; ++k) {
    if (!batching.seededPermutation) {
      if (net_.shouldStop()) {
        progress.finish("STOPPED");
        out.shouldStop = true;
        out.note = "abort during local SGD";
        return out;
      }
      trainOne(batch.inputs, batch.inputShape, batch.targets, batch.numSamples);
      continue;
    }
    // Epoch k: gather each minibatch's rows, in the contract's order, into one contiguous buffer.
    const auto order = seededPermutation(static_cast<uint64_t>(batch.numSamples), batching.seed,
                                         static_cast<uint64_t>(currentRound), static_cast<uint64_t>(k));
    for (const auto& rows : batches(order, static_cast<uint64_t>(batching.batchSize), /*dropLast=*/false)) {
      if (net_.shouldStop()) {
        progress.finish("STOPPED");
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
      trainOne(xs.data(), shape, ys.data(), n);
    }
  }

  // The updated (locally-advanced) weights, then the updated running statistics, back into the manager.
  std::vector<float> state = model.getFlatParams();
  for (double v : runningStats) state.push_back(static_cast<float>(v));
  mm_.setFlatParams(state);
  const std::string upload = mm_.serializeStateDict(batch.numSamples);

  if (net_.shouldStop()) {
    progress.finish("STOPPED");
    out.shouldStop = true;
    out.note = "abort after local SGD";
    return out;
  }

  progress.phase("UPLOADING");
  net_.submitModelUpdate(runId, clientId, currentRound, upload, batch.numSamples);
  progress.finish("WAITING");
  out.uplinkBytes = static_cast<int64_t>(upload.size());
  out.ranTraining = true;
  return out;
}
#endif  // FEDLEARN_HAS_TRAINING

}  // namespace fedlearn
