#include "fedlearn/FederatedLoop.h"

#include <stdexcept>
#include <string>

#include "fedlearn/DeComFLClient.h"
#include "fedlearn/ZerothOrderEstimator.h"

namespace fedlearn {
namespace {

std::string cfgGet(const ::google::protobuf::Map<std::string, std::string>& m,
                   const std::string& key, const std::string& fallback) {
  auto it = m.find(key);
  return it != m.end() ? it->second : fallback;
}

GradEstimateMethod parseMethod(const std::string& s) {
  return s == "central" ? GradEstimateMethod::Central : GradEstimateMethod::Forward;
}

}  // namespace

FederatedLoop::FederatedLoop(FedLearnClient& net, ModelManager& mm, std::string localTorchVersion,
                             bool allowVersionMismatch)
    : net_(net),
      mm_(mm),
      localTorchVersion_(std::move(localTorchVersion)),
      allowVersionMismatch_(allowVersionMismatch) {}

RoundOutcome FederatedLoop::deComFLRound(torch::jit::Module& model, const std::string& runId,
                                         const std::string& clientId, const DataBatch& batch) {
  RoundOutcome out;
  if (net_.shouldStop()) {
    out.shouldStop = true;
    out.note = "heartbeat/abort flag set before round";
    return out;
  }

  v2::GetDeComFLConfigResponse cfg = net_.getDeComFLConfig(runId, clientId);
  out.round = cfg.current_round();

  // E2: torch-version gate (RNG parity is undefined across torch versions).
  if (!cfg.torch_version().empty() && cfg.torch_version() != localTorchVersion_ &&
      !allowVersionMismatch_) {
    out.shouldStop = true;
    out.note = "torch_version mismatch: server '" + cfg.torch_version() + "' vs local '" +
               localTorchVersion_ + "' (refusing to train; RNG parity undefined)";
    return out;
  }

  const double eta = std::stod(cfgGet(cfg.config(), "lr", "0.001"));
  const double mu = std::stod(cfgGet(cfg.config(), "mu", "0.001"));
  const auto method = parseMethod(cfg.grad_estimate_method());

  Seeds2D seeds = FedLearnClient::fromProtoSeeds(cfg.current_seeds());
  if (seeds.empty() || seeds[0].empty()) {
    out.note = "empty seed set from server; skipping round";
    return out;
  }
  const int K = static_cast<int>(seeds.size());
  const int P = static_cast<int>(seeds[0].size());

  ZerothOrderEstimator zo(mm_, mu, method);
  DeComFLClient client(mm_, zo, eta, P, K);

  // Replay any rounds this client missed (Algorithm 2), using config lr for each round.
  if (cfg.rebuild_history().rounds_size() > 0) {
    RebuildHistory hist = FedLearnClient::fromProtoRebuildHistory(cfg.rebuild_history());
    for (auto& r : hist) r.learningRate = eta;
    client.rebuildModel(model, hist);
    if (net_.shouldStop()) {
      out.shouldStop = true;
      out.note = "abort after rebuild";
      return out;
    }
  }

  GradientScalars2D scalars = client.fit(model, seeds, batch);

  if (net_.shouldStop()) {  // E4: check between the (blocking) fit and the upload
    out.shouldStop = true;
    out.note = "abort after fit (heartbeat death / server should_stop)";
    return out;
  }

  net_.submitGradientScalars(runId, clientId, cfg.current_round(), scalars, batch.targets.numel());
  out.ranTraining = true;
  return out;
}

RoundOutcome FederatedLoop::fedAvgRound(torch::jit::Module& model, const std::string& runId,
                                        const std::string& clientId, const DataBatch& batch,
                                        int numLocalSteps, double learningRate) {
  RoundOutcome out;
  if (net_.shouldStop()) {
    out.shouldStop = true;
    out.note = "abort flag set before round";
    return out;
  }

  int currentRound = 0;
  std::string blob = net_.getGlobalModelStream(runId, clientId, &currentRound);
  out.round = currentRound;
  mm_.loadStateDict(model, blob);  // codec-validated + sha-checked by the stream layer

  // Local supervised SGD: K steps of cross-entropy + manual gradient descent on the trainable params.
  model.train();
  for (int step = 0; step < numLocalSteps; ++step) {
    if (net_.shouldStop()) {
      out.shouldStop = true;
      out.note = "abort during local SGD";
      return out;
    }
    torch::Tensor flat = mm_.getFlatParams(model).clone().set_requires_grad(true);
    mm_.setFlatParams(model, flat);
    std::vector<torch::jit::IValue> in{batch.inputs};
    torch::Tensor logits = model.forward(in).toTensor();
    torch::Tensor loss = torch::nn::functional::cross_entropy(logits, batch.targets);
    loss.backward();
    {
      torch::NoGradGuard ng;
      torch::Tensor updated = flat - learningRate * flat.grad();
      mm_.setFlatParams(model, updated.detach());
    }
  }
  model.eval();

  std::string updateBlob = mm_.serializeStateDict(model, batch.targets.numel());
  net_.submitModelUpdateStream(runId, clientId, currentRound, updateBlob, batch.targets.numel());
  out.ranTraining = true;
  return out;
}

}  // namespace fedlearn
