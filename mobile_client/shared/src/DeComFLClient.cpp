#include "fedlearn/DeComFLClient.h"

#include "fedlearn/Perturbation.h"

namespace fedlearn {

DeComFLClient::DeComFLClient(ModelManager& mm, ZerothOrderEstimator& zo, double eta, int P, int K)
    : mm_(mm), zo_(zo), eta_(eta), P_(P), K_(K) {}

GradientScalars2D DeComFLClient::fit(torch::jit::Module& model,
                                     const Seeds2D& seeds,
                                     const DataBatch& batch) {
  // Snapshot for exact revert (B1-M1: snapshot-restore, NOT running-sum subtraction).
  torch::Tensor x_initial = mm_.getFlatParams(model).clone();
  torch::Tensor x_current = x_initial.clone();
  const int64_t d = x_current.numel();

  GradientScalars2D out(K_, std::vector<double>(P_, 0.0));
  for (int k = 0; k < K_; ++k) {
    torch::Tensor delta = torch::zeros_like(x_current);
    for (int p = 0; p < P_; ++p) {
      const int64_t s = seeds[k][p];
      // Same canonical z the estimator uses (deterministic from the seed).
      torch::Tensor z = canonical_perturbation(s, d, torch::kFloat32).to(x_current.device());
      const double g = zo_.computeGradientScalar(model, x_current, s, batch.inputs, batch.targets);
      out[k][p] = g;
      delta += g * z;  // delta += g_p * z_p
    }
    // Local step with the 1/P averaging factor (matches the server after the Bug-1 fix).
    x_current = x_current - (eta_ / P_) * delta;
    mm_.setFlatParams(model, x_current);
  }

  // Exact revert: client returns to pre-round state; the server advances the global model.
  mm_.setFlatParams(model, x_initial);
  return out;  // scalars only — z is never serialized or uploaded
}

void DeComFLClient::rebuildModel(torch::jit::Module& model, const RebuildHistory& history) {
  for (const RebuildRound& round : history) {
    torch::Tensor x = mm_.getFlatParams(model);
    const int64_t d = x.numel();
    const int K = static_cast<int>(round.seeds.size());
    for (int k = 0; k < K; ++k) {
      const int P = static_cast<int>(round.seeds[k].size());
      torch::Tensor delta = torch::zeros_like(x);
      for (int p = 0; p < P; ++p) {
        torch::Tensor z = canonical_perturbation(round.seeds[k][p], d, torch::kFloat32).to(x.device());
        delta += round.gradients[k][p] * z;
      }
      x = x - (round.learningRate / P) * delta;
    }
    mm_.setFlatParams(model, x);
  }
}

}  // namespace fedlearn
