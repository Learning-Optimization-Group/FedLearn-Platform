#include "fedlearn/DeComFLClient.h"

#include "fedlearn/RandnEngine.h"

namespace fedlearn {

DeComFLClient::DeComFLClient(double eta, int P, int K) : eta_(eta), P_(P), K_(K) {}

GradientScalars2D DeComFLClient::fit(ExecutorchModel& model, const std::vector<float>& flatIn,
                                     const Seeds2D& seeds, const DataBatch& batch, double mu) {
  std::vector<float> x = flatIn;  // local working copy; flatIn is left untouched (implicit revert)
  const int64_t d = static_cast<int64_t>(x.size());

  GradientScalars2D out(K_, std::vector<double>(P_, 0.0));
  for (int k = 0; k < K_; ++k) {
    std::vector<float> delta(static_cast<size_t>(d), 0.0f);  // float32, matching the prior path
    for (int p = 0; p < P_; ++p) {
      const int64_t s = seeds[k][p];
      const std::vector<float> z = flat_randn(s, d);  // same canonical z the g-scalar uses
      const double g = etGScalarForward(model, x, z, mu, batch.inputs, batch.inputShape,
                                        batch.targets, batch.numSamples);
      out[k][p] = g;
      for (int64_t i = 0; i < d; ++i) delta[i] += static_cast<float>(g) * z[i];
    }
    // Local step with the 1/P averaging factor (matches the server after the Bug-1 fix).
    const float step = static_cast<float>(eta_ / P_);
    for (int64_t i = 0; i < d; ++i) x[i] -= step * delta[i];
  }
  return out;  // scalars only — z is never serialized or uploaded; flatIn unchanged
}

void DeComFLClient::rebuildModel(std::vector<float>& flat, const RebuildHistory& history) {
  const int64_t d = static_cast<int64_t>(flat.size());
  for (const RebuildRound& round : history) {
    const int K = static_cast<int>(round.seeds.size());
    for (int k = 0; k < K; ++k) {
      const int P = static_cast<int>(round.seeds[k].size());
      std::vector<float> delta(static_cast<size_t>(d), 0.0f);
      for (int p = 0; p < P; ++p) {
        const std::vector<float> z = flat_randn(round.seeds[k][p], d);
        const float g = static_cast<float>(round.gradients[k][p]);
        for (int64_t i = 0; i < d; ++i) delta[i] += g * z[i];
      }
      const float step = static_cast<float>(round.learningRate / P);
      for (int64_t i = 0; i < d; ++i) flat[i] -= step * delta[i];
    }
  }
}

}  // namespace fedlearn
