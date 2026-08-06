#pragma once
//
// DeComFLClient.h — on-device DeComFL client: one round (fit) + missed-round replay
// (rebuildModel). 15-LLD-mobile.md §5.2, §6.2; mirrors the DeComFL correctness spec.
//
// Libtorch-free (Phase 3c): parameters are std::vector<float>, the forward/loss runs through
// ExecutorchModel, and the perturbation is fedlearn::flat_randn (byte-exact canonical z).
//
#include <vector>

#include "fedlearn/EtZeroOrder.h"
#include "fedlearn/ExecutorchModel.h"
#include "fedlearn/RandnEngine.h"
#include "fedlearn/Types.h"

namespace fedlearn {

class DeComFLClient {
 public:
  // mu is per-round server config, so it is passed to fit() (not the ctor).
  DeComFLClient(double eta, int P, int K);

  // One DeComFL round. `flatIn` is the current global params — it is NOT modified (the client
  // reverts; the server owns the true global trajectory). Internally fit works on a copy and
  // advances it K local steps so each step's g-scalars are taken at the updated point. Returns
  // the per-(k,p) gradient scalars to upload (NEVER z). The local step uses (eta/P)*delta — the
  // 1/P factor that matches the server after the Bug-1 fix.
  //
  // Templated on the model type (any type with ExecutorchModel's loss() signature) so the round
  // loop is exercisable without an ExecuTorch runtime — which is what makes the P+1 forward-pass
  // cost contract assertable in a test. Production instantiates it with ExecutorchModel.
  template <class Model>
  GradientScalars2D fit(Model& model, const std::vector<float>& flatIn, const Seeds2D& seeds,
                        const DataBatch& batch, double mu) {
    std::vector<float> x = flatIn;  // local working copy; flatIn is left untouched (implicit revert)
    const int64_t d = static_cast<int64_t>(x.size());

    GradientScalars2D out(K_, std::vector<double>(P_, 0.0));
    for (int k = 0; k < K_; ++k) {
      std::vector<float> delta(static_cast<size_t>(d), 0.0f);  // float32, matching the prior path
      // L(x) is fixed across this step's P perturbations (x and the batch do not change here),
      // so evaluate it ONCE: P+1 forward passes per local step instead of 2P. Re-evaluated every
      // k because the local step below advances x.
      const double lossRef =
          etBaseLoss(model, x, batch.inputs, batch.inputShape, batch.targets, batch.numSamples);
      for (int p = 0; p < P_; ++p) {
        const int64_t s = seeds[k][p];
        const std::vector<float> z = flat_randn(s, d);  // same canonical z the g-scalar uses
        const double g = etGScalarForward(model, x, z, mu, batch.inputs, batch.inputShape,
                                          batch.targets, batch.numSamples, lossRef);
        out[k][p] = g;
        for (int64_t i = 0; i < d; ++i) delta[i] += static_cast<float>(g) * z[i];
      }
      // Local step with the 1/P averaging factor (matches the server after the Bug-1 fix).
      const float step = static_cast<float>(eta_ / P_);
      for (int64_t i = 0; i < d; ++i) x[i] -= step * delta[i];
    }
    return out;  // scalars only — z is never serialized or uploaded; flatIn unchanged
  }

  // Replay missed rounds (Algorithm 2) IN PLACE on `flat`: for each round, regenerate z from its
  // seed and apply x -= (lr/P) * g * z, using the SERVER-AVERAGED gradients carried in history.
  void rebuildModel(std::vector<float>& flat, const RebuildHistory& history);

 private:
  double eta_;
  int P_;
  int K_;
};

}  // namespace fedlearn
