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
  GradientScalars2D fit(ExecutorchModel& model, const std::vector<float>& flatIn,
                        const Seeds2D& seeds, const DataBatch& batch, double mu);

  // Replay missed rounds (Algorithm 2) IN PLACE on `flat`: for each round, regenerate z from its
  // seed and apply x -= (lr/P) * g * z, using the SERVER-AVERAGED gradients carried in history.
  void rebuildModel(std::vector<float>& flat, const RebuildHistory& history);

 private:
  double eta_;
  int P_;
  int K_;
};

}  // namespace fedlearn
