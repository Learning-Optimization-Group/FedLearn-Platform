#pragma once
//
// DeComFLClient.h — on-device DeComFL client: one round (fit) + missed-round replay
// (rebuildModel). 15-LLD-mobile.md §5.2, §6.2; mirrors the DeComFL correctness spec.
//
#include <torch/script.h>

#include "fedlearn/ModelManager.h"
#include "fedlearn/Types.h"
#include "fedlearn/ZerothOrderEstimator.h"

namespace fedlearn {

class DeComFLClient {
 public:
  DeComFLClient(ModelManager& mm, ZerothOrderEstimator& zo, double eta, int P, int K);

  // One DeComFL round. Returns the per-(k,p) gradient scalars to upload (NEVER z).
  // Snapshot-restore revert (B1-M1): snapshot x_initial at the top, restore EXACTLY at the
  // end (the client reverts; the server owns the true global trajectory). The local step
  // uses (eta / P) * delta — the 1/P factor that matches the server after the Bug-1 fix.
  GradientScalars2D fit(torch::jit::Module& model,
                        const Seeds2D& seeds,   // [K][P]
                        const DataBatch& batch);

  // Replay missed rounds (Algorithm 2): for each round, regenerate z from its seed and apply
  // x -= (lr / P) * g * z, using the SERVER-AVERAGED gradients carried in the history.
  void rebuildModel(torch::jit::Module& model, const RebuildHistory& history);

 private:
  ModelManager& mm_;
  ZerothOrderEstimator& zo_;
  double eta_;
  int P_;
  int K_;
};

}  // namespace fedlearn
