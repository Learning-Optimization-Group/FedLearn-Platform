#pragma once

#include "Utils.h"
#include "ZerothOrderEstimator.h"

namespace fedlearn {

/**
 * Port of framework/src/fedlearn/client/decomfl_client.py
 *
 * Computes gradient scalars using zeroth-order optimization.
 * Returns gradient scalars instead of full model parameters.
 * Implements model rebuilding for missed rounds.
 */
class DeComFLClient {
 public:
  DeComFLClient(FedLearnModule& model,
                float smoothing_param = 0.001f);

  /**
   * Algorithm 2: replay missed rounds to sync model state.
   * Regenerates perturbations from seeds + applies averaged gradient updates.
   */
  void rebuildModel(const std::vector<RebuildRound>& history,
                    float learning_rate);

  /**
   * Algorithm 4: ZO local training.
   * Returns {gradient_scalars[K][P], num_examples}.
   */
  std::pair<std::vector<std::vector<double>>, int64_t> fit(
      const Seeds2D& seeds,
      const TrainingConfig& config,
      const std::vector<std::pair<torch::Tensor, torch::Tensor>>& batches);

  void setGrpcStatusCallback(
      std::function<void(const std::string&, int, int)> callback) {
    status_callback_ = std::move(callback);
  }

 private:
  FedLearnModule& model_;
  ZerothOrderEstimator zo_estimator_;
  torch::Tensor x_current_;
  std::function<void(const std::string&, int, int)> status_callback_;
};

}  // namespace fedlearn
