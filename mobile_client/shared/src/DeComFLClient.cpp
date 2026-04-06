#include "DeComFLClient.h"

namespace fedlearn {

DeComFLClient::DeComFLClient(FedLearnModule& model,
                             float smoothing_param)
    : model_(model), zo_estimator_(smoothing_param) {
  x_current_ = fedlearn::getFlatParams(model_);
  log("DeComFLClient",
      "Initialized with " + std::to_string(x_current_.numel()) + " parameters");
}

void DeComFLClient::rebuildModel(const std::vector<RebuildRound>& history,
                                 float learning_rate) {
  if (history.empty()) return;

  log("DeComFLClient",
      "Rebuilding model from " + std::to_string(history.size()) +
          " missed rounds");

  for (const auto& round_data : history) {
    int K = static_cast<int>(round_data.seeds.size());
    int P = K > 0 ? static_cast<int>(round_data.seeds[0].size()) : 0;

    for (int k = 0; k < K; ++k) {
      auto delta = torch::zeros_like(x_current_);

      for (int p = 0; p < P; ++p) {
        auto z = zo_estimator_.generatePerturbation(round_data.seeds[k][p],
                                                     x_current_.numel());
        double g = round_data.gradients[k][p];
        delta += g * z;
      }

      x_current_ = x_current_ - (learning_rate / P) * delta;
    }
  }

  fedlearn::setFlatParams(model_, x_current_);
  log("DeComFLClient", "Model rebuild complete");
}

std::pair<std::vector<std::vector<double>>, int64_t> DeComFLClient::fit(
    const Seeds2D& seeds,
    const TrainingConfig& config,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& batches) {
  int K = static_cast<int>(seeds.size());
  int P = K > 0 ? static_cast<int>(seeds[0].size()) : 0;
  float eta = config.learning_rate;

  log("DeComFLClient",
      "Starting local training: K=" + std::to_string(K) +
          ", P=" + std::to_string(P));

  auto x_initial = x_current_.clone();
  std::vector<std::vector<double>> gradient_scalars;
  int batch_idx = 0;
  int64_t total_examples = 0;

  // Algorithm 4, Line 14: loop over local steps k = 1..K
  for (int k = 0; k < K; ++k) {
    auto delta = torch::zeros_like(x_current_);
    std::vector<double> k_gradients;

    auto& batch = batches[batch_idx % batches.size()];
    batch_idx++;
    auto inputs = batch.first;
    auto targets = batch.second;
    total_examples += inputs.size(0);

    if (status_callback_) {
      status_callback_("training", k + 1, K);
    }

    // Algorithm 4, Line 16: loop over perturbations p = 1..P
    for (int p = 0; p < P; ++p) {
      auto z = zo_estimator_.generatePerturbation(seeds[k][p],
                                                   x_current_.numel());
      double g = zo_estimator_.computeGradientScalar(model_, x_current_, z,
                                                      inputs, targets);
      k_gradients.push_back(g);
      delta += g * z;
    }

    // Algorithm 4, Line 21: update model
    x_current_ = x_current_ - (eta / P) * delta;
    gradient_scalars.push_back(k_gradients);

    if ((k + 1) % std::max(1, K / 5) == 0) {
      log("DeComFLClient",
          "Completed local step " + std::to_string(k + 1) + "/" +
              std::to_string(K));
    }
  }

  // Algorithm 4: revert model to initial state (server applies update via seeds)
  x_current_ = x_initial;
  fedlearn::setFlatParams(model_, x_current_);

  log("DeComFLClient",
      "Local training complete. Generated " +
          std::to_string(gradient_scalars.size()) + " local steps with " +
          std::to_string(gradient_scalars[0].size()) + " perturbations each");

  return {gradient_scalars, total_examples};
}

}  // namespace fedlearn
