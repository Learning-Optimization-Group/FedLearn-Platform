#include "fedlearn/DeComFLClient.h"

#include "fedlearn/RandnEngine.h"

namespace fedlearn {

DeComFLClient::DeComFLClient(double eta, int P, int K) : eta_(eta), P_(P), K_(K) {}

// fit() is a template and lives in DeComFLClient.h.

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
