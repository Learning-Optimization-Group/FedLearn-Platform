#pragma once
//
// Types.h — plain-data types for the C++ FL core (15-LLD-mobile.md §5.3).
//
#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace fedlearn {

using Seeds2D = std::vector<std::vector<int64_t>>;            // [K][P] perturbation seeds
using GradientScalars2D = std::vector<std::vector<double>>;   // [K][P] g scalars (double — wire is double, B1)

enum class GradEstimateMethod { Forward, Central };

// One on-device training batch. inputs: float tensor; targets: int64 class indices (CNN/MLP path).
struct DataBatch {
  torch::Tensor inputs;
  torch::Tensor targets;
};

// One missed round to replay in rebuildModel (Algorithm 2). gradients are the
// SERVER-AVERAGED scalars for that round (from RebuildHistory in GetDeComFLConfigResponse).
struct RebuildRound {
  int roundNumber = 0;
  Seeds2D seeds;                 // [K][P]
  GradientScalars2D gradients;   // [K][P], server-averaged
  double learningRate = 0.0;     // the eta used that round
};
using RebuildHistory = std::vector<RebuildRound>;

struct RoundConfig {
  std::string strategy;
  double learningRate = 0.0;
  double mu = 0.0;
  int numPerturbations = 0;   // P
  int numLocalSteps = 0;      // K
  GradEstimateMethod method = GradEstimateMethod::Forward;
  int64_t seed = 0;
  std::string torchVersion;
};

struct DeviceMetrics {
  int64_t peakRssBytes = 0;
  std::string thermalState;
  double batteryLevel = 0.0;
  bool batteryCharging = false;
};

struct ModelInfo {
  int64_t paramCount = 0;
  int64_t trainableParamCount = 0;
  std::string sha256;
  std::string tier;   // "1M" | "10M" | "100M" | "" if below the smallest tier
};

}  // namespace fedlearn
