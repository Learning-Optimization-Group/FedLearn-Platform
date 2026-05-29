#pragma once
//
// BridgeTypes.h — plain-C++ mirrors of the TurboModule spec types (bridge/specs/NativeFedLearnCore.ts).
//
// These have NO React Native / JSI dependency, so the FL logic in FedLearnCoreModule can be
// written and reasoned about independently of the (version-specific) codegen glue. The JSI layer
// converts these structs <-> jsi::Object field-by-field (no hand-built JSON, fixes A6 §L3).
//
#include <cstdint>
#include <string>
#include <vector>

namespace fedlearn::bridge {

struct RegisterResult {
  bool accepted = false;
  std::string message;
  int assignedRound = 0;
  int serverProtocolVersion = 0;
};

struct ServerStatus {
  std::string serverState;
  int currentRound = 0;
  int requiredClientsForRound = 0;
  int receivedUpdatesThisRound = 0;
  int activeClients = 0;
  int64_t roundDeadlineUnixMs = 0;
};

struct RoundConfig {
  std::string strategy;            // "DeComFL" | "FedAvg"
  double learningRate = 0.0;       // eta
  double mu = 0.0;                 // ZO smoothing radius
  int numPerturbations = 0;        // P
  int numLocalSteps = 0;           // K
  std::string gradEstimateMethod;  // "forward" | "central"
  int64_t seed = 0;
  std::string torchVersion;        // server's expected torch version (RNG-parity gate)
};

struct RoundResult {
  int round = 0;
  double loss = -1.0;
  double accuracy = -1.0;          // -1.0 if not evaluated
  int64_t scalarsTransmitted = 0;  // K*P for DeComFL, 0 for FedAvg
  int64_t uplinkBytes = 0;
  int64_t downlinkBytes = 0;
  int64_t computeMs = 0;
  bool reverted = false;           // DeComFL: model restored to the pre-round snapshot
};

struct DeviceMetrics {
  int64_t peakRssBytes = 0;
  std::string thermalState = "NOMINAL";  // NOMINAL|FAIR|SERIOUS|CRITICAL
  double batteryLevel = -1.0;            // 0..1; -1 if unknown
  bool batteryCharging = false;
};

struct ModelInfo {
  int64_t paramCount = 0;
  int64_t trainableParamCount = 0;
  std::string sha256;
  std::string tier;  // "1M" | "10M" | "100M"
};

struct InferResult {
  std::vector<double> logits;
  std::vector<double> probabilities;
  int argmax = -1;
};

}  // namespace fedlearn::bridge
