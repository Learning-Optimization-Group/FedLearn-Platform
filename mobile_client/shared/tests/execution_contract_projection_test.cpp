// execution_contract_projection_test.cpp — the native trainer runs exactly the training a run's execution
// contract states. The server publishes that training in the contract (proto/fedlearn/contract/v1); the phone's
// TypeScript reader projects it into a learning rate and a step count, and the shared fixture
// projection_tinynet_fedavg.json holds that projection. This test is the C++ end of the chain
// contract -> projection -> trained endpoint: it trains with the fixture's OWN numbers and must land on the
// framework's committed endpoint for THAT training (contract_local_manifest.json), within its tolerance.
#include "fedlearn/TrainableExecutorchModel.h"
#include "fixtures.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr const char* kTrainablePte = "tinynet_trainable.pte";
constexpr const char* kTrainablePteSha =
    "ff398410f7339172295386dfc6220c5f46f21eddfb8ea145daf54e6a15dae412";
constexpr int kFlatDim = 25;
// contract_local_manifest.json: endpoint_atol. Far tighter than the lr=0.1 golden's 2e-3, because the
// contract's single step at lr=1e-3 moves the parameters by only ~1.4e-4 (max_param_movement) — at 2e-3 this
// test would pass on a trainer that did nothing. kMinMovement pins that the step really happened.
constexpr float kEndpointAtol = 1e-6f;
constexpr float kMinMovement = 1e-5f;

const std::vector<std::string> kParamNames = {"base.fc1.weight", "base.fc1.bias"};

// The fixture is a flat JSON object of numbers, so a key lookup plus a number parse is enough here; the
// contract itself is never parsed this way (the typed reader does that before anything reaches C++).
std::string extractValue(const std::string& json, const std::string& key) {
  size_t pos = json.find("\"" + key + "\"");
  if (pos == std::string::npos) {
    return "";
  }
  pos = json.find(':', pos);
  if (pos == std::string::npos) {
    return "";
  }
  pos += 1;
  while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) {
    ++pos;
  }
  size_t end = pos;
  while (end < json.length() && json[end] != ',' && json[end] != '}' && json[end] != ']') {
    ++end;
  }
  return json.substr(pos, end - pos);
}

}  // namespace

TEST(ExecutionContractProjection, LocalEndpointMatchesContractGolden) {
  // Read the JSON fixture
  const std::string contractPath = std::string(CONTRACT_DIR) + "/projection_tinynet_fedavg.json";
  std::ifstream f(contractPath);
  std::ostringstream buffer;
  buffer << f.rdbuf();
  std::string json = buffer.str();

  // Extract learningRate
  std::string lrStr = extractValue(json, "learningRate");
  if (lrStr.empty()) {
    FAIL() << "Missing 'learningRate' in contract fixture: " << contractPath;
  }
  double lr = std::stod(lrStr);
  if (lr <= 0.0) {
    FAIL() << "Invalid learningRate: " << lr << ", must be positive.";
  }

  // Extract numLocalSteps
  std::string stepsStr = extractValue(json, "numLocalSteps");
  if (stepsStr.empty()) {
    FAIL() << "Missing 'numLocalSteps' in contract fixture: " << contractPath;
  }
  int steps = std::stoi(stepsStr);
  if (steps <= 0) {
    FAIL() << "Invalid numLocalSteps: " << steps << ", must be a positive integer.";
  }

  fedlearn::TrainableExecutorchModel m(fedtest::goldenPath(kTrainablePte), kTrainablePteSha, kParamNames);
  ASSERT_EQ(m.flatDim(), kFlatDim);

  // Start from the same committed initial flat vector as the framework golden.
  m.setFlatParams(fedtest::readF32(fedtest::goldenPath("zo_flat.f32")));

  const auto x = fedtest::zoInputs();   // {8,4} -> 32 floats
  const auto y = fedtest::zoTargets();  // 8 int64
  const std::vector<int64_t> xShape{8, 4};

  // Replay exactly numLocalSteps steps at learningRate
  for (int e = 0; e < steps; ++e) {
    m.trainStep(x.data(), xShape, y.data(), static_cast<int64_t>(y.size()), static_cast<float>(lr));
  }

  const auto got = m.getFlatParams();
  const auto initial = fedtest::readF32(fedtest::goldenPath("zo_flat.f32"));
  float moved = 0.0f;
  for (size_t i = 0; i < got.size(); ++i) moved = std::max(moved, std::fabs(got[i] - initial[i]));
  EXPECT_GT(moved, kMinMovement) << "the contract's training left the parameters where they started";

  const auto golden = fedtest::readF32(fedtest::goldenPath("contract_local_final.f32"));
  ASSERT_EQ(got.size(), golden.size());
  for (size_t i = 0; i < golden.size(); ++i)
    EXPECT_NEAR(got[i], golden[i], kEndpointAtol)
        << "contract-driven FedAvg endpoint diverged from the framework golden at param " << i;
}
