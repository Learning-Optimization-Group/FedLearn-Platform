// dynamic_batch_training_test.cpp — a device trains its own dataset, whose example count need not be the batch
// the programs were exported with (Stage 3). A program exported with a static batch of 8 refuses any other count
// (ExecuTorch NotSupported, seen live on a phone with a 6-example snapshot); one exported with a dynamic batch
// (pte_export max_batch) must train 6 examples to the framework's endpoint on those 6, and 8 to the usual one.
//
// Constants mirror framework/tests/fixtures/decomfl_golden/{fedavg_local,fedavg_pte}_manifest.json.
#include "fedlearn/TrainableExecutorchModel.h"
#include "fixtures.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr float kLr = 0.1f;             // fedavg_local_manifest.learning_rate
constexpr int kLocalEpochs = 5;         // fedavg_local_manifest.local_epochs
constexpr float kEndpointAtol = 2e-3f;  // fedavg_local_manifest.endpoint_atol
constexpr int kSmallBatch = 6;          // fedavg_local_manifest.small_batch_examples

constexpr const char* kStaticPte = "tinynet_trainable.pte";
constexpr const char* kStaticPteSha = "ff398410f7339172295386dfc6220c5f46f21eddfb8ea145daf54e6a15dae412";
constexpr const char* kDynamicPte = "tinynet_trainable_dynbatch.pte";
constexpr const char* kDynamicPteSha = "8df99d447a8883a7a65ace595493a15f40cc6339c1e23553fbf922ef9851f124";
const std::vector<std::string> kParamNames = {"base.fc1.weight", "base.fc1.bias"};

// Train the first `n` committed examples as one full batch for kLocalEpochs steps from the committed init.
std::vector<float> trainFirst(fedlearn::TrainableExecutorchModel& m, int n) {
  m.setFlatParams(fedtest::readF32(fedtest::goldenPath("zo_flat.f32")));
  const auto x = fedtest::zoInputs();
  const auto y = fedtest::zoTargets();
  const std::vector<int64_t> xShape{n, 4};
  for (int e = 0; e < kLocalEpochs; ++e) m.trainStep(x.data(), xShape, y.data(), n, kLr);
  return m.getFlatParams();
}

void expectNear(const std::vector<float>& got, const std::string& goldenFile) {
  const auto golden = fedtest::readF32(fedtest::goldenPath(goldenFile));
  ASSERT_EQ(got.size(), golden.size());
  for (size_t i = 0; i < golden.size(); ++i)
    EXPECT_NEAR(got[i], golden[i], kEndpointAtol) << goldenFile << " diverged at param " << i;
}

}  // namespace

TEST(DynamicBatchTraining, AStaticBatchProgramRefusesAnotherExampleCount) {
  fedlearn::TrainableExecutorchModel m(fedtest::goldenPath(kStaticPte), kStaticPteSha, kParamNames);
  EXPECT_THROW(trainFirst(m, kSmallBatch), std::runtime_error);
}

TEST(DynamicBatchTraining, ADynamicBatchProgramTrainsFewerExamplesToTheFrameworkEndpoint) {
  fedlearn::TrainableExecutorchModel m(fedtest::goldenPath(kDynamicPte), kDynamicPteSha, kParamNames);
  expectNear(trainFirst(m, kSmallBatch), "fedavg_local_final_6.f32");
}

TEST(DynamicBatchTraining, ADynamicBatchProgramTrainsTheFullBatchAsTheStaticOneDoes) {
  fedlearn::TrainableExecutorchModel m(fedtest::goldenPath(kDynamicPte), kDynamicPteSha, kParamNames);
  expectNear(trainFirst(m, 8), "fedavg_local_final.f32");
}

// Sizes can change between steps on one loaded program: a final partial minibatch follows full ones.
TEST(DynamicBatchTraining, TheExampleCountMayChangeBetweenSteps) {
  fedlearn::TrainableExecutorchModel m(fedtest::goldenPath(kDynamicPte), kDynamicPteSha, kParamNames);
  trainFirst(m, 8);
  expectNear(trainFirst(m, kSmallBatch), "fedavg_local_final_6.f32");
}
