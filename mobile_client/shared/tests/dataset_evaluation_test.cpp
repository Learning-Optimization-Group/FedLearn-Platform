// dataset_evaluation_test.cpp — a device evaluates its whole dataset after a round, in chunks no larger than the
// programs take. A dataset larger than one batch failed evaluation on a phone after every minibatch round (the
// programs take at most 8 examples per call), and the round then rejected after its upload. The chunked loss is
// the example-weighted mean of the chunk losses, which must equal torch's whole-dataset cross-entropy.
#include "fedlearn/DatasetEvaluation.h"
#include "fedlearn/ExecutorchModel.h"
#include "fedlearn/ModelExecutionError.h"
#include "fixtures.h"

#include <gtest/gtest.h>

#include <vector>

namespace {

constexpr const char* kLossPte = "tinynet_loss_dynbatch.pte";
constexpr const char* kLossSha = "810286ec8b2d5ba6b3226ba6a34c335d62ad574fb72c4f1923595ea0c8a335e9";
constexpr const char* kInferPte = "tinynet_infer_dynbatch.pte";
constexpr const char* kInferSha = "ce80af6d991a5560097f361a4220a75c0b95678f5f5366128ba927b962fdcfe0";
constexpr double kTorchInitialLoss = 1.1232997179031372;  // fedavg_minibatch_manifest.initial_dataset_loss
constexpr double kTorchInitialCorrect = 8;                 // initial_dataset_correct

struct Fixture {
  fedlearn::ExecutorchModel loss{fedtest::goldenPath(kLossPte), kLossSha};
  fedlearn::ExecutorchModel infer{fedtest::goldenPath(kInferPte), kInferSha};
  std::vector<float> flat = fedtest::readF32(fedtest::goldenPath("zo_flat.f32"));
  std::vector<float> x = fedtest::readF32(fedtest::goldenPath("minibatch_inputs.f32"));
  std::vector<int64_t> y = fedtest::readI64(fedtest::goldenPath("minibatch_targets.i64"));
  fedlearn::DataBatch data() const { return fedlearn::DataBatch{x.data(), {20, 4}, y.data(), 20}; }
};

}  // namespace

TEST(DatasetEvaluation, ChunksReproduceTheWholeDatasetLossAndAccuracy) {
  Fixture f;
  const auto m = fedlearn::evaluateDataset(f.loss, f.infer, f.flat, f.data(), /*chunk=*/8);  // 8, 8, then 4
  EXPECT_NEAR(m.loss, kTorchInitialLoss, 1e-6);
  EXPECT_DOUBLE_EQ(m.accuracy, kTorchInitialCorrect / 20);
}

// The chunks are weighted by their example counts: an unweighted mean of the three chunk losses is another number.
TEST(DatasetEvaluation, ChunkLossesAreWeightedByTheirExampleCounts) {
  Fixture f;
  const auto whole8 = fedlearn::evaluateDataset(f.loss, f.infer, f.flat, f.data(), 8);
  const auto by4 = fedlearn::evaluateDataset(f.loss, f.infer, f.flat, f.data(), 4);  // five equal chunks
  EXPECT_NEAR(whole8.loss, by4.loss, 1e-6);
}

TEST(DatasetEvaluation, AProgramThatCannotTakeTheWholeDatasetInOneCallIsWhyItChunks) {
  Fixture f;
  EXPECT_THROW(fedlearn::evaluateDataset(f.loss, f.infer, f.flat, f.data(), /*chunk=*/0),
               fedlearn::ModelExecutionError);
}

TEST(DatasetEvaluation, AnEmptyDatasetIsNotEvaluable) {
  Fixture f;
  const auto m = fedlearn::evaluateDataset(f.loss, f.infer, f.flat, fedlearn::DataBatch{}, 8);
  EXPECT_EQ(m.loss, 0.0);
  EXPECT_EQ(m.accuracy, 0.0);
}
