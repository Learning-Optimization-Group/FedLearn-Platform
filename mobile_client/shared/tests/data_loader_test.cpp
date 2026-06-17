// data_loader_test.cpp — torch-free batch loading + validation (15-LLD §13 task 10).
#include "fedlearn/DataLoader.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "fixtures.h"

TEST(DataLoader, LoadsAndValidatesGoldenBatch) {
  fedlearn::OwnedBatch b = fedlearn::DataLoader::fromRawFiles(
      fedtest::goldenPath("zo_inputs.f32"), {8, 4}, fedtest::goldenPath("zo_targets.i64"));
  EXPECT_EQ(b.inputs.size(), 32u);
  EXPECT_EQ(b.targets.size(), 8u);
  EXPECT_EQ(b.inputs, fedtest::zoInputs());
  EXPECT_EQ(b.targets, fedtest::zoTargets());

  const fedlearn::DataBatch v = b.view();
  EXPECT_EQ(v.numSamples, 8);
  EXPECT_EQ(v.inputShape, (std::vector<int64_t>{8, 4}));
  EXPECT_NE(v.inputs, nullptr);
  EXPECT_NE(v.targets, nullptr);
}

TEST(DataLoader, RejectsShapeProductMismatch) {
  // {8,5} = 40 elements, but zo_inputs.f32 has 32.
  EXPECT_THROW(fedlearn::DataLoader::fromRawFiles(fedtest::goldenPath("zo_inputs.f32"), {8, 5},
                                                  fedtest::goldenPath("zo_targets.i64")),
               std::runtime_error);
}

TEST(DataLoader, RejectsMissingFile) {
  EXPECT_THROW(fedlearn::DataLoader::fromRawFiles(fedtest::goldenPath("does_not_exist.f32"), {8, 4},
                                                  fedtest::goldenPath("zo_targets.i64")),
               std::runtime_error);
}

TEST(DataLoader, ValidateRejectsBadBatch) {
  fedlearn::DataBatch nullBatch{nullptr, {8, 4}, nullptr, 8};
  EXPECT_THROW(fedlearn::DataLoader::validate(nullBatch), std::invalid_argument);

  std::vector<float> x(32, 0.0f);
  std::vector<int64_t> y(8, 0);
  fedlearn::DataBatch mismatched{x.data(), {7, 4}, y.data(), 8};  // inputShape[0]=7 != numSamples=8
  EXPECT_THROW(fedlearn::DataLoader::validate(mismatched), std::invalid_argument);
}
