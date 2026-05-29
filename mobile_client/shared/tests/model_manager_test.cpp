// model_manager_test.cpp — verify-before-load (E8) + param counts.
#include "fedlearn/ModelManager.h"
#include "fedlearn/Sha256.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#include "fixtures.h"

TEST(ModelManager, LoadsWithCorrectShaAndReportsCounts) {
  fedlearn::ModelManager mm;
  const std::string path = fedtest::goldenPath("zo_model_tiny.pt");
  const std::string sha = fedlearn::Sha256::hexDigestFile(path);  // correct hash (Sha256 is KAT-proven)

  fedlearn::ModelInfo info;
  torch::jit::Module model = mm.loadScriptModel(path, sha, &info);
  EXPECT_EQ(info.paramCount, 43);
  EXPECT_EQ(info.trainableParamCount, 25);
  (void)model;
}

TEST(ModelManager, RejectsTamperedModel) {
  fedlearn::ModelManager mm;
  const std::string path = fedtest::goldenPath("zo_model_tiny.pt");
  EXPECT_THROW(mm.loadScriptModel(path, std::string(64, '0'), nullptr), std::runtime_error);
}
