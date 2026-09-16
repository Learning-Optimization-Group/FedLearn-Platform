// serialize_roundtrip_test.cpp — safetensors state-dict save/load symmetry + cross-language
// byte parity with the Python golden (15-LLD §13 task 6, FedAvg path). ExecuTorch, torch-free.
#include "fedlearn/ModelManager.h"

#include <gtest/gtest.h>

#include <fstream>
#include <string>
#include <vector>

#include "fixtures.h"

namespace {
std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
}  // namespace

TEST(SerializeRoundtrip, SaveThenLoadRestoresParams) {
  fedlearn::ModelManager mm = fedtest::makeManager();
  const std::vector<float> flat = fedtest::readF32(fedtest::goldenPath("zo_flat.f32"));
  ASSERT_EQ(flat.size(), 25u);
  mm.setFlatParams(flat);

  const std::string blob = mm.serializeStateDict(/*numExamples=*/7);
  ASSERT_FALSE(blob.empty());

  // Perturb, then restore from the blob.
  std::vector<float> perturbed = flat;
  for (auto& v : perturbed) v += 3.14f;
  mm.setFlatParams(perturbed);
  ASSERT_NE(mm.getFlatParams(), flat);

  mm.loadStateDict(blob);
  EXPECT_EQ(mm.getFlatParams(), flat);
}

TEST(SerializeRoundtrip, ByteMatchesPythonGolden) {
  fedlearn::ModelManager mm = fedtest::makeManager();
  mm.setFlatParams(fedtest::readF32(fedtest::goldenPath("zo_flat.f32")));
  // The golden was written by the Python serializer over the same flat + num_examples=8.
  EXPECT_EQ(mm.serializeStateDict(8), ReadFile(fedtest::goldenPath("zo_state.safetensors")))
      << "ModelManager serializeStateDict diverged from the Python safetensors golden";
}
