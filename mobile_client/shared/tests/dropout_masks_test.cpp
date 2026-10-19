// dropout_masks_test.cpp — DROPOUT_MASKS_SEEDED_V1 in the native trainer. It must reproduce the masks the framework
// draws for the same run seed, round, step and layer (the shared golden
// framework/tests/fixtures/execution_contract_v1/dropout_masks_v1.golden), so a phone's training step that uses
// dropout can be replayed exactly.
#include "fedlearn/DropoutMasks.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> goldenLines(const std::string& kind) {
  std::ifstream in(std::string(CONTRACT_DIR) + "/dropout_masks_v1.golden");
  if (!in) throw std::runtime_error("dropout_masks_v1.golden is missing");
  std::vector<std::string> out;
  for (std::string line; std::getline(in, line);) {
    if (line.rfind(kind + " ", 0) == 0) out.push_back(line.substr(kind.size() + 1));
  }
  return out;
}

}  // namespace

TEST(DropoutMasks, ReproducesEveryGoldenMask) {
  const auto lines = goldenLines("mask");
  ASSERT_GE(lines.size(), 10u);
  for (const auto& line : lines) {
    const auto colon = line.find(':');
    std::istringstream head(line.substr(0, colon));
    uint64_t n = 0, seed = 0, round = 0, step = 0, layer = 0;
    double rate = 0;
    head >> n >> rate >> seed >> round >> step >> layer;
    std::string bits = line.substr(colon + 1);
    bits.erase(0, bits.find_first_not_of(' '));
    const auto keep = fedlearn::dropoutKeep(n, rate, seed, round, step, layer);
    std::string got;
    for (bool k : keep) got += k ? '1' : '0';
    EXPECT_EQ(got, bits) << line.substr(0, colon);
  }
}

TEST(DropoutMasks, ScalesAreFloat32InvertedDropout) {
  for (const auto& line : goldenLines("scale")) {
    const auto colon = line.find(':');
    const double rate = std::stod(line.substr(0, colon));
    std::string hex = line.substr(colon + 1);
    hex.erase(0, hex.find_first_not_of(' '));
    const float scale = fedlearn::dropoutScale(rate);
    unsigned char bytes[4];
    std::memcpy(bytes, &scale, 4);
    char got[9];
    std::snprintf(got, sizeof got, "%02x%02x%02x%02x", bytes[0], bytes[1], bytes[2], bytes[3]);
    EXPECT_EQ(std::string(got), hex) << "rate " << rate;
  }
}

TEST(DropoutMasks, AMaskIsTheScaleWhereKeptAndZeroWhereDropped) {
  const auto keep = fedlearn::dropoutKeep(64, 0.3, 42, 1, 0, 0);
  const auto mask = fedlearn::dropoutMask(64, 0.3, 42, 1, 0, 0);
  ASSERT_EQ(mask.size(), 64u);
  for (size_t i = 0; i < mask.size(); ++i) EXPECT_EQ(mask[i], keep[i] ? fedlearn::dropoutScale(0.3) : 0.0f);
}

TEST(DropoutMasks, ARateOutsideZeroToOneIsRefused) {
  EXPECT_THROW(fedlearn::dropoutKeep(4, 1.0, 1, 1, 0, 0), std::invalid_argument);
  EXPECT_THROW(fedlearn::dropoutKeep(4, -0.1, 1, 1, 0, 0), std::invalid_argument);
}
