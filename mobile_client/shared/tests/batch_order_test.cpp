// batch_order_test.cpp — BATCH_ORDER_SEEDED_PERMUTATION_V1 in the native trainer. It must reproduce the permutation
// the framework draws for the same run seed, round and epoch (the shared golden
// framework/tests/fixtures/execution_contract_v1/batch_permutation_v1.golden), so a phone's multi-batch update can be
// replayed exactly from its data.
#include "fedlearn/BatchOrder.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Case {
  uint64_t n, seed, round, epoch;
  std::vector<uint64_t> perm;
};

std::vector<Case> goldenCases() {
  std::ifstream in(std::string(CONTRACT_DIR) + "/batch_permutation_v1.golden");
  if (!in) throw std::runtime_error("batch_permutation_v1.golden is missing");
  std::vector<Case> cases;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#' || line.rfind("below", 0) == 0) continue;
    const auto colon = line.find(':');
    std::istringstream head(line.substr(0, colon)), tail(line.substr(colon + 1));
    Case c;
    head >> c.n >> c.seed >> c.round >> c.epoch;
    for (uint64_t i; tail >> i;) c.perm.push_back(i);
    cases.push_back(c);
  }
  return cases;
}

}  // namespace

// Published SplitMix64 outputs for state 0: the stream is the standard generator, not a look-alike.
TEST(BatchOrder, TheStreamIsSplitMix64) {
  fedlearn::SplitMix64 rng(0);
  EXPECT_EQ(rng.next(), 0xE220A8397B1DCDAFULL);
  EXPECT_EQ(rng.next(), 0x6E789E6AA1B965F4ULL);
  EXPECT_EQ(rng.next(), 0x06C45D188009454FULL);
  EXPECT_EQ(rng.next(), 0xF88BB8A8724C81ECULL);
}

TEST(BatchOrder, ReproducesEveryGoldenPermutation) {
  const auto cases = goldenCases();
  ASSERT_GE(cases.size(), 10u);
  for (const auto& c : cases) {
    EXPECT_EQ(fedlearn::seededPermutation(c.n, c.seed, c.round, c.epoch), c.perm)
        << "n=" << c.n << " seed=" << c.seed << " round=" << c.round << " epoch=" << c.epoch;
  }
}

TEST(BatchOrder, BatchesKeepTheFinalPartialBatchUnlessDropLast) {
  std::vector<uint64_t> order{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const auto kept = fedlearn::batches(order, 4, /*dropLast=*/false);
  ASSERT_EQ(kept.size(), 3u);
  EXPECT_EQ(kept[2], (std::vector<uint64_t>{8, 9, 10}));
  EXPECT_EQ(fedlearn::batches(order, 4, /*dropLast=*/true).size(), 2u);
  EXPECT_TRUE(fedlearn::batches({}, 4, false).empty());
  EXPECT_THROW(fedlearn::batches(order, 0, false), std::invalid_argument);
}

// The rejection boundary of the unbiased draw, which ordinary seeds never reach: the golden's states make the next
// draw exactly 2^64 - 1 (rejected for bound 3) and 2^64 - 2 (the largest accepted draw).
TEST(BatchOrder, TheUnbiasedDrawRejectsExactlyTheDrawsAboveItsLimit) {
  std::ifstream in(std::string(CONTRACT_DIR) + "/batch_permutation_v1.golden");
  ASSERT_TRUE(in);
  int seen = 0;
  for (std::string line; std::getline(in, line);) {
    if (line.rfind("below", 0) != 0) continue;
    std::istringstream fields(line.substr(5));
    uint64_t state = 0, bound = 0, expected = 0;
    char colon = 0;
    fields >> state >> bound >> colon >> expected;
    EXPECT_EQ(fedlearn::SplitMix64(state).below(bound), expected) << line;
    ++seen;
  }
  EXPECT_EQ(seen, 2);
}
