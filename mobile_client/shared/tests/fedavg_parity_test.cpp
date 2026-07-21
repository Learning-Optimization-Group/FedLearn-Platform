// fedavg_parity_test.cpp — the C++ first-order (FedAvg) endpoint must match the framework's real
// LocalTrainer.fit(mu=0) within tolerance (Phase B M1c). Mirrors how et_multiround_test.cpp pins the
// zeroth-order path. TrainableExecutorchModel loads the trainable TinyNet .pte and replays the same
// K full-batch SGD steps on the committed batch; the endpoint must land on the frozen golden.
//
// Constants mirror framework/tests/fixtures/decomfl_golden/{fedavg_local,fedavg_pte}_manifest.json.
// Tolerance-based (ET backward vs torch autograd is never bit-exact across runtimes) — endpoint_atol.
#include "fedlearn/TrainableExecutorchModel.h"
#include "fixtures.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

// fedavg_local_manifest.json
constexpr float kLr = 0.1f;             // learning_rate
constexpr int kLocalEpochs = 5;         // local_epochs (== full-batch SGD steps)
constexpr float kEndpointAtol = 2e-3f;  // endpoint_atol
constexpr int kFlatDim = 25;            // flat_dim (fc1: 20 weight + 5 bias)

// fedavg_pte_manifest.json
constexpr const char* kTrainablePte = "tinynet_trainable.pte";
constexpr const char* kTrainablePteSha =
    "ff398410f7339172295386dfc6220c5f46f21eddfb8ea145daf54e6a15dae412";

// Canonical ET trainable names in the framework's named_parameters() flat order (the training-graph
// wrapper prefixes "base."). fc1.weight [5,4]=20 THEN fc1.bias [5]=5 — NOT ET's alphabetical map
// order (bias < weight), which is exactly the ordering gotcha the wrapper must correct.
const std::vector<std::string> kParamNames = {"base.fc1.weight", "base.fc1.bias"};

}  // namespace

TEST(FedAvgParity, SetThenGetFlatParamsRoundTrips) {
  fedlearn::TrainableExecutorchModel m(fedtest::goldenPath(kTrainablePte), kTrainablePteSha, kParamNames);
  ASSERT_EQ(m.flatDim(), kFlatDim);

  const auto init = fedtest::readF32(fedtest::goldenPath("zo_flat.f32"));
  ASSERT_EQ(init.size(), static_cast<size_t>(kFlatDim));
  m.setFlatParams(init);
  const auto readback = m.getFlatParams();  // must honor canonical order, not ET's alphabetical map
  ASSERT_EQ(readback.size(), init.size());
  for (size_t i = 0; i < init.size(); ++i)
    EXPECT_FLOAT_EQ(readback[i], init[i]) << "set/get flat params round-trip broke at index " << i;
}

TEST(FedAvgParity, LocalEndpointMatchesFrameworkGolden) {
  fedlearn::TrainableExecutorchModel m(fedtest::goldenPath(kTrainablePte), kTrainablePteSha, kParamNames);

  // start from the SAME committed init the framework FedAvg golden starts from (byte-identical).
  m.setFlatParams(fedtest::readF32(fedtest::goldenPath("zo_flat.f32")));

  const auto x = fedtest::zoInputs();   // {8,4} -> 32 floats
  const auto y = fedtest::zoTargets();  // 8 int64
  const std::vector<int64_t> xShape{8, 4};
  for (int e = 0; e < kLocalEpochs; ++e) {
    m.trainStep(x.data(), xShape, y.data(), static_cast<int64_t>(y.size()), kLr);
  }

  const auto got = m.getFlatParams();
  const auto golden = fedtest::readF32(fedtest::goldenPath("fedavg_local_final.f32"));
  ASSERT_EQ(got.size(), golden.size());
  for (size_t i = 0; i < golden.size(); ++i)
    EXPECT_NEAR(got[i], golden[i], kEndpointAtol)
        << "first-order FedAvg endpoint diverged from the framework golden at param " << i;
}
