// grpc_marshal_test.cpp — server-free unit tests for the proto<->core converters and the codec
// whitelist (the parts of the gRPC layer that ARE unit-testable without a running server).
// Built only under -DFEDLEARN_BUILD_GRPC=ON (needs the buf-generated proto types).
#include "fedlearn/FedLearnClient.h"

#include <gtest/gtest.h>

#include <stdexcept>

TEST(GrpcMarshal, GradientScalarsRoundTrip) {
  fedlearn::GradientScalars2D g = {{0.1, 0.2, 0.3}, {-1.0, 2.5, 0.0}};
  fedlearn::v2::GradientScalars proto = fedlearn::FedLearnClient::toProtoScalars(g);
  ASSERT_EQ(proto.local_steps_size(), 2);
  ASSERT_EQ(proto.local_steps(0).scalars_size(), 3);

  fedlearn::GradientScalars2D back = fedlearn::FedLearnClient::fromProtoScalars(proto);
  ASSERT_EQ(back.size(), 2u);
  for (size_t k = 0; k < g.size(); ++k) {
    ASSERT_EQ(back[k].size(), g[k].size());
    for (size_t p = 0; p < g[k].size(); ++p) EXPECT_DOUBLE_EQ(back[k][p], g[k][p]);
  }
}

TEST(GrpcMarshal, SeedsFromProto) {
  fedlearn::v2::PerturbationSeeds s;
  auto* a = s.add_local_steps();
  a->add_seeds(11);
  a->add_seeds(22);
  auto* b = s.add_local_steps();
  b->add_seeds(33);
  b->add_seeds(44);  // rectangular: a fixed P=2 across the K local steps (jagged is rejected below)

  fedlearn::Seeds2D out = fedlearn::FedLearnClient::fromProtoSeeds(s);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0].size(), 2u);
  EXPECT_EQ(out[1].size(), 2u);
  EXPECT_EQ(out[0][1], 22);
  EXPECT_EQ(out[1][0], 33);
}

TEST(GrpcMarshal, SeedsFromProtoRejectsJaggedMatrix) {
  // A server sending rows of different lengths is an OOB heap read in DeComFLClient::fit (P is taken from
  // row 0 and applied to every row via unchecked operator[]). Reject at the marshal boundary.
  fedlearn::v2::PerturbationSeeds s;
  auto* a = s.add_local_steps();
  a->add_seeds(11);
  a->add_seeds(22);
  s.add_local_steps()->add_seeds(33);  // only 1 seed -> jagged
  EXPECT_THROW(fedlearn::FedLearnClient::fromProtoSeeds(s), std::runtime_error);
}

TEST(GrpcMarshal, SeedsFromProtoRejectsEmptyRow) {
  // An empty perturbation row (P=0) is a divide-by-zero (learningRate / P) that NaN-poisons the model.
  fedlearn::v2::PerturbationSeeds s;
  s.add_local_steps();  // a local step with zero seeds
  EXPECT_THROW(fedlearn::FedLearnClient::fromProtoSeeds(s), std::runtime_error);
}

TEST(GrpcMarshal, RebuildHistoryRejectsGradientSeedShapeMismatch) {
  // gradients and seeds come from independent proto fields; rebuildModel indexes gradients[k][p] with
  // K,P derived from seeds — a smaller gradients matrix is an OOB heap read. Reject the mismatch.
  fedlearn::v2::RebuildHistory h;
  auto* r = h.add_rounds();
  r->mutable_seeds()->add_local_steps()->add_seeds(7);  // seeds: 1x1
  // average_gradients left empty (0 rows) vs seeds' 1 row -> shape mismatch
  EXPECT_THROW(fedlearn::FedLearnClient::fromProtoRebuildHistory(h), std::runtime_error);
}

TEST(GrpcMarshal, RebuildHistoryFromProto) {
  fedlearn::v2::RebuildHistory h;
  auto* r = h.add_rounds();
  r->set_round_number(5);
  r->mutable_seeds()->add_local_steps()->add_seeds(7);
  r->mutable_average_gradients()->add_local_steps()->add_scalars(0.5);

  fedlearn::RebuildHistory out = fedlearn::FedLearnClient::fromProtoRebuildHistory(h);
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].roundNumber, 5);
  EXPECT_EQ(out[0].seeds[0][0], 7);
  EXPECT_DOUBLE_EQ(out[0].gradients[0][0], 0.5);
  EXPECT_DOUBLE_EQ(out[0].learningRate, 0.0);  // RoundHistory has no lr; the loop fills it
}

TEST(GrpcMarshal, CodecWhitelist) {
  EXPECT_NO_THROW(fedlearn::FedLearnClient::validateCodec("safetensors"));
  EXPECT_NO_THROW(fedlearn::FedLearnClient::validateCodec("lz4+safetensors"));
  EXPECT_THROW(fedlearn::FedLearnClient::validateCodec("pickle"), std::invalid_argument);
  EXPECT_THROW(fedlearn::FedLearnClient::validateCodec(""), std::invalid_argument);
}
