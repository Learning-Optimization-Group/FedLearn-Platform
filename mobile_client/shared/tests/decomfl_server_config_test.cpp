// decomfl_server_config_test.cpp — how the phone reads a DeComFL round's server config.
//
// The server sends learning_rate and smoothing_param (grpc_servicer.py GetDeComFLConfig). The native client used to
// read "lr" and "mu", keys the server never sends, so it always trained on its own 0.001 defaults and never on the
// server's values -- unnoticed only because the server's defaults are 0.001 too. It also records whether each value
// was sent, so a round held to an execution contract can refuse one the server left out.
#include "fedlearn/DeComFLServerConfig.h"

#include <gtest/gtest.h>

#include <map>
#include <string>

TEST(DeComFLServerConfig, ReadsTheKeysTheServerSends) {
  fedlearn::DeComFLConfig out;
  fedlearn::applyDeComFLServerConfig({{"learning_rate", "0.01"}, {"smoothing_param", "0.002"}}, "forward", out);
  EXPECT_DOUBLE_EQ(out.config.learningRate, 0.01);
  EXPECT_DOUBLE_EQ(out.config.mu, 0.002);
  EXPECT_TRUE(out.learningRateSent);
  EXPECT_TRUE(out.muSent);
  EXPECT_EQ(out.config.method, fedlearn::GradEstimateMethod::Forward);
}

TEST(DeComFLServerConfig, DoesNotMistakeTheOldKeysForTheServersValues) {
  fedlearn::DeComFLConfig out;
  fedlearn::applyDeComFLServerConfig({{"lr", "0.5"}, {"mu", "0.5"}}, "forward", out);
  EXPECT_FALSE(out.learningRateSent);
  EXPECT_FALSE(out.muSent);
  EXPECT_DOUBLE_EQ(out.config.learningRate, 0.001);   // the historical default for a round with no contract
  EXPECT_DOUBLE_EQ(out.config.mu, 0.001);
}

TEST(DeComFLServerConfig, ReadsTheEstimator) {
  fedlearn::DeComFLConfig out;
  fedlearn::applyDeComFLServerConfig({}, "central", out);
  EXPECT_EQ(out.config.method, fedlearn::GradEstimateMethod::Central);
}
