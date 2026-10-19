#include "fedlearn/ModelStream.h"
#include "fedlearn/Sha256.h"
#include "fedlearn/TrainingStateError.h"
#include <gtest/gtest.h>

namespace {
fedlearn::ModelStreamPart part(int index, std::string_view data) {
  return {index, 2, index == 1, 2, "safetensors", false, 5,
          fedlearn::Sha256::hexDigest("model"), data};
}
}

TEST(ModelStream, ReassemblesACheckedSnapshotWithItsRound) {
  fedlearn::ModelStream stream(64);
  stream.append(part(0, "mo"));
  stream.append(part(1, "del"));
  EXPECT_EQ(stream.finish(), "model");
  EXPECT_EQ(stream.currentRound(), 2);
}

TEST(ModelStream, RejectsMissingIntegrityMetadataBeforeAcceptingWeights) {
  for (int change = 0; change < 4; ++change) {
    SCOPED_TRACE(change);
    fedlearn::ModelStream stream(64);
    auto first = part(0, "mo");
    if (change == 0) first.sha256.clear();
    if (change == 1) first.totalBytes = -1;
    if (change == 2) first.currentRound = 0;
    if (change == 3) first.totalBytes = 65;
    EXPECT_THROW(stream.append(first), fedlearn::TrainingStateError);
  }
}

TEST(ModelStream, RejectsCorruptionEvenWhenAllFramingLooksComplete) {
  fedlearn::ModelStream stream(64);
  stream.append(part(0, "mo"));
  stream.append(part(1, "bad"));
  EXPECT_THROW(stream.finish(), fedlearn::TrainingStateError);
}

TEST(ModelStream, RejectsAnIncompleteSnapshot) {
  fedlearn::ModelStream stream(64);
  stream.append(part(0, "mo"));
  EXPECT_THROW(stream.finish(), fedlearn::TrainingStateError);
}

TEST(ModelStream, RejectsChangedMetadataAndMissingOrRepeatedChunks) {
  for (int change = 0; change < 9; ++change) {
    SCOPED_TRACE(change);
    fedlearn::ModelStream stream(64);
    stream.append(part(0, "mo"));
    auto second = part(1, "del");
    if (change == 0) second.index = 0;
    if (change == 1) second.index = 2;
    if (change == 2) second.totalChunks = 3;
    if (change == 3) second.currentRound = 3;
    if (change == 4) second.totalBytes = 6;
    if (change == 5) second.sha256 = std::string(64, '0');
    if (change == 6) second.isFinal = false;
    if (change == 7) second.compressed = true;
    if (change == 8) second.codec = "lz4+safetensors";
    EXPECT_THROW(stream.append(second), fedlearn::TrainingStateError);
  }
}

TEST(ModelStream, RejectsAPrematureFinalMarkerAndTrailingData) {
  fedlearn::ModelStream stream(64);
  auto first = part(0, "mo");
  first.isFinal = true;
  EXPECT_THROW(stream.append(first), fedlearn::TrainingStateError);
  fedlearn::ModelStream complete(64);
  complete.append(part(0, "mo"));
  complete.append(part(1, "del"));
  EXPECT_THROW(complete.append(part(1, "del")), fedlearn::TrainingStateError);
}

TEST(ModelStream, RejectsAFinalLengthMismatch) {
  fedlearn::ModelStream stream(64);
  stream.append(part(0, "mo"));
  EXPECT_THROW(stream.append(part(1, "d")), fedlearn::TrainingStateError);
}
