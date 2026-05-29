// sha256_test.cpp — NIST known-answer vectors prove the Sha256 implementation is correct,
// which is what makes ModelManager's verify-before-load trustworthy.
#include "fedlearn/Sha256.h"

#include <gtest/gtest.h>

#include <string>

TEST(Sha256, NistKnownAnswerVectors) {
  EXPECT_EQ(fedlearn::Sha256::hexDigest(std::string("")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(fedlearn::Sha256::hexDigest(std::string("abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(fedlearn::Sha256::hexDigest(
                std::string("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}
