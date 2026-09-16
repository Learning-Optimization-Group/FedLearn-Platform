// Server trust without client identity. The platform hands a phone the FL server's certificate at enrollment, and the
// client must dial TLS against it without also holding a client keypair, which nothing provisions. resolveTlsMaterial is
// the gRPC-free half of FedLearnClient::makeChannel, so these run in the host suite, which has no gRPC.
#include "fedlearn/TlsTrust.h"

#include <gtest/gtest.h>

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const std::string kEnrollmentPem = "-----BEGIN CERTIFICATE-----\nfrom-enrollment\n-----END CERTIFICATE-----\n";

// Serves fixed files and records every path asked for; an unknown path fails the way an unreadable file does.
struct FakeFiles {
  std::map<std::string, std::string> files;
  std::vector<std::string> reads;

  fedlearn::FileReader reader() {
    return [this](const std::string& path) {
      reads.push_back(path);
      auto it = files.find(path);
      if (it == files.end()) throw std::runtime_error("cannot read " + path);
      return it->second;
    };
  }
};

}  // namespace

TEST(TlsTrust, TheEnrollmentCertificateIsTrustedWithoutReadingAnyFile) {
  FakeFiles fs;
  fs.files["/pinned/ca.pem"] = "pinned";
  const auto tls = fedlearn::resolveTlsMaterial(kEnrollmentPem, "/pinned/ca.pem", "", "", fs.reader());
  EXPECT_EQ(tls.rootCertsPem, kEnrollmentPem);
  EXPECT_TRUE(tls.clientCertChainPem.empty());
  EXPECT_TRUE(tls.clientKeyPem.empty());
  EXPECT_TRUE(fs.reads.empty());
}

TEST(TlsTrust, APinnedCertificateFileIsUsedWhenEnrollmentSentNone) {
  FakeFiles fs;
  fs.files["/pinned/ca.pem"] = "pinned";
  const auto tls = fedlearn::resolveTlsMaterial("", "/pinned/ca.pem", "", "", fs.reader());
  EXPECT_EQ(tls.rootCertsPem, "pinned");
}

TEST(TlsTrust, WithNoCertificateAtAllTheDefaultRootsApply) {
  FakeFiles fs;
  const auto tls = fedlearn::resolveTlsMaterial("", "", "", "", fs.reader());
  EXPECT_TRUE(tls.rootCertsPem.empty());
  EXPECT_TRUE(fs.reads.empty());
}

TEST(TlsTrust, AClientCertificateWithoutItsKeyIsRefused) {
  FakeFiles fs;
  EXPECT_THROW(fedlearn::resolveTlsMaterial(kEnrollmentPem, "", "/client.pem", "", fs.reader()), std::runtime_error);
  EXPECT_THROW(fedlearn::resolveTlsMaterial(kEnrollmentPem, "", "", "/client.key", fs.reader()), std::runtime_error);
}

TEST(TlsTrust, AClientKeypairIsReadWhenBothHalvesAreSet) {
  FakeFiles fs;
  fs.files["/client.pem"] = "chain";
  fs.files["/client.key"] = "key";
  const auto tls = fedlearn::resolveTlsMaterial(kEnrollmentPem, "", "/client.pem", "/client.key", fs.reader());
  EXPECT_EQ(tls.clientCertChainPem, "chain");
  EXPECT_EQ(tls.clientKeyPem, "key");
}

TEST(TlsTrust, AnUnreadableClientKeypairIsRefused) {
  FakeFiles fs;
  EXPECT_THROW(fedlearn::resolveTlsMaterial(kEnrollmentPem, "", "/missing.pem", "/missing.key", fs.reader()),
               std::runtime_error);
}
