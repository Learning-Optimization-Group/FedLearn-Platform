// Server trust without client identity. The platform hands a phone the FL server's certificate at enrollment. Before
// this the client refused TLS unless it also held a client certificate and key, which nothing provisions, so a phone
// could not dial a TLS deployment at all.
#include <gtest/gtest.h>

#include <stdexcept>

#include "fedlearn/FedLearnClient.h"

namespace {

// A real self-signed certificate, so no failure below can come from a malformed PEM.
const char* const kServerCertPem =
    "-----BEGIN CERTIFICATE-----\n"
    "MIICFjCCAbwCCQDQNiHCK2YzjTAKBggqhkjOPQQDAjAZMRcwFQYDVQQDDA5mbC5l\n"
    "eGFtcGxlLm9yZzAeFw0yNjA5MTUxOTI3NDNaFw0zNjA5MTIxOTI3NDNaMBkxFzAV\n"
    "BgNVBAMMDmZsLmV4YW1wbGUub3JnMIIBSzCCAQMGByqGSM49AgEwgfcCAQEwLAYH\n"
    "KoZIzj0BAQIhAP////8AAAABAAAAAAAAAAAAAAAA////////////////MFsEIP//\n"
    "//8AAAABAAAAAAAAAAAAAAAA///////////////8BCBaxjXYqjqT57PrvVV2mIa8\n"
    "ZR0GsMxTsPY7zjw+J9JgSwMVAMSdNgiG5wSTamZ44ROdJreBn36QBEEEaxfR8uEs\n"
    "Qkf4vOblY6RA8ncDfYEt6zOg9KE5RdiYwpZP40Li/hp/m47n60p8D54WK84zV2sx\n"
    "Xs7LtkBoN79R9QIhAP////8AAAAA//////////+85vqtpxeehPO5ysL8YyVRAgEB\n"
    "A0IABPesqCVrEpvA5EbQ3kYxsEOXLdbylvW9blzME0MmD/+iH57OWDyjfqsZomj+\n"
    "Bsa+WTeD0HVegbxHBlCZQRa3H+0wCgYIKoZIzj0EAwIDSAAwRQIhAJUpPGKoXilD\n"
    "3eCwNEeBsKh4MvaxSVf0UCx7XkT2e3z0AiAIFGDR+FG5tiliGV+v+esC3/ccL/65\n"
    "iK2U2hjSEcHR6A==\n"
    "-----END CERTIFICATE-----\n";

fedlearn::GrpcClientConfig tlsConfig() {
  fedlearn::GrpcClientConfig cfg;
  cfg.serverAddress = "fl.example.org:50001";
  cfg.useTls = true;
  cfg.caCertPem = kServerCertPem;
  return cfg;
}

}  // namespace

TEST(GrpcChannelCredentials, TlsWithOnlyTheServerCertificateBuildsAClient) {
  EXPECT_NO_THROW({ fedlearn::FedLearnClient client(tlsConfig()); });
}

TEST(GrpcChannelCredentials, AClientCertificateWithoutItsKeyIsRefused) {
  auto cfg = tlsConfig();
  cfg.clientCertPath = "/nonexistent/client.pem";
  EXPECT_THROW({ fedlearn::FedLearnClient client(cfg); }, std::runtime_error);
}

TEST(GrpcChannelCredentials, AnUnreadableClientKeypairIsRefused) {
  auto cfg = tlsConfig();
  cfg.clientCertPath = "/nonexistent/client.pem";
  cfg.clientKeyPath = "/nonexistent/client.key";
  EXPECT_THROW({ fedlearn::FedLearnClient client(cfg); }, std::runtime_error);
}
