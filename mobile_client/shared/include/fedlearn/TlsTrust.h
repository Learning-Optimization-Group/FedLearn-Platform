#pragma once
//
// TlsTrust.h — which certificates a mobile client's TLS channel trusts and presents.
//
// The platform hands a phone the FL server's certificate at enrollment (EnrollmentDto.grpcServerCertPem). This header
// holds ONLY the pure decision — which root certificates verify the server, and whether to present a client keypair —
// so it is unit-testable in the gRPC-free host suite; FedLearnClient::makeChannel copies the result into
// grpc::SslCredentialsOptions.
//
// This header must stay dependency-light (std only): no grpcpp, no proto, no libtorch.
//
#include <functional>
#include <stdexcept>
#include <string>

namespace fedlearn {

// Reads a certificate or key file; throws std::runtime_error when the file cannot be read.
using FileReader = std::function<std::string(const std::string& path)>;

struct TlsMaterial {
  std::string rootCertsPem;        // empty = gRPC's default roots
  std::string clientCertChainPem;  // empty together with clientKeyPem = no client identity (server-only TLS)
  std::string clientKeyPem;
};

// Server trust: the certificate handed out at enrollment, else a pinned file, else gRPC's default roots. Client identity
// (mTLS) is optional, but a keypair is both halves or neither.
inline TlsMaterial resolveTlsMaterial(const std::string& caCertPem, const std::string& caCertPath,
                                      const std::string& clientCertPath, const std::string& clientKeyPath,
                                      const FileReader& readFile) {
  TlsMaterial tls;
  if (!caCertPem.empty()) {
    tls.rootCertsPem = caCertPem;
  } else if (!caCertPath.empty()) {
    tls.rootCertsPem = readFile(caCertPath);
  }
  if (clientCertPath.empty() != clientKeyPath.empty()) {
    throw std::runtime_error("FedLearnClient: clientCertPath and clientKeyPath must be set together");
  }
  if (!clientCertPath.empty()) {
    tls.clientKeyPem = readFile(clientKeyPath);
    tls.clientCertChainPem = readFile(clientCertPath);
  }
  return tls;
}

}  // namespace fedlearn
