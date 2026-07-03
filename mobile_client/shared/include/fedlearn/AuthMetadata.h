#pragma once
//
// AuthMetadata.h — the gRPC metadata that authenticates a mobile client call to a
// fail-closed FL server.
//
// The server's ConnectionTokenInterceptor (framework/src/fedlearn/security/interceptor.py)
// gates protected RPCs on an "x-connection-token" metadata key carrying the HMAC-JWT the
// backend minted at enrollment (EnrollmentDto.connectionToken). This header holds ONLY the
// pure decision — which key/value pairs to attach for a given token — so it is unit-testable
// in the gRPC-free host suite; FedLearnClient applies the result to each grpc::ClientContext.
//
// This header must stay dependency-light (std only): no grpcpp, no proto, no libtorch.
//
#include <string>
#include <utility>
#include <vector>

namespace fedlearn {

// The client-metadata pairs that authenticate a call. An empty token yields NO pairs —
// the legacy / auth-off path where the client simply sends nothing and a fail-open server
// accepts it (a fail-closed server rejects it, which is the intended behaviour).
inline std::vector<std::pair<std::string, std::string>> authMetadata(const std::string& connectionToken) {
  if (connectionToken.empty()) {
    return {};
  }
  return {{"x-connection-token", connectionToken}};
}

}  // namespace fedlearn
