// auth_metadata_test.cpp — the FL-boundary auth contract for the mobile client.
//
// FedLearnClient attaches authMetadata(connectionToken_) to every gRPC ClientContext so a
// fail-closed FL server's ConnectionTokenInterceptor (which checks the "x-connection-token"
// metadata key) admits the call. This pins the pure decision — key/value selection — which is
// gRPC-free and therefore testable in this host suite; the AddMetadata application lives in
// FedLearnClient::applyAuth (built only under -DFEDLEARN_BUILD_GRPC=ON).

#include "fedlearn/AuthMetadata.h"

#include <gtest/gtest.h>

TEST(AuthMetadata, AttachesConnectionTokenUnderTheInterceptorKey) {
  const std::string jwt = "eyJhbGciOiJIUzI1NiJ9.payload.sig";
  auto md = fedlearn::authMetadata(jwt);
  ASSERT_EQ(md.size(), 1u);
  // Must match ConnectionTokenInterceptor's metadata key exactly, lowercase (gRPC lowercases keys).
  EXPECT_EQ(md[0].first, "x-connection-token");
  EXPECT_EQ(md[0].second, jwt);
}

TEST(AuthMetadata, EmptyTokenYieldsNoMetadata) {
  // The legacy / auth-off path: no token means the client sends nothing (a fail-open server
  // accepts it; a fail-closed server rejects it, which is intended).
  EXPECT_TRUE(fedlearn::authMetadata("").empty());
}
