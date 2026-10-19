#include "fedlearn/FedLearnClient.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace {
using namespace std::chrono_literals;

template <typename Predicate>
bool waitFor(Predicate predicate) {
  const auto until = std::chrono::steady_clock::now() + 3s;
  while (std::chrono::steady_clock::now() < until) {
    if (predicate()) return true;
    std::this_thread::sleep_for(2ms);
  }
  return false;
}

class RecoveryServer : public fedlearn::v2::FederatedLearningService::Service {
 public:
  std::atomic<int> heartbeatCalls{0}, heartbeatAcks{0}, registrations{0}, uploadChunks{0};
  std::atomic<bool> failHeartbeats{true}, serverStop{false}, blockHeartbeat{false};
  std::mutex mutex;
  std::condition_variable condition;
  bool releaseHeartbeat = false;

  grpc::Status RegisterClient(grpc::ServerContext*, const fedlearn::v2::RegisterClientRequest*,
                              fedlearn::v2::RegisterClientResponse* response) override {
    ++registrations;
    response->set_status(fedlearn::v2::RegisterClientResponse::ACCEPTED);
    return grpc::Status::OK;
  }

  grpc::Status GetServerStatus(grpc::ServerContext*, const fedlearn::v2::GetServerStatusRequest*,
                               fedlearn::v2::GetServerStatusResponse* response) override {
    response->set_server_state(fedlearn::v2::GetServerStatusResponse::TRAINING);
    response->set_current_round(8);
    return grpc::Status::OK;
  }

  grpc::Status Heartbeat(grpc::ServerContext*, const fedlearn::v2::HeartbeatRequest*,
                         fedlearn::v2::HeartbeatResponse* response) override {
    ++heartbeatCalls;
    if (blockHeartbeat.load()) {
      std::unique_lock<std::mutex> lock(mutex);
      condition.wait(lock, [&] { return releaseHeartbeat; });
    }
    if (failHeartbeats.load()) return {grpc::StatusCode::UNAVAILABLE, "connection lost"};
    ++heartbeatAcks;
    response->set_acknowledged(true);
    response->set_should_stop(serverStop.load());
    return grpc::Status::OK;
  }

  grpc::Status SubmitModelUpdateStream(grpc::ServerContext* context,
      grpc::ServerReader<fedlearn::v2::ModelUpdateChunk>* reader,
      fedlearn::v2::SubmitModelUpdateResponse*) override {
    fedlearn::v2::ModelUpdateChunk chunk;
    if (reader->Read(&chunk)) ++uploadChunks;
    // Leave a real streaming RPC incomplete until its client deadline expires.
    while (!context->IsCancelled()) std::this_thread::sleep_for(2ms);
    return {grpc::StatusCode::CANCELLED, "incomplete upload"};
  }
};

class GrpcRecovery : public ::testing::Test {
 protected:
  RecoveryServer service;
  std::unique_ptr<grpc::Server> server;
  fedlearn::GrpcClientConfig config;

  void SetUp() override {
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    server = builder.BuildAndStart();
    ASSERT_NE(server, nullptr);
    config.serverAddress = "127.0.0.1:" + std::to_string(port);
    config.useTls = false;
    config.heartbeatIntervalMs = 20;
    config.heartbeatFailureLimit = 3;
    config.rpcDeadlineMs = 250;
  }

  void TearDown() override {
    { std::lock_guard<std::mutex> lock(service.mutex); service.releaseHeartbeat = true; }
    service.condition.notify_all();
    server->Shutdown();
    server->Wait();
  }

  // Wait for the transport to publish either a terminal stop or a recoverable failure.
  bool awaitHeartbeatExit(fedlearn::FedLearnClient& client) {
    return waitFor([&] {
      try { return client.shouldStop(); }
      catch (const std::runtime_error&) { return true; }
    });
  }
};

TEST_F(GrpcRecovery, ExhaustedHeartbeatRequiresReconnectInsteadOfCleanStop) {
  fedlearn::FedLearnClient client(config);
  client.registerClient("run", "phone", "", 2);
  client.startHeartbeat("run", "phone", 7);
  ASSERT_TRUE(awaitHeartbeatExit(client));
  EXPECT_EQ(service.heartbeatCalls.load(), 3);
  EXPECT_THROW(client.getServerStatus("run"), std::runtime_error);
  try {
    EXPECT_FALSE(client.shouldStop());
    FAIL() << "Heartbeat loss must reject the round for re-registration";
  } catch (const std::runtime_error& error) {
    EXPECT_EQ(std::string(error.what()).find("RECONNECT_REQUIRED: "), 0u);
  }
}

TEST_F(GrpcRecovery, PartialUploadDeadlineDoesNotTurnHeartbeatLossIntoCleanStop) {
  fedlearn::FedLearnClient client(config);
  client.registerClient("run", "phone", "", 2);
  client.startHeartbeat("run", "phone", 7);
  try {
    client.submitModelUpdate("run", "phone", 8, std::string(3 * 1024 * 1024, 'x'), 16);
    FAIL() << "An incomplete upload must fail";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("gRPC code 4:"), std::string::npos);
  }
  EXPECT_EQ(service.uploadChunks.load(), 1);
  ASSERT_TRUE(awaitHeartbeatExit(client));
  EXPECT_THROW(client.shouldStop(), std::runtime_error);
}

TEST_F(GrpcRecovery, ReplacementRegistrationRestoresHeartbeatBeforeResumedStatus) {
  fedlearn::FedLearnClient stale(config);
  stale.registerClient("run", "phone", "", 2);
  stale.startHeartbeat("run", "phone", 7);
  ASSERT_TRUE(awaitHeartbeatExit(stale));
  EXPECT_THROW(stale.startHeartbeat("run", "phone", 7), std::runtime_error);
  service.failHeartbeats = false;
  fedlearn::FedLearnClient replacement(config);
  const auto registered = replacement.registerClient("run", "phone", "", 2);
  ASSERT_EQ(registered.status(), fedlearn::v2::RegisterClientResponse::ACCEPTED);
  replacement.startHeartbeat("run", "phone", 7);
  ASSERT_TRUE(waitFor([&] { return service.heartbeatAcks.load() > 0; }));
  EXPECT_EQ(service.registrations.load(), 2);
  EXPECT_FALSE(replacement.shouldStop());
  EXPECT_EQ(replacement.getServerStatus("run").current_round(), 8);
}

TEST_F(GrpcRecovery, UserStopWinsOverConcurrentHeartbeatFailure) {
  service.blockHeartbeat = true;
  config.heartbeatFailureLimit = 1;
  fedlearn::FedLearnClient client(config);
  client.startHeartbeat("run", "phone", 7);
  ASSERT_TRUE(waitFor([&] { return service.heartbeatCalls.load() == 1; }));
  client.requestAbort();
  { std::lock_guard<std::mutex> lock(service.mutex); service.releaseHeartbeat = true; }
  service.condition.notify_all();
  // Do not cancel the heartbeat while observing this transition: lifecycle
  // cancellation would bypass the failure branch whose Stop precedence matters.
  const bool stopOverwritten = waitFor([&] {
    try { return !client.shouldStop(); }
    catch (const std::runtime_error&) { return true; }
  });
  EXPECT_FALSE(stopOverwritten);
  client.stopHeartbeat();
  EXPECT_TRUE(client.shouldStop());
  client.startHeartbeat("run", "phone", 7);
  EXPECT_TRUE(client.shouldStop());
  EXPECT_EQ(service.heartbeatCalls.load(), 1);
}

TEST_F(GrpcRecovery, ServerStopRemainsTerminal) {
  service.failHeartbeats = false;
  service.serverStop = true;
  fedlearn::FedLearnClient client(config);
  client.startHeartbeat("run", "phone", 7);
  ASSERT_TRUE(awaitHeartbeatExit(client));
  EXPECT_TRUE(client.shouldStop());
}

}  // namespace
