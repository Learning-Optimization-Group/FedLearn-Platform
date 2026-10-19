#include "fedlearn/TransportLifecycle.h"
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <future>
#include <functional>
#include <memory>
#include <thread>

namespace {
using namespace std::chrono_literals;
struct State {
  std::atomic<bool> aborted{false}, heartbeat{false}, destroyed{false};
  std::function<void()> onStart = [] {}, onStop = [] {};
};
struct Client {
  std::shared_ptr<State> state;
  explicit Client(std::shared_ptr<State> s) : state(std::move(s)) {}
  ~Client() { state->destroyed = true; }
  void requestAbort() { state->aborted = true; }
  bool shouldStop() const { return state->aborted; }
  void stopHeartbeat() { auto s = state; s->onStop(); s->heartbeat = false; }
  void startHeartbeat(const std::string&, const std::string&, int) {
    auto s = state; s->onStart(); s->aborted = false; s->heartbeat = true;
  }
};
}

TEST(TransportLifecycle, ReplacementWaitsForAllStopDereferences) {
  fedlearn::TransportLifecycle<Client> slot;
  auto old = std::make_shared<State>();
  slot.replace(std::make_unique<Client>(old));
  std::promise<void> entered, release, replacing;
  auto gate = release.get_future().share();
  old->onStop = [&] { entered.set_value(); gate.wait(); };
  auto stop = std::async(std::launch::async, [&] { slot.stop(); });
  entered.get_future().wait();
  auto replacement = std::async(std::launch::async, [&] {
    replacing.set_value(); slot.replace(std::make_unique<Client>(std::make_shared<State>()));
  });
  replacing.get_future().wait();
  const auto waiting = replacement.wait_for(30ms);
  const bool destroyedBeforeStopFinished = old->destroyed;
  release.set_value(); stop.get(); replacement.get();
  EXPECT_EQ(waiting, std::future_status::timeout);
  EXPECT_FALSE(destroyedBeforeStopFinished);
  EXPECT_TRUE(old->destroyed);
}

TEST(TransportLifecycle, HeartbeatStartupAndStopCannotOverlap) {
  fedlearn::TransportLifecycle<Client> slot;
  auto state = std::make_shared<State>();
  slot.replace(std::make_unique<Client>(state));
  auto* client = slot.get();
  std::promise<void> entered, release, stopping;
  auto gate = release.get_future().share();
  state->onStart = [&] { entered.set_value(); gate.wait(); };
  auto start = std::async(std::launch::async, [&] { return slot.startHeartbeat(client, "r", "c", 1); });
  entered.get_future().wait();
  auto stop = std::async(std::launch::async, [&] { stopping.set_value(); slot.stop(); });
  stopping.get_future().wait();
  const auto waiting = stop.wait_for(30ms);
  release.set_value(); EXPECT_TRUE(start.get()); stop.get();
  EXPECT_EQ(waiting, std::future_status::timeout);
  EXPECT_FALSE(state->heartbeat);
  EXPECT_TRUE(state->aborted);
}

TEST(TransportLifecycle, LateRegistrationSuccessCannotRestartAnAbortedClient) {
  fedlearn::TransportLifecycle<Client> slot;
  auto state = std::make_shared<State>();
  slot.replace(std::make_unique<Client>(state));
  auto* client = slot.get();
  slot.stop();
  EXPECT_FALSE(slot.startHeartbeat(client, "r", "c", 1));
  EXPECT_TRUE(state->aborted);
  EXPECT_FALSE(state->heartbeat);
}

TEST(TransportLifecycle, RegistrationFailureUnwindsAfterConcurrentAbort) {
  fedlearn::TransportLifecycle<Client> slot;
  auto state = std::make_shared<State>();
  slot.replace(std::make_unique<Client>(state));
  auto* client = slot.get();
  std::promise<void> registering;
  auto registration = std::async(std::launch::async, [&] {
    registering.set_value();
    while (!client->shouldStop()) std::this_thread::yield();
    throw std::runtime_error("registration cancelled");
  });
  registering.get_future().wait();
  slot.stop();
  EXPECT_THROW(registration.get(), std::runtime_error);
  EXPECT_FALSE(state->heartbeat);
  slot.replace(std::make_unique<Client>(std::make_shared<State>()));
  EXPECT_TRUE(slot.startHeartbeat(slot.get(), "r", "c", 2));
  slot.stop();
}
