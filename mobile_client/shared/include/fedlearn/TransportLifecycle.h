#pragma once

#include <memory>
#include <mutex>
#include <string>

namespace fedlearn {

// Ordinary client operations and replacement are serialized by the bridge's
// state mutex. Stop must bypass that mutex to cancel a blocked RPC or round.
// This shorter lock protects ownership and heartbeat start/stop only.
template <typename Client>
class TransportLifecycle {
 public:
  void replace(std::unique_ptr<Client> client) {
    std::lock_guard<std::mutex> lock(mutex_);
    client_ = std::move(client);
  }

  // The caller must hold the bridge state mutex while using this pointer.
  Client* get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return client_.get();
  }

  bool startHeartbeat(Client* expected, const std::string& runId,
                      const std::string& clientId, int round) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!client_ || client_.get() != expected || client_->shouldStop()) return false;
    client_->startHeartbeat(runId, clientId, round);
    return true;
  }

  void stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (client_) {
      client_->requestAbort();
      client_->stopHeartbeat();
    }
  }

 private:
  mutable std::mutex mutex_;
  std::unique_ptr<Client> client_;
};

}  // namespace fedlearn
