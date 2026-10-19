#pragma once

#include <atomic>
#include <stdexcept>

namespace fedlearn {

// A round must not queue behind another round and later submit the same update. The guard is
// acquired before the module's broader state mutex and released even when native training throws.
class RoundFlight {
 public:
  class Guard {
   public:
    explicit Guard(RoundFlight& owner) : owner_(owner) {}
    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;
    ~Guard() { owner_.active_.store(false); }

   private:
    RoundFlight& owner_;
  };

  Guard acquire() {
    if (active_.exchange(true)) {
      throw std::runtime_error("A training round is already in progress");
    }
    return Guard(*this);
  }

 private:
  std::atomic<bool> active_{false};
};

}  // namespace fedlearn
