#pragma once

#include <stdexcept>

namespace fedlearn {

// Invalid model state or replay history cannot be repaired by reconnecting.
class TrainingStateError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline constexpr const char* kTrainingStatePrefix = "TRAINING_STATE: ";

}  // namespace fedlearn
