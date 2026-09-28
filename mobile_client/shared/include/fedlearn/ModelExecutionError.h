// ModelExecutionError.h — ExecuTorch refused to load or run a model program on this device's data.
//
// Such a failure is deterministic for the program and the data (e.g. a static-shape program given another example
// count returns NotSupported), so retrying the round or rejoining the run cannot help. The JS bridge rejects it
// with kModelExecutionPrefix, which the training loop reads as "stop now" rather than as a transient blip.
#pragma once

#include <stdexcept>
#include <string>

namespace fedlearn {

class ModelExecutionError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

inline constexpr const char* kModelExecutionPrefix = "MODEL_EXECUTION: ";

/** The message a failed native call rejects its JS promise with. */
inline std::string rejectionMessage(const std::exception& e) {
  if (dynamic_cast<const ModelExecutionError*>(&e) != nullptr) {
    return std::string(kModelExecutionPrefix) + e.what();
  }
  return e.what();
}

}  // namespace fedlearn
