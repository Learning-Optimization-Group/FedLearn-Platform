#pragma once
// DeComFLServerConfig.h — the phone's reading of a DeComFL round's server config, free of gRPC and protobuf so it is
// unit-testable in the host suite. FedLearnClient converts the proto map and calls this.
#include "fedlearn/IFedLearnClient.h"
#include "fedlearn/Types.h"

#include <map>
#include <string>

namespace fedlearn {

// Reads the keys the server sends (grpc_servicer.py GetDeComFLConfig: learning_rate, smoothing_param) and the
// estimator, recording whether each value was sent. A value the server omits keeps the historical 0.001 default,
// which only a round without an execution contract may train on.
inline void applyDeComFLServerConfig(const std::map<std::string, std::string>& config, const std::string& method,
                                     DeComFLConfig& out) {
  const auto rate = config.find("learning_rate");
  out.learningRateSent = rate != config.end();
  out.config.learningRate = out.learningRateSent ? std::stod(rate->second) : 0.001;
  const auto mu = config.find("smoothing_param");
  out.muSent = mu != config.end();
  out.config.mu = out.muSent ? std::stod(mu->second) : 0.001;
  out.config.method = method == "central" ? GradEstimateMethod::Central : GradEstimateMethod::Forward;
}

}  // namespace fedlearn
