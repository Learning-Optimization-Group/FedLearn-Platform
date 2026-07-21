#pragma once
//
// IFedLearnClient.h — the gRPC-free seam FederatedLoop depends on.
//
// FederatedLoop must be unit-testable without gRPC/proto (the libtorch-free, gRPC-free ET test
// suite). So it depends on THIS abstract interface returning CORE types (no v2::*, no grpcpp),
// not on the concrete FedLearnClient. The real FedLearnClient implements it (proto<->core
// marshaling); unit tests supply a mock. This header must include ONLY Types.h + <string>.
//
#include <string>

#include "fedlearn/Types.h"

namespace fedlearn {

// The DeComFL per-round server config, marshaled to core types (no proto).
struct DeComFLConfig {
  bool shouldStop = false;       // server should_stop
  int currentRound = 0;
  RoundConfig config;            // learningRate, mu, method (P/K derived from seeds)
  Seeds2D seeds;                 // [K][P]
  RebuildHistory rebuildHistory; // missed rounds to replay (server-averaged g)
};

// Abstract gRPC client seam: FederatedLoop depends on THIS, not on FedLearnClient. The real
// FedLearnClient implements it (proto<->core marshaling); unit tests supply a mock. NO gRPC/proto
// headers here — this header must compile in the libtorch-free, gRPC-free ET test suite.
class IFedLearnClient {
 public:
  virtual ~IFedLearnClient() = default;
  virtual bool shouldStop() const = 0;
  virtual DeComFLConfig getDeComFLConfig(const std::string& runId, const std::string& clientId) = 0;
  // Unified scalar upload (Constraint 7): seeds + g-scalars + num_examples. For DeComFL the server
  // already knows the seeds (it sent them); for FedAvg the client generated them — uploading seeds
  // in both cases unifies the wire. Implementations may ignore seeds for the DeComFL path.
  virtual void submitGradientScalars(const std::string& runId, const std::string& clientId,
                                     int trainedOnRound, const Seeds2D& seeds,
                                     const GradientScalars2D& gradients, int64_t numExamples) = 0;
  // FedAvg global-model download: returns the verified safetensors blob; sets *outCurrentRound.
  virtual std::string getGlobalModelStream(const std::string& runId, const std::string& clientId,
                                           int* outCurrentRound) = 0;
  // FedAvg (first-order) weight-update upload: the client trained locally with real gradients and
  // uploads the resulting model weight blob (safetensors, ModelManager::serializeStateDict) for
  // server-side aggregation — the model-blob analogue of submitGradientScalars. Proto-free; the
  // concrete client marshals it onto SubmitModelUpdateStream.
  virtual void submitModelUpdate(const std::string& runId, const std::string& clientId,
                                 int trainedOnRound, const std::string& modelBlob,
                                 int64_t numExamples) = 0;
};

}  // namespace fedlearn
