#include "fedlearn/FedLearnClient.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "fedlearn/Sha256.h"

// gRPC C++ API used here (stable across recent gRPC releases): grpc::CreateCustomChannel,
// grpc::SslCredentials / grpc::InsecureChannelCredentials, grpc::ClientContext,
// grpc::ClientReader / grpc::ClientWriter, grpc::Status. Generated stub:
// fedlearn::v2::FederatedLearningService::NewStub. If a gRPC version renames any of these the
// build (CI mobile.yml) catches it — these are not host-verifiable without the runtime.

namespace fedlearn {
namespace {

std::string readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.good()) throw std::runtime_error("FedLearnClient: cannot read cert/key file: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string statusStr(const grpc::Status& s) {
  std::ostringstream ss;
  ss << "gRPC code " << static_cast<int>(s.error_code()) << ": " << s.error_message();
  return ss.str();
}

}  // namespace

void FedLearnClient::validateCodec(const std::string& codec) {
  if (codec != "safetensors" && codec != "lz4+safetensors") {
    throw std::invalid_argument("FedLearnClient: rejected wire codec '" + codec +
                                "' (allowed: safetensors, lz4+safetensors)");
  }
}

std::shared_ptr<grpc::Channel> FedLearnClient::makeChannel() const {
  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(cfg_.maxMessageBytes);
  args.SetMaxSendMessageSize(cfg_.maxMessageBytes);

  std::shared_ptr<grpc::ChannelCredentials> creds;
  if (cfg_.useTls) {
    if (cfg_.caCertPath.empty() || cfg_.clientCertPath.empty() || cfg_.clientKeyPath.empty()) {
      throw std::runtime_error("FedLearnClient: TLS+mTLS requires caCertPath, clientCertPath, clientKeyPath");
    }
    grpc::SslCredentialsOptions ssl;
    ssl.pem_root_certs = readFile(cfg_.caCertPath);
    ssl.pem_private_key = readFile(cfg_.clientKeyPath);
    ssl.pem_cert_chain = readFile(cfg_.clientCertPath);
    creds = grpc::SslCredentials(ssl);  // mTLS: client presents its cert; CN binds identity (R6)
  } else {
    // Dev-only. A release RN build never sets useTls=false (FEDLEARN_ALLOW_INSECURE_GRPC, E13).
    creds = grpc::InsecureChannelCredentials();
  }
  return grpc::CreateCustomChannel(cfg_.serverAddress, creds, args);
}

FedLearnClient::FedLearnClient(const GrpcClientConfig& cfg) : cfg_(cfg) {
  trainingChannel_ = makeChannel();
  heartbeatChannel_ = makeChannel();  // a SEPARATE channel so heartbeat is independent of fit()
  trainingStub_ = v2::FederatedLearningService::NewStub(trainingChannel_);
  heartbeatStub_ = v2::FederatedLearningService::NewStub(heartbeatChannel_);
}

FedLearnClient::~FedLearnClient() { stopHeartbeat(); }

// ---------------------------------------------------------------------------
// Unary control
// ---------------------------------------------------------------------------
v2::RegisterClientResponse FedLearnClient::registerClient(const std::string& runId,
                                                          const std::string& clientId,
                                                          const std::string& enrollmentToken,
                                                          int protocolVersion) {
  v2::RegisterClientRequest req;
  req.set_client_id(clientId);
  req.set_run_id(runId);
  req.set_protocol_version(protocolVersion);
  req.set_enrollment_token(enrollmentToken);

  v2::RegisterClientResponse resp;
  grpc::ClientContext ctx;
  grpc::Status s = trainingStub_->RegisterClient(&ctx, req, &resp);
  if (!s.ok()) throw std::runtime_error("RegisterClient failed: " + statusStr(s));
  return resp;
}

v2::GetServerStatusResponse FedLearnClient::getServerStatus(const std::string& runId) {
  v2::GetServerStatusRequest req;
  req.set_run_id(runId);
  v2::GetServerStatusResponse resp;
  grpc::ClientContext ctx;
  grpc::Status s = trainingStub_->GetServerStatus(&ctx, req, &resp);
  if (!s.ok()) throw std::runtime_error("GetServerStatus failed: " + statusStr(s));
  return resp;
}

v2::GetDeComFLConfigResponse FedLearnClient::getDeComFLConfig(const std::string& runId,
                                                             const std::string& clientId) {
  v2::GetDeComFLConfigRequest req;
  req.set_client_id(clientId);
  req.set_run_id(runId);
  v2::GetDeComFLConfigResponse resp;
  grpc::ClientContext ctx;
  grpc::Status s = trainingStub_->GetDeComFLConfig(&ctx, req, &resp);
  if (!s.ok()) throw std::runtime_error("GetDeComFLConfig failed: " + statusStr(s));
  return resp;
}

v2::SubmitGradientScalarsResponse FedLearnClient::submitGradientScalars(
    const std::string& runId, const std::string& clientId, int trainedOnRound,
    const GradientScalars2D& gradients, int64_t numExamples) {
  v2::SubmitGradientScalarsRequest req;
  req.set_client_id(clientId);
  req.set_run_id(runId);
  req.set_trained_on_round(trainedOnRound);
  req.set_num_examples(numExamples);
  *req.mutable_gradients() = toProtoScalars(gradients);

  v2::SubmitGradientScalarsResponse resp;
  grpc::ClientContext ctx;
  grpc::Status s = trainingStub_->SubmitGradientScalars(&ctx, req, &resp);
  if (!s.ok()) throw std::runtime_error("SubmitGradientScalars failed: " + statusStr(s));
  return resp;  // resp.bytes_received() ~ K*P*8 — the comm-cost number
}

// ---------------------------------------------------------------------------
// FedAvg streaming
// ---------------------------------------------------------------------------
std::string FedLearnClient::getGlobalModelStream(const std::string& runId,
                                                 const std::string& clientId, int* outCurrentRound) {
  v2::GetGlobalModelRequest req;
  req.set_client_id(clientId);
  req.set_run_id(runId);

  grpc::ClientContext ctx;
  std::unique_ptr<grpc::ClientReader<v2::ModelChunk>> reader(
      trainingStub_->GetGlobalModelStream(&ctx, req));

  std::string blob;
  v2::ModelChunk chunk;
  int64_t declaredTotal = -1;
  std::string declaredSha;
  int currentRound = 0;
  bool first = true;
  while (reader->Read(&chunk)) {
    if (first) {
      validateCodec(chunk.codec());            // E7-style: reject unknown codec
      declaredTotal = chunk.total_bytes();
      declaredSha = chunk.sha256();
      currentRound = chunk.current_round();
      if (declaredTotal > cfg_.maxMessageBytes * static_cast<int64_t>(4096)) {
        throw std::runtime_error("getGlobalModelStream: declared total_bytes exceeds sane cap");
      }
      first = false;
    }
    blob.append(chunk.chunk_data());
    if (declaredTotal >= 0 && static_cast<int64_t>(blob.size()) > declaredTotal) {
      throw std::runtime_error("getGlobalModelStream: reassembled size exceeds declared total_bytes");
    }
  }
  grpc::Status s = reader->Finish();
  if (!s.ok()) throw std::runtime_error("GetGlobalModelStream failed: " + statusStr(s));

  if (!declaredSha.empty() && Sha256::hexDigest(blob) != declaredSha) {
    throw std::runtime_error("getGlobalModelStream: sha256 mismatch on reassembled model blob");
  }
  if (outCurrentRound) *outCurrentRound = currentRound;
  return blob;
}

v2::SubmitModelUpdateResponse FedLearnClient::submitModelUpdateStream(
    const std::string& runId, const std::string& clientId, int trainedOnRound,
    const std::string& modelBlob, int64_t numExamples, bool compressed) {
  const int64_t total = static_cast<int64_t>(modelBlob.size());
  const std::string sha = Sha256::hexDigest(modelBlob);
  const std::string codec = compressed ? "lz4+safetensors" : "safetensors";
  const size_t kChunk = 1u << 20;  // 1 MB chunks
  int totalChunks = static_cast<int>((total + static_cast<int64_t>(kChunk) - 1) / static_cast<int64_t>(kChunk));
  if (totalChunks == 0) totalChunks = 1;

  v2::SubmitModelUpdateResponse resp;
  grpc::ClientContext ctx;
  std::unique_ptr<grpc::ClientWriter<v2::ModelUpdateChunk>> writer(
      trainingStub_->SubmitModelUpdateStream(&ctx, &resp));

  for (int i = 0; i < totalChunks; ++i) {
    const size_t off = static_cast<size_t>(i) * kChunk;
    const size_t len = std::min(kChunk, static_cast<size_t>(total) - off);
    v2::ModelUpdateChunk c;
    c.set_client_id(clientId);
    c.set_run_id(runId);
    c.set_trained_on_round(trainedOnRound);
    c.set_chunk_index(i);
    c.set_total_chunks(totalChunks);
    c.set_chunk_data(modelBlob.data() + off, len);
    c.set_is_final_chunk(i == totalChunks - 1);
    c.set_num_examples(numExamples);
    c.set_codec(codec);
    c.set_compressed(compressed);
    c.set_total_bytes(total);
    c.set_sha256(sha);
    if (!writer->Write(c)) break;  // broken stream
  }
  writer->WritesDone();
  grpc::Status s = writer->Finish();
  if (!s.ok()) throw std::runtime_error("SubmitModelUpdateStream failed: " + statusStr(s));
  return resp;
}

void FedLearnClient::reportClientMetrics(const v2::ReportClientMetricsRequest& metrics) {
  v2::ReportClientMetricsResponse resp;
  grpc::ClientContext ctx;
  // Best-effort: telemetry must never break a training round, so the status is ignored.
  trainingStub_->ReportClientMetrics(&ctx, metrics, &resp);
}

// ---------------------------------------------------------------------------
// Dual heartbeat (own thread + own channel)
// ---------------------------------------------------------------------------
void FedLearnClient::startHeartbeat(const std::string& runId, const std::string& clientId,
                                    int currentRound) {
  stopHeartbeat();  // ensure no prior thread
  heartbeatStop_.store(false);
  abortFlag_.store(false);

  heartbeatThread_ = std::thread([this, runId, clientId, currentRound]() {
    int consecutiveFailures = 0;
    while (!heartbeatStop_.load()) {
      v2::HeartbeatRequest req;
      req.set_client_id(clientId);
      req.set_run_id(runId);
      req.set_status("TRAINING");
      req.set_current_round(currentRound);

      v2::HeartbeatResponse resp;
      grpc::ClientContext ctx;
      grpc::Status s = heartbeatStub_->Heartbeat(&ctx, req, &resp);

      if (!s.ok()) {
        if (++consecutiveFailures >= cfg_.heartbeatFailureLimit) {
          abortFlag_.store(true);  // M-H3: heartbeat death is now VISIBLE to the training loop
          break;
        }
      } else {
        consecutiveFailures = 0;
        if (resp.should_stop()) {  // server told us to abort (deadline / quorum-lost / stopped)
          abortFlag_.store(true);
          break;
        }
      }
      // Sleep the interval, but wake promptly on stop.
      for (int slept = 0; slept < cfg_.heartbeatIntervalMs && !heartbeatStop_.load(); slept += 50) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
  });
}

void FedLearnClient::stopHeartbeat() {
  heartbeatStop_.store(true);
  if (heartbeatThread_.joinable()) heartbeatThread_.join();
}

// ---------------------------------------------------------------------------
// proto <-> core marshaling
// ---------------------------------------------------------------------------
v2::GradientScalars FedLearnClient::toProtoScalars(const GradientScalars2D& g) {
  v2::GradientScalars out;
  for (const auto& step : g) {
    v2::LocalStepGradients* ls = out.add_local_steps();
    for (double v : step) ls->add_scalars(v);
  }
  return out;
}

GradientScalars2D FedLearnClient::fromProtoScalars(const v2::GradientScalars& g) {
  GradientScalars2D out;
  out.reserve(g.local_steps_size());
  for (const auto& step : g.local_steps()) {
    out.emplace_back(step.scalars().begin(), step.scalars().end());
  }
  return out;
}

Seeds2D FedLearnClient::fromProtoSeeds(const v2::PerturbationSeeds& s) {
  Seeds2D out;
  out.reserve(s.local_steps_size());
  for (const auto& step : s.local_steps()) {
    out.emplace_back(step.seeds().begin(), step.seeds().end());
  }
  return out;
}

RebuildHistory FedLearnClient::fromProtoRebuildHistory(const v2::RebuildHistory& h) {
  RebuildHistory out;
  out.reserve(h.rounds_size());
  for (const auto& r : h.rounds()) {
    RebuildRound rr;
    rr.roundNumber = r.round_number();
    rr.seeds = fromProtoSeeds(r.seeds());
    rr.gradients = fromProtoScalars(r.average_gradients());
    // RoundHistory carries no learning rate; FederatedLoop sets rr.learningRate from config["lr"].
    rr.learningRate = 0.0;
    out.push_back(std::move(rr));
  }
  return out;
}

}  // namespace fedlearn
