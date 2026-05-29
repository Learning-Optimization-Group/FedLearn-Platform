#include "FedLearnCoreModule.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <thread>

#include <ReactCommon/TurboModuleUtils.h>  // react::createPromiseAsJSIValue, react::Promise (RN)

#include "fedlearn/DataLoader.h"
#include "fedlearn/ZerothOrderEstimator.h"

#include <torch/torch.h>

namespace fedlearn::bridge {
namespace {

// ---- struct -> jsi::Object converters (typed, field-by-field; NO hand-built JSON) ----
jsi::Value toJs(jsi::Runtime& rt, const RegisterResult& r) {
  jsi::Object o(rt);
  o.setProperty(rt, "accepted", r.accepted);
  o.setProperty(rt, "message", jsi::String::createFromUtf8(rt, r.message));
  o.setProperty(rt, "assignedRound", static_cast<double>(r.assignedRound));
  o.setProperty(rt, "serverProtocolVersion", static_cast<double>(r.serverProtocolVersion));
  return o;
}
jsi::Value toJs(jsi::Runtime& rt, const ServerStatus& s) {
  jsi::Object o(rt);
  o.setProperty(rt, "serverState", jsi::String::createFromUtf8(rt, s.serverState));
  o.setProperty(rt, "currentRound", static_cast<double>(s.currentRound));
  o.setProperty(rt, "requiredClientsForRound", static_cast<double>(s.requiredClientsForRound));
  o.setProperty(rt, "receivedUpdatesThisRound", static_cast<double>(s.receivedUpdatesThisRound));
  o.setProperty(rt, "activeClients", static_cast<double>(s.activeClients));
  o.setProperty(rt, "roundDeadlineUnixMs", static_cast<double>(s.roundDeadlineUnixMs));
  return o;
}
jsi::Value toJs(jsi::Runtime& rt, const RoundResult& r) {
  jsi::Object o(rt);
  o.setProperty(rt, "round", static_cast<double>(r.round));
  o.setProperty(rt, "loss", r.loss);
  o.setProperty(rt, "accuracy", r.accuracy);
  o.setProperty(rt, "scalarsTransmitted", static_cast<double>(r.scalarsTransmitted));
  o.setProperty(rt, "uplinkBytes", static_cast<double>(r.uplinkBytes));
  o.setProperty(rt, "downlinkBytes", static_cast<double>(r.downlinkBytes));
  o.setProperty(rt, "computeMs", static_cast<double>(r.computeMs));
  o.setProperty(rt, "reverted", r.reverted);
  return o;
}
jsi::Value toJs(jsi::Runtime& rt, const ModelInfo& m) {
  jsi::Object o(rt);
  o.setProperty(rt, "paramCount", static_cast<double>(m.paramCount));
  o.setProperty(rt, "trainableParamCount", static_cast<double>(m.trainableParamCount));
  o.setProperty(rt, "sha256", jsi::String::createFromUtf8(rt, m.sha256));
  o.setProperty(rt, "tier", jsi::String::createFromUtf8(rt, m.tier));
  return o;
}
jsi::Value toJs(jsi::Runtime& rt, const DeviceMetrics& d) {
  jsi::Object o(rt);
  o.setProperty(rt, "peakRssBytes", static_cast<double>(d.peakRssBytes));
  o.setProperty(rt, "thermalState", jsi::String::createFromUtf8(rt, d.thermalState));
  o.setProperty(rt, "batteryLevel", d.batteryLevel);
  o.setProperty(rt, "batteryCharging", d.batteryCharging);
  return o;
}
jsi::Value toJs(jsi::Runtime& rt, const InferResult& r) {
  jsi::Object o(rt);
  jsi::Array logits(rt, r.logits.size());
  jsi::Array probs(rt, r.probabilities.size());
  for (size_t i = 0; i < r.logits.size(); ++i) logits.setValueAtIndex(rt, i, r.logits[i]);
  for (size_t i = 0; i < r.probabilities.size(); ++i) probs.setValueAtIndex(rt, i, r.probabilities[i]);
  o.setProperty(rt, "logits", logits);
  o.setProperty(rt, "probabilities", probs);
  o.setProperty(rt, "argmax", static_cast<double>(r.argmax));
  return o;
}

// jsi RoundConfig object -> bridge::RoundConfig
RoundConfig roundConfigFromJs(jsi::Runtime& rt, const jsi::Object& o) {
  RoundConfig c;
  c.strategy = o.getProperty(rt, "strategy").asString(rt).utf8(rt);
  c.learningRate = o.getProperty(rt, "learningRate").asNumber();
  c.mu = o.getProperty(rt, "mu").asNumber();
  c.numPerturbations = static_cast<int>(o.getProperty(rt, "numPerturbations").asNumber());
  c.numLocalSteps = static_cast<int>(o.getProperty(rt, "numLocalSteps").asNumber());
  c.gradEstimateMethod = o.getProperty(rt, "gradEstimateMethod").asString(rt).utf8(rt);
  c.seed = static_cast<int64_t>(o.getProperty(rt, "seed").asNumber());
  c.torchVersion = o.getProperty(rt, "torchVersion").asString(rt).utf8(rt);
  return c;
}

// Minimal flat-array JSON parser: "[a, b, c]" -> vector<float>. The RN layer passes the input
// sample as a JSON number array already shaped for the model (Model Testing screen, §6.5).
std::vector<float> parseFloatArray(const std::string& json) {
  std::vector<float> out;
  size_t i = 0;
  const size_t n = json.size();
  while (i < n) {
    char ch = json[i];
    if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.') {
      size_t consumed = 0;
      try {
        out.push_back(std::stof(json.substr(i), &consumed));
      } catch (...) {
        throw std::runtime_error("infer: malformed number in input JSON");
      }
      i += consumed;
    } else {
      ++i;
    }
  }
  if (out.empty()) throw std::runtime_error("infer: input JSON contained no numbers");
  return out;
}

}  // namespace

FedLearnCoreModule::FedLearnCoreModule(std::shared_ptr<react::CallInvoker> jsInvoker,
                                       std::string dataDir)
    // "NativeFedLearnCore" must match TurboModuleRegistry.getEnforcing(...) in the spec.
    : react::TurboModule("NativeFedLearnCore", jsInvoker),
      jsInvoker_(std::move(jsInvoker)),
      dataDir_(std::move(dataDir)) {
  localTorchVersion_ = deviceTorchVersion();
}

FedLearnCoreModule::~FedLearnCoreModule() {
  if (net_) net_->stopHeartbeat();
}

void FedLearnCoreModule::setMetricsProvider(std::function<DeviceMetrics()> provider) {
  metricsProvider_ = std::move(provider);
}

void FedLearnCoreModule::setTrainingDataFromFiles(const std::string& inputsF32Path,
                                                  const std::vector<int64_t>& inputShape,
                                                  const std::string& targetsI64Path) {
  std::lock_guard<std::mutex> lk(stateMutex_);
  trainingBatch_ = fedlearn::DataLoader::fromRawFiles(inputsF32Path, inputShape, targetsI64Path);
  dataLoaded_ = true;
}

std::string FedLearnCoreModule::deviceTorchVersion() {
#if defined(TORCH_VERSION_MAJOR) && defined(TORCH_VERSION_MINOR) && defined(TORCH_VERSION_PATCH)
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%d.%d.%d", TORCH_VERSION_MAJOR, TORCH_VERSION_MINOR,
                TORCH_VERSION_PATCH);
  return std::string(buf);
#else
  // Must match framework/tests/fixtures/decomfl_golden/manifest.json's torch_version.
  return "2.12.0";
#endif
}

// ============================================================================
// PURE LOGIC (no JSI) — the portable, verified-in-principle core
// ============================================================================
void FedLearnCoreModule::requireReady() const {
  if (!modelLoaded_) throw std::runtime_error("loadModel must be called before a round");
  if (!net_) throw std::runtime_error("registerClient must be called before a round");
  if (!dataLoaded_)
    throw std::runtime_error("on-device training data not set (wired by the RN app layer, task 14)");
}

RegisterResult FedLearnCoreModule::doRegister(const std::string& serverAddress,
                                              const std::string& runId, const std::string& clientId,
                                              const std::string& enrollmentToken, bool useTls) {
  std::lock_guard<std::mutex> lk(stateMutex_);
  fedlearn::GrpcClientConfig cfg;
  cfg.serverAddress = serverAddress;
  cfg.useTls = useTls;
  cfg.caCertPath = dataDir_ + "/certs/ca.pem";
  cfg.clientCertPath = dataDir_ + "/certs/client.pem";
  cfg.clientKeyPath = dataDir_ + "/certs/client.key";

  net_ = std::make_unique<fedlearn::FedLearnClient>(cfg);
  clientId_ = clientId;
  // A dev (insecure) build also tolerates a torch-version mismatch with a warning (E2);
  // release (TLS) refuses it.
  loop_ = std::make_unique<fedlearn::FederatedLoop>(*net_, mm_, localTorchVersion_,
                                                    /*allowVersionMismatch=*/!useTls);

  v2::RegisterClientResponse resp =
      net_->registerClient(runId, clientId, enrollmentToken, kProtocolVersion);

  RegisterResult out;
  out.accepted = (resp.status() == v2::RegisterClientResponse::ACCEPTED);
  out.message = resp.message();
  out.assignedRound = resp.assigned_round();
  out.serverProtocolVersion = resp.protocol_version();
  if (out.accepted) {
    net_->startHeartbeat(runId, clientId, resp.assigned_round());  // dual-heartbeat (§6.4)
  }
  return out;
}

ServerStatus FedLearnCoreModule::doGetServerStatus(const std::string& runId) {
  std::lock_guard<std::mutex> lk(stateMutex_);
  if (!net_) throw std::runtime_error("registerClient must be called before getServerStatus");
  v2::GetServerStatusResponse r = net_->getServerStatus(runId);
  ServerStatus s;
  s.serverState = v2::GetServerStatusResponse::ServerState_Name(r.server_state());
  s.currentRound = r.current_round();
  s.requiredClientsForRound = r.required_clients_for_round();
  s.receivedUpdatesThisRound = r.received_updates_this_round();
  s.activeClients = r.active_clients();
  s.roundDeadlineUnixMs = r.round_deadline_unix_ms();
  return s;
}

void FedLearnCoreModule::doStop() {
  std::lock_guard<std::mutex> lk(stateMutex_);
  if (net_) net_->stopHeartbeat();
}

ModelInfo FedLearnCoreModule::doLoadModel(const std::string& modelPath,
                                          const std::string& expectedSha256) {
  std::lock_guard<std::mutex> lk(stateMutex_);
  fedlearn::ModelInfo info;
  model_ = mm_.loadScriptModel(modelPath, expectedSha256, &info);  // verify-before-load (E8)
  modelLoaded_ = true;
  ModelInfo out;
  out.paramCount = info.paramCount;
  out.trainableParamCount = info.trainableParamCount;
  out.sha256 = info.sha256;
  out.tier = info.tier.empty() ? "1M" : info.tier;  // below 1M still reports the smallest tier
  return out;
}

void FedLearnCoreModule::evalBatch(double& outLoss, double& outAccuracy) {
  torch::NoGradGuard no_grad;
  std::vector<torch::jit::IValue> in{trainingBatch_.inputs};
  torch::Tensor logits = model_.forward(in).toTensor();
  outLoss = torch::nn::functional::cross_entropy(logits, trainingBatch_.targets).item<double>();
  torch::Tensor pred = logits.argmax(1);
  outAccuracy = pred.eq(trainingBatch_.targets).to(torch::kFloat).mean().item<double>();
}

RoundResult FedLearnCoreModule::doRunDeComFLRound(const std::string& runId, const RoundConfig& cfg) {
  std::lock_guard<std::mutex> lk(stateMutex_);
  requireReady();
  const auto t0 = std::chrono::steady_clock::now();
  fedlearn::RoundOutcome outcome = loop_->deComFLRound(model_, runId, clientId_, trainingBatch_);
  const auto t1 = std::chrono::steady_clock::now();
  if (outcome.shouldStop) {
    throw std::runtime_error("STOP: " + outcome.note);  // RN treats a STOP-prefixed reject as a clean stop
  }
  RoundResult r;
  r.round = outcome.round;
  r.reverted = true;  // DeComFL snapshot-restore invariant
  r.scalarsTransmitted = static_cast<int64_t>(cfg.numLocalSteps) * cfg.numPerturbations;
  r.uplinkBytes = r.scalarsTransmitted * 8;             // K*P doubles uploaded (the O(K*P) wedge)
  r.downlinkBytes = r.scalarsTransmitted * 8;           // K*P int64 seeds downloaded (approx)
  r.computeMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  evalBatch(r.loss, r.accuracy);                        // model is reverted -> pre-round loss
  return r;
}

RoundResult FedLearnCoreModule::doRunFedAvgRound(const std::string& runId, const RoundConfig& cfg) {
  std::lock_guard<std::mutex> lk(stateMutex_);
  requireReady();
  const auto t0 = std::chrono::steady_clock::now();
  fedlearn::RoundOutcome outcome =
      loop_->fedAvgRound(model_, runId, clientId_, trainingBatch_, cfg.numLocalSteps, cfg.learningRate);
  const auto t1 = std::chrono::steady_clock::now();
  if (outcome.shouldStop) throw std::runtime_error("STOP: " + outcome.note);
  RoundResult r;
  r.round = outcome.round;
  r.reverted = false;  // FedAvg keeps the locally-trained weights to upload
  r.scalarsTransmitted = 0;
  r.computeMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  evalBatch(r.loss, r.accuracy);
  return r;
}

InferResult FedLearnCoreModule::doInfer(const std::string& inputJson) {
  std::lock_guard<std::mutex> lk(stateMutex_);
  if (!modelLoaded_) throw std::runtime_error("loadModel must be called before infer");
  std::vector<float> feats = parseFloatArray(inputJson);
  torch::NoGradGuard no_grad;
  torch::Tensor x =
      torch::from_blob(feats.data(), {1, static_cast<int64_t>(feats.size())}, torch::kFloat32).clone();
  std::vector<torch::jit::IValue> in{x};
  torch::Tensor logits = model_.forward(in).toTensor().squeeze(0);
  torch::Tensor probs = torch::softmax(logits, 0);  // REAL softmax (kills the exp(-loss) fake, C5 §3)

  InferResult r;
  const int64_t c = logits.numel();
  r.logits.reserve(c);
  r.probabilities.reserve(c);
  auto la = logits.contiguous();
  auto pa = probs.contiguous();
  for (int64_t i = 0; i < c; ++i) {
    r.logits.push_back(la[i].item<double>());
    r.probabilities.push_back(pa[i].item<double>());
  }
  r.argmax = static_cast<int>(logits.argmax(0).item<int64_t>());
  return r;
}

DeviceMetrics FedLearnCoreModule::doGetDeviceMetrics() {
  if (metricsProvider_) return metricsProvider_();  // platform supplies thermal/battery
  DeviceMetrics d;
#if defined(__linux__) || defined(__ANDROID__)
  if (FILE* f = std::fopen("/proc/self/statm", "r")) {
    long pages_total = 0, pages_resident = 0;
    if (std::fscanf(f, "%ld %ld", &pages_total, &pages_resident) == 2) {
      d.peakRssBytes = static_cast<int64_t>(pages_resident) * static_cast<int64_t>(::sysconf(_SC_PAGESIZE));
    }
    std::fclose(f);
  }
#endif
  // thermalState/battery require platform APIs (set a metricsProvider_ from the Android/iOS layer).
  return d;
}

// ============================================================================
// JSI LAYER (RN New Architecture, version-specific — see the header banner).
// Each method: capture args as plain C++, run the matching do* on a worker thread, and
// resolve/reject the JS Promise via the CallInvoker. createPromiseAsJSIValue + Promise are RN
// helpers; reconcile signatures against the generated CxxSpec.
// ============================================================================
namespace {
// Run `work` (blocking, on a worker) then `build` the jsi result on the JS thread.
template <typename Work, typename Build>
jsi::Value promiseFrom(jsi::Runtime& rt, std::shared_ptr<react::CallInvoker> invoker, Work work,
                       Build build) {
  return react::createPromiseAsJSIValue(
      rt, [invoker, work, build](jsi::Runtime& rt2, std::shared_ptr<react::Promise> promise) {
        std::thread([&rt2, invoker, promise, work, build]() {
          try {
            auto result = work();  // blocking C++
            invoker->invokeAsync([&rt2, promise, build, result]() { promise->resolve(build(rt2, result)); });
          } catch (const std::exception& e) {
            std::string msg = e.what();
            invoker->invokeAsync([promise, msg]() { promise->reject(msg); });
          }
        }).detach();
      });
}
}  // namespace

jsi::Value FedLearnCoreModule::registerClient(jsi::Runtime& rt, jsi::String serverAddress,
                                              jsi::String runId, jsi::String clientId,
                                              jsi::String enrollmentToken, bool useTls) {
  std::string addr = serverAddress.utf8(rt), run = runId.utf8(rt), cid = clientId.utf8(rt),
              tok = enrollmentToken.utf8(rt);
  return promiseFrom(
      rt, jsInvoker_, [this, addr, run, cid, tok, useTls]() { return doRegister(addr, run, cid, tok, useTls); },
      [](jsi::Runtime& r, const RegisterResult& v) { return toJs(r, v); });
}

jsi::Value FedLearnCoreModule::getServerStatus(jsi::Runtime& rt, jsi::String runId) {
  std::string run = runId.utf8(rt);
  return promiseFrom(
      rt, jsInvoker_, [this, run]() { return doGetServerStatus(run); },
      [](jsi::Runtime& r, const ServerStatus& v) { return toJs(r, v); });
}

jsi::Value FedLearnCoreModule::stop(jsi::Runtime& rt) {
  return promiseFrom(
      rt, jsInvoker_, [this]() { doStop(); return true; },
      [](jsi::Runtime&, const bool&) { return jsi::Value::undefined(); });
}

jsi::Value FedLearnCoreModule::loadModel(jsi::Runtime& rt, jsi::String modelPath,
                                         jsi::String expectedSha256) {
  std::string path = modelPath.utf8(rt), sha = expectedSha256.utf8(rt);
  return promiseFrom(
      rt, jsInvoker_, [this, path, sha]() { return doLoadModel(path, sha); },
      [](jsi::Runtime& r, const ModelInfo& v) { return toJs(r, v); });
}

jsi::Value FedLearnCoreModule::runDeComFLRound(jsi::Runtime& rt, jsi::String runId, jsi::Object config) {
  std::string run = runId.utf8(rt);
  RoundConfig cfg = roundConfigFromJs(rt, config);
  return promiseFrom(
      rt, jsInvoker_, [this, run, cfg]() { return doRunDeComFLRound(run, cfg); },
      [](jsi::Runtime& r, const RoundResult& v) { return toJs(r, v); });
}

jsi::Value FedLearnCoreModule::runFedAvgRound(jsi::Runtime& rt, jsi::String runId, jsi::Object config) {
  std::string run = runId.utf8(rt);
  RoundConfig cfg = roundConfigFromJs(rt, config);
  return promiseFrom(
      rt, jsInvoker_, [this, run, cfg]() { return doRunFedAvgRound(run, cfg); },
      [](jsi::Runtime& r, const RoundResult& v) { return toJs(r, v); });
}

jsi::Value FedLearnCoreModule::infer(jsi::Runtime& rt, jsi::String inputJson) {
  std::string in = inputJson.utf8(rt);
  return promiseFrom(
      rt, jsInvoker_, [this, in]() { return doInfer(in); },
      [](jsi::Runtime& r, const InferResult& v) { return toJs(r, v); });
}

jsi::Value FedLearnCoreModule::getDeviceMetrics(jsi::Runtime& rt) {
  return promiseFrom(
      rt, jsInvoker_, [this]() { return doGetDeviceMetrics(); },
      [](jsi::Runtime& r, const DeviceMetrics& v) { return toJs(r, v); });
}

}  // namespace fedlearn::bridge
