#include "FedLearnCoreModule.h"

#include <sys/stat.h>

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <thread>

#include <ReactCommon/TurboModuleUtils.h>  // react::createPromiseAsJSIValue, react::Promise (RN)

#include "DeviceState.h"
#include "fedlearn/DataLoader.h"
#include "fedlearn/EvalMetrics.h"
#include "fedlearn/Sha256.h"

namespace fedlearn::bridge {
namespace {

// Minimal, dependency-free base64 decoder (skips '=' padding + whitespace). Used to stage downloaded
// bundle files: provisionTrainingBundle base64-encodes each binary and passes it over JSI.
std::string base64Decode(const std::string& in) {
  auto sextet = [](unsigned char c) -> int {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;  // '=' padding, whitespace, or newline
  };
  std::string out;
  out.reserve(in.size() * 3 / 4);
  int buf = 0, bits = 0;
  for (unsigned char c : in) {
    const int v = sextet(c);
    if (v < 0) continue;
    buf = (buf << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<char>((buf >> bits) & 0xFF));
    }
  }
  return out;
}

// Case-insensitive hex-digest compare without an early exit on the first differing byte (the XOR
// accumulator runs the full length regardless of where a mismatch sits). Not load-bearing secrecy —
// the expected digest arrives from the backend, not a key — but it costs nothing to avoid handing a
// tamperer a position oracle. Case-insensitive because Sha256::hexDigest is lowercase while the
// backend's encoder is not contractually pinned to a case.
bool hexDigestEquals(const std::string& a, const std::string& b) {
  if (a.size() != b.size()) return false;
  unsigned diff = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    diff |= static_cast<unsigned>(std::tolower(static_cast<unsigned char>(a[i]))) ^
            static_cast<unsigned>(std::tolower(static_cast<unsigned char>(b[i])));
  }
  return diff == 0;
}

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
    // The CxxSpec base sets the module name ("NativeFedLearnCore") and wires methodMap_ to delegate
    // JS calls to this class's methods (registerClient, loadModel, runDeComFLRound, the setters, …).
    : react::NativeFedLearnCoreCxxSpec<FedLearnCoreModule>(jsInvoker),
      jsInvoker_(std::move(jsInvoker)),
      dataDir_(std::move(dataDir)) {}

FedLearnCoreModule::~FedLearnCoreModule() {
  if (net_) net_->requestAbort();  // trip the abort so any in-flight round returns promptly
  joinAllWorkers();                // wait for worker threads BEFORE members are destroyed (no UAF)
  if (net_) net_->stopHeartbeat();
}

void FedLearnCoreModule::reapFinishedWorkers() {
  // caller holds workersMutex_. Join+drop workers that have finished so thread handles don't leak
  // across a long training session.
  for (auto it = workers_.begin(); it != workers_.end();) {
    if (it->done->load()) {
      if (it->thread.joinable()) it->thread.join();
      it = workers_.erase(it);
    } else {
      ++it;
    }
  }
}

void FedLearnCoreModule::joinAllWorkers() {
  std::vector<Worker> pending;
  { std::lock_guard<std::mutex> lk(workersMutex_); pending.swap(workers_); }
  for (auto& w : pending) {
    if (w.thread.joinable()) w.thread.join();
  }
}

void FedLearnCoreModule::setMetricsProvider(std::function<DeviceMetrics()> provider) {
  metricsProvider_ = std::move(provider);
}

void FedLearnCoreModule::applyTrainingDataFromFiles(const std::string& inputsF32Path,
                                                    const std::vector<int64_t>& inputShape,
                                                    const std::string& targetsI64Path) {
  std::lock_guard<std::mutex> lk(stateMutex_);
  // fromRawFiles returns an OwnedBatch (owns the std::vectors); keep it in a member and expose a view.
  // Assigning the OwnedBatch temporary to a non-owning DataBatch would dangle into freed storage and be
  // read every round (evalBatch / deComFLRound / fedAvgRound).
  trainingOwner_ = fedlearn::DataLoader::fromRawFiles(inputsF32Path, inputShape, targetsI64Path);
  trainingBatch_ = trainingOwner_.view();
  dataLoaded_ = true;
}

void FedLearnCoreModule::applyModelManifest(const ModelManifest& manifest) {
  std::lock_guard<std::mutex> lk(stateMutex_);
  manifest_ = manifest;
  manifestSet_ = true;
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
  // No torch_version gate: RandnEngine makes the perturbation RNG version-independent (T9).
  loop_ = std::make_unique<fedlearn::FederatedLoop>(*net_, mm_);

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
  // Do NOT take stateMutex_: a running round holds it for its ENTIRE duration, so acquiring it here
  // would block stop() until the round already finished (a no-op mid-round). Flip the abort flag
  // lock-free so the round's shouldStop() poll breaks out, then stop the heartbeat.
  if (net_) {
    net_->requestAbort();
    net_->stopHeartbeat();
  }
}

ModelInfo FedLearnCoreModule::doLoadModel(const std::string& modelPath,
                                          const std::string& expectedSha256) {
  std::lock_guard<std::mutex> lk(stateMutex_);
  if (!manifestSet_) {
    throw std::runtime_error(
        "setModelManifest must be called before loadModel (ExecuTorch weights-as-inputs needs the "
        "param layout + infer .pte from the model's sidecar manifest)");
  }
  fedlearn::ModelInfo info;
  // ModelManager OWNS the trainable params (the .pte is weight-free) and sha256-verifies before
  // load (E8). model_ is the loss graph; inferModel_ is the separate forward(flat,x)->logits graph.
  mm_.loadModel(modelPath, expectedSha256, manifest_.paramLayout, manifest_.totalParamCount, &info);
  model_ = std::make_unique<fedlearn::ExecutorchModel>(modelPath, expectedSha256);
  inferModel_ =
      std::make_unique<fedlearn::ExecutorchModel>(manifest_.inferPtePath, manifest_.inferSha256);
  modelLoaded_ = true;
  ModelInfo out;
  out.paramCount = info.paramCount;
  out.trainableParamCount = info.trainableParamCount;
  out.sha256 = info.sha256;
  out.tier = info.tier.empty() ? "1M" : info.tier;  // below 1M still reports the smallest tier
  return out;
}

void FedLearnCoreModule::evalBatch(double& outLoss, double& outAccuracy) {
  // ExecuTorch weights-as-inputs: loss graph -> cross-entropy; infer graph -> logits for argmax.
  const std::vector<float>& flat = mm_.getFlatParams();
  const fedlearn::DataBatch& b = trainingBatch_;
  const int64_t n = b.numSamples;

  // MO-6 bounds guard: an unstaged / 0-sample batch must not reach the model (cross-entropy mean over 0
  // samples is NaN) nor the accuracy path. Report a neutral (0,0) — "not evaluable" — rather than crash.
  if (n <= 0 || b.inputs == nullptr || b.targets == nullptr) {
    outLoss = 0.0;
    outAccuracy = 0.0;
    return;
  }

  outLoss = static_cast<double>(model_->loss(flat, b.inputs, b.inputShape, b.targets, n));

  // REAL accuracy: argmax of the infer logits vs targets (no NaN, no exp(-loss) fake). argmaxCorrect is
  // bounds-safe — it guards the empty/short/ragged infer output that a naive logits[row*classes] loop
  // would OOB-read (n>0 but empty logits -> classes 0 -> logits[0] on an empty buffer). See EvalMetrics.h.
  const std::vector<float> logits = inferModel_->infer(flat, b.inputs, b.inputShape);
  const fedlearn::AccuracyCount acc = fedlearn::argmaxCorrect(logits, b.targets, n);
  outAccuracy =
      acc.scored > 0 ? static_cast<double>(acc.correct) / static_cast<double>(acc.scored) : 0.0;
}

RoundResult FedLearnCoreModule::doRunDeComFLRound(const std::string& runId, const RoundConfig& cfg) {
  std::lock_guard<std::mutex> lk(stateMutex_);
  requireReady();
  const auto t0 = std::chrono::steady_clock::now();
  fedlearn::RoundOutcome outcome = loop_->deComFLRound(*model_, runId, clientId_, trainingBatch_);
  const auto t1 = std::chrono::steady_clock::now();
  if (outcome.shouldStop) {
    throw std::runtime_error("STOP: " + outcome.note);  // RN treats a STOP-prefixed reject as a clean stop
  }
  RoundResult r;
  r.round = outcome.round;
  r.reverted = true;  // DeComFL snapshot-restore invariant
  // Report the SERVER-authoritative K/P actually used this round (the server may override the client
  // cfg), not cfg.numLocalSteps/numPerturbations.
  r.scalarsTransmitted = static_cast<int64_t>(outcome.scalarsK) * outcome.scalarsP;
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
  // FedAvg is now ZO-SGD (Constraint 7): K ZO-SGD steps, each averaging P forward-difference
  // estimates; the upload is the per-(k,p) seeds + g-scalars, NOT a weight blob.
  fedlearn::RoundOutcome outcome =
      loop_->fedAvgRound(*model_, runId, clientId_, trainingBatch_, cfg.numLocalSteps,
                         cfg.learningRate, cfg.mu, cfg.numPerturbations);
  const auto t1 = std::chrono::steady_clock::now();
  if (outcome.shouldStop) throw std::runtime_error("STOP: " + outcome.note);
  RoundResult r;
  r.round = outcome.round;
  r.reverted = false;  // FedAvg ZO-SGD keeps the locally-advanced params
  r.scalarsTransmitted = static_cast<int64_t>(outcome.scalarsK) * outcome.scalarsP;  // actual K/P used
  r.uplinkBytes = r.scalarsTransmitted * 8;  // K*P doubles uploaded (scalar wedge, not a blob)
  r.computeMs = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  evalBatch(r.loss, r.accuracy);
  return r;
}

InferResult FedLearnCoreModule::doInfer(const std::string& inputJson) {
  std::lock_guard<std::mutex> lk(stateMutex_);
  if (!modelLoaded_) throw std::runtime_error("loadModel must be called before infer");
  std::vector<float> feats = parseFloatArray(inputJson);

  // Infer graph: forward(flat, x) -> logits, one sample (shape {1, features}).
  const std::vector<float> logits =
      inferModel_->infer(mm_.getFlatParams(), feats.data(), {1, static_cast<int64_t>(feats.size())});
  const int64_t c = static_cast<int64_t>(logits.size());

  InferResult r;
  r.logits.reserve(static_cast<size_t>(c));
  r.probabilities.reserve(static_cast<size_t>(c));

  // REAL softmax over the logits (kills the exp(-loss) fake, C5 §3): numerically-stable.
  double maxLogit = c > 0 ? static_cast<double>(logits[0]) : 0.0;
  int argmax = 0;
  for (int64_t i = 0; i < c; ++i) {
    const double v = static_cast<double>(logits[static_cast<size_t>(i)]);
    if (v > maxLogit) { maxLogit = v; argmax = static_cast<int>(i); }
  }
  double sumExp = 0.0;
  for (int64_t i = 0; i < c; ++i) sumExp += std::exp(static_cast<double>(logits[static_cast<size_t>(i)]) - maxLogit);
  for (int64_t i = 0; i < c; ++i) {
    const double l = static_cast<double>(logits[static_cast<size_t>(i)]);
    r.logits.push_back(l);
    r.probabilities.push_back(sumExp > 0.0 ? std::exp(l - maxLogit) / sumExp : 0.0);
  }
  r.argmax = argmax;
  return r;
}

DeviceMetrics FedLearnCoreModule::doGetDeviceMetrics() {
  if (metricsProvider_) return metricsProvider_();  // full custom override (rarely needed)
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
  // thermal/battery are pushed in from the platform layer (Android DeviceState.kt / iOS
  // DeviceState.swift -> FedLearnCoreSetDeviceState). RSS comes from /proc above.
  const PlatformDeviceState ps = getPlatformDeviceState();
  d.thermalState = ps.thermalState;
  d.batteryLevel = ps.batteryLevel;
  d.batteryCharging = ps.batteryCharging;
  return d;
}

std::string FedLearnCoreModule::doStageBundleFile(const std::string& filename,
                                                  const std::string& base64Data,
                                                  const std::string& expectedSha256) {
  // basename only — strip any path components so a server-supplied name can't escape the bundle dir.
  std::string name = filename;
  const auto slash = name.find_last_of("/\\");
  if (slash != std::string::npos) name = name.substr(slash + 1);
  if (name.empty() || name == "." || name == "..") {
    throw std::runtime_error("stageBundleFile: invalid filename");
  }
  const std::string bytes = base64Decode(base64Data);
  // Untrusted-input rule (MO-7): verify the DECODED bytes against the bundle's declared hash BEFORE
  // anything touches disk — inputs.f32/targets.i64 are fed straight to training with no later check
  // (unlike the .pte graphs, which loadModel re-verifies). A tampered file is never staged.
  if (expectedSha256.empty()) {
    throw std::runtime_error("stageBundleFile: no expected sha256 declared for " + name);
  }
  const std::string actual = Sha256::hexDigest(bytes);
  if (!hexDigestEquals(actual, expectedSha256)) {
    throw std::runtime_error("stageBundleFile: sha256 mismatch for " + name + " (expected " +
                             expectedSha256 + ", got " + actual + ")");
  }
  const std::string dir = dataDir_ + "/bundle";
  ::mkdir(dir.c_str(), 0700);  // idempotent; ignore EEXIST
  const std::string path = dir + "/" + name;
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("stageBundleFile: cannot open " + path);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  f.flush();
  if (!f) throw std::runtime_error("stageBundleFile: write failed for " + path);
  return path;
}

// ============================================================================
// JSI LAYER (RN New Architecture, version-specific — see the header banner).
// Each method: capture args as plain C++, run the matching do* on a worker thread, and
// resolve/reject the JS Promise via the CallInvoker. createPromiseAsJSIValue + Promise are RN
// helpers; reconcile signatures against the generated CxxSpec.
// ============================================================================
// Run `work` (blocking, on a worker) then `build` the jsi result — the jsi::Runtime is touched ONLY on
// the JS thread, never on the worker and never through a reference captured across the async boundary.
// The worker thread is TRACKED (not detached) so ~FedLearnCoreModule can join it before members
// are destroyed — a detached worker (which still reaches module state through `work`) would otherwise
// resume against a freed module (use-after-free). The worker captures ONLY value-copyable, runtime-
// independent state (no jsi::Runtime, no `this`). It hands the plain-C++ result back to the JS thread via
// the CallInvoker; the runtime used to build/resolve is the one the invoker passes to the callback AT
// EXECUTION TIME (RN 0.80: react::CallFunc == std::function<void(jsi::Runtime&)>), which the invoker runs
// on the JS thread while that runtime is live. So no captured Runtime& is ever dereferenced off-thread or
// after teardown. (The previous shape captured the executor's rt2 across the worker; by reference-collapse
// that bound to the real runtime, but left it exposed to a use-after-free if the JS runtime was torn down
// mid-round — e.g. an RN instance reload while a long round is in flight; MO-9.)
template <typename Work, typename Build>
jsi::Value FedLearnCoreModule::runOnWorker(jsi::Runtime& rt, Work work, Build build) {
  auto invoker = jsInvoker_;
  return react::createPromiseAsJSIValue(
      rt, [this, invoker, work, build](jsi::Runtime& rt2, std::shared_ptr<react::Promise> promise) {
        // MO-18: react::Promise is a LongLivedObject holding a jsi::Runtime& + resolve_/reject_
        // jsi::Functions, but createPromiseAsJSIValue does NOT register it in the collection — so nothing
        // force-releases its JS handles on the JS thread when the runtime is torn down (RN reload / Fast
        // Refresh mid-round), and the last shared_ptr could run ~jsi::Function against a dead/off-thread
        // runtime (Pointer::invalidate UAF). Adopt RN's own AsyncPromise/AsyncCallback contract
        // (LongLivedObject.h:24-29): the per-runtime collection owns the SOLE strong ref; off the JS
        // thread we hold only a weak_ptr. At teardown ~TurboModuleBinding clear()s the collection on the
        // JS thread while the runtime is still valid, destroying resolve_/reject_ safely; a late
        // worker/callback lock()s weak -> null -> no-op. No jsi::Function is ever destroyed off the JS
        // thread or against a dead runtime. (add() runs here, synchronously on the JS thread.)
        react::LongLivedObjectCollection::get(rt2).add(promise);
        std::weak_ptr<react::Promise> weak = promise;
        auto done = std::make_shared<std::atomic<bool>>(false);
        // The worker captures no jsi::Runtime, no `this`, and only a WEAK ref to the promise: work/build
        // are self-contained callables and the result crosses back as a plain C++ value. (The tracked-not-
        // detached worker + destructor joinAllWorkers separately guard the `this` that `work` captures — a
        // different UAF; MO-9.)
        std::thread t([invoker, weak, work, build, done]() {
          try {
            auto result = work();  // blocking C++ (touches module state via `work`; joined before teardown)
            // Marshal back onto the JS thread. `jsRt` is supplied by the CallInvoker at execution time on
            // the JS thread; lock the weak promise (null after teardown) before touching any jsi value, and
            // allowRelease() so the collection drops its strong ref on the JS thread post-resolve.
            invoker->invokeAsync([weak, build, result](jsi::Runtime& jsRt) {
              if (auto p = weak.lock()) {
                p->resolve(build(jsRt, result));
                p->allowRelease();
              }
            });
          } catch (const std::exception& e) {
            std::string msg = e.what();
            invoker->invokeAsync([weak, msg](jsi::Runtime&) {
              if (auto p = weak.lock()) {
                p->reject(msg);
                p->allowRelease();
              }
            });
          } catch (...) {
            // A non-std::exception escaping the worker would std::terminate the whole app; reject
            // instead so the promise is always settled (and its collection ref always released).
            invoker->invokeAsync([weak](jsi::Runtime&) {
              if (auto p = weak.lock()) {
                p->reject("unknown native error");
                p->allowRelease();
              }
            });
          }
          done->store(true);
        });
        std::lock_guard<std::mutex> lk(workersMutex_);
        reapFinishedWorkers();
        workers_.push_back(Worker{std::move(t), done});
      });
}

jsi::Value FedLearnCoreModule::registerClient(jsi::Runtime& rt, jsi::String serverAddress,
                                              jsi::String runId, jsi::String clientId,
                                              jsi::String enrollmentToken, bool useTls) {
  std::string addr = serverAddress.utf8(rt), run = runId.utf8(rt), cid = clientId.utf8(rt),
              tok = enrollmentToken.utf8(rt);
  return runOnWorker(
      rt, [this, addr, run, cid, tok, useTls]() { return doRegister(addr, run, cid, tok, useTls); },
      [](jsi::Runtime& r, const RegisterResult& v) { return toJs(r, v); });
}

jsi::Value FedLearnCoreModule::getServerStatus(jsi::Runtime& rt, jsi::String runId) {
  std::string run = runId.utf8(rt);
  return runOnWorker(
      rt, [this, run]() { return doGetServerStatus(run); },
      [](jsi::Runtime& r, const ServerStatus& v) { return toJs(r, v); });
}

jsi::Value FedLearnCoreModule::stop(jsi::Runtime& rt) {
  return runOnWorker(
      rt, [this]() { doStop(); return true; },
      [](jsi::Runtime&, const bool&) { return jsi::Value::undefined(); });
}

jsi::Value FedLearnCoreModule::setModelManifest(jsi::Runtime& rt, jsi::Object manifest) {
  // Unmarshal on the JS thread (jsi values are runtime/thread-bound) into the plain native struct,
  // then apply on a worker via the platform-hook overload.
  ModelManifest m;
  jsi::Array layout = manifest.getProperty(rt, "paramLayout").asObject(rt).asArray(rt);
  const size_t n = layout.size(rt);
  m.paramLayout.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    jsi::Object ps = layout.getValueAtIndex(rt, i).asObject(rt);
    fedlearn::ParamSpec spec;
    spec.name = ps.getProperty(rt, "name").asString(rt).utf8(rt);
    jsi::Array shape = ps.getProperty(rt, "shape").asObject(rt).asArray(rt);
    const size_t sn = shape.size(rt);
    spec.shape.reserve(sn);
    for (size_t j = 0; j < sn; ++j) {
      spec.shape.push_back(static_cast<int64_t>(shape.getValueAtIndex(rt, j).asNumber()));
    }
    m.paramLayout.push_back(std::move(spec));
  }
  m.totalParamCount = static_cast<int64_t>(manifest.getProperty(rt, "totalParamCount").asNumber());
  m.inferPtePath = manifest.getProperty(rt, "inferPtePath").asString(rt).utf8(rt);
  m.inferSha256 = manifest.getProperty(rt, "inferSha256").asString(rt).utf8(rt);
  return runOnWorker(
      rt, [this, m]() { applyModelManifest(m); return true; },
      [](jsi::Runtime&, const bool&) { return jsi::Value::undefined(); });
}

jsi::Value FedLearnCoreModule::setTrainingDataFromFiles(jsi::Runtime& rt, jsi::String inputsF32Path,
                                                        jsi::Array inputShape,
                                                        jsi::String targetsI64Path) {
  std::string ip = inputsF32Path.utf8(rt), tp = targetsI64Path.utf8(rt);
  std::vector<int64_t> shape;
  const size_t sn = inputShape.size(rt);
  shape.reserve(sn);
  for (size_t j = 0; j < sn; ++j) {
    shape.push_back(static_cast<int64_t>(inputShape.getValueAtIndex(rt, j).asNumber()));
  }
  return runOnWorker(
      rt, [this, ip, shape, tp]() { applyTrainingDataFromFiles(ip, shape, tp); return true; },
      [](jsi::Runtime&, const bool&) { return jsi::Value::undefined(); });
}

jsi::Value FedLearnCoreModule::stageBundleFile(jsi::Runtime& rt, jsi::String filename,
                                               jsi::String base64Data, jsi::String expectedSha256) {
  std::string name = filename.utf8(rt), b64 = base64Data.utf8(rt), sha = expectedSha256.utf8(rt);
  return runOnWorker(
      rt, [this, name, b64, sha]() { return doStageBundleFile(name, b64, sha); },
      [](jsi::Runtime& r, const std::string& p) { return jsi::String::createFromUtf8(r, p); });
}

jsi::Value FedLearnCoreModule::loadModel(jsi::Runtime& rt, jsi::String modelPath,
                                         jsi::String expectedSha256) {
  std::string path = modelPath.utf8(rt), sha = expectedSha256.utf8(rt);
  return runOnWorker(
      rt, [this, path, sha]() { return doLoadModel(path, sha); },
      [](jsi::Runtime& r, const ModelInfo& v) { return toJs(r, v); });
}

jsi::Value FedLearnCoreModule::runDeComFLRound(jsi::Runtime& rt, jsi::String runId, jsi::Object config) {
  std::string run = runId.utf8(rt);
  RoundConfig cfg = roundConfigFromJs(rt, config);
  return runOnWorker(
      rt, [this, run, cfg]() { return doRunDeComFLRound(run, cfg); },
      [](jsi::Runtime& r, const RoundResult& v) { return toJs(r, v); });
}

jsi::Value FedLearnCoreModule::runFedAvgRound(jsi::Runtime& rt, jsi::String runId, jsi::Object config) {
  std::string run = runId.utf8(rt);
  RoundConfig cfg = roundConfigFromJs(rt, config);
  return runOnWorker(
      rt, [this, run, cfg]() { return doRunFedAvgRound(run, cfg); },
      [](jsi::Runtime& r, const RoundResult& v) { return toJs(r, v); });
}

jsi::Value FedLearnCoreModule::infer(jsi::Runtime& rt, jsi::String inputJson) {
  std::string in = inputJson.utf8(rt);
  return runOnWorker(
      rt, [this, in]() { return doInfer(in); },
      [](jsi::Runtime& r, const InferResult& v) { return toJs(r, v); });
}

jsi::Value FedLearnCoreModule::getDeviceMetrics(jsi::Runtime& rt) {
  return runOnWorker(
      rt, [this]() { return doGetDeviceMetrics(); },
      [](jsi::Runtime& r, const DeviceMetrics& v) { return toJs(r, v); });
}

}  // namespace fedlearn::bridge
