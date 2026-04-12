#include "FedLearnClient.h"
#include <torch/torch.h>
#include <chrono>

namespace fedlearn {

static grpc::ChannelArguments makeChannelArgs() {
  grpc::ChannelArguments args;
  args.SetInt(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, 1024 * 1024 * 1024);
  args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, 1024 * 1024 * 1024);
  args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 120000);
  args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 60000);
  args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
  args.SetInt(GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA, 0);
  args.SetInt(GRPC_ARG_HTTP2_MIN_SENT_PING_INTERVAL_WITHOUT_DATA_MS, 120000);
  args.SetInt(GRPC_ARG_MAX_CONNECTION_IDLE_MS, 7200000);
  args.SetInt(GRPC_ARG_MAX_CONNECTION_AGE_MS, 14400000);
  return args;
}

FedLearnClient::FedLearnClient(const std::string& server_address,
                               const std::string& client_id)
    : client_id_(client_id) {
  auto args = makeChannelArgs();
  channel_ = grpc::CreateCustomChannel(
      server_address, grpc::InsecureChannelCredentials(), args);
  stub_ = fedlearn::v1::FederatedLearningService::NewStub(channel_);

  heartbeat_channel_ = grpc::CreateCustomChannel(
      server_address, grpc::InsecureChannelCredentials(), args);
  heartbeat_stub_ =
      fedlearn::v1::FederatedLearningService::NewStub(heartbeat_channel_);

  connected_ = true;
  log("FedLearnClient", "Created channels to " + server_address);
}

FedLearnClient::~FedLearnClient() {
  close();
}

bool FedLearnClient::registerClient() {
  ::fedlearn::v1::RegisterClientRequest req;
  req.set_client_id(client_id_);

  ::fedlearn::v1::RegisterClientResponse res;
  grpc::ClientContext ctx;

  auto status = stub_->RegisterClient(&ctx, req, &res);
  if (!status.ok()) {
    log("FedLearnClient",
        "Registration failed: " + status.error_message());
    return false;
  }

  bool accepted =
      res.status() == ::fedlearn::v1::RegisterClientResponse::ACCEPTED;
  log("FedLearnClient",
      accepted ? "Registration accepted" : "Registration rejected");
  return accepted;
}

// Helper: parse ModelParameters proto → OrderedDict<string, torch::Tensor>
static OrderedDict protoToTensors(
    const ::fedlearn::v1::ModelParameters& proto) {
  OrderedDict tensors;
  for (const auto& kv : proto.tensors()) {
    const auto& t = kv.second;
    // Build shape
    std::vector<int64_t> shape(t.dims().begin(), t.dims().end());
    // Determine dtype
    torch::ScalarType dtype = torch::kFloat32;
    if (t.dtype() == "float64" || t.dtype() == "torch.float64")
      dtype = torch::kFloat64;
    else if (t.dtype() == "int64" || t.dtype() == "torch.int64")
      dtype = torch::kInt64;
    else if (t.dtype() == "int32" || t.dtype() == "torch.int32")
      dtype = torch::kInt32;
    else if (t.dtype() == "float16" || t.dtype() == "torch.float16")
      dtype = torch::kFloat16;

    // Create tensor from raw bytes
    const std::string& raw = t.data();
    auto options = torch::TensorOptions().dtype(dtype);
    auto tensor = torch::from_blob(
        const_cast<char*>(raw.data()), shape, options).clone();
    tensors[kv.first] = tensor;
  }
  return tensors;
}

// Helper: OrderedDict → ModelParameters proto
static void tensorsToProto(const OrderedDict& tensors,
                            ::fedlearn::v1::ModelParameters* proto,
                            int64_t num_examples) {
  proto->set_num_examples_trained(num_examples);
  auto* tmap = proto->mutable_tensors();
  for (const auto& kv : tensors) {
    ::fedlearn::v1::Tensor t;
    auto cpu = kv.second.contiguous().cpu();
    // dtype string (numpy convention as Python side expects)
    std::string dtype_str;
    if (cpu.scalar_type() == torch::kFloat32) dtype_str = "float32";
    else if (cpu.scalar_type() == torch::kFloat64) dtype_str = "float64";
    else if (cpu.scalar_type() == torch::kFloat16) dtype_str = "float16";
    else if (cpu.scalar_type() == torch::kInt64) dtype_str = "int64";
    else if (cpu.scalar_type() == torch::kInt32) dtype_str = "int32";
    else dtype_str = "float32";
    t.set_dtype(dtype_str);
    for (int d : cpu.sizes()) t.add_dims(d);
    t.set_data(cpu.data_ptr(), cpu.nbytes());
    (*tmap)[kv.first] = std::move(t);
  }
}

FedLearnClient::GlobalModelResult FedLearnClient::getGlobalModel() {
  GlobalModelResult result{};

  ::fedlearn::v1::GetGlobalModelRequest req;
  req.set_client_id(client_id_);

  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::hours(1));

  ::fedlearn::v1::GetGlobalModelResponse response;
  auto status = stub_->GetGlobalModel(&ctx, req, &response);

  if (!status.ok()) {
    log("FedLearnClient",
        "GetGlobalModel failed: " + status.error_message());
    return result;
  }

  result.current_round = response.current_round();
  for (const auto& kv : response.config()) {
    result.config[kv.first] = kv.second;
  }

  result.tensors = protoToTensors(response.parameters());

  int64_t total_params = 0;
  for (const auto& kv : result.tensors) total_params += kv.second.numel();

  result.success = true;
  log("FedLearnClient",
      "GetGlobalModel complete: " + std::to_string(result.tensors.size()) +
          " tensors, " +
          std::to_string((total_params * 4) / (1024 * 1024)) + " MB, round " +
          std::to_string(result.current_round));
  return result;
}

bool FedLearnClient::submitUpdate(const OrderedDict& params,
                                  int64_t num_examples,
                                  int32_t round_number) {
  grpc::ClientContext ctx;
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::hours(1));

  ::fedlearn::v1::SubmitModelUpdateReque request;
  request.set_client_id(client_id_);
  request.set_trained_on_round(round_number);
  tensorsToProto(params, request.mutable_parameters(), num_examples);

  int64_t total_params = 0;
  for (const auto& kv : params) total_params += kv.second.numel();
  log("FedLearnClient",
      "Uploading " + std::to_string(params.size()) + " tensors (" +
          std::to_string((total_params * 4) / (1024 * 1024)) + " MB)");

  ::fedlearn::v1::SubmitModelUpdateResponse response;
  auto status = stub_->SubmitModelUpdate(&ctx, request, &response);

  if (!status.ok()) {
    log("FedLearnClient",
        "SubmitModelUpdate failed: " + status.error_message());
    return false;
  }

  log("FedLearnClient", "Model update accepted (round " +
      std::to_string(round_number) + ")");
  return response.received();
}

void FedLearnClient::startHeartbeat(int interval_seconds) {
  if (heartbeat_active_) return;
  heartbeat_interval_ = interval_seconds;
  heartbeat_active_ = true;

  heartbeat_thread_ = std::thread([this]() { heartbeatLoop(); });
  log("FedLearnClient",
      "Heartbeat started (every " + std::to_string(heartbeat_interval_) +
          "s)");
}

void FedLearnClient::stopHeartbeat() {
  heartbeat_active_ = false;
  if (heartbeat_thread_.joinable()) {
    heartbeat_thread_.join();
  }
  log("FedLearnClient", "Heartbeat stopped");
}

void FedLearnClient::updateStatus(const std::string& status, int step,
                                  int total) {
  std::lock_guard<std::mutex> lock(status_mutex_);
  current_status_ = status;
  current_step_ = step;
  total_steps_ = total;
}

void FedLearnClient::heartbeatLoop() {
  while (heartbeat_active_) {
    try {
      ::fedlearn::v1::HeartbeatRequest req;
      req.set_client_id(client_id_);

      {
        std::lock_guard<std::mutex> lock(status_mutex_);
        req.set_status(current_status_);
        req.set_current_step(current_step_);
        req.set_total_steps(total_steps_);
        req.set_current_round(current_round_);
      }

      ::fedlearn::v1::HeartbeatResponse res;
      grpc::ClientContext ctx;
      ctx.set_deadline(
          std::chrono::system_clock::now() + std::chrono::seconds(30));

      auto status = heartbeat_stub_->Heartbeat(&ctx, req, &res);

      if (status.ok() && res.should_stop()) {
        log("FedLearnClient", "Server requested training stop");
        heartbeat_active_ = false;
        break;
      }
    } catch (...) {
      // Heartbeat errors should not crash the thread
    }

    for (int i = 0; i < heartbeat_interval_ * 10 && heartbeat_active_; ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}

FedLearnClient::DeComFLConfigResult FedLearnClient::getDeComFLConfig() {
  DeComFLConfigResult result{};
  result.success = false;

  ::fedlearn::v1::GetDeComFLConfigRequest req;
  req.set_client_id(client_id_);

  ::fedlearn::v1::GetDeComFLConfigResponse res;
  grpc::ClientContext ctx;

  auto status = stub_->GetDeComFLConfig(&ctx, req, &res);
  if (!status.ok()) {
    log("FedLearnClient",
        "GetDeComFLConfig failed: " + status.error_message());
    return result;
  }

  result.current_round = res.current_round();
  if (result.current_round == -1) {
    result.success = true;
    return result;
  }

  // Parse seeds [K][P]
  for (const auto& local_step : res.current_seeds().local_steps()) {
    std::vector<int32_t> step_seeds;
    for (auto seed : local_step.seeds()) {
      step_seeds.push_back(seed);
    }
    result.seeds.push_back(step_seeds);
  }

  // Parse rebuild history
  for (const auto& round_hist : res.rebuild_history().rounds()) {
    RebuildRound rr;
    rr.round_number = round_hist.round_number();

    for (const auto& ls : round_hist.seeds().local_steps()) {
      std::vector<int32_t> s;
      for (auto seed : ls.seeds()) s.push_back(seed);
      rr.seeds.push_back(s);
    }

    for (const auto& lg : round_hist.average_gradients().local_steps()) {
      std::vector<double> g;
      for (auto scalar : lg.scalars()) g.push_back(scalar);
      rr.gradients.push_back(g);
    }

    result.rebuild_history.push_back(rr);
  }

  for (auto& kv : res.config()) {
    result.config[kv.first] = kv.second;
  }

  result.success = true;
  log("FedLearnClient",
      "Got DeComFL config for round " +
          std::to_string(result.current_round) + ": " +
          std::to_string(result.seeds.size()) + " local steps, " +
          std::to_string(result.rebuild_history.size()) + " missed rounds");
  return result;
}

bool FedLearnClient::submitGradientScalars(
    const std::vector<std::vector<double>>& scalars,
    int64_t num_examples, int32_t round_number) {
  ::fedlearn::v1::SubmitGradientScalarsRequest req;
  req.set_client_id(client_id_);
  req.set_trained_on_round(round_number);
  req.set_num_examples(num_examples);

  for (const auto& k_grads : scalars) {
    auto* local_step = req.mutable_gradients()->add_local_steps();
    for (double g : k_grads) {
      local_step->add_scalars(g);
    }
  }

  ::fedlearn::v1::SubmitGradientScalarsResponse res;
  grpc::ClientContext ctx;

  auto status = stub_->SubmitGradientScalars(&ctx, req, &res);
  if (!status.ok()) {
    log("FedLearnClient",
        "SubmitGradientScalars failed: " + status.error_message());
    return false;
  }

  log("FedLearnClient", "Gradient scalars accepted");
  return res.received();
}

bool FedLearnClient::waitForConnected(int timeout_seconds) {
  auto deadline = std::chrono::system_clock::now() +
                  std::chrono::seconds(timeout_seconds);
  bool ok = channel_->WaitForConnected(deadline);
  if (!ok) {
    log("FedLearnClient", "Channel failed to connect within " +
        std::to_string(timeout_seconds) + "s");
    connected_ = false;
  }
  return ok;
}

void FedLearnClient::close() {
  stopHeartbeat();
  connected_ = false;
  log("FedLearnClient", "Disconnected");
}

}  // namespace fedlearn
