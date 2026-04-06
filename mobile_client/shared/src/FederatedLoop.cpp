#include "FederatedLoop.h"

namespace fedlearn {

FederatedLoop::FederatedLoop(FedLearnClient& grpc_client,
                             ModelManager& model_mgr, DataLoader& data_loader)
    : grpc_client_(grpc_client),
      model_mgr_(model_mgr),
      data_loader_(data_loader) {
  status_.phase = "idle";
}

FederatedLoop::~FederatedLoop() {
  stop();
}

void FederatedLoop::startFedAvg(const FedAvgConfig& config) {
  if (running_) {
    log("FederatedLoop", "Already running, stop first");
    return;
  }
  running_ = true;
  loop_thread_ = std::thread([this, config]() { fedAvgLoop(config); });
}

void FederatedLoop::startDeComFL(const DeComFLConfig& config) {
  if (running_) {
    log("FederatedLoop", "Already running, stop first");
    return;
  }
  running_ = true;
  loop_thread_ = std::thread([this, config]() { deComFLLoop(config); });
}

void FederatedLoop::stop() {
  running_ = false;
  if (loop_thread_.joinable()) {
    loop_thread_.join();
  }
  setStatus("stopped");
}

FederatedLoop::Status FederatedLoop::getStatus() const {
  std::lock_guard<std::mutex> lock(status_mutex_);
  return status_;
}

void FederatedLoop::setStatus(const std::string& phase, int32_t round,
                              float loss, float accuracy, int step, int total,
                              const std::string& error) {
  std::lock_guard<std::mutex> lock(status_mutex_);
  status_.phase = phase;
  if (round >= 0) status_.round = round;
  if (loss >= 0) status_.loss = loss;
  if (accuracy >= 0) status_.accuracy = accuracy;
  if (step >= 0) status_.step = step;
  if (total >= 0) status_.total_steps = total;
  status_.error_message = error;
}

// Port of framework/src/fedlearn/client/client.py start_client (lines 33-123)
void FederatedLoop::fedAvgLoop(FedAvgConfig config) {
  int32_t last_completed_round = -1;

  setStatus("registering");
  if (!grpc_client_.registerClient()) {
    setStatus("error", -1, -1, -1, -1, -1, "Registration failed");
    running_ = false;
    return;
  }

  grpc_client_.startHeartbeat();

  while (running_) {
    try {
      // 1. Fetch global model
      setStatus("fetching");
      grpc_client_.updateStatus("fetching_model", 0, 0);

      auto model_result = grpc_client_.getGlobalModel();
      if (!model_result.success) {
        setStatus("error", -1, -1, -1, -1, -1, "Failed to fetch model");
        std::this_thread::sleep_for(std::chrono::seconds(10));
        continue;
      }

      int32_t server_round = model_result.current_round;

      if (server_round == -1) {
        log("FederatedLoop", "Server finished training. Stopping.");
        break;
      }

      if (server_round > last_completed_round) {
        log("FederatedLoop",
            "Starting local training for round " +
                std::to_string(server_round));
        setStatus("training", server_round);
        grpc_client_.setCurrentRound(server_round);
        grpc_client_.updateStatus("training", 0, 1);

        // 2. Apply global model weights
        model_mgr_.setStateDict(model_result.tensors);

        // 3. Local training
        auto batches = data_loader_.getBatches(config.batch_size, true);
        int total_steps = config.local_epochs * static_cast<int>(batches.size());
        int step = 0;
        float last_loss = 0.0f;

        for (int epoch = 0; epoch < config.local_epochs && running_; ++epoch) {
          for (auto& [input, target] : batches) {
            if (!running_) break;
            last_loss =
                model_mgr_.trainStep(input, target, config.learning_rate);
            step++;
            grpc_client_.updateStatus("training", step, total_steps);
            setStatus("training", server_round, last_loss, -1, step,
                      total_steps);
          }
        }

        if (!running_) break;

        // 4. Submit update
        setStatus("uploading", server_round);
        grpc_client_.updateStatus("submitting_update", 0, 0);

        auto state_dict = model_mgr_.getStateDict();
        if (grpc_client_.submitUpdate(state_dict,
                                       data_loader_.numTrainSamples(),
                                       server_round)) {
          log("FederatedLoop",
              "Submitted update for round " + std::to_string(server_round));
          last_completed_round = server_round;
          setStatus("waiting", server_round, last_loss);
          grpc_client_.updateStatus("idle", 0, 0);
        } else {
          setStatus("error", server_round, -1, -1, -1, -1,
                    "Failed to submit update");
          grpc_client_.updateStatus("error", 0, 0);
        }
      } else {
        log("FederatedLoop",
            "Server still in round " + std::to_string(server_round) +
                ". Waiting...");
        setStatus("waiting", server_round);
        grpc_client_.updateStatus("waiting", 0, 0);
        std::this_thread::sleep_for(std::chrono::seconds(5));
      }

    } catch (const std::exception& e) {
      log("FederatedLoop", std::string("Error: ") + e.what());
      setStatus("error", -1, -1, -1, -1, -1, e.what());
      std::this_thread::sleep_for(std::chrono::seconds(10));
    }
  }

  grpc_client_.stopHeartbeat();
  setStatus("stopped");
  running_ = false;
  log("FederatedLoop", "FedAvg loop stopped");
}

// Port of framework/src/fedlearn/client/decomfl_start.py (lines 13-111)
void FederatedLoop::deComFLLoop(DeComFLConfig config) {
  int32_t last_completed_round = -1;

  setStatus("registering");
  if (!grpc_client_.registerClient()) {
    setStatus("error", -1, -1, -1, -1, -1, "Registration failed");
    running_ = false;
    return;
  }

  grpc_client_.startHeartbeat();

  DeComFLClient decomfl_client(model_mgr_.getModule(), config.smoothing_param);

  while (running_) {
    try {
      setStatus("fetching");
      grpc_client_.updateStatus("fetching_config", 0, 0);

      auto config_result = grpc_client_.getDeComFLConfig();
      if (!config_result.success) {
        setStatus("error", -1, -1, -1, -1, -1, "Failed to fetch config");
        std::this_thread::sleep_for(std::chrono::seconds(10));
        continue;
      }

      int32_t server_round = config_result.current_round;

      if (server_round == -1) {
        log("FederatedLoop", "Server finished training. Stopping.");
        break;
      }

      if (server_round > last_completed_round) {
        log("FederatedLoop",
            "Starting DeComFL training for round " +
                std::to_string(server_round));
        setStatus("training", server_round);
        grpc_client_.setCurrentRound(server_round);

        // Rebuild model if needed
        if (!config_result.rebuild_history.empty()) {
          grpc_client_.updateStatus("rebuilding", 0, 0);
          float lr = config.learning_rate;
          if (config_result.config.count("learning_rate")) {
            lr = std::stof(config_result.config.at("learning_rate"));
          }
          decomfl_client.rebuildModel(config_result.rebuild_history, lr);
        }

        grpc_client_.updateStatus("training", 0, 1);

        TrainingConfig tc;
        tc.seeds = config_result.seeds;
        tc.learning_rate = config.learning_rate;
        if (config_result.config.count("learning_rate"))
          tc.learning_rate =
              std::stof(config_result.config.at("learning_rate"));
        if (config_result.config.count("smoothing_param"))
          tc.smoothing_param =
              std::stof(config_result.config.at("smoothing_param"));

        auto batches = data_loader_.getBatches(config.batch_size, true);
        auto [gradient_scalars, num_examples] =
            decomfl_client.fit(config_result.seeds, tc, batches);

        // Submit gradient scalars
        setStatus("uploading", server_round);
        grpc_client_.updateStatus("submitting_update", 0, 0);

        if (grpc_client_.submitGradientScalars(gradient_scalars, num_examples,
                                                server_round)) {
          log("FederatedLoop",
              "Submitted gradient scalars for round " +
                  std::to_string(server_round));
          last_completed_round = server_round;
          setStatus("waiting", server_round);
          grpc_client_.updateStatus("idle", 0, 0);
        } else {
          setStatus("error", server_round, -1, -1, -1, -1,
                    "Failed to submit gradient scalars");
        }
      } else {
        setStatus("waiting", server_round);
        grpc_client_.updateStatus("waiting", 0, 0);
        std::this_thread::sleep_for(std::chrono::seconds(5));
      }

    } catch (const std::exception& e) {
      log("FederatedLoop", std::string("DeComFL error: ") + e.what());
      setStatus("error", -1, -1, -1, -1, -1, e.what());
      std::this_thread::sleep_for(std::chrono::seconds(10));
    }
  }

  grpc_client_.stopHeartbeat();
  setStatus("stopped");
  running_ = false;
  log("FederatedLoop", "DeComFL loop stopped");
}

}  // namespace fedlearn
