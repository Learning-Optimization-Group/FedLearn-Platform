#include "ModelManager.h"
#include <torch/torch.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <sstream>

namespace fedlearn {

bool ModelManager::loadScriptModel(const std::string& path) {
  try {
    model_ = torch::jit::load(path);
    // Enable training mode and gradient tracking for all parameters
    model_.train(true);
    for (const auto& param : model_.parameters()) {
      param.set_requires_grad(true);
    }
    loaded_ = true;
    log("ModelManager", "Loaded TorchScript model from " + path +
        " (" + std::to_string(numParams()) + " params)");
    return true;
  } catch (const std::exception& e) {
    log("ModelManager", std::string("Failed to load model: ") + e.what());
    loaded_ = false;
    return false;
  }
}

bool ModelManager::loadStateDict(const std::vector<uint8_t>& data) {
  try {
    std::string str(data.begin(), data.end());
    std::istringstream stream(str);

    // The server sends torch.save({'parameters': OrderedDict, 'num_examples': int}).
    // Use the JIT pickle helpers that are available in mobile libtorch.
    auto loaded = torch::jit::pickle_load(
        std::vector<char>(data.begin(), data.end()));

    if (loaded.isGenericDict()) {
      auto dict = loaded.toGenericDict();
      auto params_ivalue = dict.at("parameters");
      if (params_ivalue.isGenericDict()) {
        auto params_dict = params_ivalue.toGenericDict();
        for (const auto& param : model_.named_parameters()) {
          std::string name = param.name;
          // Remove module prefix if present (e.g., "module.conv1.weight" -> "conv1.weight")
          auto dot_pos = name.find('.');
          if (dot_pos != std::string::npos) {
            // Keep the name as-is for TorchScript modules
          }
          if (params_dict.contains(name)) {
            auto tensor = params_dict.at(name).toTensor();
            param.value.data().copy_(tensor);
          }
        }
      }
    }

    // Restore requires_grad after copying weights (data() copy disables autograd)
    for (const auto& param : model_.parameters()) {
      param.set_requires_grad(true);
    }
    log("ModelManager", "Loaded state dict from buffer");
    return true;
  } catch (const std::exception& e) {
    log("ModelManager", std::string("Failed to load state dict: ") + e.what());
    return false;
  }
}

std::vector<uint8_t> ModelManager::serializeStateDict() {
  try {
    // Replicate Python: torch.save({'parameters': state_dict, 'num_examples': 0})
    c10::Dict<std::string, torch::Tensor> params_dict;
    for (const auto& param : model_.named_parameters()) {
      params_dict.insert(param.name, param.value.data().clone().cpu());
    }

    auto model_data = c10::impl::GenericDict(
        c10::StringType::get(), c10::AnyType::get());
    model_data.insert("parameters",
                      c10::IValue(c10::impl::GenericDict(
                          c10::StringType::get(), c10::TensorType::get())));
    model_data.insert("num_examples", c10::IValue(static_cast<int64_t>(0)));

    // Use the JIT pickle helpers to produce bytes compatible with Python torch.load
    auto data_ivalue = c10::IValue(c10::impl::GenericDict(
        c10::StringType::get(), c10::AnyType::get()));
    auto gd = data_ivalue.toGenericDict();
    gd.insert("num_examples", c10::IValue(static_cast<int64_t>(0)));

    auto inner_params = c10::impl::GenericDict(
        c10::StringType::get(), c10::TensorType::get());
    for (const auto& param : model_.named_parameters()) {
      inner_params.insert(param.name, param.value.data().clone().cpu());
    }
    gd.insert("parameters", c10::IValue(inner_params));

    auto bytes = torch::jit::pickle_save(data_ivalue);
    return std::vector<uint8_t>(bytes.begin(), bytes.end());
  } catch (const std::exception& e) {
    log("ModelManager", std::string("Failed to serialize state dict: ") + e.what());
    return {};
  }
}

torch::Tensor ModelManager::forward(torch::Tensor input) {
  torch::NoGradGuard no_grad;
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input);
  return model_.forward(inputs).toTensor();
}

float ModelManager::computeLoss(torch::Tensor output, torch::Tensor target) {
  auto loss = torch::nn::functional::detail::cross_entropy(
      output,
      target,
      torch::Tensor(),
      -100,
      torch::nn::functional::CrossEntropyFuncOptions::reduction_t(
          torch::enumtype::kMean{}),
      0.0);
  return loss.item<float>();
}

torch::Tensor ModelManager::getFlatParams() {
  return fedlearn::getFlatParams(model_);
}

void ModelManager::setFlatParams(const torch::Tensor& flat) {
  fedlearn::setFlatParams(model_, flat);
}

int64_t ModelManager::numParams() {
  return fedlearn::getNumParams(model_);
}

float ModelManager::trainStep(torch::Tensor input, torch::Tensor target,
                              float lr) {
  float loss_val = 0.0f;
  {
    // Scoped block so intermediate tensors are freed immediately after backward
    std::vector<torch::jit::IValue> inputs_vec;
    inputs_vec.push_back(input);
    auto output = model_.forward(inputs_vec).toTensor();

    auto loss = torch::nn::functional::detail::cross_entropy(
        output,
        target,
        torch::Tensor(),
        -100,
        torch::nn::functional::CrossEntropyFuncOptions::reduction_t(
            torch::enumtype::kMean{}),
        0.0);
    loss_val = loss.item<float>();

    // Backward: compute gradients, then immediately release the computation graph
    loss.backward();
  }  // output, loss, and computation graph freed here

  // SGD parameter update
  {
    torch::NoGradGuard no_grad;
    for (const auto& param : model_.parameters()) {
      if (param.grad().defined()) {
        param.data().sub_(param.grad() * lr);
        param.grad().zero_();
      }
    }
  }
  return loss_val;
}

OrderedDict ModelManager::getStateDict() {
  OrderedDict state;
  for (const auto& param : model_.named_parameters()) {
    state[param.name] = param.value.data().clone().cpu();
  }
  return state;
}

void ModelManager::setStateDict(const OrderedDict& state_dict) {
  for (const auto& param : model_.named_parameters()) {
    auto it = state_dict.find(param.name);
    if (it != state_dict.end()) {
      param.value.data().copy_(it->second);
    }
  }
  // Ensure all parameters still require gradients after the copy
  for (const auto& param : model_.parameters()) {
    param.set_requires_grad(true);
  }
}

}  // namespace fedlearn
