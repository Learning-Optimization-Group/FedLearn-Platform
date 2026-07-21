// train_smoke.cpp — minimal on-device forward+backward proof (Phase-A de-risk for first-order
// on-device training, i.e. "mobile supports all algorithms").
//
// Loads a TRAINABLE .pte (exported with a captured backward graph by export_xor_trainable.py),
// then runs ExecuTorch's training extension: TrainingModule.execute_forward_backward gives the
// loss + gradients, and optimizer/sgd applies the update — the exact API a native FedAvg/FedProx
// round will use. It trains XOR for a few thousand steps and asserts the loss collapses, proving
// ExecuTorch really does on-device backprop (it is otherwise an inference-only runtime, which is
// why the mobile client is currently zeroth-order/DeComFL only and MO-4 fail-closed-refuses
// first-order training).
//
// Build + run: see run_training_smoke_macos.sh. Exit 0 = backprop works.
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/training/module/training_module.h>
#include <executorch/extension/training/optimizer/sgd.h>
#include <executorch/runtime/platform/runtime.h>

#include <cstdio>
#include <memory>
#include <utility>
#include <vector>

using executorch::extension::FileDataLoader;
using executorch::extension::make_tensor_ptr;
using executorch::extension::TensorPtr;
using executorch::extension::training::TrainingModule;
using executorch::extension::training::optimizer::SGD;
using executorch::extension::training::optimizer::SGDOptions;

int main(int argc, char** argv) {
  executorch::runtime::runtime_init();
  if (argc < 2) {
    printf("usage: %s <trainable.pte>\n", argv[0]);
    return 2;
  }
  auto loader_res = FileDataLoader::from(argv[1]);
  if (!loader_res.ok()) {
    printf("FAIL: cannot open %s\n", argv[1]);
    return 1;
  }
  auto loader = std::make_unique<FileDataLoader>(std::move(loader_res.get()));
  TrainingModule mod(std::move(loader));

  // The 4 XOR points (x:[1,2] f32, y:[1] int64) — the whole training set.
  std::vector<std::pair<TensorPtr, TensorPtr>> data;
  data.push_back({make_tensor_ptr<float>({1, 2}, {1, 1}), make_tensor_ptr<int64_t>({1}, {0})});
  data.push_back({make_tensor_ptr<float>({1, 2}, {0, 0}), make_tensor_ptr<int64_t>({1}, {0})});
  data.push_back({make_tensor_ptr<float>({1, 2}, {1, 0}), make_tensor_ptr<int64_t>({1}, {1})});
  data.push_back({make_tensor_ptr<float>({1, 2}, {0, 1}), make_tensor_ptr<int64_t>({1}, {1})});

  auto params = mod.named_parameters("forward");
  if (!params.ok()) {
    printf("FAIL: named_parameters error %d\n", static_cast<int>(params.error()));
    return 1;
  }
  SGD optimizer(params.get(), SGDOptions{0.1});

  float first_loss = -1.f, last_loss = -1.f;
  const int steps = 5000;
  for (int i = 0; i < steps; ++i) {
    auto& d = data[i % data.size()];
    auto res = mod.execute_forward_backward("forward", {*d.first, *d.second});
    if (!res.ok()) {
      printf("FAIL: execute_forward_backward error %d at step %d\n", static_cast<int>(res.error()), i);
      return 1;
    }
    const float loss = res.get()[0].toTensor().const_data_ptr<float>()[0];
    if (i == 0) first_loss = loss;
    if (i >= steps - 4) last_loss = (last_loss < 0 ? loss : last_loss + loss);  // sum last epoch
    auto grads = mod.named_gradients("forward");
    if (!grads.ok()) {
      printf("FAIL: named_gradients error %d\n", static_cast<int>(grads.error()));
      return 1;
    }
    optimizer.step(grads.get());
  }
  last_loss /= 4.f;  // average over the final epoch (4 points)
  printf("first_loss=%.5f  last_loss(avg over final epoch)=%.5f\n", first_loss, last_loss);
  if (last_loss < first_loss * 0.5f) {
    printf("PASS: on-device forward+backward reduced XOR loss by >2x — ET backprop works.\n");
    return 0;
  }
  printf("FAIL: loss did not decrease enough (backprop suspect)\n");
  return 1;
}
