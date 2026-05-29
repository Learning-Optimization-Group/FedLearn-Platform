#include "fedlearn/DataLoader.h"

#include <fstream>
#include <numeric>
#include <stdexcept>

namespace fedlearn {
namespace {

template <typename T>
std::vector<T> readBinary(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.good()) throw std::runtime_error("DataLoader: cannot read file: " + path);
  std::vector<T> out;
  T x;
  while (f.read(reinterpret_cast<char*>(&x), sizeof(T))) out.push_back(x);
  return out;
}

}  // namespace

void DataLoader::validate(const DataBatch& batch) {
  if (!batch.inputs.defined() || !batch.targets.defined()) {
    throw std::invalid_argument("DataLoader::validate: inputs/targets undefined");
  }
  if (!batch.inputs.is_floating_point()) {
    throw std::invalid_argument("DataLoader::validate: inputs must be a floating tensor");
  }
  if (batch.targets.scalar_type() != at::kLong) {
    throw std::invalid_argument("DataLoader::validate: targets must be int64 (class indices)");
  }
  if (batch.targets.dim() != 1) {
    throw std::invalid_argument("DataLoader::validate: targets must be 1-D");
  }
  if (batch.inputs.size(0) != batch.targets.size(0)) {
    throw std::invalid_argument("DataLoader::validate: batch dimension mismatch (inputs vs targets)");
  }
}

DataBatch DataLoader::fromRawFiles(const std::string& inputsF32Path,
                                   const std::vector<int64_t>& inputShape,
                                   const std::string& targetsI64Path) {
  if (inputShape.empty()) {
    throw std::runtime_error("DataLoader::fromRawFiles: inputShape must be non-empty");
  }
  int64_t expected = std::accumulate(inputShape.begin(), inputShape.end(), int64_t{1},
                                     std::multiplies<int64_t>());

  std::vector<float> inBytes = readBinary<float>(inputsF32Path);
  if (static_cast<int64_t>(inBytes.size()) != expected) {
    throw std::runtime_error("DataLoader::fromRawFiles: inputs element count != product(inputShape)");
  }
  std::vector<int64_t> tgt = readBinary<int64_t>(targetsI64Path);

  // .clone() so the tensors own their storage after the local buffers go out of scope.
  torch::Tensor inputs =
      torch::from_blob(inBytes.data(), inputShape, torch::kFloat32).clone();
  torch::Tensor targets =
      torch::from_blob(tgt.data(), {static_cast<int64_t>(tgt.size())}, torch::kLong).clone();

  DataBatch batch{inputs, targets};
  validate(batch);
  return batch;
}

}  // namespace fedlearn
