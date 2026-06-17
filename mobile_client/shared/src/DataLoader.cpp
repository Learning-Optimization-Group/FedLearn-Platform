#include "fedlearn/DataLoader.h"

#include <fstream>
#include <functional>
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
  if (batch.inputs == nullptr || batch.targets == nullptr) {
    throw std::invalid_argument("DataLoader::validate: inputs/targets are null");
  }
  if (batch.numSamples <= 0) {
    throw std::invalid_argument("DataLoader::validate: numSamples must be > 0");
  }
  if (batch.inputShape.empty()) {
    throw std::invalid_argument("DataLoader::validate: inputShape must be non-empty");
  }
  if (batch.inputShape[0] != batch.numSamples) {
    throw std::invalid_argument("DataLoader::validate: batch dimension mismatch (inputShape[0] vs numSamples)");
  }
}

OwnedBatch DataLoader::fromRawFiles(const std::string& inputsF32Path,
                                    const std::vector<int64_t>& inputShape,
                                    const std::string& targetsI64Path) {
  if (inputShape.empty()) {
    throw std::runtime_error("DataLoader::fromRawFiles: inputShape must be non-empty");
  }
  const int64_t expected = std::accumulate(inputShape.begin(), inputShape.end(), int64_t{1},
                                           std::multiplies<int64_t>());

  OwnedBatch b;
  b.inputs = readBinary<float>(inputsF32Path);
  if (static_cast<int64_t>(b.inputs.size()) != expected) {
    throw std::runtime_error("DataLoader::fromRawFiles: inputs element count != product(inputShape)");
  }
  b.inputShape = inputShape;
  b.targets = readBinary<int64_t>(targetsI64Path);

  validate(b.view());
  return b;
}

}  // namespace fedlearn
