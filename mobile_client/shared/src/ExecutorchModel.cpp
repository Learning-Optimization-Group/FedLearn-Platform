#include "fedlearn/ExecutorchModel.h"
#include "fedlearn/Sha256.h"

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/method_meta.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/runtime.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace fedlearn {
namespace {

using executorch::aten::ScalarType;
using executorch::aten::SizesType;
using executorch::extension::FileDataLoader;
using executorch::extension::make_tensor_ptr;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::HierarchicalAllocator;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::MemoryManager;
using executorch::runtime::Method;
using executorch::runtime::MethodMeta;
using executorch::runtime::Program;
using executorch::runtime::Span;

void ensureRuntimeInit() {
  static std::once_flag once;
  std::call_once(once, [] { executorch::runtime::runtime_init(); });
}

[[noreturn]] void fail(const std::string& what, Error e) {
  throw std::runtime_error("ExecutorchModel: " + what + " (error " +
                           std::to_string(static_cast<int>(e)) + ")");
}

// Scratch arena for the method's non-planned allocations (4 MB is ample for the small FL models).
constexpr size_t kMethodArenaBytes = 4 * 1024 * 1024;

}  // namespace

// Members are unique_ptr-wrapped so their addresses are stable (Method holds raw pointers into
// the Program, MemoryManager and planned buffers — none may move after the Method is created).
struct ExecutorchModel::Impl {
  std::unique_ptr<FileDataLoader> loader;
  std::unique_ptr<Program> program;
  std::vector<std::unique_ptr<uint8_t[]>> planned;
  std::vector<Span<uint8_t>> planned_spans;
  std::unique_ptr<HierarchicalAllocator> planned_alloc;
  std::vector<uint8_t> method_arena;
  std::unique_ptr<MemoryAllocator> method_alloc;
  std::unique_ptr<MemoryManager> mm;
  std::unique_ptr<Method> method;
  int64_t flat_dim = 0;
};

ExecutorchModel::ExecutorchModel(const std::string& ptePath, const std::string& expectedSha256)
    : impl_(std::make_unique<Impl>()) {
  // Untrusted-input rule: verify the hash BEFORE handing bytes to ExecuTorch.
  const std::string actual = Sha256::hexDigestFile(ptePath);
  if (actual != expectedSha256) {
    throw std::runtime_error("ExecutorchModel: sha256 mismatch for '" + ptePath +
                             "' (expected " + expectedSha256 + ", got " + actual + ")");
  }
  ensureRuntimeInit();

  auto loaderRes = FileDataLoader::from(ptePath.c_str());
  if (!loaderRes.ok()) fail("FileDataLoader::from failed", loaderRes.error());
  impl_->loader = std::make_unique<FileDataLoader>(std::move(*loaderRes));

  auto progRes = Program::load(impl_->loader.get());
  if (!progRes.ok()) fail("Program::load failed", progRes.error());
  impl_->program = std::make_unique<Program>(std::move(*progRes));

  auto metaRes = impl_->program->method_meta("forward");
  if (!metaRes.ok()) fail("method_meta(forward) failed", metaRes.error());
  const MethodMeta& meta = metaRes.get();

  // flat_dim = element count of input 0 (the trainable parameter vector). The forward signature
  // is f(flat, x, y), so input 0 must exist; fail loudly rather than silently report flat_dim 0.
  if (meta.num_inputs() < 1) {
    throw std::runtime_error("ExecutorchModel: model 'forward' exposes no inputs");
  }
  auto in0 = meta.input_tensor_meta(0);
  if (!in0.ok()) fail("input_tensor_meta(0) failed", in0.error());
  {
    int64_t n = 1;
    for (int32_t d : in0.get().sizes()) n *= d;
    impl_->flat_dim = n;
  }

  // Planned memory buffers (sized from the method metadata). reserve() so planned_spans' storage
  // does not reallocate — HierarchicalAllocator below captures planned_spans.data() as a raw,
  // non-owning pointer that must stay valid for the Method's lifetime.
  impl_->planned_spans.reserve(meta.num_memory_planned_buffers());
  for (size_t i = 0; i < meta.num_memory_planned_buffers(); ++i) {
    auto szRes = meta.memory_planned_buffer_size(i);
    if (!szRes.ok()) fail("memory_planned_buffer_size failed", szRes.error());
    const auto sz = static_cast<size_t>(szRes.get());
    impl_->planned.push_back(std::make_unique<uint8_t[]>(sz));
    impl_->planned_spans.emplace_back(impl_->planned.back().get(), sz);
  }
  impl_->planned_alloc = std::make_unique<HierarchicalAllocator>(
      Span<Span<uint8_t>>(impl_->planned_spans.data(), impl_->planned_spans.size()));

  impl_->method_arena.resize(kMethodArenaBytes);
  impl_->method_alloc = std::make_unique<MemoryAllocator>(
      static_cast<uint32_t>(impl_->method_arena.size()), impl_->method_arena.data());
  impl_->mm = std::make_unique<MemoryManager>(impl_->method_alloc.get(), impl_->planned_alloc.get());

  auto methodRes = impl_->program->load_method("forward", impl_->mm.get());
  if (!methodRes.ok()) fail("load_method(forward) failed", methodRes.error());
  impl_->method = std::make_unique<Method>(std::move(*methodRes));
}

ExecutorchModel::~ExecutorchModel() = default;

int64_t ExecutorchModel::flatDim() const { return impl_->flat_dim; }

float ExecutorchModel::loss(const std::vector<float>& flat,
                            const float* x, const std::vector<int64_t>& xShape,
                            const int64_t* y, int64_t n) {
  // Narrow each dimension to SizesType (int32 in lean mode), failing loudly on overflow rather
  // than silently truncating a >2^31 dimension to a wrong shape.
  auto toSize = [](int64_t d) -> SizesType {
    if (d < 0 ||
        static_cast<uint64_t>(d) > static_cast<uint64_t>(std::numeric_limits<SizesType>::max())) {
      throw std::runtime_error("ExecutorchModel: dimension " + std::to_string(d) +
                               " exceeds the model index type range");
    }
    return static_cast<SizesType>(d);
  };
  std::vector<SizesType> flatSizes{toSize(static_cast<int64_t>(flat.size()))};
  std::vector<SizesType> xSizes;
  xSizes.reserve(xShape.size());
  for (int64_t d : xShape) xSizes.push_back(toSize(d));
  std::vector<SizesType> ySizes{toSize(n)};

  // make_tensor_ptr ALIASES the caller-owned buffers (no deleter, no copy). They stay valid for
  // this whole call — which spans set_input + execute, where ExecuTorch reads them — and the
  // scalar result is copied out before returning. A forward pass only reads its inputs, so the
  // const_cast to the non-const API never results in a write through the alias.
  auto tFlat = make_tensor_ptr(flatSizes, const_cast<float*>(flat.data()), ScalarType::Float);
  auto tX = make_tensor_ptr(xSizes, const_cast<float*>(x), ScalarType::Float);
  auto tY = make_tensor_ptr(ySizes, const_cast<int64_t*>(y), ScalarType::Long);

  Method& method = *impl_->method;
  if (auto e = method.set_input(*tFlat, 0); e != Error::Ok) fail("set_input(flat) failed", e);
  if (auto e = method.set_input(*tX, 1); e != Error::Ok) fail("set_input(x) failed", e);
  if (auto e = method.set_input(*tY, 2); e != Error::Ok) fail("set_input(y) failed", e);
  if (auto e = method.execute(); e != Error::Ok) fail("execute failed", e);

  if (method.outputs_size() != 1) {
    throw std::runtime_error("ExecutorchModel: expected 1 output, got " +
                             std::to_string(method.outputs_size()));
  }
  EValue out;
  if (auto e = method.get_outputs(&out, 1); e != Error::Ok) fail("get_outputs failed", e);
  if (!out.isTensor()) {
    throw std::runtime_error("ExecutorchModel: output 0 is not a tensor");
  }
  const auto& t = out.toTensor();
  if (t.scalar_type() != ScalarType::Float) {
    throw std::runtime_error("ExecutorchModel: expected Float output, got scalar_type " +
                             std::to_string(static_cast<int>(t.scalar_type())));
  }
  if (t.numel() < 1) {
    throw std::runtime_error("ExecutorchModel: output tensor is empty");
  }
  return t.const_data_ptr<float>()[0];
}

std::vector<float> ExecutorchModel::infer(const std::vector<float>& flat,
                                          const float* x, const std::vector<int64_t>& xShape) {
  // Narrow each dimension to SizesType (int32 in lean mode), failing loudly on overflow rather
  // than silently truncating a >2^31 dimension to a wrong shape.
  auto toSize = [](int64_t d) -> SizesType {
    if (d < 0 ||
        static_cast<uint64_t>(d) > static_cast<uint64_t>(std::numeric_limits<SizesType>::max())) {
      throw std::runtime_error("ExecutorchModel: dimension " + std::to_string(d) +
                               " exceeds the model index type range");
    }
    return static_cast<SizesType>(d);
  };
  std::vector<SizesType> flatSizes{toSize(static_cast<int64_t>(flat.size()))};
  std::vector<SizesType> xSizes;
  xSizes.reserve(xShape.size());
  for (int64_t d : xShape) xSizes.push_back(toSize(d));

  // make_tensor_ptr ALIASES the caller-owned buffers (no deleter, no copy). They stay valid for
  // this whole call — which spans set_input + execute, where ExecuTorch reads them. The infer
  // graph has two inputs: flat (Float) and x (Float). No y input (no cross-entropy).
  auto tFlat = make_tensor_ptr(flatSizes, const_cast<float*>(flat.data()), ScalarType::Float);
  auto tX = make_tensor_ptr(xSizes, const_cast<float*>(x), ScalarType::Float);

  Method& method = *impl_->method;
  if (auto e = method.set_input(*tFlat, 0); e != Error::Ok) fail("set_input(flat) failed", e);
  if (auto e = method.set_input(*tX, 1); e != Error::Ok) fail("set_input(x) failed", e);
  if (auto e = method.execute(); e != Error::Ok) fail("execute failed", e);

  if (method.outputs_size() != 1) {
    throw std::runtime_error("ExecutorchModel: expected 1 output, got " +
                             std::to_string(method.outputs_size()));
  }
  EValue out;
  if (auto e = method.get_outputs(&out, 1); e != Error::Ok) fail("get_outputs failed", e);
  if (!out.isTensor()) {
    throw std::runtime_error("ExecutorchModel: output 0 is not a tensor");
  }
  const auto& t = out.toTensor();
  if (t.scalar_type() != ScalarType::Float) {
    throw std::runtime_error("ExecutorchModel: expected Float output, got scalar_type " +
                             std::to_string(static_cast<int>(t.scalar_type())));
  }
  if (t.numel() < 1) {
    throw std::runtime_error("ExecutorchModel: output tensor is empty");
  }
  // Copy ALL logits (batch × num_classes) into a flat vector and return it.
  const float* data = t.const_data_ptr<float>();
  return std::vector<float>(data, data + t.numel());
}

}  // namespace fedlearn
