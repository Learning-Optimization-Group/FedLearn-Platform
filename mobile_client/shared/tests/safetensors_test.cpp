// safetensors_test.cpp — the cross-language state-dict codec contract (Phase 3c T5).
//
// Asserts the C++ saveSafetensors produces BYTE-IDENTICAL output to the Python-frozen golden
// (framework/.../zo_state.safetensors, written by safetensors_codec.py), and that load/save
// round-trips. Byte-parity is what keeps a C++ client and the Python server agreeing on the
// FedAvg state-dict wire format. Torch-free (no libtorch, no ExecuTorch).
//
// GOLDEN_DIR is injected by CMake and points at framework/tests/fixtures/decomfl_golden.
#include "fedlearn/Safetensors.h"

#include <gtest/gtest.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef GOLDEN_DIR
#define GOLDEN_DIR "."
#endif

namespace {

std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  EXPECT_TRUE(f.good()) << "cannot open: " << path;
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

template <typename T>
std::vector<T> ReadBin(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<T> out;
  T x;
  while (f.read(reinterpret_cast<char*>(&x), sizeof(T))) out.push_back(x);
  return out;
}

}  // namespace

TEST(Safetensors, LoadGoldenMatchesFlatVector) {
  const std::string blob = ReadFile(std::string(GOLDEN_DIR) + "/zo_state.safetensors");
  fedlearn::MetadataList meta;
  const auto tensors = fedlearn::loadSafetensors(blob, &meta);

  ASSERT_EQ(tensors.size(), 2u);
  EXPECT_EQ(tensors[0].name, "fc1.weight");
  EXPECT_EQ(tensors[0].shape, (std::vector<int64_t>{5, 4}));
  EXPECT_EQ(tensors[0].data.size(), 20u);
  EXPECT_EQ(tensors[1].name, "fc1.bias");
  EXPECT_EQ(tensors[1].shape, (std::vector<int64_t>{5}));
  EXPECT_EQ(tensors[1].data.size(), 5u);

  ASSERT_EQ(meta.size(), 1u);
  EXPECT_EQ(meta[0].first, "num_examples");
  EXPECT_EQ(meta[0].second, "8");

  // Concatenated tensor data reconstructs the committed trainable flat vector.
  const auto flat = ReadBin<float>(std::string(GOLDEN_DIR) + "/zo_flat.f32");
  ASSERT_EQ(flat.size(), 25u);
  std::vector<float> recon;
  for (const auto& t : tensors) recon.insert(recon.end(), t.data.begin(), t.data.end());
  ASSERT_EQ(recon.size(), 25u);
  for (size_t i = 0; i < recon.size(); ++i) EXPECT_FLOAT_EQ(recon[i], flat[i]) << "at " << i;
}

TEST(Safetensors, ResaveIsByteIdenticalToPythonGolden) {
  const std::string golden = ReadFile(std::string(GOLDEN_DIR) + "/zo_state.safetensors");
  fedlearn::MetadataList meta;
  const auto tensors = fedlearn::loadSafetensors(golden, &meta);
  const std::string resaved = fedlearn::saveSafetensors(tensors, meta);
  EXPECT_EQ(resaved, golden) << "C++ saveSafetensors diverged from the Python safetensors golden";
}

TEST(Safetensors, RoundTripsSyntheticTensors) {
  std::vector<fedlearn::NamedTensor> in = {
      {"w", {2, 3}, {1.0f, 2.5f, -3.0f, 0.0f, 4.25f, 9.0f}},
      {"b", {2}, {0.5f, -0.5f}},
  };
  fedlearn::MetadataList meta = {{"num_examples", "16"}, {"note", "synthetic"}};
  const std::string blob = fedlearn::saveSafetensors(in, meta);

  fedlearn::MetadataList outMeta;
  const auto out = fedlearn::loadSafetensors(blob, &outMeta);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0].name, "w");
  EXPECT_EQ(out[0].shape, (std::vector<int64_t>{2, 3}));
  EXPECT_EQ(out[0].data, in[0].data);
  EXPECT_EQ(out[1].data, in[1].data);
  EXPECT_EQ(outMeta, meta);
  // Re-save is stable.
  EXPECT_EQ(fedlearn::saveSafetensors(out, outMeta), blob);
}

TEST(Safetensors, RejectsMalformedBlob) {
  EXPECT_THROW(fedlearn::loadSafetensors("short"), std::runtime_error);
  // An 8-byte length claiming a huge header (e.g. a legacy pickle blob) must error, not over-read.
  std::string bad(16, '\xff');
  EXPECT_THROW(fedlearn::loadSafetensors(bad), std::runtime_error);
}
