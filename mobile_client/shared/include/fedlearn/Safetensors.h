#pragma once
//
// Safetensors.h — deterministic safetensors-shaped state-dict codec (torch-free, float32-only).
//
// Byte-identical to framework/src/fedlearn/communication/safetensors_codec.py. This is the
// cross-language wire format for FedAvg model state-dicts, replacing the legacy torch.jit pickle blob.
// Format:  u64_le(header_len) ++ header_json_utf8 ++ raw_f32_data
// where header_json is compact (no spaces) with tensor entries in stored order, each
// {"dtype":"F32","shape":[...],"data_offsets":[s,e]}, then an optional "__metadata__" object
// of string->string. A length/JSON sniff makes a legacy pickle blob fail loudly, not silently.
//
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace fedlearn {

struct NamedTensor {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> data;  // row-major float32, length == product(shape)
};

using MetadataList = std::vector<std::pair<std::string, std::string>>;

// Serialize named float32 tensors (in the given order) + string metadata.
std::string saveSafetensors(const std::vector<NamedTensor>& tensors, const MetadataList& metadata);

// Inverse: parse a safetensors blob into named tensors (in stored order); fills `metadata` if
// non-null. Throws std::runtime_error on a malformed blob (incl. a legacy pickle blob).
std::vector<NamedTensor> loadSafetensors(const std::string& blob, MetadataList* metadata = nullptr);

}  // namespace fedlearn
