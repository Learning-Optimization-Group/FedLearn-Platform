#include "fedlearn/DtypeMap.h"

#include <stdexcept>
#include <unordered_map>

namespace fedlearn {

namespace {

// Mirrors _SAFE_DTYPES in framework/src/fedlearn/communication/serializer.py exactly.
// Keep this list and the Python list in lockstep.
const std::unordered_map<std::string, DType>& nameToDtype() {
  static const std::unordered_map<std::string, DType> kMap = {
      {"float16", DType::Float16},
      {"float32", DType::Float32},
      {"float64", DType::Float64},
      {"int8", DType::Int8},
      {"int16", DType::Int16},
      {"int32", DType::Int32},
      {"int64", DType::Int64},
      {"uint8", DType::UInt8},
      {"bool", DType::Bool},
      {"bfloat16", DType::BFloat16},
  };
  return kMap;
}

}  // namespace

DType dtypeFromString(const std::string& name) {
  const auto& m = nameToDtype();
  auto it = m.find(name);
  if (it == m.end()) {
    throw std::invalid_argument("DtypeMap: unsafe/unknown dtype '" + name + "'");
  }
  return it->second;
}

std::string stringFromDtype(DType dtype) {
  for (const auto& kv : nameToDtype()) {
    if (kv.second == dtype) {
      return kv.first;
    }
  }
  throw std::invalid_argument("DtypeMap: dtype is not in the SAFE_DTYPES whitelist");
}

bool isSafeDtype(const std::string& name) {
  return nameToDtype().count(name) > 0;
}

}  // namespace fedlearn
