#include "fedlearn/DtypeMap.h"

#include <stdexcept>
#include <unordered_map>

namespace fedlearn {

namespace {

// Mirrors _SAFE_DTYPES in framework/src/fedlearn/communication/serializer.py exactly.
// Keep this list and the Python list in lockstep.
const std::unordered_map<std::string, at::ScalarType>& nameToDtype() {
  static const std::unordered_map<std::string, at::ScalarType> kMap = {
      {"float16", at::kHalf},
      {"float32", at::kFloat},
      {"float64", at::kDouble},
      {"int8", at::kChar},
      {"int16", at::kShort},
      {"int32", at::kInt},
      {"int64", at::kLong},
      {"uint8", at::kByte},
      {"bool", at::kBool},
      {"bfloat16", at::kBFloat16},
  };
  return kMap;
}

}  // namespace

at::ScalarType dtypeFromString(const std::string& name) {
  const auto& m = nameToDtype();
  auto it = m.find(name);
  if (it == m.end()) {
    throw std::invalid_argument("DtypeMap: unsafe/unknown dtype '" + name + "'");
  }
  return it->second;
}

std::string stringFromDtype(at::ScalarType dtype) {
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
