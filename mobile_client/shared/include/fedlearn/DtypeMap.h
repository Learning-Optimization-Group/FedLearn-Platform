#pragma once
//
// DtypeMap.h — string <-> dtype with a SAFE_DTYPES whitelist (torch-free).
//
// Mirrors the Python whitelist in framework/src/fedlearn/communication/serializer.py
// (_SAFE_DTYPES). Any dtype crossing the wire or loaded from disk must be on this list;
// an unknown dtype string is a HARD ERROR (15-LLD-mobile.md §9 E7 / §13 task 5), never a
// silent default — this blocks arbitrary-dtype injection from a malformed server payload.
//
// Carries no torch/ExecuTorch dependency: the dtype is a self-contained enum, not a torch
// ScalarType, so this near-universally-included header stays dependency-light.
//
#include <cstdint>
#include <string>

namespace fedlearn {

// The whitelisted dtypes (1:1 with the Python _SAFE_DTYPES list, same order).
enum class DType : uint8_t {
  Float16,
  Float32,
  Float64,
  Int8,
  Int16,
  Int32,
  Int64,
  UInt8,
  Bool,
  BFloat16,
};

// Returns the DType for a whitelisted dtype name (e.g. "float32").
// Throws std::invalid_argument if the name is not in the whitelist.
DType dtypeFromString(const std::string& name);

// Returns the canonical lowercase name (e.g. "float32") for a whitelisted DType.
// Throws std::invalid_argument if the dtype is not in the whitelist.
std::string stringFromDtype(DType dtype);

// True iff name is a whitelisted dtype.
bool isSafeDtype(const std::string& name);

}  // namespace fedlearn
