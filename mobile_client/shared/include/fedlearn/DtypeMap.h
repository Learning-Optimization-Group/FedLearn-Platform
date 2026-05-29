#pragma once
//
// DtypeMap.h — string <-> torch dtype with a SAFE_DTYPES whitelist.
//
// Mirrors the Python whitelist in framework/src/fedlearn/communication/serializer.py
// (_SAFE_DTYPES). Any dtype crossing the wire or loaded from disk must be on this list;
// an unknown dtype string is a HARD ERROR (15-LLD-mobile.md §9 E7 / §13 task 5), never a
// silent default — this blocks arbitrary-dtype injection from a malformed server payload.
//
#include <string>
#include <torch/torch.h>

namespace fedlearn {

// Returns the torch ScalarType for a whitelisted dtype name (e.g. "float32").
// Throws std::invalid_argument if the name is not in the whitelist.
at::ScalarType dtypeFromString(const std::string& name);

// Returns the canonical lowercase name (e.g. "float32") for a whitelisted dtype.
// Throws std::invalid_argument if the dtype is not in the whitelist.
std::string stringFromDtype(at::ScalarType dtype);

// True iff name is a whitelisted dtype.
bool isSafeDtype(const std::string& name);

}  // namespace fedlearn
