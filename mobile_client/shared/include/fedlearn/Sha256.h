#pragma once
//
// Sha256.h — dependency-free FIPS 180-4 SHA-256 (Secure Hash Algorithm 256-bit).
//
// Used by ModelManager to verify a model file's hash BEFORE torch::jit::load (the
// untrusted-input rule, 15-LLD §13 task 6 / E8). Self-contained so the mobile core needs
// no OpenSSL dependency. Correctness is pinned by sha256_test.cpp (NIST known-answer).
//
#include <cstddef>
#include <cstdint>
#include <string>

namespace fedlearn {

class Sha256 {
 public:
  // Lowercase hex digest of the given bytes.
  static std::string hexDigest(const uint8_t* data, size_t len);
  static std::string hexDigest(const std::string& s) {
    return hexDigest(reinterpret_cast<const uint8_t*>(s.data()), s.size());
  }
  // Lowercase hex digest of a file's contents. Throws std::runtime_error if unreadable.
  static std::string hexDigestFile(const std::string& path);
};

}  // namespace fedlearn
