#pragma once

#include <cstdint>
#include <map>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace fedlearn::secagg {

// Finite field prime: p = 2^31 - 1 = 2,147,483,647 (Mersenne prime M31)
inline constexpr int64_t kPrime = 2147483647LL;

// Default fixed-point quantization scale (1,000,000)
inline constexpr int64_t kDefaultScale = 1000000LL;

/**
 * Normalizes any signed 64-bit integer into the canonical field range [0, p - 1].
 */
inline int64_t normalize_mod(int64_t a) noexcept {
  int64_t rem = a % kPrime;
  return (rem < 0) ? (rem + kPrime) : rem;
}

/**
 * Modular addition: (a + b) mod p in [0, p - 1].
 * Overflow-safe: inputs normalized into [0, p - 1], sum <= 2p - 2 < 2^32.
 */
inline int64_t add_mod(int64_t a, int64_t b) noexcept {
  int64_t na = normalize_mod(a);
  int64_t nb = normalize_mod(b);
  int64_t sum = na + nb;
  return (sum >= kPrime) ? (sum - kPrime) : sum;
}

/**
 * Modular subtraction: (a - b) mod p in [0, p - 1].
 * Underflow-safe: diff in [-(p - 1), p - 1].
 */
inline int64_t sub_mod(int64_t a, int64_t b) noexcept {
  int64_t na = normalize_mod(a);
  int64_t nb = normalize_mod(b);
  int64_t diff = na - nb;
  return (diff < 0) ? (diff + kPrime) : diff;
}

/**
 * Modular multiplication: (a * b) mod p in [0, p - 1].
 * Overflow-safe: na, nb < 2^31 - 1, product < 4.62 * 10^18 < 2^63 - 1 (fits in uint64_t / int64_t).
 */
inline int64_t mul_mod(int64_t a, int64_t b) noexcept {
  uint64_t na = static_cast<uint64_t>(normalize_mod(a));
  uint64_t nb = static_cast<uint64_t>(normalize_mod(b));
  return static_cast<int64_t>((na * nb) % static_cast<uint64_t>(kPrime));
}

/**
 * Modular exponentiation: (base^exp) mod p in [0, p - 1].
 */
int64_t pow_mod(int64_t base, int64_t exp) noexcept;

/**
 * Modular multiplicative inverse via Fermat's Little Theorem:
 * a^(p - 2) mod p.
 * Throws std::invalid_argument if a == 0 mod p.
 */
int64_t inv_mod(int64_t a);

/**
 * Fixed-point quantization of a floating-point value to F_p:
 * round(value * scale) mod p.
 */
inline int64_t quantize(double value, int64_t scale = kDefaultScale) {
  if (scale <= 0) {
    throw std::invalid_argument("Quantization scale must be positive.");
  }
  if (!std::isfinite(value)) {
    throw std::invalid_argument("Cannot quantize non-finite float value.");
  }
  double scaled = std::round(value * static_cast<double>(scale));
  double reduced = std::fmod(scaled, static_cast<double>(kPrime));
  int64_t rem = static_cast<int64_t>(reduced);
  return (rem < 0) ? (rem + kPrime) : rem;
}

/**
 * De-quantization from F_p to float64, accounting for signedness:
 * values > p / 2 represent negative numbers (n - p).
 */
inline double dequantize(int64_t val, int64_t scale = kDefaultScale) {
  if (scale <= 0) {
    throw std::invalid_argument("Dequantization scale must be positive.");
  }
  int64_t n = normalize_mod(val);
  int64_t signed_val = (n > (kPrime / 2)) ? (n - kPrime) : n;
  return static_cast<double>(signed_val) / static_cast<double>(scale);
}

/**
 * Computes Shamir shares for a secret vector using explicit random coefficients.
 *
 * @param secret The secret vector in F_p.
 * @param num_shares Total number of shares to generate (evaluated at x = 1, 2, ..., num_shares).
 * @param threshold Reconstruction threshold T (degree = T - 1).
 * @param random_coefficients Vector of size (T - 1), where each element is a coefficient vector of length secret.size().
 * @return Map of x -> share_vector for x in {1, ..., num_shares}.
 */
std::map<int, std::vector<int64_t>> compute_shamir_shares(
    const std::vector<int64_t>& secret,
    int num_shares,
    int threshold,
    const std::vector<std::vector<int64_t>>& random_coefficients);

/**
 * Computes Shamir shares for a secret vector using a 64-bit seed to deterministically
 * generate the (threshold - 1) random coefficient vectors.
 */
std::map<int, std::vector<int64_t>> compute_shamir_shares_seeded(
    const std::vector<int64_t>& secret,
    int num_shares,
    int threshold,
    uint64_t seed);

/**
 * Reconstructs the secret vector from at least `threshold` shares using Lagrange interpolation at x = 0.
 *
 * @param shares Map of x -> share_vector (at least `threshold` entries required).
 * @param threshold Privacy/reconstruction threshold.
 * @return The reconstructed secret vector in F_p.
 */
std::vector<int64_t> shamir_reconstruct(
    const std::map<int, std::vector<int64_t>>& shares,
    int threshold);

/**
 * Aggregates a list of share vectors by element-wise modular addition.
 * S[k] = sum_i shares[i][k] mod p.
 */
std::vector<int64_t> aggregate_shares(
    const std::vector<std::vector<int64_t>>& shares);

/**
 * Masks a continuous floating-point gradient vector with a field mask vector:
 * y[k] = (quantize(values[k], scale) + mask[k]) mod p.
 */
std::vector<int64_t> mask_vector(
    const std::vector<double>& values,
    const std::vector<int64_t>& mask,
    int64_t scale = kDefaultScale);

/**
 * End-to-end server unmasking:
 * Recovers sum_i values[i] from surviving masked vectors and surviving aggregated shares.
 */
std::vector<double> recover_aggregate(
    const std::vector<std::vector<int64_t>>& masked_values,
    const std::map<int, std::vector<int64_t>>& summed_shares,
    int threshold,
    int64_t scale = kDefaultScale);

}  // namespace fedlearn::secagg

// C-linkage export interface for cross-language verification and JNI wrappers
extern "C" {

int64_t lightsecagg_add_mod(int64_t a, int64_t b);
int64_t lightsecagg_sub_mod(int64_t a, int64_t b);
int64_t lightsecagg_mul_mod(int64_t a, int64_t b);
int64_t lightsecagg_inv_mod(int64_t a);
int64_t lightsecagg_quantize(double value, int64_t scale);
double lightsecagg_dequantize(int64_t val, int64_t scale);

int32_t lightsecagg_compute_shamir_shares(
    const int64_t* secret,
    int32_t length,
    int32_t num_shares,
    int32_t threshold,
    const int64_t* flattened_coeffs,
    int64_t* out_shares);

int32_t lightsecagg_aggregate_shares(
    const int64_t* flattened_shares,
    int32_t num_vectors,
    int32_t length,
    int64_t* out_aggregate);

int32_t lightsecagg_mask_vector(
    const double* values,
    const int64_t* mask,
    int32_t length,
    int64_t scale,
    int64_t* out_masked);

int32_t lightsecagg_reconstruct_secret(
    const int32_t* xs,
    const int64_t* flattened_shares,
    int32_t num_shares,
    int32_t length,
    int32_t threshold,
    int64_t* out_secret);

}  // extern "C"

