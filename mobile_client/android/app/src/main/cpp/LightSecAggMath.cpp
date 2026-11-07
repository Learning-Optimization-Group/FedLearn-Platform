#include "LightSecAggMath.h"

#include <random>

namespace fedlearn::secagg {

int64_t pow_mod(int64_t base, int64_t exp) noexcept {
  int64_t res = 1;
  int64_t b = normalize_mod(base);
  int64_t e = exp;
  while (e > 0) {
    if (e & 1) {
      res = mul_mod(res, b);
    }
    b = mul_mod(b, b);
    e >>= 1;
  }
  return res;
}

int64_t inv_mod(int64_t a) {
  int64_t na = normalize_mod(a);
  if (na == 0) {
    throw std::invalid_argument("Modular inverse of 0 modulo p does not exist.");
  }
  // Fermat's Little Theorem: a^(p-2) = a^(-1) mod p
  return pow_mod(na, kPrime - 2);
}

std::map<int, std::vector<int64_t>> compute_shamir_shares(
    const std::vector<int64_t>& secret,
    int num_shares,
    int threshold,
    const std::vector<std::vector<int64_t>>& random_coefficients) {
  if (num_shares < 1) {
    throw std::invalid_argument("num_shares must be at least 1.");
  }
  if (threshold < 1 || threshold > num_shares) {
    throw std::invalid_argument("threshold must be in [1, num_shares].");
  }
  if (static_cast<int>(random_coefficients.size()) != threshold - 1) {
    throw std::invalid_argument("random_coefficients size must equal threshold - 1.");
  }

  const size_t length = secret.size();
  for (size_t deg = 0; deg < random_coefficients.size(); ++deg) {
    if (random_coefficients[deg].size() != length) {
      throw std::invalid_argument("Coefficient vector length mismatch with secret.");
    }
  }

  std::map<int, std::vector<int64_t>> shares;

  for (int x = 1; x <= num_shares; ++x) {
    std::vector<int64_t> share(length, 0);

    for (size_t k = 0; k < length; ++k) {
      // Horner's method from degree (threshold - 1) down to degree 0 (secret)
      int64_t acc = 0;
      if (threshold > 1) {
        acc = normalize_mod(random_coefficients[threshold - 2][k]);
        for (int d = threshold - 3; d >= 0; --d) {
          acc = add_mod(mul_mod(acc, x), random_coefficients[d][k]);
        }
        acc = add_mod(mul_mod(acc, x), secret[k]);
      } else {
        // Degree 0 polynomial: f(x) = secret
        acc = normalize_mod(secret[k]);
      }
      share[k] = acc;
    }

    shares[x] = std::move(share);
  }

  return shares;
}

std::map<int, std::vector<int64_t>> compute_shamir_shares_seeded(
    const std::vector<int64_t>& secret,
    int num_shares,
    int threshold,
    uint64_t seed) {
  if (threshold < 1 || threshold > num_shares) {
    throw std::invalid_argument("threshold must be in [1, num_shares].");
  }

  const size_t length = secret.size();
  std::vector<std::vector<int64_t>> random_coeffs;
  random_coeffs.reserve(threshold - 1);

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int64_t> dist(0, kPrime - 1);

  for (int d = 0; d < threshold - 1; ++d) {
    std::vector<int64_t> coeff_vec(length);
    for (size_t k = 0; k < length; ++k) {
      coeff_vec[k] = dist(rng);
    }
    random_coeffs.push_back(std::move(coeff_vec));
  }

  return compute_shamir_shares(secret, num_shares, threshold, random_coeffs);
}

std::vector<int64_t> shamir_reconstruct(
    const std::map<int, std::vector<int64_t>>& shares,
    int threshold) {
  if (threshold <= 0) {
    throw std::invalid_argument("threshold must be positive.");
  }
  if (static_cast<int>(shares.size()) < threshold) {
    throw std::invalid_argument("Reconstruction requires at least threshold shares.");
  }

  std::vector<int> xs;
  xs.reserve(threshold);
  for (const auto& kv : shares) {
    if (kv.first <= 0) {
      throw std::invalid_argument("Share evaluation points (x) must be positive.");
    }
    xs.push_back(kv.first);
    if (static_cast<int>(xs.size()) == threshold) {
      break;
    }
  }

  const size_t length = shares.at(xs[0]).size();
  for (int x : xs) {
    if (shares.at(x).size() != length) {
      throw std::invalid_argument("All shares must have identical vector length.");
    }
  }

  std::vector<int64_t> out(length, 0);

  for (int i : xs) {
    // Lagrange basis polynomial evaluated at x = 0:
    // lambda_i = prod_{j in xs, j != i} (-j) / (i - j)  (mod PRIME)
    int64_t num = 1;
    int64_t den = 1;
    for (int j : xs) {
      if (j == i) continue;
      num = mul_mod(num, -j);
      den = mul_mod(den, i - j);
    }
    int64_t coeff = mul_mod(num, inv_mod(den));
    const auto& share_vec = shares.at(i);
    for (size_t k = 0; k < length; ++k) {
      int64_t term = mul_mod(share_vec[k], coeff);
      out[k] = add_mod(out[k], term);
    }
  }

  return out;
}

std::vector<int64_t> aggregate_shares(
    const std::vector<std::vector<int64_t>>& shares) {
  if (shares.empty()) {
    throw std::invalid_argument("aggregate_shares requires at least one share vector.");
  }

  const size_t length = shares[0].size();
  for (size_t i = 1; i < shares.size(); ++i) {
    if (shares[i].size() != length) {
      throw std::invalid_argument("All share vectors must have equal dimension.");
    }
  }

  std::vector<int64_t> total(length, 0);
  for (const auto& s : shares) {
    for (size_t k = 0; k < length; ++k) {
      total[k] = add_mod(total[k], s[k]);
    }
  }

  return total;
}

std::vector<int64_t> mask_vector(
    const std::vector<double>& values,
    const std::vector<int64_t>& mask,
    int64_t scale) {
  if (values.size() != mask.size()) {
    throw std::invalid_argument("Values and mask dimensions must match.");
  }

  const size_t length = values.size();
  std::vector<int64_t> y(length);
  for (size_t k = 0; k < length; ++k) {
    int64_t q = quantize(values[k], scale);
    y[k] = add_mod(q, mask[k]);
  }

  return y;
}

std::vector<double> recover_aggregate(
    const std::vector<std::vector<int64_t>>& masked_values,
    const std::map<int, std::vector<int64_t>>& summed_shares,
    int threshold,
    int64_t scale) {
  if (masked_values.empty()) {
    throw std::invalid_argument("recover_aggregate requires at least one masked value vector.");
  }

  const size_t length = masked_values[0].size();
  for (size_t i = 1; i < masked_values.size(); ++i) {
    if (masked_values[i].size() != length) {
      throw std::invalid_argument("Masked value dimension mismatch.");
    }
  }

  // 1. Sum masked values modulo p
  std::vector<int64_t> masked_total(length, 0);
  for (const auto& y : masked_values) {
    for (size_t k = 0; k < length; ++k) {
      masked_total[k] = add_mod(masked_total[k], y[k]);
    }
  }

  // 2. Interpolate summed shares at x = 0 to recover total mask
  std::vector<int64_t> mask_total = shamir_reconstruct(summed_shares, threshold);
  if (mask_total.size() != length) {
    throw std::runtime_error("Reconstructed mask dimension mismatch.");
  }

  // 3. Subtract mask sum modulo p and dequantize
  std::vector<double> out(length);
  for (size_t k = 0; k < length; ++k) {
    int64_t unmasked = sub_mod(masked_total[k], mask_total[k]);
    out[k] = dequantize(unmasked, scale);
  }

  return out;
}

}  // namespace fedlearn::secagg

extern "C" {

int64_t lightsecagg_add_mod(int64_t a, int64_t b) {
  return fedlearn::secagg::add_mod(a, b);
}

int64_t lightsecagg_sub_mod(int64_t a, int64_t b) {
  return fedlearn::secagg::sub_mod(a, b);
}

int64_t lightsecagg_mul_mod(int64_t a, int64_t b) {
  return fedlearn::secagg::mul_mod(a, b);
}

int64_t lightsecagg_inv_mod(int64_t a) {
  try {
    return fedlearn::secagg::inv_mod(a);
  } catch (...) {
    return -1;
  }
}

int64_t lightsecagg_quantize(double value, int64_t scale) {
  try {
    return fedlearn::secagg::quantize(value, scale);
  } catch (...) {
    return -1;
  }
}

double lightsecagg_dequantize(int64_t val, int64_t scale) {
  try {
    return fedlearn::secagg::dequantize(val, scale);
  } catch (...) {
    return 0.0;
  }
}

int32_t lightsecagg_compute_shamir_shares(
    const int64_t* secret,
    int32_t length,
    int32_t num_shares,
    int32_t threshold,
    const int64_t* flattened_coeffs,
    int64_t* out_shares) {
  if (!secret || !out_shares || length <= 0 || num_shares < 1 || threshold < 1 || threshold > num_shares) {
    return -1;
  }
  if (threshold > 1 && !flattened_coeffs) {
    return -2;
  }

  std::vector<int64_t> secret_vec(secret, secret + length);
  std::vector<std::vector<int64_t>> coeffs;
  coeffs.reserve(threshold - 1);
  for (int d = 0; d < threshold - 1; ++d) {
    coeffs.emplace_back(flattened_coeffs + d * length, flattened_coeffs + (d + 1) * length);
  }

  try {
    auto shares = fedlearn::secagg::compute_shamir_shares(secret_vec, num_shares, threshold, coeffs);
    for (int x = 1; x <= num_shares; ++x) {
      const auto& sh = shares[x];
      std::copy(sh.begin(), sh.end(), out_shares + (x - 1) * length);
    }
    return 0;
  } catch (...) {
    return -3;
  }
}

int32_t lightsecagg_aggregate_shares(
    const int64_t* flattened_shares,
    int32_t num_vectors,
    int32_t length,
    int64_t* out_aggregate) {
  if (!flattened_shares || !out_aggregate || num_vectors <= 0 || length <= 0) {
    return -1;
  }

  std::vector<std::vector<int64_t>> shares;
  shares.reserve(num_vectors);
  for (int i = 0; i < num_vectors; ++i) {
    shares.emplace_back(flattened_shares + i * length, flattened_shares + (i + 1) * length);
  }

  try {
    auto agg = fedlearn::secagg::aggregate_shares(shares);
    std::copy(agg.begin(), agg.end(), out_aggregate);
    return 0;
  } catch (...) {
    return -2;
  }
}

int32_t lightsecagg_mask_vector(
    const double* values,
    const int64_t* mask,
    int32_t length,
    int64_t scale,
    int64_t* out_masked) {
  if (!values || !mask || !out_masked || length <= 0 || scale <= 0) {
    return -1;
  }

  std::vector<double> vals(values, values + length);
  std::vector<int64_t> m(mask, mask + length);

  try {
    auto masked = fedlearn::secagg::mask_vector(vals, m, scale);
    std::copy(masked.begin(), masked.end(), out_masked);
    return 0;
  } catch (...) {
    return -2;
  }
}

int32_t lightsecagg_reconstruct_secret(
    const int32_t* xs,
    const int64_t* flattened_shares,
    int32_t num_shares,
    int32_t length,
    int32_t threshold,
    int64_t* out_secret) {
  if (!xs || !flattened_shares || !out_secret || num_shares < threshold || length <= 0 || threshold <= 0) {
    return -1;
  }

  std::map<int, std::vector<int64_t>> shares;
  for (int i = 0; i < num_shares; ++i) {
    shares[xs[i]] = std::vector<int64_t>(flattened_shares + i * length, flattened_shares + (i + 1) * length);
  }

  try {
    auto rec = fedlearn::secagg::shamir_reconstruct(shares, threshold);
    std::copy(rec.begin(), rec.end(), out_secret);
    return 0;
  } catch (...) {
    return -2;
  }
}

}  // extern "C"

