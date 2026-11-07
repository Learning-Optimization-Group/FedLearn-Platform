#include "LightSecAggMath.h"

#include <cassert>
#include <iostream>
#include <random>
#include <cmath>

using namespace fedlearn::secagg;

void test_basic_arithmetic() {
  std::cout << "[RUN] test_basic_arithmetic..." << std::endl;

  // normalize_mod
  assert(normalize_mod(0) == 0);
  assert(normalize_mod(kPrime) == 0);
  assert(normalize_mod(kPrime - 1) == kPrime - 1);
  assert(normalize_mod(kPrime + 1) == 1);
  assert(normalize_mod(-1) == kPrime - 1);
  assert(normalize_mod(-kPrime) == 0);
  assert(normalize_mod(-kPrime - 1) == kPrime - 1);

  // add_mod
  assert(add_mod(0, 0) == 0);
  assert(add_mod(0, kPrime - 1) == kPrime - 1);
  assert(add_mod(kPrime - 1, 1) == 0);
  assert(add_mod(kPrime - 1, kPrime - 1) == kPrime - 2);
  assert(add_mod(-1, 1) == 0);
  assert(add_mod(100, 200) == 300);

  // sub_mod
  assert(sub_mod(0, 0) == 0);
  assert(sub_mod(0, 1) == kPrime - 1);
  assert(sub_mod(1, kPrime - 1) == 2);
  assert(sub_mod(kPrime - 1, kPrime - 1) == 0);
  assert(sub_mod(10, 20) == kPrime - 10);
  assert(sub_mod(20, 10) == 10);

  // mul_mod
  assert(mul_mod(0, 12345) == 0);
  assert(mul_mod(1, kPrime - 1) == kPrime - 1);
  // (p - 1)^2 == (-1)^2 == 1 (mod p)
  assert(mul_mod(kPrime - 1, kPrime - 1) == 1);
  // 2^30 * 2^30 = 2^60 = 2^29 * 2^31 == 2^29 (mod 2^31 - 1)
  int64_t pow2_30 = 1LL << 30;
  int64_t pow2_29 = 1LL << 29;
  assert(mul_mod(pow2_30, pow2_30) == pow2_29);

  // inv_mod
  assert(inv_mod(1) == 1);
  assert(inv_mod(kPrime - 1) == kPrime - 1);
  // 2 * 2^30 = 2^31 == 1 (mod p)
  assert(inv_mod(2) == (1LL << 30));
  assert(mul_mod(2, inv_mod(2)) == 1);

  bool threw_zero_inv = false;
  try {
    inv_mod(0);
  } catch (const std::invalid_argument&) {
    threw_zero_inv = true;
  }
  assert(threw_zero_inv);

  // Random tests for inv_mod: a * inv(a) == 1 mod p
  std::mt19937_64 rng(1337);
  std::uniform_int_distribution<int64_t> dist(1, kPrime - 1);
  for (int i = 0; i < 500; ++i) {
    int64_t a = dist(rng);
    int64_t inv = inv_mod(a);
    assert(mul_mod(a, inv) == 1);
  }

  std::cout << "[PASS] test_basic_arithmetic" << std::endl;
}

void test_quantization() {
  std::cout << "[RUN] test_quantization..." << std::endl;

  assert(quantize(0.0) == 0);
  assert(dequantize(0) == 0.0);

  assert(quantize(1.0) == 1000000LL);
  assert(dequantize(1000000LL) == 1.0);

  assert(quantize(-1.0) == kPrime - 1000000LL);
  assert(dequantize(kPrime - 1000000LL) == -1.0);

  double val = 0.123456;
  int64_t q = quantize(val);
  assert(q == 123456LL);
  double dq = dequantize(q);
  assert(std::abs(dq - val) < 1e-9);

  double neg_val = -0.654321;
  int64_t q_neg = quantize(neg_val);
  assert(q_neg == kPrime - 654321LL);
  double dq_neg = dequantize(q_neg);
  assert(std::abs(dq_neg - neg_val) < 1e-9);

  std::cout << "[PASS] test_quantization" << std::endl;
}

void test_shamir_deterministic() {
  std::cout << "[RUN] test_shamir_deterministic..." << std::endl;

  // Secret vector: 3 coordinates
  std::vector<int64_t> secret = {42, 100, kPrime - 1};
  int num_shares = 3;
  int threshold = 2; // degree 1 polynomial: f(x) = secret + a1 * x
  std::vector<std::vector<int64_t>> coeffs = {
      {10, 20, 30}
  };

  auto shares = compute_shamir_shares(secret, num_shares, threshold, coeffs);
  assert(shares.size() == 3);

  // x = 1: secret + a1 = {52, 120, (p-1 + 30) % p = 29}
  assert(shares[1][0] == 52);
  assert(shares[1][1] == 120);
  assert(shares[1][2] == 29);

  // x = 2: secret + 2*a1 = {62, 140, (p-1 + 60) % p = 59}
  assert(shares[2][0] == 62);
  assert(shares[2][1] == 140);
  assert(shares[2][2] == 59);

  // x = 3: secret + 3*a1 = {72, 160, (p-1 + 90) % p = 89}
  assert(shares[3][0] == 72);
  assert(shares[3][1] == 160);
  assert(shares[3][2] == 89);

  // Test reconstruction with all combinations of 2 shares
  std::map<int, std::vector<int64_t>> subset12 = {{1, shares[1]}, {2, shares[2]}};
  std::map<int, std::vector<int64_t>> subset13 = {{1, shares[1]}, {3, shares[3]}};
  std::map<int, std::vector<int64_t>> subset23 = {{2, shares[2]}, {3, shares[3]}};

  auto rec12 = shamir_reconstruct(subset12, threshold);
  auto rec13 = shamir_reconstruct(subset13, threshold);
  auto rec23 = shamir_reconstruct(subset23, threshold);

  assert(rec12 == secret);
  assert(rec13 == secret);
  assert(rec23 == secret);

  std::cout << "[PASS] test_shamir_deterministic" << std::endl;
}

void test_shamir_thresholds_and_dimensions() {
  std::cout << "[RUN] test_shamir_thresholds_and_dimensions..." << std::endl;

  std::vector<int> dimensions = {1, 16, 128, 512};
  std::vector<std::pair<int, int>> configs = {
      {1, 1},
      {2, 3},
      {3, 5},
      {4, 7},
      {5, 10}
  };

  std::mt19937_64 rng(42);
  std::uniform_int_distribution<int64_t> dist(0, kPrime - 1);

  for (int dim : dimensions) {
    std::vector<int64_t> secret(dim);
    for (int k = 0; k < dim; ++k) {
      secret[k] = dist(rng);
    }

    for (const auto& [threshold, num_shares] : configs) {
      auto shares = compute_shamir_shares_seeded(secret, num_shares, threshold, rng());
      assert(static_cast<int>(shares.size()) == num_shares);

      // Select first threshold shares
      std::map<int, std::vector<int64_t>> subset;
      for (int x = 1; x <= threshold; ++x) {
        subset[x] = shares[x];
      }

      auto reconstructed = shamir_reconstruct(subset, threshold);
      assert(reconstructed == secret);

      // Also select last threshold shares
      std::map<int, std::vector<int64_t>> subset_last;
      for (int x = num_shares - threshold + 1; x <= num_shares; ++x) {
        subset_last[x] = shares[x];
      }
      auto reconstructed_last = shamir_reconstruct(subset_last, threshold);
      assert(reconstructed_last == secret);
    }
  }

  std::cout << "[PASS] test_shamir_thresholds_and_dimensions" << std::endl;
}

void test_lightsecagg_multi_client_simulation() {
  std::cout << "[RUN] test_lightsecagg_multi_client_simulation..." << std::endl;

  const int num_clients = 6;
  const int threshold = 4;
  const int dim = 64;

  std::mt19937_64 rng(999);
  std::uniform_real_distribution<double> val_dist(-2.0, 2.0);
  std::uniform_int_distribution<int64_t> mask_dist(0, kPrime - 1);

  // Each client has floating-point vector x_i and field mask z_i
  std::vector<std::vector<double>> client_values(num_clients, std::vector<double>(dim));
  std::vector<std::vector<int64_t>> client_masks(num_clients, std::vector<int64_t>(dim));
  std::vector<std::vector<int64_t>> masked_updates(num_clients);

  // Shares held by holder j from dealer i: dealer_shares[i][j]
  std::vector<std::map<int, std::vector<int64_t>>> dealer_shares(num_clients);

  for (int i = 0; i < num_clients; ++i) {
    for (int k = 0; k < dim; ++k) {
      client_values[i][k] = val_dist(rng);
      client_masks[i][k] = mask_dist(rng);
    }
    // Step 1: Client i secret-shares mask z_i
    dealer_shares[i] = compute_shamir_shares_seeded(client_masks[i], num_clients, threshold, rng());
    // Step 2: Client i masks its update: y_i = q(x_i) + z_i mod p
    masked_updates[i] = mask_vector(client_values[i], client_masks[i]);
  }

  // Suppose client 5 drops out! Surviving cohort: {0, 1, 2, 3, 4} (size 5 >= threshold 4)
  std::vector<int> survivors = {0, 1, 2, 3, 4};

  // Step 3: Each surviving holder j sums shares it holds from surviving dealers:
  // S_j = sum_{i in survivors} s_{i -> (j+1)} mod p
  std::map<int, std::vector<int64_t>> summed_shares_by_holder;
  for (int holder_idx : survivors) {
    int x = holder_idx + 1; // share index
    std::vector<std::vector<int64_t>> shares_to_sum;
    for (int dealer_idx : survivors) {
      shares_to_sum.push_back(dealer_shares[dealer_idx][x]);
    }
    summed_shares_by_holder[x] = aggregate_shares(shares_to_sum);
  }

  // Step 4: Server collects masked updates of survivors
  std::vector<std::vector<int64_t>> surviving_masked_updates;
  for (int i : survivors) {
    surviving_masked_updates.push_back(masked_updates[i]);
  }

  // Step 5: Server performs one-shot recovery
  std::vector<double> recovered = recover_aggregate(
      surviving_masked_updates, summed_shares_by_holder, threshold);

  // Calculate expected sum of survivors
  std::vector<double> expected(dim, 0.0);
  for (int i : survivors) {
    for (int k = 0; k < dim; ++k) {
      expected[k] += client_values[i][k];
    }
  }

  for (int k = 0; k < dim; ++k) {
    double err = std::abs(recovered[k] - expected[k]);
    assert(err < 1e-4); // Within quantization tolerance
  }

  std::cout << "[PASS] test_lightsecagg_multi_client_simulation" << std::endl;
}

void test_error_handling() {
  std::cout << "[RUN] test_error_handling..." << std::endl;

  // threshold > num_shares
  bool caught = false;
  try {
    compute_shamir_shares_seeded({1, 2, 3}, 3, 4, 123);
  } catch (const std::invalid_argument&) {
    caught = true;
  }
  assert(caught);

  // threshold = 0
  caught = false;
  try {
    compute_shamir_shares_seeded({1, 2, 3}, 3, 0, 123);
  } catch (const std::invalid_argument&) {
    caught = true;
  }
  assert(caught);

  // fewer than threshold shares for reconstruction
  caught = false;
  try {
    std::map<int, std::vector<int64_t>> shares = {{1, {10, 20}}};
    shamir_reconstruct(shares, 2);
  } catch (const std::invalid_argument&) {
    caught = true;
  }
  assert(caught);

  // dimension mismatch in aggregate_shares
  caught = false;
  try {
    aggregate_shares({{1, 2}, {1, 2, 3}});
  } catch (const std::invalid_argument&) {
    caught = true;
  }
  assert(caught);

  // empty shares in aggregate_shares
  caught = false;
  try {
    aggregate_shares({});
  } catch (const std::invalid_argument&) {
    caught = true;
  }
  assert(caught);

  // NaN in quantize
  caught = false;
  try {
    quantize(std::nan(""));
  } catch (const std::invalid_argument&) {
    caught = true;
  }
  assert(caught);

  std::cout << "[PASS] test_error_handling" << std::endl;
}

int main() {
  std::cout << "Starting LightSecAgg C++ Unit Tests..." << std::endl;
  test_basic_arithmetic();
  test_quantization();
  test_shamir_deterministic();
  test_shamir_thresholds_and_dimensions();
  test_lightsecagg_multi_client_simulation();
  test_error_handling();
  std::cout << "All LightSecAgg C++ Unit Tests PASSED!" << std::endl;
  return 0;
}
