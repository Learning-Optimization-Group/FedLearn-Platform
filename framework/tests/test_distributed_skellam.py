import math

import numpy as np
import pytest

from fedlearn.privacy.distributed_skellam import (
    INTEGER_ORDERS, calibrate_mu, conditional_round, decode, encode, fwht, largest_prime_below, next_pow2,
    plan_distributed_skellam, rotation_signs, rounded_norm_bound_sq, skellam_epsilon, skellam_rdp,
    wrapped_coordinates)
from fedlearn.privacy.dp_accountant import required_noise_multiplier


def _log_poisson(k, lam):
    return k * math.log(lam) - lam - math.lgamma(k + 1)


def _skellam_log_pmf(mu, support):
    """log P(X = k) for X = Poisson(mu/2) - Poisson(mu/2), by direct convolution in log space."""
    lam = mu / 2.0
    top = int(lam + 40 * math.sqrt(lam) + 50)
    out = {}
    for k in support:
        terms = [_log_poisson(j + abs(k), lam) + _log_poisson(j, lam) for j in range(top)]
        m = max(terms)
        out[k] = m + math.log(sum(math.exp(t - m) for t in terms))
    return out


def test_primes_for_bit_widths():
    assert largest_prime_below(2 ** 12) == 4093
    assert largest_prime_below(2 ** 16) == 65521
    assert largest_prime_below(2 ** 31) == 2 ** 31 - 1
    with pytest.raises(ValueError):
        largest_prime_below(2)


def test_fwht_is_orthonormal_and_self_inverse():
    h = np.array([[1.0]])
    while h.shape[0] < 8:
        h = np.block([[h, h], [h, -h]])
    x = np.random.default_rng(0).normal(size=(3, 8))
    assert np.allclose(fwht(x), x @ h.T / math.sqrt(8))
    assert np.allclose(fwht(fwht(x)), x)
    assert np.allclose(np.linalg.norm(fwht(x), axis=1), np.linalg.norm(x, axis=1))
    assert next_pow2(1026) == 2048 and next_pow2(1024) == 1024
    with pytest.raises(ValueError):
        fwht(np.ones(6))


def test_rdp_matches_corollary_and_approaches_gaussian():
    a, mu, d1, d2 = 5, 1000.0, 30.0, 10.0
    expected = a * d2 ** 2 / (2 * mu) + min(((2 * a - 1) * d2 ** 2 + 6 * d1) / (4 * mu ** 2), 3 * d1 / (2 * mu))
    assert skellam_rdp(a, mu, d1, d2) == pytest.approx(expected)
    big = 1e8
    assert skellam_rdp(a, big, d1, d2) / (a * d2 ** 2 / (2 * big)) == pytest.approx(1.0, rel=1e-5)
    for bad in (1, 2.5):
        with pytest.raises(ValueError):
            skellam_rdp(bad, mu, d1, d2)


@pytest.mark.parametrize("mu,shift,alpha", [(20.0, 1, 2), (20.0, 1, 4), (60.0, 2, 3)])
def test_theorem_bound_dominates_exact_renyi_divergence(mu, shift, alpha):
    """Theorem 3.5 (1-D, Delta1 = Delta2 = shift) upper-bounds the exact Renyi divergence."""
    support = range(-int(mu + 40 * math.sqrt(mu) + 50), int(mu + 40 * math.sqrt(mu) + 50))
    logq = _skellam_log_pmf(mu, sorted(set(support) | {k - shift for k in support}))
    logs = [alpha * logq[k - shift] + (1 - alpha) * logq[k] for k in support]
    m = max(logs)
    exact = (m + math.log(sum(math.exp(v - m) for v in logs))) / (alpha - 1)
    bound = skellam_rdp(alpha, mu, shift, shift)
    assert exact <= bound + 1e-12
    assert exact >= alpha * shift ** 2 / (2 * mu) * 0.8  # the bound is not vacuous


def test_calibration_hits_target_and_is_minimal():
    mu = calibrate_mu(4.0, 2000.0, 100.0, 20, 1e-5)
    eps, order = skellam_epsilon(mu, 2000.0, 100.0, 20, 1e-5)
    assert eps <= 4.0 and eps == pytest.approx(4.0, abs=1e-6)
    assert skellam_epsilon(mu * 0.999, 2000.0, 100.0, 20, 1e-5)[0] > 4.0
    assert order in INTEGER_ORDERS


def test_plan_converges_to_gaussian_rdp_at_high_precision_and_respects_field():
    for eps in (8.0, 4.0, 1.0):
        plan = plan_distributed_skellam(eps, 1e-5, 20, 0.05, 1026, 80, bits=31)
        z_eff = plan.sum_noise_std / (2 * 0.05)
        assert z_eff == pytest.approx(required_noise_multiplier(eps, 1.0, 20, 1e-5), rel=0.01)
        assert plan.epsilon <= eps + 1e-9
        assert 2 * plan.k * plan.signal_bound_std <= plan.prime
    low = plan_distributed_skellam(4.0, 1e-5, 20, 0.05, 1026, 80, bits=14)
    high = plan_distributed_skellam(4.0, 1e-5, 20, 0.05, 1026, 80, bits=20)
    assert low.sum_noise_std > high.sum_noise_std
    with pytest.raises(ValueError):
        plan_distributed_skellam(1.0, 1e-5, 20, 0.05, 1026, 80, bits=12)
    add_remove = plan_distributed_skellam(4.0, 1e-5, 20, 0.05, 1026, 80, bits=20, adjacency="add_remove")
    assert add_remove.sum_noise_std == pytest.approx(high.sum_noise_std / 2, rel=0.01)


def test_conditional_rounding_is_unbiased_and_bounded():
    rng = np.random.default_rng(1)
    values = np.full(1000, 0.3)
    draws = np.stack([conditional_round(values, math.inf, rng)[0] for _ in range(200)])
    assert draws.mean() == pytest.approx(0.3, abs=0.01)
    bound = rounded_norm_bound_sq(float(np.linalg.norm(values)), 1000, math.exp(-0.5))
    rounded, tries = conditional_round(values, bound, rng)
    assert float(rounded @ rounded) <= bound and tries >= 1
    with pytest.raises(RuntimeError):
        conditional_round(values, 0.0, rng, max_tries=5)


def test_aggregate_noise_has_variance_mu():
    plan = plan_distributed_skellam(4.0, 1e-5, 20, 0.05, 1026, 40, bits=16)
    signs = rotation_signs(3, plan.d_pad)
    rng = np.random.default_rng(2)
    total = sum(encode(np.zeros(plan.d), plan, signs, rng)[1] for _ in range(plan.clients))
    assert total.var() == pytest.approx(plan.mu, rel=0.1)
    assert abs(total.mean()) < 4 * math.sqrt(plan.mu / plan.d_pad)


def test_encode_decode_recovers_the_sum_of_clipped_updates():
    plan = plan_distributed_skellam(1e4, 1e-5, 1, 0.05, 37, 10, bits=31)
    signs = rotation_signs(4, plan.d_pad)
    rng = np.random.default_rng(5)
    updates = [rng.normal(size=37) * 0.001 for _ in range(9)] + [np.ones(37)]  # the last one is clipped
    expected = sum(updates[:9]) + np.ones(37) * 0.05 / math.sqrt(37)
    field = sum(encode(u, plan, signs, rng)[0] for u in updates) % plan.prime
    estimate = decode(field, plan, signs)
    tolerance = 6 * (plan.sum_noise_std + math.sqrt(plan.clients) / plan.scale)
    assert np.abs(estimate - expected).max() < tolerance
    # Nothing on the wire stops a client skipping the clip: a norm-0.5 upload (10x the clip) arrives whole.
    big = np.full(37, 0.5 / math.sqrt(37))
    field = sum(encode(u, plan, signs, rng, clip=u is not big)[0] for u in updates[:9] + [big]) % plan.prime
    assert np.abs(decode(field, plan, signs) - (sum(updates[:9]) + big)).max() < tolerance


def test_wrapped_coordinates_counts_out_of_range_sums():
    plan = plan_distributed_skellam(4.0, 1e-5, 20, 0.05, 1026, 80, bits=12)
    half = plan.prime // 2
    assert wrapped_coordinates(np.array([0, half, half + 1, -half, -half - 1]), plan) == 2
    with pytest.raises(ValueError):
        encode(np.zeros(5), plan, rotation_signs(0, plan.d_pad), np.random.default_rng(0))
