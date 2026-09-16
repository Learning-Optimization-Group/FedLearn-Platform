"""Tests for the pure-Python RDP accountant of the Sampled Gaussian Mechanism.

Correctness gates (design doc `docs/plans/dp-ffa-lora-design.md`, FR-13):

  1. q=1 analytic oracle (LOAD-BEARING, done-when #1): an *independent* recomputation
     of ``ε(δ) = min_{α>1} [ steps·α/(2z²) + ln(1/δ)/(α−1) ]`` (never calling
     ``compute_rdp``), plus the calculus closed form ``ε* = steps/(2z²) +
     2·√(steps·ln(1/δ)/(2z²))``. The accountant must match both.
  2. q<1 SGM reference vectors from Opacus 1.6.0 (see banner on the literals below).
     Opacus is a ONE-OFF oracle used to mint these numbers — it is NOT a runtime or
     CI dependency and is never imported here or added to requirements.
  3. Monotonicity: ε ↑ as z ↓, ε ↑ as steps ↑, ε ↑ as q ↑.
  4. ``required_noise_multiplier`` round-trips.
  5. ``RDPAccountant`` matches ``compute_rdp`` + ``get_epsilon`` for homogeneous and
     heterogeneous round sequences.

A DELIBERATE bound choice, documented here so nobody "fixes" it later:
This module's ``get_epsilon`` uses the CLASSIC Mironov (2017) RDP→(ε,δ) conversion
``ε = min_α [ rdp(α) + ln(1/δ)/(α−1) ]`` — mandated by the spec and by the q=1
analytic oracle. Opacus' ``get_privacy_spent`` uses the TIGHTER Balle et al. (2020)
bound, which yields a *smaller* ε (by ~12–26% on the cases below). That gap is a
conversion-bound choice, NOT a mechanism error: the underlying per-order RDP produced
by ``compute_rdp`` matches Opacus to ~1e-9 (asserted directly in
``test_compute_rdp_matches_opacus_per_order``). Our ε is therefore *conservative*
(never under-reports privacy loss) relative to Opacus — the safe direction for a
privacy accountant. Both numbers are recorded below for transparency.
"""

import math

import numpy as np
import pytest

from fedlearn.privacy.dp_accountant import (
    DEFAULT_ORDERS,
    RDPAccountant,
    compute_rdp,
    get_epsilon,
    required_noise_multiplier,
)


# --------------------------------------------------------------------------------------
# Independent oracles (NO call into compute_rdp)
# --------------------------------------------------------------------------------------
def _analytic_eps_q1_grid(z, steps, delta, orders):
    """ε for the non-subsampled Gaussian composed `steps` times, minimised over `orders`.

    ε(α) = steps·α/(2z²) + ln(1/δ)/(α−1). Independent of compute_rdp.
    """
    best_e, best_a = math.inf, None
    for a in orders:
        e = steps * a / (2.0 * z * z) + math.log(1.0 / delta) / (a - 1.0)
        if e < best_e:
            best_e, best_a = e, a
    return best_e, best_a


def _analytic_eps_q1_closed_form(z, steps, delta):
    """Continuous optimum of the same objective, via calculus.

    d/dα [ c1·α + c2/(α−1) ] = 0  →  α* = 1 + √(c2/c1),
    ε* = c1 + 2·√(c1·c2), with c1 = steps/(2z²), c2 = ln(1/δ).
    """
    c1 = steps / (2.0 * z * z)
    c2 = math.log(1.0 / delta)
    return c1 + 2.0 * math.sqrt(c1 * c2)


def _classic_eps_from_rdp(rdp, orders, delta):
    """The classic Mironov RDP→ε conversion, recomputed independently of get_epsilon."""
    best_e, best_a = math.inf, None
    for a, r in zip(orders, rdp):
        if not math.isfinite(r):
            continue
        e = r + math.log(1.0 / delta) / (a - 1.0)
        if e < best_e:
            best_e, best_a = e, a
    return best_e, best_a


# --------------------------------------------------------------------------------------
# Opacus 1.6.0 reference vectors — GENERATED ONCE, hardcoded, NOT a runtime dependency.
# Provenance:  opacus 1.6.0  (opacus.accountants.analysis.rdp.compute_rdp)
#   per-order RDP:  compute_rdp(q=q, noise_multiplier=z, steps=1, orders=[...])
#   end-to-end ε :  compute_rdp(..., steps=T) → both bounds (classic & Balle/opacus)
# Do NOT import opacus anywhere in the committed suite; these literals are the oracle.
# --------------------------------------------------------------------------------------

# Single-step per-order RDP  (q=0.01, z=1.0, steps=1) — exercises int & fractional α.
OPACUS_RDP_Q001_Z1 = {
    1.1: 9.241470982855839e-05,
    1.5: 0.00012725374351154145,
    2.0: 0.00017181342207453428,
    2.5: 0.00021757533233085147,
    3.0: 0.00026463757458466063,
    4.0: 0.00036315404891075525,
    5.0: 0.000468667242169153,
    7.9: 0.0008694429577251905,
    8.4: 0.001033969548280792,
    10.0: 0.03827041889494864,
    15.5: 2.8272852759704428,
    16.0: 3.087850783696245,
    32.0: 11.246275937048072,
    63.0: 26.820552875528232,
    64.0: 27.32173187455178,
    128.0: 59.358568631445074,
}

# Single-step per-order RDP  (q=0.05, z=1.2, steps=1).
OPACUS_RDP_Q005_Z12 = {
    1.5: 0.0018250340180205305,
    2.0: 0.0025033545202484825,
    4.4: 0.006497075426399607,
    8.0: 0.021782161012633073,
    16.0: 2.360716814281843,
    32.0: 8.01874232137674,
}

# End-to-end ε over the full DEFAULT_ORDERS grid.  `eps_classic` = Opacus RDP fed
# through the classic Mironov bound (what THIS module computes); `eps_opacus_tighter`
# = Opacus' own get_privacy_spent (Balle et al.), recorded for transparency only.
OPACUS_EPS_CASES = [
    # (q,     z,   steps, delta,  eps_classic,        eps_opacus_tighter)
    (0.01, 1.0, 1000, 1e-5, 2.537982880184644, 2.1013652716430564),
    (0.001, 0.8, 5000, 1e-6, 2.0089898324077726, 1.5923077586416754),
    (0.05, 1.2, 500, 1e-5, 6.634692261720458, 5.9281503749478786),
]


# --------------------------------------------------------------------------------------
# 0. API / grid invariants
# --------------------------------------------------------------------------------------
def test_default_orders_exact_definition():
    expected = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)) + [128, 256, 512]
    assert DEFAULT_ORDERS == expected
    # α = 1 (which would divide by zero in the ε conversion) must never appear.
    assert all(a > 1.0 for a in DEFAULT_ORDERS)


def test_compute_rdp_length_matches_orders():
    rdp = compute_rdp(q=0.01, noise_multiplier=1.0, steps=5)
    assert len(rdp) == len(DEFAULT_ORDERS)


# --------------------------------------------------------------------------------------
# 1. q=1 analytic oracle  (LOAD-BEARING)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "z,steps,delta",
    [
        (1.0, 100, 1e-5),
        (2.0, 50, 1e-5),
        (4.0, 20, 1e-6),
        (0.7, 500, 1e-5),
        (10.0, 10, 1e-7),
    ],
)
def test_q1_reduces_to_plain_gaussian_rdp(z, steps, delta):
    """compute_rdp(q=1) must equal the closed form steps·α/(2z²) at every order."""
    rdp = compute_rdp(q=1.0, noise_multiplier=z, steps=steps)
    for a, r in zip(DEFAULT_ORDERS, rdp):
        assert r == pytest.approx(steps * a / (2.0 * z * z), rel=1e-12)


@pytest.mark.parametrize(
    "z,steps,delta",
    [
        (1.0, 100, 1e-5),
        (2.0, 50, 1e-5),
        (4.0, 20, 1e-6),
        (0.7, 500, 1e-5),
        (10.0, 10, 1e-7),
    ],
)
def test_q1_get_epsilon_matches_independent_grid_oracle(z, steps, delta):
    """Accountant ε at q=1 must equal the independent grid min to ~machine precision."""
    oracle_eps, oracle_order = _analytic_eps_q1_grid(z, steps, delta, DEFAULT_ORDERS)
    rdp = compute_rdp(q=1.0, noise_multiplier=z, steps=steps)
    eps, order = get_epsilon(rdp, delta=delta)
    assert eps == pytest.approx(oracle_eps, rel=1e-9)
    assert order == pytest.approx(oracle_order, rel=1e-12)


@pytest.mark.parametrize(
    "z,steps,delta",
    [
        (1.0, 100, 1e-5),
        (4.0, 20, 1e-6),
        (2.0, 200, 1e-5),
    ],
)
def test_q1_get_epsilon_near_continuous_closed_form(z, steps, delta):
    """Grid-restricted ε must sit just above the calculus continuum optimum (≤1%)."""
    eps_star = _analytic_eps_q1_closed_form(z, steps, delta)
    rdp = compute_rdp(q=1.0, noise_multiplier=z, steps=steps)
    eps, _ = get_epsilon(rdp, delta=delta)
    assert eps >= eps_star - 1e-9  # grid min can never beat the continuum
    assert eps <= eps_star * 1.01  # grid is dense enough to be within 1%


# --------------------------------------------------------------------------------------
# 2. q<1 SGM — direct mechanism check against Opacus per-order RDP
# --------------------------------------------------------------------------------------
def test_compute_rdp_matches_opacus_per_order():
    """The heart of the correctness gate: our SGM RDP == Opacus SGM RDP, per order,
    for BOTH integer and fractional α. Bound-independent (no ε conversion here)."""
    for (q, z, ref) in [
        (0.01, 1.0, OPACUS_RDP_Q001_Z1),
        (0.05, 1.2, OPACUS_RDP_Q005_Z12),
    ]:
        orders = list(ref.keys())
        rdp = compute_rdp(q=q, noise_multiplier=z, steps=1, orders=orders)
        for a, r in zip(orders, rdp):
            assert r == pytest.approx(ref[a], rel=1e-6), f"q={q} z={z} α={a}"


def test_q_lt_1_end_to_end_epsilon_matches_opacus_classic_bound():
    """End-to-end ε (our compute_rdp → our classic get_epsilon) vs Opacus RDP through
    the SAME classic bound. Same conversion on both sides ⇒ this isolates the
    mechanism and must match tightly (≪ the 1–2% budget)."""
    for (q, z, steps, delta, eps_classic, _eps_tighter) in OPACUS_EPS_CASES:
        rdp = compute_rdp(q=q, noise_multiplier=z, steps=steps)
        eps, _ = get_epsilon(rdp, delta=delta)
        assert eps == pytest.approx(eps_classic, rel=1e-4), f"q={q} z={z} T={steps}"


def test_q_lt_1_epsilon_conservative_vs_opacus_tighter_bound():
    """Sanity: our (classic) ε is >= Opacus' (Balle) ε and within a documented gap.
    Conservative = the safe direction for a privacy accountant."""
    for (q, z, steps, delta, _eps_classic, eps_tighter) in OPACUS_EPS_CASES:
        rdp = compute_rdp(q=q, noise_multiplier=z, steps=steps)
        eps, _ = get_epsilon(rdp, delta=delta)
        assert eps >= eps_tighter - 1e-9
        assert eps <= eps_tighter * 1.30  # observed 12–26% classic-vs-Balle gap


# --------------------------------------------------------------------------------------
# 3. Monotonicity
# --------------------------------------------------------------------------------------
def test_epsilon_decreases_as_noise_increases():
    delta = 1e-5
    zs = [0.5, 0.8, 1.0, 2.0, 4.0, 8.0]
    eps = [get_epsilon(compute_rdp(q=0.01, noise_multiplier=z, steps=1000), delta=delta)[0] for z in zs]
    for a, b in zip(eps, eps[1:]):
        assert b < a  # ε strictly decreases as z increases


def test_epsilon_increases_with_steps():
    delta = 1e-5
    steps_list = [10, 100, 1000, 5000]
    eps = [get_epsilon(compute_rdp(q=0.01, noise_multiplier=1.0, steps=s), delta=delta)[0] for s in steps_list]
    for a, b in zip(eps, eps[1:]):
        assert b > a


def test_epsilon_increases_with_sampling_rate():
    delta = 1e-5
    qs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    eps = [get_epsilon(compute_rdp(q=q, noise_multiplier=1.5, steps=500), delta=delta)[0] for q in qs]
    for a, b in zip(eps, eps[1:]):
        assert b > a


# --------------------------------------------------------------------------------------
# 4. required_noise_multiplier round-trip
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "target_eps,q,steps,delta",
    [
        (3.0, 0.01, 1000, 1e-5),
        (6.0, 0.05, 500, 1e-5),
        (1.5, 0.001, 5000, 1e-6),
        (8.0, 0.02, 2000, 1e-5),
    ],
)
def test_required_noise_multiplier_round_trip(target_eps, q, steps, delta):
    z = required_noise_multiplier(target_eps, q, steps, delta)
    assert z > 0
    eps, _ = get_epsilon(compute_rdp(q=q, noise_multiplier=z, steps=steps), delta=delta)
    assert eps <= target_eps + 1e-6                 # solved z meets the target
    assert eps >= target_eps * 0.98                 # ... and is not wildly over-shooting


def test_required_noise_multiplier_monotone_in_target():
    q, steps, delta = 0.01, 1000, 1e-5
    z_loose = required_noise_multiplier(8.0, q, steps, delta)
    z_tight = required_noise_multiplier(2.0, q, steps, delta)
    assert z_tight > z_loose  # a tighter ε target needs more noise


def test_required_noise_multiplier_infeasible_raises():
    # Absurdly tiny ε target that no z ≤ 1e6 can reach for this many steps.
    with pytest.raises(ValueError):
        required_noise_multiplier(1e-9, 0.5, 1_000_000, 1e-6)


# --------------------------------------------------------------------------------------
# 5. RDPAccountant
# --------------------------------------------------------------------------------------
def test_accountant_matches_compute_rdp_for_homogeneous_rounds():
    q, z, rounds, delta = 0.01, 1.1, 200, 1e-5
    acct = RDPAccountant()
    for _ in range(rounds):
        acct.step(noise_multiplier=z, sample_rate=q, num_steps=1)
    eps_acct, order_acct = acct.get_privacy_spent(delta)

    rdp = compute_rdp(q=q, noise_multiplier=z, steps=rounds)
    eps_ref, order_ref = get_epsilon(rdp, delta=delta)
    assert eps_acct == pytest.approx(eps_ref, rel=1e-12)
    assert order_acct == pytest.approx(order_ref, rel=1e-12)


def test_accountant_num_steps_equivalent_to_repeated_steps():
    q, z, delta = 0.02, 1.3, 1e-5
    a1 = RDPAccountant()
    a1.step(noise_multiplier=z, sample_rate=q, num_steps=350)
    a2 = RDPAccountant()
    for _ in range(350):
        a2.step(noise_multiplier=z, sample_rate=q)
    e1, _ = a1.get_privacy_spent(delta)
    e2, _ = a2.get_privacy_spent(delta)
    assert e1 == pytest.approx(e2, rel=1e-12)


def test_accountant_sums_heterogeneous_rounds():
    """Heterogeneous rounds: running RDP is the SUM of per-round RDP vectors."""
    delta = 1e-5
    schedule = [
        (0.01, 1.0, 300),
        (0.05, 1.5, 100),
        (0.02, 0.8, 50),
    ]
    acct = RDPAccountant()
    for (q, z, n) in schedule:
        acct.step(noise_multiplier=z, sample_rate=q, num_steps=n)
    eps_acct, order_acct = acct.get_privacy_spent(delta)

    total = np.zeros(len(DEFAULT_ORDERS))
    for (q, z, n) in schedule:
        total = total + np.asarray(compute_rdp(q=q, noise_multiplier=z, steps=n))
    eps_ref, order_ref = get_epsilon(list(total), delta=delta)

    assert eps_acct == pytest.approx(eps_ref, rel=1e-12)
    assert order_acct == pytest.approx(order_ref, rel=1e-12)


def test_accountant_privacy_grows_monotonically_over_rounds():
    q, z, delta = 0.01, 1.0, 1e-5
    acct = RDPAccountant()
    prev = 0.0
    for _ in range(5):
        acct.step(noise_multiplier=z, sample_rate=q, num_steps=100)
        eps, _ = acct.get_privacy_spent(delta)
        assert eps > prev
        prev = eps


# --------------------------------------------------------------------------------------
# 6. Guards / edge cases
# --------------------------------------------------------------------------------------
def test_q_zero_gives_zero_rdp():
    """q=0 touches no data ⇒ zero RDP at every order.

    Note the *converted* ε is not exactly 0: the classic bound leaves a residual
    ln(1/δ)/(α_max − 1) term that only vanishes as α → ∞, so on a finite order grid
    ε = ln(1/δ)/(α_max − 1) (a known RDP→(ε,δ) conversion artifact; Opacus agrees).
    """
    rdp = compute_rdp(q=0.0, noise_multiplier=1.0, steps=1000)
    assert all(r == 0.0 for r in rdp)
    delta = 1e-5
    eps, order = get_epsilon(rdp, delta=delta)
    assert order == max(DEFAULT_ORDERS)
    assert eps == pytest.approx(math.log(1.0 / delta) / (max(DEFAULT_ORDERS) - 1.0), rel=1e-12)


def test_zero_steps_gives_zero_rdp():
    rdp = compute_rdp(q=0.01, noise_multiplier=1.0, steps=0)
    assert all(r == 0.0 for r in rdp)


def test_negative_steps_raises():
    with pytest.raises(ValueError):
        compute_rdp(q=0.01, noise_multiplier=1.0, steps=-1)


def test_nonpositive_noise_raises():
    with pytest.raises(ValueError):
        compute_rdp(q=0.01, noise_multiplier=0.0, steps=10)
    with pytest.raises(ValueError):
        compute_rdp(q=0.01, noise_multiplier=-1.0, steps=10)


def test_invalid_sample_rate_raises():
    with pytest.raises(ValueError):
        compute_rdp(q=-0.1, noise_multiplier=1.0, steps=10)
    with pytest.raises(ValueError):
        compute_rdp(q=1.5, noise_multiplier=1.0, steps=10)


def test_invalid_delta_raises():
    rdp = compute_rdp(q=0.01, noise_multiplier=1.0, steps=100)
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError):
            get_epsilon(rdp, delta=bad)


def test_custom_orders_are_respected():
    orders = [2.0, 4.0, 8.0, 16.0]
    rdp = compute_rdp(q=0.01, noise_multiplier=1.0, steps=100, orders=orders)
    assert len(rdp) == len(orders)
    eps, order = get_epsilon(rdp, orders=orders, delta=1e-5)
    assert order in orders
