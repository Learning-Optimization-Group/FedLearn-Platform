"""FR-13 — adversarial hardening for the from-scratch RDP accountant + DP mechanism.

These are NEW edge cases not covered by ``test_dp_accountant.py`` / ``test_dp_mechanism.py``.
Each asserts a behaviour that the module already gets RIGHT; a case where the module misbehaves
is reported to the integrator instead of committed here (see the audit's defect list).

Coverage added:
  A1. ``required_noise_multiplier`` solver CONTRACT (ε(z) <= target) across extreme regimes the
      existing round-trip never touches: q=1 (non-subsampled), q -> 0 (tiny), a very loose target
      (solver saturates near z_min), a tight target (large z), and steps=1 — plus a wide monotone
      sweep (smaller target ⇒ larger z).
  A2. ``get_epsilon`` min-over-orders is robust to a non-finite (inf/nan) RDP entry in the grid:
      the classic bound never SELECTS a non-finite order, and an all-inf curve degrades to
      (inf, first_order) without raising.
  C1. ``dp_aggregate`` clip boundary EXACTNESS: a delta with L2 norm exactly = S is left unscaled
      (to float32), a delta ≫ S is scaled to exactly S, and the clipped norm never exceeds S.
  C2. Zero-norm delta (client == global): no divide-by-zero, zero contribution, no NaN — alone and
      mixed into a cohort.
  C3. Noise calibration follows the exact per-coordinate std z*S/N, INCLUDING the 1/N law (std at
      N=5 is 2/5 of std at N=2 for the same z, S) — measured across coordinates in single draws.
"""

import math
from collections import OrderedDict

import pytest
import torch

from fedlearn.privacy.dp_accountant import (
    DEFAULT_ORDERS,
    compute_rdp,
    get_epsilon,
    required_noise_multiplier,
)
from fedlearn.privacy.dp_mechanism import dp_aggregate


B_KEY = "m.lora_B.weight"
H_KEY = "score.weight"
A_KEY = "m.lora_A.weight"


def _global(b_shape=(2, 1), h_len=1):
    """Aggregatable-key reference at zero, so each client's delta == its uploaded value."""
    return OrderedDict(
        [
            (A_KEY, torch.tensor([[1.0, 2.0]])),
            (B_KEY, torch.zeros(*b_shape)),
            (H_KEY, torch.zeros(h_len)),
        ]
    )


def _delta_norm(out, ref, keys):
    clipped = OrderedDict((k, out[k] - ref[k].float()) for k in keys)
    return float(torch.sqrt(sum((t * t).sum() for t in clipped.values())))


# ======================================================================================
# A1. required_noise_multiplier — solver contract at extreme regimes
# ======================================================================================
@pytest.mark.parametrize(
    "target,q,steps,delta",
    [
        (0.05, 0.001, 100, 1e-6),   # very tight ε -> large z
        (2.0, 1.0, 500, 1e-6),      # q=1 non-subsampled Gaussian
        (0.5, 1.0, 500, 1e-6),      # q=1, tighter
        (1.0, 1e-6, 1000, 1e-6),    # q -> 0 (tiny but > 0)
        (1.0e6, 0.5, 1000, 1e-6),   # absurdly loose target: solver saturates toward z_min
        (3.0, 0.01, 1, 1e-5),       # steps = 1
        (8.0, 0.02, 2000, 1e-5),    # large steps
    ],
)
def test_required_noise_multiplier_solver_always_satisfies_target(target, q, steps, delta):
    """The returned z MUST make the accounted ε <= target — the solver's core contract —
    even at q=1, q->0, steps=1, and extreme targets the existing round-trip never exercises."""
    z = required_noise_multiplier(target, q, steps, delta)
    assert 1e-3 <= z <= 1e6           # within the search bracket
    eps = get_epsilon(compute_rdp(q, z, steps), delta)[0]
    assert eps <= target + 1e-9       # ε(z) <= target, the invariant the bisection guarantees


def test_required_noise_multiplier_strictly_monotone_over_wide_range():
    """A tighter ε target must never need LESS noise — over a decade-wide target sweep."""
    q, steps, delta = 0.01, 1000, 1e-5
    targets = [50.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.7, 0.3]
    zs = [required_noise_multiplier(t, q, steps, delta) for t in targets]
    for z_loose, z_tight in zip(zs, zs[1:]):
        assert z_tight > z_loose      # target ↓  ⇒  z ↑ (strict)


def test_required_noise_multiplier_q1_roundtrips_through_closed_form():
    """At q=1 the accounted ε(z) is the plain-Gaussian closed form; the solved z must land it
    just under target with no subsampling amplification in play."""
    target, steps, delta = 1.5, 400, 1e-6
    z = required_noise_multiplier(target, q=1.0, steps=steps, delta=delta)
    # Independent q=1 closed form: ε* = steps/(2z²) + 2·√(steps·ln(1/δ)/(2z²)).
    c1 = steps / (2.0 * z * z)
    c2 = math.log(1.0 / delta)
    eps_star = c1 + 2.0 * math.sqrt(c1 * c2)
    assert eps_star <= target + 1e-9
    # The module minimises over the DISCRETE order grid, so its ε sits just above the continuum
    # optimum (grid can't beat the continuum) yet within grid density (<=1%), and still <= target.
    eps_mod = get_epsilon(compute_rdp(1.0, z, steps), delta)[0]
    assert eps_mod >= eps_star - 1e-9
    assert eps_mod <= eps_star * 1.01
    assert eps_mod <= target + 1e-9


# ======================================================================================
# A2. get_epsilon — min-over-orders robust to non-finite RDP entries
# ======================================================================================
def test_get_epsilon_ignores_inf_or_nan_rdp_entry():
    """A non-finite RDP value at any single order must not be SELECTED as the minimising order,
    so the reported (ε, order) is identical to the clean curve."""
    clean = compute_rdp(q=0.01, noise_multiplier=1.0, steps=1000)
    eps_clean, order_clean = get_epsilon(clean, delta=1e-5)

    for bad in (math.inf, math.nan):
        for idx in (0, len(clean) // 2, len(clean) - 1):
            poisoned = list(clean)
            poisoned[idx] = bad
            eps, order = get_epsilon(poisoned, delta=1e-5)
            assert eps == pytest.approx(eps_clean, rel=1e-12)
            assert order == pytest.approx(order_clean, rel=1e-12)


def test_get_epsilon_all_inf_curve_degrades_without_raising():
    """A degenerate all-inf RDP curve returns (inf, first_order) rather than crashing."""
    all_inf = [math.inf] * len(DEFAULT_ORDERS)
    eps, order = get_epsilon(all_inf, delta=1e-5)
    assert math.isinf(eps)
    assert order == float(DEFAULT_ORDERS[0])


# ======================================================================================
# C1. dp_aggregate — clip boundary exactness
# ======================================================================================
def test_clip_delta_at_exactly_S_is_left_unscaled():
    """||delta|| == S must pass through unscaled (to float32): B=(3,4) => norm exactly 5 == S."""
    ref = _global()
    keys = [B_KEY, H_KEY]
    S = 5.0
    delta = OrderedDict([(B_KEY, torch.tensor([[3.0], [4.0]])), (H_KEY, torch.tensor([0.0]))])
    out = dp_aggregate([("c", delta, 1)], ref, keys, clip_norm=S, noise_multiplier=0.0)
    # Single client, z=0 => out - ref == the (un)clipped delta exactly.
    assert torch.allclose(out[B_KEY], delta[B_KEY], atol=0.0, rtol=0.0)
    assert torch.allclose(out[H_KEY], delta[H_KEY], atol=0.0, rtol=0.0)
    assert _delta_norm(out, ref, keys) <= S + 1e-6


def test_clip_oversized_delta_scaled_to_exactly_S():
    """||delta|| = 50 ≫ S=5 must be scaled to land AT the boundary (norm == S), never above."""
    ref = _global()
    keys = [B_KEY, H_KEY]
    S = 5.0
    delta = OrderedDict([(B_KEY, torch.tensor([[30.0], [40.0]])), (H_KEY, torch.tensor([0.0]))])
    out = dp_aggregate([("c", delta, 1)], ref, keys, clip_norm=S, noise_multiplier=0.0)
    norm = _delta_norm(out, ref, keys)
    assert norm == pytest.approx(S, rel=1e-5)
    assert norm <= S + 1e-6                                    # clip is the safe direction
    # Direction preserved: clipped == delta * (S / ||delta||).
    assert torch.allclose(out[B_KEY], delta[B_KEY] * (S / 50.0), rtol=1e-5)


# ======================================================================================
# C2. dp_aggregate — zero-norm delta: no divide-by-zero, no NaN
# ======================================================================================
def test_zero_norm_delta_single_client_is_safe():
    """A client whose params equal the global (delta == 0) yields zero contribution, not NaN."""
    ref = _global()
    keys = [B_KEY, H_KEY]
    zero = OrderedDict([(B_KEY, torch.zeros(2, 1)), (H_KEY, torch.zeros(1))])
    out = dp_aggregate([("c", zero, 1)], ref, keys, clip_norm=1.0, noise_multiplier=0.0)
    assert not torch.isnan(out[B_KEY]).any() and not torch.isnan(out[H_KEY]).any()
    assert torch.equal(out[B_KEY], torch.zeros(2, 1))          # ref(0) + mean_delta(0)
    assert torch.equal(out[H_KEY], torch.zeros(1))


def test_zero_norm_client_mixed_into_cohort_contributes_zero():
    """One zero-delta client among non-zero clients contributes 0 to the uniform average (no
    divide-by-zero in its clip), so the mean is (0 + 2)/2 = 1 per coordinate."""
    ref = _global()
    keys = [B_KEY, H_KEY]
    zero = OrderedDict([(B_KEY, torch.zeros(2, 1)), (H_KEY, torch.zeros(1))])
    two = OrderedDict([(B_KEY, torch.full((2, 1), 2.0)), (H_KEY, torch.tensor([2.0]))])
    out = dp_aggregate(
        [("z", zero, 1), ("t", two, 1)], ref, keys, clip_norm=1e9, noise_multiplier=0.0
    )
    assert not torch.isnan(out[B_KEY]).any()
    assert torch.allclose(out[B_KEY], torch.full((2, 1), 1.0))  # uniform (0 + 2)/2
    assert torch.allclose(out[H_KEY], torch.tensor([1.0]))


# ======================================================================================
# C3. Noise calibration — exact per-coordinate std z*S/N and the 1/N law
# ======================================================================================
def _pure_noise_std_across_coords(n_clients, z, S, k_coords, seed):
    """Aggregate n zero-delta clients (mean_delta == 0) over a big head so the output IS pure
    per-coordinate noise; return the sample std across its coordinates."""
    ref = OrderedDict(
        [
            (A_KEY, torch.tensor([[1.0, 2.0]])),
            (B_KEY, torch.zeros(2, 1)),
            (H_KEY, torch.zeros(k_coords)),
        ]
    )
    keys = [B_KEY, H_KEY]
    zero = OrderedDict([(B_KEY, torch.zeros(2, 1)), (H_KEY, torch.zeros(k_coords))])
    results = [(f"c{i}", zero, 1) for i in range(n_clients)]
    g = torch.Generator().manual_seed(seed)
    out = dp_aggregate(results, ref, keys, clip_norm=S, noise_multiplier=z, generator=g)
    noise = out[H_KEY] - ref[H_KEY].float()                    # ref(0) + 0 + noise
    return float(noise.std(unbiased=True))


def test_noise_std_is_exactly_z_S_over_N_and_scales_as_one_over_N():
    z, S, K = 2.0, 6.0, 8000
    std_n2 = _pure_noise_std_across_coords(2, z, S, K, seed=11)
    std_n5 = _pure_noise_std_across_coords(5, z, S, K, seed=11)

    # Absolute calibration: per-coordinate std == z*S/N.
    assert std_n2 == pytest.approx(z * S / 2, rel=0.06)
    assert std_n5 == pytest.approx(z * S / 5, rel=0.06)
    # The 1/N law: std(N=2) / std(N=5) == 5/2, NOT 1/sqrt(N) or a constant.
    assert std_n2 / std_n5 == pytest.approx(2.5, rel=0.08)


# C4. Non-finite client update must be REJECTED (audit fix) — a NaN/Inf coordinate defeats the L2
# clip (scale = min(1.0, S/(NaN)) = 1.0; Inf*scale(0) = NaN), silently corrupting the aggregated
# coordinate for EVERY client and voiding the sensitivity bound the (eps,delta) claim rests on.
@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_dp_aggregate_rejects_a_non_finite_client_update(bad_val):
    B, H = "m.lora_B.weight", "score.weight"
    gp = OrderedDict([("m.lora_A.weight", torch.tensor([[1.0, 2.0]])),
                      (B, torch.zeros(2, 1)), (H, torch.zeros(1))])
    good = OrderedDict([(B, torch.tensor([[1.0], [1.0]])), (H, torch.tensor([1.0]))])
    bad = OrderedDict([(B, torch.tensor([[bad_val], [1.0]])), (H, torch.tensor([1.0]))])
    with pytest.raises(ValueError):
        dp_aggregate([("bad", bad, 1), ("good", good, 1)], gp, [B, H],
                     clip_norm=5.0, noise_multiplier=0.0)
