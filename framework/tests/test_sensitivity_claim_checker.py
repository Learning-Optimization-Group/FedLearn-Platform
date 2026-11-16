"""CI privacy gate: Planted Defect Suite & Sensitivity Claim Checker.

Validates that:
1. Four sound paths pass without false alarms (max observed distance <= claimed sensitivity).
2. Five known composition defects are reliably falsified by adversarial witness search:
   - Defect 1: Trimmed mean claiming nominal mean sensitivity 2C/N.
   - Defect 2: Moving center with un-noised anchor across rounds claiming 2tau/N instead of 2tau*t/N.
   - Defect 3: Reported-weight mean claiming uniform sensitivity 2C/N.
   - Defect 4: Realised-cohort sensitivity claiming 2C/N when actual participating cohort is smaller.
   - Defect 5: Coordinate median claiming 1/N sensitivity.
"""
from __future__ import annotations

import math
import pytest
import torch

from fedlearn.privacy.sensitivity import (
    GLOBAL_SENSITIVITY_REGISTRY,
    clip_tensor_rows,
    find_sensitivity_witness,
)
from fedlearn.server.robust_aggregation import (
    centered_clip,
    trimmed_mean,
)


# =============================================================================================
# 1. Sound Paths (Must NOT be falsified — Zero False Alarms)
# =============================================================================================

def test_sound_path_clipped_mean_replace_one():
    n, dim, clip_c = 20, 10, 2.0
    claim = GLOBAL_SENSITIVITY_REGISTRY.get_claim("mean", "replace_one")
    claimed_sens = claim.compute(num_clients=n, clip_norm=clip_c)
    assert math.isclose(claimed_sens, 2.0 * clip_c / n)

    def agg(x: torch.Tensor) -> torch.Tensor:
        clipped = clip_tensor_rows(x, max_norm=clip_c)
        return clipped.mean(dim=0)

    res = find_sensitivity_witness(
        aggregator_fn=agg,
        claimed_sensitivity=claimed_sens,
        num_clients=n,
        dim=dim,
        clip_norm=clip_c,
        num_random_trials=100,
    )
    assert not res.falsified, f"Sound mean falsely falsified: {res}"
    assert res.breach_factor <= 1.0001


def test_sound_path_centered_clip_fixed_centre_replace_one():
    n, dim, clip_c, tau = 20, 10, 2.0, 0.5
    claim = GLOBAL_SENSITIVITY_REGISTRY.get_claim("centered_clip", "replace_one")
    claimed_sens = claim.compute(num_clients=n, clip_norm=clip_c, tau=tau)
    assert math.isclose(claimed_sens, 2.0 * tau / n)

    fixed_centre = torch.zeros(dim)

    def agg(x: torch.Tensor) -> torch.Tensor:
        clipped = clip_tensor_rows(x, max_norm=clip_c)
        return centered_clip(clipped, centre=fixed_centre, tau=tau, iterations=1)

    res = find_sensitivity_witness(
        aggregator_fn=agg,
        claimed_sensitivity=claimed_sens,
        num_clients=n,
        dim=dim,
        clip_norm=clip_c,
        num_random_trials=100,
    )
    assert not res.falsified, f"Sound centered_clip falsely falsified: {res}"
    assert res.breach_factor <= 1.0001


def test_sound_path_trimmed_mean_certified_replace_one():
    n, dim, clip_c, beta = 20, 10, 2.0, 0.2
    claim = GLOBAL_SENSITIVITY_REGISTRY.get_claim("trimmed_mean", "replace_one")
    claimed_sens = claim.compute(num_clients=n, clip_norm=clip_c, trim_ratio=beta)
    # k = floor(0.2 * 20) = 4, kept = 12
    assert math.isclose(claimed_sens, 2.0 * clip_c / 12.0)

    def agg(x: torch.Tensor) -> torch.Tensor:
        clipped = clip_tensor_rows(x, max_norm=clip_c)
        return trimmed_mean(clipped, trim_ratio=beta)

    res = find_sensitivity_witness(
        aggregator_fn=agg,
        claimed_sensitivity=claimed_sens,
        num_clients=n,
        dim=dim,
        clip_norm=clip_c,
        num_random_trials=100,
    )
    assert not res.falsified, f"Sound trimmed mean falsely falsified: {res}"
    assert res.breach_factor <= 1.0001


def test_sound_registry_fails_closed_on_unregistered_rule():
    with pytest.raises(ValueError, match="is not registered in the DP Sensitivity Registry"):
        GLOBAL_SENSITIVITY_REGISTRY.get_claim("coordinate_median", "replace_one")

    with pytest.raises(ValueError, match="is not registered in the DP Sensitivity Registry"):
        GLOBAL_SENSITIVITY_REGISTRY.get_claim("krum", "replace_one")


# =============================================================================================
# 2. Planted Defects (Must BE Falsified by the Checker)
# =============================================================================================

def test_planted_defect_1_trimmed_mean_nominal_sensitivity():
    """Defect: Aggregator trims outliers but claims nominal mean sensitivity 2C/N."""
    n, dim, clip_c, beta = 20, 10, 2.0, 0.2
    # Flawed claim: 2C/N instead of 2C/(N - 2k)
    flawed_claim = 2.0 * clip_c / n  # 0.20, whereas true sensitivity is 2*2/12 = 0.333

    def agg(x: torch.Tensor) -> torch.Tensor:
        clipped = clip_tensor_rows(x, max_norm=clip_c)
        return trimmed_mean(clipped, trim_ratio=beta)

    res = find_sensitivity_witness(
        aggregator_fn=agg,
        claimed_sensitivity=flawed_claim,
        num_clients=n,
        dim=dim,
        clip_norm=clip_c,
        num_random_trials=50,
    )
    assert res.falsified, "Checker failed to catch flawed trimmed-mean sensitivity!"
    assert res.breach_factor >= 1.5, f"Expected breach factor >= 1.5, got {res.breach_factor:.3f}"


def test_planted_defect_2_unnoised_anchor_drift_across_rounds():
    """Defect: Centered clipping carries an un-noised history center across rounds,
    claiming per-round sensitivity 2*tau/N instead of the cumulative 2*tau*t/N."""
    n, dim, clip_c, tau, rounds = 20, 5, 2.0, 0.5, 10
    claimed_per_round = 2.0 * tau / n  # 0.05

    # Simulate multi-round execution on two neighboring histories differing in client 0
    c1, c2 = torch.zeros(dim), torch.zeros(dim)
    gen = torch.Generator().manual_seed(42)

    for t in range(1, rounds + 1):
        honest = clip_tensor_rows(torch.randn(n - 1, dim, generator=gen), max_norm=clip_c)
        # Client 0 sends tau * e0 in D1, and -tau * e0 in D2
        d1 = torch.cat([torch.tensor([[tau] + [0.0] * (dim - 1)]), honest], dim=0)
        d2 = torch.cat([torch.tensor([[-tau] + [0.0] * (dim - 1)]), honest], dim=0)

        # Update un-noised moving centers
        c1 = c1 + centered_clip(d1, centre=c1, tau=tau, iterations=1)
        c2 = c2 + centered_clip(d2, centre=c2, tau=tau, iterations=1)

    drift = float(torch.linalg.vector_norm(c1 - c2))
    breach = drift / claimed_per_round
    # After 10 rounds, drift is ~ 10 * (2*tau/N) = 10 * 0.05 = 0.50 (10x breach)
    assert drift > claimed_per_round * 5.0, f"Drift {drift:.4f} did not reveal expected multi-round breach"
    assert breach >= 8.0, f"Breach factor {breach:.2f} less than expected ~10x"


def test_planted_defect_3_reported_weight_mean():
    """Defect: Strategy weights clients by self-reported example counts under DP,
    claiming uniform 2C/N sensitivity."""
    n, dim, clip_c = 20, 5, 2.0
    flawed_claimed_sens = 2.0 * clip_c / n  # 0.20

    # Attacker reports 10x the examples of honest clients
    weights = torch.ones(n)
    weights[0] = 10.0  # Attacker weight
    norm_weights = weights / weights.sum()

    def agg(x: torch.Tensor) -> torch.Tensor:
        clipped = clip_tensor_rows(x, max_norm=clip_c)
        return (clipped * norm_weights.unsqueeze(1)).sum(dim=0)

    res = find_sensitivity_witness(
        aggregator_fn=agg,
        claimed_sensitivity=flawed_claimed_sens,
        num_clients=n,
        dim=dim,
        clip_norm=clip_c,
    )
    assert res.falsified, "Checker failed to catch reported-weight mean sensitivity breach!"
    assert res.breach_factor > 1.5, f"Breach factor was {res.breach_factor:.3f}"


def test_planted_defect_4_realised_cohort_sensitivity():
    """Defect: Noise is calibrated assuming N clients always participate,
    but only n_realised < N show up, inflating sensitivity by N / n_realised."""
    n_expected = 50
    n_realised = 20
    clip_c = 1.0

    claimed_sens = 2.0 * clip_c / n_expected  # 0.04
    actual_sens = 2.0 * clip_c / n_realised   # 0.10
    assert actual_sens > claimed_sens

    def agg_realised(x: torch.Tensor) -> torch.Tensor:
        clipped = clip_tensor_rows(x, max_norm=clip_c)
        return clipped.mean(dim=0)

    res = find_sensitivity_witness(
        aggregator_fn=agg_realised,
        claimed_sensitivity=claimed_sens,
        num_clients=n_realised,
        dim=5,
        clip_norm=clip_c,
    )
    assert res.falsified, "Checker failed to catch realised cohort mismatch!"
    assert res.breach_factor >= 2.4, f"Breach factor was {res.breach_factor:.3f}"


def test_planted_defect_5_coordinate_median_order_n_sensitivity():
    """Defect: Claiming coordinate-wise median has 1/N global sensitivity."""
    n, dim, clip_c = 21, 5, 1.0  # odd cohort
    flawed_claimed_sens = 2.0 * clip_c / n  # claims 2/21 = 0.095

    def agg_median(x: torch.Tensor) -> torch.Tensor:
        clipped = clip_tensor_rows(x, max_norm=clip_c)
        return torch.median(clipped, dim=0).values

    # In median, 1 client can jump between middle elements separated by O(C), not O(C/N)
    res = find_sensitivity_witness(
        aggregator_fn=agg_median,
        claimed_sensitivity=flawed_claimed_sens,
        num_clients=n,
        dim=dim,
        clip_norm=clip_c,
        num_random_trials=100,
    )
    assert res.falsified, "Checker failed to falsify coordinate median 1/N claim!"
    assert res.breach_factor >= 2.0, f"Expected breach >= 2x, got {res.breach_factor:.3f}"


def test_planted_defect_6_krum_order_n_sensitivity():
    """Defect: Claiming distance-based selection rule (Krum) has 2C/N global sensitivity."""
    from fedlearn.server.robust_aggregation import krum_select

    n, dim, clip_c, f = 10, 5, 2.0, 1
    # Flawed claim: 2C/N instead of order-C sensitivity
    flawed_claimed_sens = 2.0 * clip_c / n  # claims 2*2/10 = 0.40

    def agg_krum(x: torch.Tensor) -> torch.Tensor:
        clipped = clip_tensor_rows(x, max_norm=clip_c)
        idx = krum_select(clipped, num_byzantine=f)
        return clipped[idx]

    res = find_sensitivity_witness(
        aggregator_fn=agg_krum,
        claimed_sensitivity=flawed_claimed_sens,
        num_clients=n,
        dim=dim,
        clip_norm=clip_c,
        num_random_trials=50,
    )
    assert res.falsified, "Checker failed to falsify Krum 2C/N claim!"
    assert res.breach_factor >= 4.0, f"Expected order-N breach factor >= 4x, got {res.breach_factor:.3f}"


def test_sound_path_fixed_denominator_mean_add_remove():
    """Sound path under add_remove: sum over a fixed denominator D claiming C/D must not be falsified."""
    n, dim, clip_c, d_denom = 20, 10, 2.0, 50.0
    claimed_sens = clip_c / d_denom

    def agg(x: torch.Tensor) -> torch.Tensor:
        clipped = clip_tensor_rows(x, max_norm=clip_c)
        return clipped.sum(dim=0) / d_denom

    res = find_sensitivity_witness(
        aggregator_fn=agg,
        claimed_sensitivity=claimed_sens,
        num_clients=n,
        dim=dim,
        clip_norm=clip_c,
        adjacency="add_remove",
        num_random_trials=100,
    )
    assert not res.falsified, f"Sound fixed-denominator mean falsely falsified: {res}"
    assert res.breach_factor <= 1.0001


def test_planted_defect_reporting_cohort_mean_add_remove():
    """Planted defect: a mean over the reporting cohort claiming C/n under add/remove must be falsified (ratio ~1.9)."""
    n, dim, clip_c = 20, 10, 2.0
    claimed_sens = clip_c / n  # Flawed claim: C/n

    def agg(x: torch.Tensor) -> torch.Tensor:
        clipped = clip_tensor_rows(x, max_norm=clip_c)
        return clipped.mean(dim=0)

    res = find_sensitivity_witness(
        aggregator_fn=agg,
        claimed_sensitivity=claimed_sens,
        num_clients=n,
        dim=dim,
        clip_norm=clip_c,
        adjacency="add_remove",
        num_random_trials=100,
    )
    assert res.falsified, f"Reporting cohort mean was not falsified under add/remove: {res}"
    assert pytest.approx(res.breach_factor, rel=0.05) == (2.0 * n) / (n + 1)


def test_sound_path_fixed_denominator_centered_clip_add_remove():
    """Sound path under add_remove: centered clip over fixed denominator D claiming min(C, tau)/D."""
    n, dim, clip_c, tau, d_denom = 20, 10, 2.0, 0.5, 50.0
    claim = GLOBAL_SENSITIVITY_REGISTRY.get_claim("centered_clip", "add_remove")
    claimed_sens = claim.compute(num_clients=int(d_denom), clip_norm=clip_c, tau=tau)
    assert math.isclose(claimed_sens, tau / d_denom)

    def agg(x: torch.Tensor) -> torch.Tensor:
        clipped = clip_tensor_rows(x, max_norm=clip_c)
        norms = torch.linalg.vector_norm(clipped, dim=1, keepdim=True).clamp_min(1e-12)
        shrink = torch.clamp(tau / norms, max=1.0)
        cclipped = clipped * shrink
        return cclipped.sum(dim=0) / d_denom

    res = find_sensitivity_witness(
        aggregator_fn=agg,
        claimed_sensitivity=claimed_sens,
        num_clients=n,
        dim=dim,
        clip_norm=clip_c,
        adjacency="add_remove",
        num_random_trials=100,
    )
    assert not res.falsified, f"Sound fixed-denominator centered clip falsely falsified: {res}"
    assert res.breach_factor <= 1.0001


def test_planted_defect_fixed_denominator_trimmed_mean_add_remove():
    """Planted defect: fixed-denominator trimmed mean under add/remove claiming C/(D - 2k) is falsified (ratio >= 1.3)."""
    n, clip_c, beta, d_denom = 20, 2.0, 0.1, 50.0
    k = int(beta * d_denom)
    d_trim = d_denom - 2 * k
    claimed_sens = clip_c / d_trim

    def agg(x: torch.Tensor) -> torch.Tensor:
        clipped = clip_tensor_rows(x, max_norm=clip_c)
        n_cur = clipped.shape[0]
        if n_cur <= 2 * k:
            return torch.zeros(clipped.shape[1], dtype=clipped.dtype)
        sX, _ = torch.sort(clipped, dim=0)
        retained = sX[k : n_cur - k]
        return retained.sum(dim=0) / d_trim

    dB = 20
    dim = 1 + dB
    a = clip_c / math.sqrt(dB)
    c1 = math.sqrt(0.8) * clip_c
    base = torch.zeros(n, dim, dtype=torch.float64)
    for t in range(k):
        base[t, 0] = c1
        for j in range(dB):
            if j % k == t:
                base[t, 1 + j] = -a
    base[k:, 1:] = -a
    new = torch.zeros(1, dim, dtype=torch.float64)
    new[0, 0] = c1 - 1e-3

    out_base = agg(base)
    out_new = agg(torch.cat([base, new], dim=0))
    dist = float(torch.linalg.vector_norm(out_new - out_base))
    breach_factor = dist / claimed_sens
    assert breach_factor >= 1.3, f"Expected breach factor >= 1.3x, got {breach_factor:.4f}"


def test_unregistered_add_remove_claim_for_trimmed_mean():
    """Trimmed mean has no registered add/remove claim; centered clipping does."""
    assert not GLOBAL_SENSITIVITY_REGISTRY.is_registered("trimmed_mean", "add_remove")
    with pytest.raises(ValueError, match="not registered"):
        GLOBAL_SENSITIVITY_REGISTRY.get_claim("trimmed_mean", "add_remove")
    assert GLOBAL_SENSITIVITY_REGISTRY.is_registered("centered_clip", "add_remove")
    claim_cc = GLOBAL_SENSITIVITY_REGISTRY.get_claim("centered_clip", "add_remove")
    assert claim_cc.adjacency == "add_remove"


def test_generic_search_falsifies_f2_trimmed_mean_add_remove_with_no_hint():
    """F3 acceptance test: With no hint from the test, generic search falsifies the F2 claim."""
    n, dim, clip_c, beta, d_denom = 20, 21, 2.0, 0.1, 50.0
    k = int(beta * d_denom)
    d_trim = d_denom - 2 * k
    claimed_sens = clip_c / d_trim

    def agg(x: torch.Tensor) -> torch.Tensor:
        clipped = clip_tensor_rows(x, max_norm=clip_c)
        n_cur = clipped.shape[0]
        if n_cur <= 2 * k:
            return torch.zeros(clipped.shape[1], dtype=clipped.dtype)
        sX, _ = torch.sort(clipped, dim=0)
        retained = sX[k : n_cur - k]
        return retained.sum(dim=0) / d_trim

    res = find_sensitivity_witness(
        aggregator_fn=agg,
        claimed_sensitivity=claimed_sens,
        num_clients=n,
        dim=dim,
        clip_norm=clip_c,
        adjacency="add_remove",
        num_random_trials=50,
    )
    assert res.falsified, f"Checker failed to falsify F2 claim with no hint: {res}"
    assert res.breach_factor >= 1.3, f"Expected breach factor >= 1.3x, got {res.breach_factor:.4f}"


@pytest.mark.parametrize("dim", [21, 64])
@pytest.mark.parametrize("adjacency", ["replace_one", "add_remove"])
def test_generic_search_zero_false_alarms_sound_paths(dim, adjacency):
    """F3 acceptance test: Zero false alarms on fixed-denominator mean and centered clipping at dims 21 and 64."""
    from fedlearn.server.robust_aggregation import centered_clip
    n, clip_c, d_denom = 20, 2.0, 50.0

    # Fixed-denominator mean
    claimed_m = (clip_c / d_denom) if adjacency == "add_remove" else (2.0 * clip_c / n)
    def agg_mean(x: torch.Tensor) -> torch.Tensor:
        clipped = clip_tensor_rows(x, max_norm=clip_c)
        return (clipped.sum(dim=0) / d_denom) if adjacency == "add_remove" else clipped.mean(dim=0)

    res_m = find_sensitivity_witness(
        aggregator_fn=agg_mean,
        claimed_sensitivity=claimed_m,
        num_clients=n,
        dim=dim,
        clip_norm=clip_c,
        adjacency=adjacency,
        num_random_trials=20,
    )
    assert not res_m.falsified, f"False alarm on mean (dim={dim}, adj={adjacency}): {res_m}"
    assert res_m.breach_factor <= 1.0001

    # Centered clipping
    tau = 0.5
    claimed_cc = (min(clip_c, tau) / d_denom) if adjacency == "add_remove" else (2.0 * min(clip_c, tau) / n)
    def agg_cc(x: torch.Tensor) -> torch.Tensor:
        clipped = clip_tensor_rows(x, max_norm=clip_c)
        flat = clipped.reshape(clipped.shape[0], -1)
        if adjacency == "add_remove":
            norms = torch.linalg.vector_norm(flat, dim=1, keepdim=True).clamp_min(1e-12)
            shrink = torch.clamp(tau / norms, max=1.0)
            return (flat * shrink).sum(dim=0) / d_denom
        else:
            return centered_clip(flat, torch.zeros_like(flat[0]), tau=tau, iterations=1)

    res_cc = find_sensitivity_witness(
        aggregator_fn=agg_cc,
        claimed_sensitivity=claimed_cc,
        num_clients=n,
        dim=dim,
        clip_norm=clip_c,
        adjacency=adjacency,
        num_random_trials=20,
    )
    assert not res_cc.falsified, f"False alarm on centered_clip (dim={dim}, adj={adjacency}): {res_cc}"
    assert res_cc.breach_factor <= 1.0001


def test_iterations_run_counts_what_ran():
    """F3 acceptance test: iterations_run counts the actual number of candidate pairs evaluated."""
    call_count = [0]

    def counted_agg(x: torch.Tensor) -> torch.Tensor:
        call_count[0] += 1
        return x.mean(dim=0)

    res = find_sensitivity_witness(
        aggregator_fn=counted_agg,
        claimed_sensitivity=0.2,
        num_clients=10,
        dim=5,
        clip_norm=1.0,
        num_random_trials=15,
        adjacency="replace_one",
    )
    # Each pair evaluation calls aggregator_fn twice
    assert res.iterations_run > 0
    assert call_count[0] == res.iterations_run * 2



