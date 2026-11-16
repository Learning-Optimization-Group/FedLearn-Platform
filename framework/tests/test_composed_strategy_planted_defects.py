"""Planted-defect tests targeting ComposedStrategy itself.

For each registered sensitivity claim in GLOBAL_SENSITIVITY_REGISTRY, two tests:
1. A sound configuration passes the sensitivity checker (no false alarm).
2. A planted defect (wrong sensitivity / wrong denominator / reporting-cohort denominator)
   is detected.

Each test builds ComposedStrategy through its constructor, uses its preview or
_pipeline method as the aggregation target, and labels clearly which registered
claim it drives.

Also tests:
- Failure paths route through the release ledger under DP with strict_dp_ledger.
- Timings censorship under DP with strict_dp_ledger.
"""
from collections import OrderedDict
import math
from typing import List, Tuple

import pytest
import torch

from fedlearn.privacy.sensitivity import (
    GLOBAL_SENSITIVITY_REGISTRY,
    SensitivityClaim,
    find_sensitivity_witness,
)
from fedlearn.server.composed_strategy import ComposedStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KEYS = ("w", "b")


def _global():
    return OrderedDict(w=torch.tensor([[0.5, -0.5, 1.0]]), b=torch.tensor([0.1]))


def _global_dim(dim: int):
    """A single-key global of the given dimension."""
    return OrderedDict(w=torch.zeros(dim))


def _results_from_deltas(glob, deltas: torch.Tensor) -> List[Tuple[str, "OrderedDict", int]]:
    """Build (cid, params, n_examples) results from an (N, dim) delta tensor for a single-key model."""
    key = list(glob.keys())[0]
    out = []
    for i in range(deltas.shape[0]):
        p = OrderedDict()
        p[key] = glob[key] + deltas[i]
        out.append((f"c{i}", p, 1))
    return out


def _strategy_aggregator(strategy: ComposedStrategy, glob):
    """Return an aggregator function that drives ComposedStrategy.preview (noise-free)."""
    key = list(glob.keys())[0]

    def agg(x: torch.Tensor) -> torch.Tensor:
        results = _results_from_deltas(glob, x)
        out = strategy.preview(results, server_round=0)
        if out is None:
            return glob[key].clone()
        return out[key] - glob[key]  # return the delta, not the absolute params

    return agg


def _results(seed, n=10, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    out = []
    glob = _global()
    for i in range(n):
        p = OrderedDict((k, v + scale * torch.randn(v.shape, generator=g)) for k, v in glob.items())
        out.append((f"c{i}", p, 10 + i))
    return out


def _copy(results):
    return [(c, OrderedDict((k, v.clone()) for k, v in p.items()), n) for c, p, n in results]


# ===========================================================================
# Claim 1: ("mean", "replace_one") — sensitivity 2C/N
# Target: ComposedStrategy with rule="mean", adjacency="replace_one"
# ===========================================================================

class TestMeanReplaceOne:
    """Drives: ('mean', 'replace_one') registered claim."""

    def test_sound_configuration_passes(self):
        """A correct mean under replace_one at 2C/N is not falsified."""
        n, dim, C = 20, 10, 2.0
        glob = _global_dim(dim)
        s = ComposedStrategy(glob, rule="mean", clip_norm=C, central_noise_multiplier=0.0)
        claim = GLOBAL_SENSITIVITY_REGISTRY.get_claim("mean", "replace_one")
        claimed_sens = claim.compute(num_clients=n, clip_norm=C)
        assert math.isclose(claimed_sens, 2 * C / n)

        agg = _strategy_aggregator(s, glob)
        res = find_sensitivity_witness(
            aggregator_fn=agg, claimed_sensitivity=claimed_sens,
            num_clients=n, dim=dim, clip_norm=C, num_random_trials=50,
        )
        assert not res.falsified, f"Sound mean replace_one falsely falsified: {res}"
        assert res.breach_factor <= 1.0001

    def test_planted_defect_wrong_sensitivity_detected(self):
        """Defect: noise sized to C/N (half the correct 2C/N) is detected."""
        n, dim, C = 20, 10, 2.0
        glob = _global_dim(dim)
        s = ComposedStrategy(glob, rule="mean", clip_norm=C, central_noise_multiplier=0.0)
        # Planted defect: claim C/N instead of 2C/N
        wrong_sens = C / n

        agg = _strategy_aggregator(s, glob)
        res = find_sensitivity_witness(
            aggregator_fn=agg, claimed_sensitivity=wrong_sens,
            num_clients=n, dim=dim, clip_norm=C, num_random_trials=50,
        )
        assert res.falsified, f"Planted defect (half sensitivity) not detected: {res}"
        assert res.breach_factor >= 1.9, f"Expected breach >= 1.9, got {res.breach_factor:.3f}"

    def test_planted_defect_via_monkeypatched_registry_detected(self, monkeypatch):
        """Defect: monkeypatch the claim registry for ('mean', 'replace_one') to return C/N.
        Target: ComposedStrategy._sensitivity reads the monkeypatched claim."""
        n, dim, C = 20, 10, 2.0
        glob = _global_dim(dim)
        s = ComposedStrategy(
            glob, rule="mean", clip_norm=C, central_noise_multiplier=1.0,
            adjacency="replace_one", public_cohort=n, noise_seed=1,
        )
        # Monkeypatch the registry claim to under-noise (return C/N instead of 2C/N)
        def defective_calc(num_clients: int, clip_norm: float, **kwargs) -> float:
            return float(clip_norm) / float(num_clients)

        defective_claim = SensitivityClaim(
            rule_name="mean", adjacency="replace_one", formula_latex=r"\frac{C}{N}",
            centre_provenance="none", cohort_semantics="public_constant", calculator=defective_calc,
        )
        monkeypatch.setattr(GLOBAL_SENSITIVITY_REGISTRY, "_registry",
                            {**GLOBAL_SENSITIVITY_REGISTRY._registry, ("mean", "replace_one"): defective_claim})

        wrong_sens = s._sensitivity(n)
        assert math.isclose(wrong_sens, C / n)

        agg = _strategy_aggregator(s, glob)
        res = find_sensitivity_witness(
            aggregator_fn=agg, claimed_sensitivity=wrong_sens,
            num_clients=n, dim=dim, clip_norm=C, num_random_trials=50,
        )
        assert res.falsified, f"Planted defect (monkeypatched registry C/N) not detected: {res}"
        assert res.breach_factor >= 1.9, f"Expected breach >= 1.9, got {res.breach_factor:.3f}"


# ===========================================================================
# Claim 2: ("mean", "add_remove") — sensitivity C/D (fixed denominator)
# Target: ComposedStrategy with rule="mean", fixed_denominator, adjacency="add_remove"
# ===========================================================================

class TestMeanAddRemove:
    """Drives: ('mean', 'add_remove') registered claim."""

    def test_sound_configuration_passes(self):
        """A correct fixed-denominator mean under add_remove at C/D is not falsified."""
        n, dim, C, D = 20, 10, 2.0, 50.0
        glob = _global_dim(dim)
        s = ComposedStrategy(
            glob, rule="mean", clip_norm=C, central_noise_multiplier=2.0,
            adjacency="add_remove", fixed_denominator=D, noise_seed=1,
        )
        claim = GLOBAL_SENSITIVITY_REGISTRY.get_claim("mean", "add_remove")
        claimed_sens = claim.compute(num_clients=D, clip_norm=C)
        assert math.isclose(claimed_sens, C / D)

        agg = _strategy_aggregator(s, glob)
        res = find_sensitivity_witness(
            aggregator_fn=agg, claimed_sensitivity=claimed_sens,
            num_clients=n, dim=dim, clip_norm=C,
            adjacency="add_remove", num_random_trials=50,
        )
        assert not res.falsified, f"Sound mean add_remove falsely falsified: {res}"
        assert res.breach_factor <= 1.0001

    def test_planted_defect_reporting_cohort_mean_detected(self):
        """Defect: ComposedStrategy built without fixed_denominator divides by the
        reporting cohort n, but claimed sensitivity is sized to fixed denominator C/D.
        Target: ComposedStrategy constructor built without fixed_denominator."""
        n, dim, C, D = 20, 10, 2.0, 50.0
        glob = _global_dim(dim)
        # Built through constructor WITHOUT fixed_denominator:
        s_defective = ComposedStrategy(
            glob, rule="mean", clip_norm=C, central_noise_multiplier=0.0,
        )
        # Claim sensitivity sized to fixed denominator C/D = 0.04
        claimed_sens = C / D

        agg = _strategy_aggregator(s_defective, glob)
        res = find_sensitivity_witness(
            aggregator_fn=agg, claimed_sensitivity=claimed_sens,
            num_clients=n, dim=dim, clip_norm=C,
            adjacency="add_remove", num_random_trials=50,
        )
        assert res.falsified, f"Reporting-cohort mean defect not detected: {res}"
        assert res.breach_factor >= 1.8, f"Expected breach >= 1.8, got {res.breach_factor:.3f}"

    def test_planted_defect_under_noised_denominator_detected(self):
        """Defect: ComposedStrategy built with fixed_denominator D, but noise is sized
        to an inflated denominator 2*D (claimed sensitivity C/(2*D), halving noise)."""
        n, dim, C, D = 20, 10, 2.0, 50.0
        glob = _global_dim(dim)
        s = ComposedStrategy(
            glob, rule="mean", clip_norm=C, central_noise_multiplier=2.0,
            adjacency="add_remove", fixed_denominator=D, noise_seed=1,
        )
        # Defect: claim sensitivity sized with denominator 2*D
        wrong_sens = C / (2.0 * D)

        agg = _strategy_aggregator(s, glob)
        res = find_sensitivity_witness(
            aggregator_fn=agg, claimed_sensitivity=wrong_sens,
            num_clients=n, dim=dim, clip_norm=C,
            adjacency="add_remove", num_random_trials=50,
        )
        assert res.falsified, f"Inflated denominator defect not detected: {res}"
        assert res.breach_factor >= 1.9, f"Expected breach >= 1.9, got {res.breach_factor:.3f}"


# ===========================================================================
# Claim 3: ("trimmed_mean", "replace_one") — sensitivity 2C/(N - 2k)
# Target: ComposedStrategy with rule="trimmed_mean", adjacency="replace_one"
# ===========================================================================

class TestTrimmedMeanReplaceOne:
    """Drives: ('trimmed_mean', 'replace_one') registered claim."""

    def test_sound_configuration_passes(self):
        """A correct trimmed mean under replace_one at 2C/(N-2k) is not falsified."""
        n, dim, C, beta = 20, 10, 2.0, 0.2
        glob = _global_dim(dim)
        s = ComposedStrategy(
            glob, rule="trimmed_mean", clip_norm=C,
            rule_params={"trim_ratio": beta}, central_noise_multiplier=0.0,
        )
        claim = GLOBAL_SENSITIVITY_REGISTRY.get_claim("trimmed_mean", "replace_one")
        k = int(beta * n)
        kept = n - 2 * k
        claimed_sens = claim.compute(num_clients=n, clip_norm=C, trim_ratio=beta)
        assert math.isclose(claimed_sens, 2 * C / kept)

        agg = _strategy_aggregator(s, glob)
        res = find_sensitivity_witness(
            aggregator_fn=agg, claimed_sensitivity=claimed_sens,
            num_clients=n, dim=dim, clip_norm=C, num_random_trials=50,
        )
        assert not res.falsified, f"Sound trimmed_mean replace_one falsely falsified: {res}"
        assert res.breach_factor <= 1.0001

    def test_planted_defect_mean_sensitivity_detected(self):
        """Defect: noise sized to 2C/N (nominal mean sensitivity) for trimmed mean is detected."""
        n, dim, C, beta = 20, 10, 2.0, 0.2
        glob = _global_dim(dim)
        s = ComposedStrategy(
            glob, rule="trimmed_mean", clip_norm=C,
            rule_params={"trim_ratio": beta}, central_noise_multiplier=0.0,
        )
        # Planted defect: claim 2C/N instead of 2C/(N-2k)
        wrong_sens = 2 * C / n  # 0.2, correct is 2*2/12 ≈ 0.333

        agg = _strategy_aggregator(s, glob)
        res = find_sensitivity_witness(
            aggregator_fn=agg, claimed_sensitivity=wrong_sens,
            num_clients=n, dim=dim, clip_norm=C, num_random_trials=50,
        )
        assert res.falsified, f"Planted defect (mean sensitivity for trimmed mean) not detected: {res}"
        assert res.breach_factor >= 1.5, f"Expected breach >= 1.5, got {res.breach_factor:.3f}"

    def test_planted_defect_via_monkeypatched_registry_detected(self, monkeypatch):
        """Defect: monkeypatch registry claim ('trimmed_mean', 'replace_one') to ignore trim_ratio.
        Target: ComposedStrategy._sensitivity reads the monkeypatched claim."""
        n, dim, C, beta = 20, 10, 2.0, 0.2
        glob = _global_dim(dim)
        s = ComposedStrategy(
            glob, rule="trimmed_mean", clip_norm=C,
            rule_params={"trim_ratio": beta}, central_noise_multiplier=1.0,
            adjacency="replace_one", public_cohort=n, noise_seed=1,
        )
        # Monkeypatch registry claim to omit the denominator reduction
        def defective_calc(num_clients: int, clip_norm: float, **kwargs) -> float:
            return 2.0 * float(clip_norm) / float(num_clients)

        defective_claim = SensitivityClaim(
            rule_name="trimmed_mean", adjacency="replace_one",
            formula_latex=r"\frac{2C}{N}", centre_provenance="none",
            cohort_semantics="public_constant", calculator=defective_calc,
        )
        monkeypatch.setattr(GLOBAL_SENSITIVITY_REGISTRY, "_registry",
                            {**GLOBAL_SENSITIVITY_REGISTRY._registry, ("trimmed_mean", "replace_one"): defective_claim})

        wrong_sens = s._sensitivity(n)
        assert math.isclose(wrong_sens, 2.0 * C / n)

        agg = _strategy_aggregator(s, glob)
        res = find_sensitivity_witness(
            aggregator_fn=agg, claimed_sensitivity=wrong_sens,
            num_clients=n, dim=dim, clip_norm=C, num_random_trials=50,
        )
        assert res.falsified, f"Planted defect (monkeypatched registry for trimmed_mean) not detected: {res}"
        assert res.breach_factor >= 1.5, f"Expected breach >= 1.5, got {res.breach_factor:.3f}"


# ===========================================================================
# Claim 4: ("centered_clip", "replace_one") — sensitivity 2*min(C, tau)/N
# Target: ComposedStrategy with rule="centered_clip", adjacency="replace_one"
# ===========================================================================

class TestCenteredClipReplaceOne:
    """Drives: ('centered_clip', 'replace_one') registered claim."""

    def test_sound_configuration_passes(self):
        """A correct centered_clip under replace_one at 2*min(C,tau)/N is not falsified."""
        n, dim, C, tau = 20, 10, 2.0, 0.5
        glob = _global_dim(dim)
        s = ComposedStrategy(
            glob, rule="centered_clip", clip_norm=C,
            rule_params={"tau": tau, "iterations": 1}, central_noise_multiplier=0.0,
        )
        claim = GLOBAL_SENSITIVITY_REGISTRY.get_claim("centered_clip", "replace_one")
        claimed_sens = claim.compute(num_clients=n, clip_norm=C, tau=tau)
        assert math.isclose(claimed_sens, 2 * min(C, tau) / n)

        agg = _strategy_aggregator(s, glob)
        res = find_sensitivity_witness(
            aggregator_fn=agg, claimed_sensitivity=claimed_sens,
            num_clients=n, dim=dim, clip_norm=C, num_random_trials=50,
        )
        assert not res.falsified, f"Sound centered_clip replace_one falsely falsified: {res}"
        assert res.breach_factor <= 1.0001

    def test_planted_defect_under_noised_tau_detected(self):
        """Defect: noise sized to 2*tau_small/N with tau_small=0.01 (actual tau=0.5).
        Target: ComposedStrategy with tau=0.5."""
        n, dim, C, tau = 20, 10, 2.0, 0.5
        glob = _global_dim(dim)
        s = ComposedStrategy(
            glob, rule="centered_clip", clip_norm=C,
            rule_params={"tau": tau, "iterations": 1}, central_noise_multiplier=0.0,
        )
        # Planted defect: claim sensitivity for tau=0.01 instead of tau=0.5
        wrong_sens = 2 * 0.01 / n

        agg = _strategy_aggregator(s, glob)
        res = find_sensitivity_witness(
            aggregator_fn=agg, claimed_sensitivity=wrong_sens,
            num_clients=n, dim=dim, clip_norm=C, num_random_trials=50,
        )
        assert res.falsified, f"Under-noised centered_clip defect not detected: {res}"
        assert res.breach_factor >= 10.0, f"Expected breach >= 10x, got {res.breach_factor:.3f}"

    def test_planted_defect_half_sensitivity_detected(self):
        """Defect: noise sized to min(C,tau)/N (missing factor of 2 under replace_one).
        Target: ComposedStrategy with rule="centered_clip"."""
        n, dim, C, tau = 20, 10, 2.0, 0.5
        glob = _global_dim(dim)
        s = ComposedStrategy(
            glob, rule="centered_clip", clip_norm=C,
            rule_params={"tau": tau, "iterations": 1}, central_noise_multiplier=0.0,
        )
        # Planted defect: missing replace_one factor of 2
        wrong_sens = min(C, tau) / n

        agg = _strategy_aggregator(s, glob)
        res = find_sensitivity_witness(
            aggregator_fn=agg, claimed_sensitivity=wrong_sens,
            num_clients=n, dim=dim, clip_norm=C, num_random_trials=50,
        )
        assert res.falsified, f"Half sensitivity defect not detected: {res}"
        assert res.breach_factor >= 1.9, f"Expected breach >= 1.9, got {res.breach_factor:.3f}"


# ===========================================================================
# Claim 5: ("centered_clip", "add_remove") — sensitivity min(C, tau)/D
# Target: ComposedStrategy with rule="centered_clip", fixed_denominator, adjacency="add_remove"
# ===========================================================================

class TestCenteredClipAddRemove:
    """Drives: ('centered_clip', 'add_remove') registered claim."""

    def test_sound_configuration_passes(self):
        """A correct fixed-denominator centered_clip under add_remove at min(C,tau)/D is not falsified."""
        n, dim, C, tau, D = 20, 10, 2.0, 0.5, 50.0
        glob = _global_dim(dim)
        s = ComposedStrategy(
            glob, rule="centered_clip", clip_norm=C,
            rule_params={"tau": tau, "iterations": 1},
            central_noise_multiplier=2.0, adjacency="add_remove",
            fixed_denominator=D, noise_seed=1,
        )
        claim = GLOBAL_SENSITIVITY_REGISTRY.get_claim("centered_clip", "add_remove")
        claimed_sens = claim.compute(num_clients=D, clip_norm=C, tau=tau)
        assert math.isclose(claimed_sens, min(C, tau) / D)

        agg = _strategy_aggregator(s, glob)
        res = find_sensitivity_witness(
            aggregator_fn=agg, claimed_sensitivity=claimed_sens,
            num_clients=n, dim=dim, clip_norm=C,
            adjacency="add_remove", num_random_trials=50,
        )
        assert not res.falsified, f"Sound centered_clip add_remove falsely falsified: {res}"
        assert res.breach_factor <= 1.0001

    def test_planted_defect_reporting_cohort_denominator_detected(self):
        """Defect: ComposedStrategy built without fixed_denominator divides by reporting
        cohort n instead of D, while claimed sensitivity is sized to D.
        Target: ComposedStrategy constructor without fixed_denominator."""
        n, dim, C, tau, D = 20, 10, 2.0, 0.5, 50.0
        glob = _global_dim(dim)
        # Built through constructor WITHOUT fixed_denominator:
        s_defective = ComposedStrategy(
            glob, rule="centered_clip", clip_norm=C,
            rule_params={"tau": tau, "iterations": 1},
            central_noise_multiplier=0.0,
        )
        # Claimed sensitivity is min(C, tau) / D = 0.5 / 50 = 0.01
        claimed_sens = min(C, tau) / D

        agg = _strategy_aggregator(s_defective, glob)
        res = find_sensitivity_witness(
            aggregator_fn=agg, claimed_sensitivity=claimed_sens,
            num_clients=n, dim=dim, clip_norm=C,
            adjacency="add_remove", num_random_trials=50,
        )
        assert res.falsified, f"Reporting-cohort denominator defect not detected: {res}"
        assert res.breach_factor >= 1.8, f"Expected breach >= 1.8, got {res.breach_factor:.3f}"

    def test_planted_defect_under_noised_denominator_detected(self):
        """Defect: ComposedStrategy built with fixed_denominator D, but noise sized
        to an inflated denominator 2*D (claimed sensitivity min(C,tau)/(2*D))."""
        n, dim, C, tau, D = 20, 10, 2.0, 0.5, 50.0
        glob = _global_dim(dim)
        s = ComposedStrategy(
            glob, rule="centered_clip", clip_norm=C,
            rule_params={"tau": tau, "iterations": 1},
            central_noise_multiplier=2.0, adjacency="add_remove",
            fixed_denominator=D, noise_seed=1,
        )
        # Defect: claim sensitivity sized with denominator 2*D
        wrong_sens = min(C, tau) / (2.0 * D)

        agg = _strategy_aggregator(s, glob)
        res = find_sensitivity_witness(
            aggregator_fn=agg, claimed_sensitivity=wrong_sens,
            num_clients=n, dim=dim, clip_norm=C,
            adjacency="add_remove", num_random_trials=50,
        )
        assert res.falsified, f"Inflated denominator defect not detected: {res}"
        assert res.breach_factor >= 1.9, f"Expected breach >= 1.9, got {res.breach_factor:.3f}"


# ===========================================================================
# Failure path ledger tests
# ===========================================================================

class TestFailurePathLedger:
    """Failure paths go through the release ledger under DP with strict_dp_ledger."""

    def test_all_invalid_failure_path_censors_stats_under_dp(self):
        """When every client update is invalid or filtered, the failure path must still
        censor aggregated/dropped/filtered under DP with strict_dp_ledger, and the
        message must carry no client count."""
        glob = _global()
        s = ComposedStrategy(
            glob, rule="mean", clip_norm=0.4,
            central_noise_multiplier=1.0, noise_seed=1,
            strict_dp_ledger=True,
        )
        # Make all updates invalid (NaN)
        res = _results(1, n=5)
        for _, p, _ in res:
            p["w"][0, 0] = float("nan")

        out = s.aggregate_fit(0, res)
        assert out is None
        st = s.last_round_stats
        assert st["failed"] is True
        # Under strict_dp_ledger with DP, the failure message must not carry counts
        assert "5" not in (st["message"] or ""), f"Message leaks count: {st['message']}"
        assert "0" not in (st["message"] or "") or "round 0" in (st["message"] or ""), \
            f"Message leaks zero count: {st['message']}"
        # aggregated, dropped, filtered, timings must be censored
        assert st["aggregated"] is None
        assert st["dropped"] == []
        assert st["filtered"] == []
        assert st["timings"] is None

    def test_public_cohort_mismatch_failure_path_censors_under_dp(self):
        """When the reporting cohort doesn't match public_cohort, the failure message
        must not contain the actual client count under strict_dp_ledger."""
        glob = _global()
        s = ComposedStrategy(
            glob, rule="mean", clip_norm=0.5,
            central_noise_multiplier=1.0, noise_seed=12,
            adjacency="replace_one", public_cohort=100,
            strict_dp_ledger=True,
        )
        res = _results(42, n=8)
        out = s.aggregate_fit(0, res)
        assert out is None
        st = s.last_round_stats
        assert st["failed"] is True
        # The message must not leak how many clients reported
        assert "8" not in (st["message"] or ""), f"Message leaks count '8': {st['message']}"
        assert "public_cohort" in (st["message"] or "")
        # aggregated, dropped, filtered, timings must be censored
        assert st["aggregated"] is None
        assert st["dropped"] == []
        assert st["filtered"] == []
        assert st["timings"] is None

    def test_empty_cohort_without_fixed_denominator_failure_path_censors_under_dp(self):
        """When cohort is empty and fixed_denominator is None under DP, failure stats
        must censor aggregated, dropped, filtered, timings."""
        glob = _global()
        s = ComposedStrategy(
            glob, rule="mean", clip_norm=0.5,
            central_noise_multiplier=1.0, noise_seed=1,
            strict_dp_ledger=True,
        )
        out = s.aggregate_fit(0, [])
        assert out is None
        st = s.last_round_stats
        assert st["failed"] is True
        assert st["aggregated"] is None
        assert st["dropped"] == []
        assert st["filtered"] == []
        assert st["timings"] is None

    def test_rule_refusal_failure_path_censors_under_dp(self):
        """When a rule precondition fails under DP with strict_dp_ledger, failure stats
        must censor aggregated, dropped, filtered, timings."""
        glob = _global()
        s = ComposedStrategy(
            glob, rule="mean", clip_norm=0.5,
            central_noise_multiplier=1.0, noise_seed=1,
            strict_dp_ledger=True,
        )
        res = _results(1, n=4)
        # Induce a rule failure in _reduce_deltas
        def failing_reduce(*args, **kwargs):
            raise ValueError("simulated rule refusal")

        s._reduce_deltas = failing_reduce
        out = s.aggregate_fit(0, res)
        assert out is None
        st = s.last_round_stats
        assert st["failed"] is True
        assert st["aggregated"] is None
        assert st["dropped"] == []
        assert st["filtered"] == []
        assert st["timings"] is None

    def test_failure_path_non_dp_preserves_existing_behaviour(self):
        """Without DP (z == 0), failure paths should keep existing behaviour unchanged."""
        glob = _global()
        s = ComposedStrategy(
            glob, rule="mean", clip_norm=0.4,
            central_noise_multiplier=0.0,
        )
        res = _results(1, n=5)
        for _, p, _ in res:
            p["w"][0, 0] = float("nan")

        out = s.aggregate_fit(0, res)
        assert out is None
        st = s.last_round_stats
        assert st["failed"] is True
        # Non-DP: aggregated, dropped should contain the actual values
        assert st["aggregated"] == 0
        assert len(st["dropped"]) == 5
        assert set(st["timings"]) == {"clip", "rule", "noise", "optimizer"}

    def test_failure_path_relaxed_ledger_preserves_counts_under_dp(self):
        """With strict_dp_ledger=False, failure paths keep actual counts even under DP."""
        glob = _global()
        with pytest.warns(UserWarning, match="strict_dp_ledger is False"):
            s = ComposedStrategy(
                glob, rule="mean", clip_norm=0.5,
                central_noise_multiplier=1.0, noise_seed=12,
                adjacency="replace_one", public_cohort=100,
                strict_dp_ledger=False,
            )
        res = _results(42, n=8)
        out = s.aggregate_fit(0, res)
        assert out is None
        st = s.last_round_stats
        assert st["failed"] is True
        assert st["aggregated"] == 8
        assert "8" in (st["message"] or "")
        assert set(st["timings"]) == {"clip", "rule", "noise", "optimizer"}


# ===========================================================================
# Timings censorship under DP
# ===========================================================================

class TestTimingsCensorship:
    """Under DP with strict_dp_ledger, stats['timings'] must not be released."""

    def test_timings_not_released_under_dp(self):
        """Under central DP with strict_dp_ledger, timings must be censored (None)."""
        glob = _global()
        s = ComposedStrategy(
            glob, rule="trimmed_mean", clip_norm=0.1,
            rule_params={"trim_ratio": 0.2},
            central_noise_multiplier=2.0, noise_seed=1,
            strict_dp_ledger=True,
        )
        s.aggregate_fit(0, _results(5))
        st = s.last_round_stats
        assert st.get("timings") is None, \
            f"Under DP with strict_dp_ledger, timings must be None, got: {st.get('timings')}"

    def test_timings_preserved_without_dp(self):
        """Without DP (z == 0), timings must retain the full per-stage breakdown."""
        glob = _global()
        s = ComposedStrategy(
            glob, rule="trimmed_mean", clip_norm=0.1,
            rule_params={"trim_ratio": 0.2},
            central_noise_multiplier=0.0,
        )
        s.aggregate_fit(0, _results(5))
        st = s.last_round_stats
        assert set(st["timings"]) == {"clip", "rule", "noise", "optimizer"}

    def test_timings_preserved_under_dp_with_relaxed_ledger(self):
        """With strict_dp_ledger=False, timings retain per-stage breakdown even under DP."""
        glob = _global()
        with pytest.warns(UserWarning, match="strict_dp_ledger is False"):
            s = ComposedStrategy(
                glob, rule="trimmed_mean", clip_norm=0.1,
                rule_params={"trim_ratio": 0.2},
                central_noise_multiplier=2.0, noise_seed=1,
                strict_dp_ledger=False,
            )
        s.aggregate_fit(0, _results(5))
        st = s.last_round_stats
        assert set(st["timings"]) == {"clip", "rule", "noise", "optimizer"}

    def test_timings_in_empty_cohort_dp_path(self):
        """Empty cohort under DP with strict_dp_ledger: timings must be censored."""
        glob = _global()
        s = ComposedStrategy(
            glob, rule="mean", clip_norm=1.0,
            central_noise_multiplier=1.0,
            adjacency="add_remove", fixed_denominator=50.0,
            noise_seed=123, strict_dp_ledger=True,
        )
        out = s.aggregate_fit(1, [])
        assert out is not None
        st = s.last_round_stats
        assert st.get("timings") is None, \
            f"Under DP, empty cohort timings must be None, got: {st.get('timings')}"


class TestLedgerHardening:
    """Security-review follow-ups: a refused rule's message and a zero noise scale under DP."""

    def test_rule_refusal_message_carries_no_count_under_dp(self):
        """Robust-rule preconditions interpolate the reporting count into their ValueError; under DP
        with the strict ledger that text must not reach the released message."""
        s = ComposedStrategy(
            _global(), rule="mean", clip_norm=0.5,
            central_noise_multiplier=1.0, noise_seed=1, strict_dp_ledger=True,
        )

        def failing_reduce(*args, **kwargs):
            raise ValueError("Krum requires n >= 2f + 3 for its guarantee; got n=7, f=3 (needs 9).")

        s._reduce_deltas = failing_reduce
        assert s.aggregate_fit(0, _results(1, n=7)) is None
        st = s.last_round_stats
        assert st["failed"] is True
        assert not any(ch.isdigit() for ch in st["message"].split(":", 1)[1]), st["message"]

    def test_rule_refusal_message_keeps_detail_without_dp(self):
        s = ComposedStrategy(_global(), rule="mean", clip_norm=0.5)

        def failing_reduce(*args, **kwargs):
            raise ValueError("got n=7")

        s._reduce_deltas = failing_reduce
        assert s.aggregate_fit(0, _results(1, n=7)) is None
        assert "got n=7" in s.last_round_stats["message"]

    def test_zero_noise_scale_under_dp_fails_closed(self):
        """If z > 0 but the noise scale evaluates to zero, the round must not release an un-noised
        aggregate or raw statistics."""
        s = ComposedStrategy(
            _global(), rule="mean", clip_norm=0.5,
            central_noise_multiplier=1.0, noise_seed=1, strict_dp_ledger=True,
        )
        s._sensitivity = lambda n: 0.0
        assert s.aggregate_fit(0, _results(1, n=4)) is None
        st = s.last_round_stats
        assert st["failed"] is True
        assert not any(ch.isdigit() for ch in st["message"].split(":", 1)[1]), st["message"]
        assert st["aggregated"] is None and st["dropped"] == [] and st["filtered"] == []
        assert st["timings"] is None
        for key in ("aggregate_delta_norm", "clip_rate", "mean_shrink", "orig_norms", "kept_share"):
            assert st.get(key) is None, key
