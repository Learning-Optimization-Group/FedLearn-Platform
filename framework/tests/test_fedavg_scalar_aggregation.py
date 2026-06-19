"""
Tests for FedAvgAggregator.aggregate_scalars — server-side reconstruction of
per-client model updates from ZO gradient scalars (DECISION D1).

The correctness gate is a 2-client toy round proving:
  scalar path (aggregate_scalars) == weight-delta path (aggregate) within atol=1e-5.
"""
import pytest
import torch
from collections import OrderedDict

from fedlearn.server.strategy import FedAvgAggregator
from fedlearn.estimators.perturbation import canonical_perturbation


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def make_global_params() -> OrderedDict:
    """Deterministic tiny model: fc.weight (3×4) + fc.bias (3) → d=15."""
    torch.manual_seed(0)
    return OrderedDict([
        ("fc.weight", torch.randn(3, 4, dtype=torch.float32)),
        ("fc.bias",   torch.randn(3,    dtype=torch.float32)),
    ])


# Round hyper-params
ETA = 0.01
K   = 2   # local steps
P   = 3   # perturbations per step

# Client 1 — 8 examples
SEEDS_C1 = [
    [101, 202, 303],   # k=0
    [404, 505, 606],   # k=1
]
GRADS_C1 = [
    [0.1,  -0.2,  0.3],
    [-0.4,  0.5, -0.6],
]
N_C1 = 8

# Client 2 — 16 examples
SEEDS_C2 = [
    [707, 808, 909],
    [111, 222, 333],
]
GRADS_C2 = [
    [0.7, -0.8,  0.9],
    [-0.1,  0.2, -0.3],
]
N_C2 = 16


# ---------------------------------------------------------------------------
# Helper: compute Δ_c for one client (reference implementation of T9 math)
# ---------------------------------------------------------------------------

def _compute_delta(seeds, gradients, eta, P, d):
    """Δ_c = (eta/P) · Σ_{k,p} g[k][p] · canonical_perturbation(seed[k][p], d)."""
    delta = torch.zeros(d, dtype=torch.float32)
    for k_seeds, k_grads in zip(seeds, gradients):
        for seed, g in zip(k_seeds, k_grads):
            delta += g * canonical_perturbation(seed, d)
    delta *= eta / P
    return delta


def _flatten(params: OrderedDict) -> torch.Tensor:
    return torch.cat([t.view(-1) for t in params.values()]).float()


def _unflatten(flat: torch.Tensor, template: OrderedDict) -> OrderedDict:
    out = OrderedDict()
    offset = 0
    for name, t in template.items():
        n = t.numel()
        out[name] = flat[offset:offset + n].view_as(t).clone()
        offset += n
    return out


# ---------------------------------------------------------------------------
# Correctness: scalar path == weight-delta path within 1e-5
# ---------------------------------------------------------------------------

class TestAggregateScalarsEquivalence:

    def test_scalar_path_matches_weight_delta_path(self):
        """The scalar aggregation must produce the same result as the weight-delta
        FedAvg path (aggregate) within atol=1e-5 for a 2-client toy round."""
        global_params = make_global_params()
        flat = _flatten(global_params)
        d = flat.numel()

        # --- Reference: weight-delta path ---
        delta_c1 = _compute_delta(SEEDS_C1, GRADS_C1, ETA, P, d)
        delta_c2 = _compute_delta(SEEDS_C2, GRADS_C2, ETA, P, d)

        flat_c1 = flat - delta_c1
        flat_c2 = flat - delta_c2

        params_c1 = _unflatten(flat_c1, global_params)
        params_c2 = _unflatten(flat_c2, global_params)

        ref = FedAvgAggregator().aggregate([
            ("c1", params_c1, N_C1),
            ("c2", params_c2, N_C2),
        ])

        # --- Scalar path ---
        got = FedAvgAggregator().aggregate_scalars(
            global_params=global_params,
            results=[
                ("c1", SEEDS_C1, GRADS_C1, N_C1),
                ("c2", SEEDS_C2, GRADS_C2, N_C2),
            ],
            eta=ETA,
            num_perturbations=P,
        )

        assert got.keys() == global_params.keys(), "Key sets must match"

        for key in global_params:
            assert got[key].shape == global_params[key].shape, (
                f"Shape mismatch for {key}: got {got[key].shape}"
            )
            assert torch.allclose(got[key], ref[key], atol=1e-5), (
                f"Scalar vs weight-delta mismatch for '{key}': "
                f"max_diff={( got[key] - ref[key]).abs().max().item():.2e}"
            )


# ---------------------------------------------------------------------------
# Single-client: output == global_old - Δ_c exactly (no averaging distortion)
# ---------------------------------------------------------------------------

class TestAggregateScalarsSingleClient:

    def test_single_client_equals_global_minus_delta(self):
        global_params = make_global_params()
        flat = _flatten(global_params)
        d = flat.numel()

        delta_c1 = _compute_delta(SEEDS_C1, GRADS_C1, ETA, P, d)
        expected_flat = flat - delta_c1
        expected = _unflatten(expected_flat, global_params)

        got = FedAvgAggregator().aggregate_scalars(
            global_params=global_params,
            results=[("c1", SEEDS_C1, GRADS_C1, N_C1)],
            eta=ETA,
            num_perturbations=P,
        )

        for key in global_params:
            assert torch.allclose(got[key], expected[key], atol=1e-6), (
                f"Single-client mismatch for '{key}'"
            )


# ---------------------------------------------------------------------------
# Guard: malformed payloads raise ValueError
# ---------------------------------------------------------------------------

class TestAggregateScalarsValidation:

    def _base_results(self):
        return [("c1", SEEDS_C1, GRADS_C1, N_C1)]

    def test_ragged_kp_seeds_gradients_mismatch_raises(self):
        """len(seeds) != len(gradients) must raise ValueError."""
        global_params = make_global_params()
        bad_results = [("c1", SEEDS_C1, GRADS_C1[:-1], N_C1)]  # gradients has K-1 rows
        with pytest.raises(ValueError, match="[Kk].*mismatch|seeds.*gradients|len"):
            FedAvgAggregator().aggregate_scalars(
                global_params=global_params,
                results=bad_results,
                eta=ETA,
                num_perturbations=P,
            )

    def test_ragged_inner_p_dimension_raises(self):
        """seeds[k] and gradients[k] having different lengths must raise ValueError."""
        global_params = make_global_params()
        bad_grads = [
            [0.1, -0.2],          # k=0 has P=2 instead of P=3
            [-0.4,  0.5, -0.6],
        ]
        bad_results = [("c1", SEEDS_C1, bad_grads, N_C1)]
        with pytest.raises(ValueError):
            FedAvgAggregator().aggregate_scalars(
                global_params=global_params,
                results=bad_results,
                eta=ETA,
                num_perturbations=P,
            )

    def test_all_zero_examples_raises(self):
        """All entries with n<=0 must raise ValueError after sanitization."""
        global_params = make_global_params()
        bad_results = [
            ("c1", SEEDS_C1, GRADS_C1, 0),
            ("c2", SEEDS_C2, GRADS_C2, -5),
        ]
        with pytest.raises(ValueError, match="No valid updates"):
            FedAvgAggregator().aggregate_scalars(
                global_params=global_params,
                results=bad_results,
                eta=ETA,
                num_perturbations=P,
            )

    def test_invalid_entries_dropped_valid_kept(self):
        """A mix of n<=0 and n>0: the zero entry is dropped; valid entry aggregated."""
        global_params = make_global_params()
        # c1 has n=0 (dropped), c2 is valid
        results = [
            ("c1", SEEDS_C1, GRADS_C1, 0),
            ("c2", SEEDS_C2, GRADS_C2, N_C2),
        ]
        # Should not raise; result must equal single-client c2 outcome
        got = FedAvgAggregator().aggregate_scalars(
            global_params=global_params,
            results=results,
            eta=ETA,
            num_perturbations=P,
        )
        expected = FedAvgAggregator().aggregate_scalars(
            global_params=global_params,
            results=[("c2", SEEDS_C2, GRADS_C2, N_C2)],
            eta=ETA,
            num_perturbations=P,
        )
        for key in global_params:
            assert torch.allclose(got[key], expected[key], atol=1e-6)

    def test_max_samples_cap_applied(self):
        """num_examples > MAX_SAMPLES is capped; equal-weight average across 2 clients."""
        global_params = make_global_params()
        results = [
            ("c1", SEEDS_C1, GRADS_C1, 200_000),
            ("c2", SEEDS_C2, GRADS_C2, 200_000),
        ]
        # Just verify it runs and produces a result (not a crash)
        got = FedAvgAggregator().aggregate_scalars(
            global_params=global_params,
            results=results,
            eta=ETA,
            num_perturbations=P,
        )
        assert got.keys() == global_params.keys()
