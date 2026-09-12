"""P2-2 slice 1 — additive secure aggregation over the DeComFL scalar channel.

WHY THIS COMPOSES SO CHEAPLY, which is the whole point of doing it here:
``DeComFL.aggregate_fit`` only ever reads ``sum(grad_scalars[k][p] for ... in clients)`` and
``len(clients)`` — an individual client's scalars are never used. That is precisely the
precondition additive secure aggregation needs, so masking drops in without touching the
aggregation math. And the payload is K*P SCALARS (80 bytes at K=1,P=10) rather than a
d-dimensional vector, so the masking cost does not scale with the model.

Masks live in a finite field (integers mod q) over fixed-point-quantised scalars, NOT in float.
Float masks cancel only approximately — (g+m) + (g'-m) loses low bits of g when m is large — and
"approximately" would undermine the exactness claim these tests pin.
"""
import pytest
import torch

from fedlearn.security.secure_aggregation import (
    DEFAULT_MODULUS,
    DEFAULT_SCALE,
    dequantize,
    mask_contribution,
    masking_cost,
    pairwise_mask,
    quantize,
    unmask_sum,
)


def test_pairwise_masks_over_the_whole_cohort_sum_to_zero():
    """The one property everything else rests on: masks cancel exactly in the aggregate."""
    cohort = ["c0", "c1", "c2", "c3"]
    masks = [pairwise_mask(c, cohort, round_seed=7, length=5) for c in cohort]

    total = torch.zeros(5, dtype=torch.int64)
    for m in masks:
        total = (total + m) % DEFAULT_MODULUS
    assert torch.equal(total, torch.zeros(5, dtype=torch.int64))


def test_a_single_clients_mask_is_not_trivially_zero():
    """A mask of all zeros would satisfy the cancellation test while hiding nothing."""
    cohort = ["c0", "c1", "c2"]
    m = pairwise_mask("c0", cohort, round_seed=7, length=8)
    assert not torch.equal(m, torch.zeros(8, dtype=torch.int64))


def test_masks_are_deterministic_for_the_same_round_and_differ_across_rounds():
    cohort = ["c0", "c1", "c2"]
    a = pairwise_mask("c0", cohort, round_seed=7, length=4)
    again = pairwise_mask("c0", cohort, round_seed=7, length=4)
    other_round = pairwise_mask("c0", cohort, round_seed=8, length=4)

    assert torch.equal(a, again), "same round must reproduce — the server cannot re-derive otherwise"
    assert not torch.equal(a, other_round), "mask reuse across rounds leaks the difference of inputs"


def test_pairwise_mask_rejects_a_client_outside_the_cohort():
    with pytest.raises(ValueError, match="cohort"):
        pairwise_mask("stranger", ["c0", "c1", "c2"], round_seed=7, length=4)


# --------------------------------------------------------------------------------------------------
# The claim that matters: masking changes nothing about the aggregate
# --------------------------------------------------------------------------------------------------
def test_masked_aggregate_recovers_the_exact_quantised_sum():
    """Exactness, not approximation — the integer sum must match to the bit."""
    cohort = ["a", "b", "c", "d"]
    values = {
        "a": [0.5, -1.25, 0.0],
        "b": [-0.5, 2.0, 3.5],
        "c": [1.0, 0.125, -2.25],
        "d": [0.25, -0.75, 1.0],
    }
    masked = [
        mask_contribution(values[c], client_id=c, cohort=cohort, round_seed=11)
        for c in cohort
    ]
    recovered = unmask_sum(masked)

    expected = quantize([sum(values[c][i] for c in cohort) for i in range(3)])
    assert torch.equal(recovered, expected)


def test_masked_aggregate_matches_the_plain_float_sum_within_quantisation_error():
    cohort = ["a", "b", "c"]
    values = {"a": [0.001234, -0.5], "b": [-0.000987, 0.25], "c": [0.5, 0.125]}
    masked = [mask_contribution(values[c], client_id=c, cohort=cohort, round_seed=3) for c in cohort]

    recovered = dequantize(unmask_sum(masked))
    plain = torch.tensor([sum(values[c][i] for c in cohort) for i in range(2)])
    assert torch.allclose(recovered, plain, atol=1e-5)


def test_an_individual_masked_contribution_hides_the_value():
    """If a single masked contribution were close to the plaintext, the scheme would be useless."""
    cohort = ["a", "b", "c"]
    v = [0.5, -1.25]
    masked = mask_contribution(v, client_id="a", cohort=cohort, round_seed=11)
    assert not torch.allclose(dequantize(masked), torch.tensor(v), atol=1.0)


def test_a_dropped_client_corrupts_the_aggregate_which_is_why_dropout_resilience_is_slice_2():
    """Pins the KNOWN limitation rather than leaving it undocumented.

    Pairwise masking only cancels over the full cohort. If one client's contribution is missing,
    its peers' halves of the shared pairs survive and the aggregate is garbage — it does not
    degrade gracefully. Removing this failure is exactly what LightSecAgg's one-shot reconstruction
    buys, and it is the next slice.
    """
    cohort = ["a", "b", "c"]
    values = {"a": [1.0], "b": [2.0], "c": [3.0]}
    masked = [mask_contribution(values[c], client_id=c, cohort=cohort, round_seed=5) for c in cohort]

    without_c = dequantize(unmask_sum(masked[:2]))
    assert not torch.allclose(without_c, torch.tensor([3.0]), atol=1e-3)


# --------------------------------------------------------------------------------------------------
# The headline claim: masking cost is O(d) over the gradient channel and O(1) over the ZO channel
# --------------------------------------------------------------------------------------------------
def test_gradient_channel_masking_cost_grows_linearly_with_model_dimension():
    small = masking_cost("fedavg", model_dim=1_000)
    large = masking_cost("fedavg", model_dim=1_000_000)
    assert small["mask_elements"] == 1_000
    assert large["mask_elements"] == 1_000_000
    assert large["mask_bytes"] == 1000 * small["mask_bytes"]


def test_zeroth_order_channel_masking_cost_is_constant_in_model_dimension():
    """This is the whole argument for composing secure aggregation with DeComFL."""
    small = masking_cost("decomfl", model_dim=1_000, K=1, P=10)
    large = masking_cost("decomfl", model_dim=1_000_000, K=1, P=10)
    assert small["mask_elements"] == 10
    assert large["mask_elements"] == 10, "ZO masking must not scale with d"
    assert small["mask_bytes"] == large["mask_bytes"]


def test_zeroth_order_masking_cost_scales_with_K_times_P_not_d():
    assert masking_cost("decomfl", model_dim=10**9, K=5, P=20)["mask_elements"] == 100


def test_masking_cost_advantage_is_reported_and_grows_with_dimension():
    at_10k = masking_cost("decomfl", model_dim=10_000, K=1, P=10)["advantage_vs_gradient_channel"]
    at_1m = masking_cost("decomfl", model_dim=1_000_000, K=1, P=10)["advantage_vs_gradient_channel"]
    assert at_10k == pytest.approx(1_000)
    assert at_1m == pytest.approx(100_000)


def test_masking_cost_rejects_an_unknown_channel():
    with pytest.raises(ValueError, match="channel"):
        masking_cost("carrier-pigeon", model_dim=10)


def test_dequantised_aggregate_is_within_the_quantisation_bound_not_bit_identical():
    """Pins the honest precision claim: exact in the field, bounded after dequantisation.

    Each client's value is rounded to the nearest 1/scale, contributing at most 0.5/scale of
    error, so an n-client sum is within n/(2*scale) of the plain float sum. Asserting equality
    here would be wrong, and asserting a loose tolerance would hide a regression in `scale`.
    """
    cohort = ["a", "b", "c", "d"]
    vals = {
        "a": [0.0012345, -0.5000004], "b": [-0.0009876, 0.2500001],
        "c": [0.4999999, 0.1250003], "d": [0.3333333, -0.6666667],
    }
    masked = [mask_contribution(vals[c], client_id=c, cohort=cohort, round_seed=1) for c in cohort]
    secure = dequantize(unmask_sum(masked)).to(torch.float64)
    plain = torch.tensor([sum(vals[c][i] for c in cohort) for i in range(2)], dtype=torch.float64)

    bound = len(cohort) / (2 * DEFAULT_SCALE)
    assert not torch.equal(secure, plain), "exactness here would mean quantisation was skipped"
    assert torch.all((secure - plain).abs() <= bound), f"error exceeds n/(2*scale) = {bound}"
