"""FR-12 — Byzantine-robust aggregation: coordinate-wise median + beta-trimmed-mean, server-side
L2 norm clipping of each client update, non-finite (NaN/Inf) rejection, and a Byzantine-fraction
guard that refuses to aggregate when the estimated malicious fraction exceeds the estimator's
breakdown point.

The math these tests pin (a reviewer who knows FL will check these against their definitions):
  - coordinate-wise median: per-coordinate statistical median across the client axis (for an even
    client count, the mean of the two central order statistics — NOT torch.median's lower-middle).
  - beta-trimmed-mean: per-coordinate, sort the n client values, drop k = floor(beta*n) from EACH
    end, average the remaining n - 2k. beta = 0 is the plain mean.
  - global-norm clipping: scale a whole update by min(1, S / ||delta||_2), where ||delta||_2 is the
    L2 norm over ALL tensors of the update concatenated (mirrors torch.nn.utils.clip_grad_norm_).
"""
from collections import OrderedDict

import pytest
import torch

from fedlearn.server.robust_aggregation import (
    RobustAggregator,
    clip_l2_norm,
    coordinate_wise_median,
    trimmed_mean,
)
from fedlearn.server.strategy import FedAvgAggregator


# --------------------------------------------------------------------------------------------------
# Pure-math correctness: coordinate-wise median
# --------------------------------------------------------------------------------------------------
def test_coordinate_wise_median_odd_count_is_middle_order_statistic():
    stacked = torch.tensor([[1.0, 10.0], [3.0, 30.0], [2.0, 20.0]])  # 3 clients x dim-2
    out = coordinate_wise_median(stacked)
    assert torch.allclose(out, torch.tensor([2.0, 20.0]))


def test_coordinate_wise_median_even_count_averages_two_central_order_statistics():
    # 4 clients: true median is the mean of the two central values per coordinate, NOT torch.median's
    # lower-middle. Column 0 sorted -> [1,2,3,4] -> mean(2,3)=2.5; column 1 -> [10,20,30,40] -> 25.
    stacked = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    out = coordinate_wise_median(stacked)
    assert torch.allclose(out, torch.tensor([2.5, 25.0]))


def test_coordinate_wise_median_ignores_a_single_extreme_outlier():
    # One wild client cannot move the median off the honest cluster (odd count -> middle element).
    stacked = torch.tensor([[5.0], [5.0], [1e9]])
    out = coordinate_wise_median(stacked)
    assert torch.allclose(out, torch.tensor([5.0]))


# --------------------------------------------------------------------------------------------------
# Pure-math correctness: beta-trimmed-mean
# --------------------------------------------------------------------------------------------------
def test_trimmed_mean_drops_floor_beta_n_from_each_end():
    # n=5, beta=0.2 -> k=floor(1.0)=1 -> drop min and max -> mean(2,3,4)=3.
    stacked = torch.tensor([[1.0], [2.0], [3.0], [4.0], [100.0]])
    out = trimmed_mean(stacked, trim_ratio=0.2)
    assert torch.allclose(out, torch.tensor([3.0]))


def test_trimmed_mean_beta_zero_is_plain_mean():
    stacked = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    out = trimmed_mean(stacked, trim_ratio=0.0)
    assert torch.allclose(out, torch.tensor([2.5]))


def test_trimmed_mean_trims_both_ends_symmetrically():
    # n=6, beta=1/3 -> k=floor(2.0)=2 -> drop 2 lowest + 2 highest -> mean of middle 2.
    stacked = torch.tensor([[-1000.0], [1.0], [2.0], [3.0], [4.0], [1000.0]])
    out = trimmed_mean(stacked, trim_ratio=1.0 / 3.0)
    assert torch.allclose(out, torch.tensor([2.5]))  # mean(2,3)


def test_trimmed_mean_rejects_over_trim():
    stacked = torch.tensor([[1.0], [2.0]])
    with pytest.raises(ValueError, match="trim"):
        trimmed_mean(stacked, trim_ratio=0.5)  # k=1 -> would remove everything


# --------------------------------------------------------------------------------------------------
# Pure-math correctness: global L2 norm clipping
# --------------------------------------------------------------------------------------------------
def test_clip_l2_norm_scales_an_over_budget_update_to_the_bound():
    update = OrderedDict({"w": torch.tensor([3.0, 4.0])})  # L2 norm = 5
    clipped, orig = clip_l2_norm(update, max_norm=1.0)
    assert orig == pytest.approx(5.0)
    total = torch.sqrt(sum((t * t).sum() for t in clipped.values()))
    assert total.item() == pytest.approx(1.0, abs=1e-5)
    assert torch.allclose(clipped["w"], torch.tensor([0.6, 0.8]), atol=1e-5)


def test_clip_l2_norm_is_identity_below_budget():
    update = OrderedDict({"w": torch.tensor([0.3, 0.4])})  # norm 0.5 < 1.0
    clipped, orig = clip_l2_norm(update, max_norm=1.0)
    assert orig == pytest.approx(0.5)
    assert torch.allclose(clipped["w"], update["w"])


def test_clip_l2_norm_uses_the_global_norm_across_all_tensors():
    # norm over BOTH tensors concatenated = sqrt(9+16+0) = 5; clip to 2.5 -> scale 0.5.
    update = OrderedDict({"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([0.0, 0.0])})
    clipped, _ = clip_l2_norm(update, max_norm=2.5)
    assert torch.allclose(clipped["a"], torch.tensor([1.5, 2.0]))
    assert torch.allclose(clipped["b"], torch.tensor([0.0, 0.0]))


# --------------------------------------------------------------------------------------------------
# Helpers for the aggregator-level tests
# --------------------------------------------------------------------------------------------------
def _params(vec) -> OrderedDict:
    return OrderedDict({"w": torch.tensor(vec, dtype=torch.float32)})


def _init(dim: int) -> OrderedDict:
    return OrderedDict({"w": torch.zeros(dim, dtype=torch.float32)})


# --------------------------------------------------------------------------------------------------
# DoD: non-finite rejection — a NaN/Inf client is dropped and the round completes on the honest ones
# --------------------------------------------------------------------------------------------------
def test_robust_aggregation_drops_non_finite_client_and_completes_on_honest():
    agg = RobustAggregator(
        initial_parameters=_init(3), method="median", clip_norm=None, byzantine_fraction=0.0,
    )
    honest = [
        (None, _params([1.0, 2.0, 3.0]), 100),
        (None, _params([1.2, 2.2, 3.2]), 100),
        (None, _params([0.8, 1.8, 2.8]), 100),
    ]
    poisoned = (None, _params([float("nan"), float("inf"), 3.0]), 100)

    result = agg.aggregate_fit(1, honest + [poisoned])

    assert result is not None
    assert torch.isfinite(result["w"]).all()
    # The NaN client was dropped -> the result is the honest-only coordinate-wise median.
    honest_stack = torch.stack([e[1]["w"] for e in honest])
    assert torch.allclose(result["w"], coordinate_wise_median(honest_stack))
    assert agg.last_round_failed is False


def test_robust_aggregation_refuses_when_every_client_is_non_finite():
    agg = RobustAggregator(initial_parameters=_init(2), method="median", byzantine_fraction=0.0)
    result = agg.aggregate_fit(1, [(None, _params([float("nan"), 1.0]), 100)])
    assert result is None
    assert agg.last_round_failed is True


# --------------------------------------------------------------------------------------------------
# DoD: a 100x-norm attacker is clipped and cannot move the global beyond the clip bound S
# --------------------------------------------------------------------------------------------------
def test_norm_clip_bounds_the_global_move_of_a_100x_attacker():
    S = 1.0
    # trim_ratio=0 -> plain mean, so CLIPPING is the only thing bounding the attacker's pull.
    agg = RobustAggregator(
        initial_parameters=_init(4), method="trimmed_mean", trim_ratio=0.0,
        clip_norm=S, byzantine_fraction=0.0,
    )
    g0 = agg._global["w"].clone()
    honest = (None, _params([0.0, 0.0, 0.0, 0.0]), 100)          # delta 0
    attacker = (None, _params([1e4, 0.0, 0.0, 0.0]), 100)        # delta norm 1e4 (100x+ budget)

    result = agg.aggregate_fit(1, [honest, attacker])

    move = torch.norm(result["w"] - g0).item()
    assert move <= S + 1e-5, f"clipped attacker moved the global by {move} > S={S}"

    # Control: unclipped FedAvg over the SAME raw updates is dragged thousands of units away.
    fedavg = FedAvgAggregator().aggregate([honest, attacker])
    assert torch.norm(fedavg["w"] - g0).item() > 100 * S


# --------------------------------------------------------------------------------------------------
# DoD: Byzantine-fraction guard — refuse when the estimated malicious fraction exceeds tolerance
# --------------------------------------------------------------------------------------------------
def test_byzantine_guard_refuses_trimmed_mean_when_estimate_exceeds_trim_ratio():
    agg = RobustAggregator(
        initial_parameters=_init(2), method="trimmed_mean", trim_ratio=0.1, byzantine_fraction=0.3,
    )
    g_before = agg._global["w"].clone()
    result = agg.aggregate_fit(1, [(None, _params([1.0, 1.0]), 100), (None, _params([2.0, 2.0]), 100)])
    assert result is None
    assert agg.last_round_failed is True
    assert "exceeds" in (agg.last_round_message or "").lower()
    assert torch.allclose(agg._global["w"], g_before)  # global untouched on refusal


def test_byzantine_guard_allows_trimmed_mean_at_the_tolerance_boundary():
    agg = RobustAggregator(
        initial_parameters=_init(2), method="trimmed_mean", trim_ratio=0.2, byzantine_fraction=0.2,
    )
    result = agg.aggregate_fit(1, [(None, _params([1.0, 1.0]), 100), (None, _params([3.0, 3.0]), 100)])
    assert result is not None
    assert agg.last_round_failed is False


def test_byzantine_guard_median_tolerates_up_to_one_half():
    agg = RobustAggregator(initial_parameters=_init(2), method="median", byzantine_fraction=0.6)
    assert agg.aggregate_fit(1, [(None, _params([1.0, 1.0]), 100)]) is None
    assert agg.last_round_failed is True

    ok = RobustAggregator(initial_parameters=_init(2), method="median", byzantine_fraction=0.4)
    assert ok.aggregate_fit(1, [(None, _params([1.0, 1.0]), 100)]) is not None
    assert ok.last_round_failed is False


# --------------------------------------------------------------------------------------------------
# DoD: under a 20%-malicious attack, robust aggregators stay close to the honest mean while FedAvg
# is pulled far off. Two attack shapes: gradient scaling (large shift) and label-flip (sign flip).
# --------------------------------------------------------------------------------------------------
def _attack_cohort(malicious_value: float, dim: int = 4):
    """8 honest clients tightly clustered at 5.0 (exact honest mean 5.0) + 2 malicious clients."""
    offsets = [-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04]
    honest = [(None, _params([5.0 + o] * dim), 100) for o in offsets]
    malicious = [(None, _params([malicious_value] * dim), 100) for _ in range(2)]  # 2/10 = 20%
    honest_mean = torch.stack([e[1]["w"] for e in honest]).mean(dim=0)  # exactly 5.0
    return honest, malicious, honest_mean


@pytest.mark.parametrize("method", ["median", "trimmed_mean"])
def test_robust_stays_near_honest_mean_under_gradient_scaling_while_fedavg_does_not(method):
    honest, malicious, honest_mean = _attack_cohort(malicious_value=105.0)  # +100 scaling attack
    updates = honest + malicious

    agg = RobustAggregator(
        initial_parameters=_init(4), method=method, trim_ratio=0.2,
        clip_norm=None, byzantine_fraction=0.2,
    )
    robust = agg.aggregate_fit(1, updates)
    fedavg = FedAvgAggregator().aggregate(list(updates))

    robust_err = torch.norm(robust["w"] - honest_mean).item()
    fedavg_err = torch.norm(fedavg["w"] - honest_mean).item()

    assert robust_err < 0.5, f"{method} drifted {robust_err} from the honest mean"
    assert fedavg_err > 5.0, f"FedAvg only drifted {fedavg_err}; attack too weak to be meaningful"
    assert fedavg_err > 10 * robust_err


@pytest.mark.parametrize("method", ["median", "trimmed_mean"])
def test_robust_stays_near_honest_mean_under_label_flip_while_fedavg_does_not(method):
    honest, malicious, honest_mean = _attack_cohort(malicious_value=-50.0)  # sign-flipped payload
    updates = honest + malicious

    agg = RobustAggregator(
        initial_parameters=_init(4), method=method, trim_ratio=0.2,
        clip_norm=None, byzantine_fraction=0.2,
    )
    robust = agg.aggregate_fit(1, updates)
    fedavg = FedAvgAggregator().aggregate(list(updates))

    robust_err = torch.norm(robust["w"] - honest_mean).item()
    fedavg_err = torch.norm(fedavg["w"] - honest_mean).item()

    assert robust_err < 0.5
    assert fedavg_err > 5.0
    assert fedavg_err > 10 * robust_err


# --------------------------------------------------------------------------------------------------
# Wiring: RobustAggregator is a Strategy and is reachable through the factory
# --------------------------------------------------------------------------------------------------
def test_robust_aggregator_is_a_strategy_and_factory_registered():
    from fedlearn.server.strategy import Strategy
    from fedlearn.server.strategy_factory import create_strategy

    agg = create_strategy("robust", initial_parameters=_init(2), method="median")
    assert isinstance(agg, Strategy)
    assert agg.initialize_parameters() is not None


def test_robust_aggregator_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        RobustAggregator(initial_parameters=_init(2), method="nonsense")


def test_robust_aggregator_rejects_out_of_range_trim_ratio():
    with pytest.raises(ValueError, match="trim"):
        RobustAggregator(initial_parameters=_init(2), method="trimmed_mean", trim_ratio=0.5)


# --------------------------------------------------------------------------------------------------
# FR-19 — key/shape homogeneity: a malformed client must be dropped, never crash or wipe the global
# --------------------------------------------------------------------------------------------------
def test_robust_drops_shape_mismatched_client_instead_of_crashing():
    """A client whose tensor shape differs from the global must be dropped, not crash aggregate_fit.

    _robust_reduce does torch.stack([c[key] for c in clients]); a wrong-shape client raises a
    RuntimeError deep inside aggregation — an uncaught crash of the aggregation thread after the
    client was already accepted. Two honest dim-2 clients + one dim-3 attacker must aggregate to
    the median over the honest two.
    """
    agg = RobustAggregator(initial_parameters=_init(2), method="median", byzantine_fraction=0.0)
    honest = [(None, _params([1.0, 1.0]), 100), (None, _params([3.0, 3.0]), 100)]
    malformed = (None, OrderedDict({"w": torch.tensor([9.0, 9.0, 9.0])}), 100)  # dim-3
    result = agg.aggregate_fit(1, honest + [malformed])
    assert result is not None
    assert set(result.keys()) == {"w"}
    assert torch.allclose(result["w"], torch.tensor([2.0, 2.0]))  # median of [1, 3] per coord


def test_robust_empty_client_does_not_wipe_the_global():
    """An empty state_dict as the FIRST client must not silently reduce the aggregate to {}.

    _robust_reduce templates on clients[0].keys(); an empty clients[0] yields an empty output,
    which is then persisted as the new global — silently wiping the model. The malformed client
    must be dropped and the honest survivors aggregated instead.
    """
    agg = RobustAggregator(initial_parameters=_init(2), method="median", byzantine_fraction=0.0)
    empty = (None, OrderedDict(), 100)  # malformed empty state_dict, positioned first
    honest = [(None, _params([1.0, 1.0]), 100), (None, _params([3.0, 3.0]), 100)]
    result = agg.aggregate_fit(1, [empty] + honest)
    assert result is not None
    assert set(result.keys()) == {"w"}
    assert torch.allclose(result["w"], torch.tensor([2.0, 2.0]))
