"""FR-12 extension — selection-based Byzantine-robust rules: Krum, Multi-Krum, Bulyan.

These differ structurally from the coordinate-wise estimators already in the module. Median and
trimmed-mean reduce each parameter coordinate independently; Krum and its descendants SELECT whole
clients using L2 distances over the client's ENTIRE flattened update. Applying them per-parameter
would be a different (and wrong) algorithm, so they take a flattened ``[n_clients, D]`` matrix.

The math these tests pin, against the papers a reviewer would check:
  - Krum (Blanchard et al., NIPS 2017, https://arxiv.org/abs/1703.02757): for each client i, sum
    the squared L2 distances to its ``n - f - 2`` CLOSEST other clients; select the argmin. Requires
    ``n >= 2f + 3``.
  - Multi-Krum (same paper): take the ``m`` lowest-scoring clients by the Krum score and average
    them; ``m = 1`` reduces to Krum.
"""
from collections import OrderedDict

import pytest
import torch

from fedlearn.server.robust_aggregation import (
    bulyan_aggregate,
    centered_clip,
    krum_select,
    multi_krum_select,
)
from fedlearn.server.robust_aggregation import RobustAggregator


def test_krum_selects_the_client_at_the_centre_of_the_honest_cluster():
    # n=5, f=1 -> each score sums the n-f-2 = 2 smallest squared distances to other clients.
    # Values chosen so the scores are hand-computable and the argmin is unique:
    #   c0=0.0   -> two smallest of {1, 2.25, 9, 10000}      = 1    + 2.25 = 3.25
    #   c1=1.0   -> two smallest of {1, 0.25, 4, 9801}       = 0.25 + 1    = 1.25  <- min
    #   c2=1.5   -> two smallest of {2.25, 0.25, 2.25, ...}  = 0.25 + 2.25 = 2.50
    #   c3=3.0   -> two smallest of {9, 4, 2.25, 9409}       = 2.25 + 4    = 6.25
    #   c4=100.0 -> two smallest of {10000, 9801, 9604, 9409}= 9604 + 9409 = 19013
    stacked = torch.tensor([[0.0], [1.0], [1.5], [3.0], [100.0]])
    assert krum_select(stacked, num_byzantine=1) == 1


def test_krum_requires_n_at_least_2f_plus_3():
    # n=4, f=1 needs n >= 5; the neighbour count n-f-2 = 1 is defined but the guarantee is not.
    stacked = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="2f \\+ 3|2\\*f"):
        krum_select(stacked, num_byzantine=1)


# --------------------------------------------------------------------------------------------------
# Multi-Krum
# --------------------------------------------------------------------------------------------------
def test_multi_krum_returns_the_m_lowest_scoring_clients_in_score_order():
    # Same fixture as the Krum test, whose hand-computed scores are
    #   c0=3.25, c1=1.25, c2=2.50, c3=6.25, c4=19013
    # so ascending by score the order is c1, c2, c0, c3, c4.
    stacked = torch.tensor([[0.0], [1.0], [1.5], [3.0], [100.0]])
    assert multi_krum_select(stacked, num_byzantine=1, num_select=3) == [1, 2, 0]


def test_multi_krum_with_m_equals_one_reduces_to_krum():
    stacked = torch.tensor([[0.0], [1.0], [1.5], [3.0], [100.0]])
    assert multi_krum_select(stacked, num_byzantine=1, num_select=1) == [
        krum_select(stacked, num_byzantine=1)
    ]


def test_multi_krum_rejects_selecting_more_clients_than_exist():
    stacked = torch.tensor([[0.0], [1.0], [1.5], [3.0], [100.0]])
    with pytest.raises(ValueError, match="num_select"):
        multi_krum_select(stacked, num_byzantine=1, num_select=6)


# --------------------------------------------------------------------------------------------------
# Bulyan
# --------------------------------------------------------------------------------------------------
def test_bulyan_requires_n_at_least_4f_plus_3():
    # Bulyan's guarantee needs a strictly larger cohort than Krum's: n >= 4f + 3, not 2f + 3.
    # n=6, f=1 satisfies Krum (>=5) but NOT Bulyan (>=7), which is exactly the case worth pinning.
    stacked = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    with pytest.raises(ValueError, match="4f \\+ 3|4\\*f"):
        bulyan_aggregate(stacked, num_byzantine=1)


def test_bulyan_ignores_an_extreme_attacker_that_dominates_the_plain_mean():
    # n=7, f=1 -> selection keeps theta = n - 2f = 5 clients, then averages the beta = theta - 2f = 3
    # values nearest the per-coordinate median of those 5. Six honest clients sit in [0.9, 1.1];
    # one attacker sits at 1000, which drags the plain mean to ~143.
    stacked = torch.tensor([[1.0], [1.1], [0.9], [1.05], [0.95], [1.02], [1000.0]])
    out = bulyan_aggregate(stacked, num_byzantine=1)

    assert out.shape == (1,)
    assert 0.9 <= float(out[0]) <= 1.1, f"Bulyan was pulled off the honest cluster: {out}"
    assert float(stacked.mean()) > 100, "fixture no longer exercises the attack"


# --------------------------------------------------------------------------------------------------
# Centered Clipping
# --------------------------------------------------------------------------------------------------
def test_centered_clipping_with_a_large_tau_reduces_to_the_plain_mean():
    # One iteration is v <- v + mean_i clip(x_i - v, tau). With tau large enough that nothing is
    # clipped, the update telescopes to v + mean(x_i) - v = mean(x_i), for ANY starting centre.
    stacked = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    out = centered_clip(stacked, centre=torch.tensor([-5.0, 7.0]), tau=1e6, iterations=1)
    assert torch.allclose(out, stacked.mean(dim=0), atol=1e-5)


def test_centered_clipping_bounds_the_move_from_the_centre_by_tau():
    # The guarantee that makes this robust: every clipped term has norm <= tau, so their mean does
    # too, so one iteration moves the centre by at most tau NO MATTER what an attacker sends.
    stacked = torch.tensor([[1.0], [1.1], [0.9], [1.05], [10_000.0]])
    centre = torch.tensor([1.0])
    tau = 0.5
    out = centered_clip(stacked, centre=centre, tau=tau, iterations=1)

    assert torch.linalg.vector_norm(out - centre) <= tau + 1e-5
    assert float(stacked.mean()) > 1000, "fixture no longer exercises the attack"


def test_centered_clipping_rejects_non_positive_tau():
    stacked = torch.tensor([[1.0], [2.0], [3.0]])
    with pytest.raises(ValueError, match="tau"):
        centered_clip(stacked, centre=torch.tensor([0.0]), tau=0.0, iterations=1)


# --------------------------------------------------------------------------------------------------
# Strategy integration — the selection rules must see the WHOLE update, not one key at a time
# --------------------------------------------------------------------------------------------------
def _params(**kw) -> OrderedDict:
    return OrderedDict((k, torch.tensor(v, dtype=torch.float32)) for k, v in kw.items())


def _multi_key_cohort():
    """Five clients over TWO parameter keys; client 4 is the attacker, and it is only extreme in
    the SECOND key. A per-key rule would still be poisoned on that key; a rule that scores the
    concatenated update rejects the client outright."""
    honest = [
        _params(a=[1.00, 1.00], b=[5.00]),
        _params(a=[1.05, 0.95], b=[5.05]),
        _params(a=[0.95, 1.05], b=[4.95]),
        _params(a=[1.02, 0.98], b=[5.02]),
    ]
    attacker = _params(a=[1.01, 0.99], b=[900.0])
    return honest, attacker


def test_krum_strategy_selects_a_whole_honest_client_across_all_keys():
    honest, attacker = _multi_key_cohort()
    agg = RobustAggregator(
        initial_parameters=_params(a=[0.0, 0.0], b=[0.0]),
        method="krum", byzantine_fraction=0.0,
    )
    result = agg.aggregate_fit(1, [(None, p, 100) for p in honest + [attacker]])

    assert result is not None
    # Krum returns ONE client's update verbatim -- so the result must equal some honest client
    # exactly, on every key, and must not be the attacker.
    assert any(
        torch.allclose(result["a"], h["a"]) and torch.allclose(result["b"], h["b"])
        for h in honest
    ), f"Krum did not return an honest client verbatim: {result}"
    assert not torch.allclose(result["b"], attacker["b"])


@pytest.mark.parametrize("method,expected", [
    ("median", 0.5),
    ("krum", 0.5),
    ("multi_krum", 0.5),
    ("bulyan", 0.25),
    ("centered_clip", 0.5),
])
def test_tolerance_reports_each_rules_breakdown_point(method, expected):
    agg = RobustAggregator(initial_parameters=_params(a=[0.0]), method=method)
    assert agg.tolerance == pytest.approx(expected)
