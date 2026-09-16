"""FR-12 / paper C3 — adversarial breakdown-point + invariant tests for the Byzantine-robust
aggregator.

These complement ``test_robust_aggregation.py`` (which pins the pure-math definitions and the
*config* Byzantine-fraction guard). Here we drive the estimator with REAL malicious cohorts and
assert the *empirical* breakdown boundary matches the theory (Yin et al. 2018):

  - beta-trimmed-mean tolerates strictly up to a per-end trim count k = floor(beta*n): it stays on
    the honest cluster while the number of Byzantine clients m <= k, and is poisoned the moment
    m > k (a surviving Byzantine value re-enters the kept window).
  - coordinate-wise median tolerates strictly < n/2 Byzantine clients; a Byzantine majority owns the
    median, and — because the estimator is the *interpolating* median (mean of the two central order
    statistics, deliberately not torch.median's lower-middle) — an exact even split is already
    dragged halfway toward the adversary.

Plus the UNWEIGHTED invariant (paper's stated design): ``num_examples`` must never weight the robust
aggregate, or an attacker who inflates its own reported count reclaims the very leverage the
estimator exists to remove.
"""
from collections import OrderedDict

import pytest
import torch

from fedlearn.server.robust_aggregation import RobustAggregator
from fedlearn.server.strategy import FedAvgAggregator


def _p(vec):
    return OrderedDict({"w": torch.tensor(vec, dtype=torch.float32)})


def _init(dim):
    return OrderedDict({"w": torch.zeros(dim, dtype=torch.float32)})


# --------------------------------------------------------------------------------------------------
# Breakdown point: beta-trimmed-mean is robust up to f = beta and breaks the instant f > beta.
# The existing suite only checks the operator's byzantine_fraction *config* guard; this drives the
# estimator with an actual malicious cohort at f just below / at / above the trim fraction.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_malicious, expect_robust",
    [(1, True), (2, True), (3, False)],  # n=10, beta=0.2 -> k=2: f=0.1,0.2 robust; f=0.3 breaks
)
def test_trimmed_mean_breakdown_at_beta_boundary(n_malicious, expect_robust):
    n, beta, honest_val, mal_val = 10, 0.2, 1.0, 1e6  # k = floor(0.2*10) = 2
    honest = [(None, _p([honest_val, honest_val]), 100) for _ in range(n - n_malicious)]
    malicious = [(None, _p([mal_val, mal_val]), 100) for _ in range(n_malicious)]

    # byzantine_fraction=0 keeps the config guard OUT of the way so we observe the estimator itself.
    agg = RobustAggregator(
        initial_parameters=_init(2), method="trimmed_mean", trim_ratio=beta, byzantine_fraction=0.0,
    )
    out = agg.aggregate_fit(1, honest + malicious)["w"]

    if expect_robust:
        # m <= k: every malicious value is trimmed away -> aggregate sits exactly on the honest value.
        assert torch.allclose(out, torch.full((2,), honest_val)), (
            f"trimmed-mean should be robust at f={n_malicious / n:.1f} <= beta={beta}, got {out}"
        )
    else:
        # m > k: a Byzantine value survives the trim window and drags the mean far off the honest cluster.
        assert (out > 100.0).all(), (
            f"trimmed-mean must break at f={n_malicious / n:.1f} > beta={beta} (theory), got {out}"
        )


def test_median_holds_where_trimmed_mean_breaks_on_the_same_cohort():
    """On the SAME f>beta cohort that breaks trimmed-mean(0.2), the median (tolerance 0.5) is still
    robust — the two estimators have different breakdown points and it shows empirically."""
    n = 10
    honest = [(None, _p([1.0]), 100) for _ in range(n - 3)]      # 7 honest
    malicious = [(None, _p([1e6]), 100) for _ in range(3)]        # 3 malicious -> f=0.3

    tm = RobustAggregator(initial_parameters=_init(1), method="trimmed_mean",
                          trim_ratio=0.2, byzantine_fraction=0.0)
    med = RobustAggregator(initial_parameters=_init(1), method="median", byzantine_fraction=0.0)

    assert tm.aggregate_fit(1, honest + malicious)["w"].item() > 100.0    # trimmed-mean broken
    assert med.aggregate_fit(1, honest + malicious)["w"].item() == pytest.approx(1.0)  # median holds


# --------------------------------------------------------------------------------------------------
# Breakdown point: coordinate-wise median tolerates strictly < n/2 Byzantine clients.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_malicious, expect_robust",
    [(2, True), (3, False)],  # n=5: 2/5<1/2 robust; 3/5 is a majority -> owns the median
)
def test_median_breakdown_at_half_boundary_odd_n(n_malicious, expect_robust):
    n = 5
    honest = [(None, _p([1.0]), 100) for _ in range(n - n_malicious)]
    malicious = [(None, _p([1e6]), 100) for _ in range(n_malicious)]

    agg = RobustAggregator(initial_parameters=_init(1), method="median", byzantine_fraction=0.0)
    out = agg.aggregate_fit(1, honest + malicious)["w"].item()

    if expect_robust:
        assert out == pytest.approx(1.0)
    else:
        assert out == pytest.approx(1e6)  # Byzantine majority fully owns the median


def test_interpolating_median_is_dragged_by_an_exact_even_split():
    """The estimator is the INTERPOLATING median (mean of the two central order statistics), so its
    tolerance is strictly < 1/2: at an EXACT even split it already sits halfway to the adversary.

    2 honest@1 + 2 malicious@1e6 -> sorted [1, 1, 1e6, 1e6] -> mean of the two central = (1+1e6)/2.
    This is deliberate, documented behaviour (torch.quantile(0.5) over torch.median's lower-middle),
    and worth pinning: an even cohort split exactly on the breakdown point is NOT protected.
    """
    agg = RobustAggregator(initial_parameters=_init(1), method="median", byzantine_fraction=0.0)
    out = agg.aggregate_fit(
        1,
        [(None, _p([1.0]), 100), (None, _p([1.0]), 100),
         (None, _p([1e6]), 100), (None, _p([1e6]), 100)],
    )["w"].item()
    assert out == pytest.approx((1.0 + 1e6) / 2.0)


# --------------------------------------------------------------------------------------------------
# UNWEIGHTED invariant: num_examples must NOT weight the robust aggregate. An attacker who inflates
# its own reported count must gain ZERO leverage — a weighted median/trimmed-mean would be the bug.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("method, trim", [("median", 0.1), ("trimmed_mean", 0.0)])
def test_num_examples_does_not_weight_the_robust_aggregate(method, trim):
    updates_balanced = [(None, _p([1.0]), 100), (None, _p([2.0]), 100), (None, _p([3.0]), 100)]
    # Same client VALUES; only the reported counts change — one client claims ~1e9 examples.
    updates_skewed = [(None, _p([1.0]), 1), (None, _p([2.0]), 1), (None, _p([3.0]), 999_999_999)]

    a_bal = RobustAggregator(initial_parameters=_init(1), method=method,
                             trim_ratio=trim, byzantine_fraction=0.0)
    a_skew = RobustAggregator(initial_parameters=_init(1), method=method,
                              trim_ratio=trim, byzantine_fraction=0.0)
    bal = a_bal.aggregate_fit(1, updates_balanced)["w"]
    skew = a_skew.aggregate_fit(1, updates_skewed)["w"]

    # Robust aggregate is invariant to the reported counts (unweighted by design).
    assert torch.allclose(bal, skew), f"{method} leaked num_examples weighting: {bal} vs {skew}"
    assert bal.item() == pytest.approx(2.0)

    # Contrast: FedAvg IS num-examples-weighted, so the inflated-count client drags it toward 3.0
    # (capped at MAX_SAMPLES=100k, but still a large pull — proving the counts are load-bearing there).
    fedavg = FedAvgAggregator().aggregate(list(updates_skewed))["w"].item()
    assert fedavg > 2.9, f"FedAvg should be dragged by the inflated count, got {fedavg}"


# --------------------------------------------------------------------------------------------------
# Degenerate cohorts: N=1 and N=2 must reduce to the (interpolating) middle for BOTH estimators,
# with the trim count k=floor(beta*n) collapsing to 0 so nothing is dropped.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("method, trim", [("median", 0.1), ("trimmed_mean", 0.3)])
def test_degenerate_single_and_two_client_cohorts(method, trim):
    a1 = RobustAggregator(initial_parameters=_init(1), method=method,
                          trim_ratio=trim, byzantine_fraction=0.0)
    single = a1.aggregate_fit(1, [(None, _p([7.0]), 100)])
    assert single is not None and single["w"].item() == pytest.approx(7.0)  # N=1 -> the lone client

    a2 = RobustAggregator(initial_parameters=_init(1), method=method,
                          trim_ratio=trim, byzantine_fraction=0.0)
    # N=2, beta<0.5 -> k=floor(beta*2)=0 -> nothing trimmed; both estimators give the two-value mean.
    two = a2.aggregate_fit(1, [(None, _p([4.0]), 100), (None, _p([8.0]), 100)])
    assert two is not None and two["w"].item() == pytest.approx(6.0)
