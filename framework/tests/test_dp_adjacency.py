# framework/tests/test_dp_adjacency.py
"""Which neighbour relation does ``dp_aggregate``'s noise calibrate?

The accountant treats ``z`` as noise std / L2 sensitivity. ``dp_aggregate`` adds std ``z*S/n``, so its
epsilon is right exactly when the released mean moves by at most ``S/n`` between neighbours. The first
tests pin how far it moves under each neighbour relation; the rest cover the ``adjacency`` option
(replace-one doubles the noise) and the fixed public denominator the DP strategies enforce.

- Fixed cohort, one client's delta swapped for zero (add/remove with a fixed public denominator):
  shift ``S/n``, matching the calibration.
- Fixed cohort, one client's delta swapped for any other clipped delta (replace-one): shift up to
  ``2S/n``, so the effective multiplier is ``z/2``.
- Cohort size allowed to change (add/remove with a data-dependent denominator): shift up to about
  ``2S/n`` as well. Live DP runs never release on a changed cohort (``require_live_dp_accounting``
  forces ``min_clients == clients_per_round``), and the DP strategies now refuse one
  (``public_cohort=clients_per_round``), so this case only applies to bare ``dp_aggregate`` calls.
"""

from collections import OrderedDict

import pytest
import torch

from fedlearn.privacy.dp_mechanism import dp_aggregate

KEY = "w"
S = 1.0
D = 4


def _update(vec):
    return (OrderedDict([(KEY, torch.tensor(vec, dtype=torch.float32))]), 1)


def _release(updates):
    """The noiseless (z=0) release: the clipped uniform mean added to a zero global."""
    glob = OrderedDict([(KEY, torch.zeros(D))])
    return dp_aggregate(updates, glob, [KEY], clip_norm=S, noise_multiplier=0.0)[KEY]


def _unit(i=0, scale=1.0):
    v = [0.0] * D
    v[i] = scale
    return v


@pytest.mark.parametrize("n", [2, 10, 80])
def test_zeroing_one_client_on_a_fixed_cohort_moves_the_mean_by_s_over_n(n):
    others = [_update(_unit(1)) for _ in range(n - 1)]
    with_client = _release(others + [_update(_unit(0))])
    zeroed = _release(others + [_update([0.0] * D)])
    assert torch.linalg.vector_norm(with_client - zeroed).item() == pytest.approx(S / n)


@pytest.mark.parametrize("n", [2, 10, 80])
def test_replacing_one_client_on_a_fixed_cohort_moves_the_mean_by_twice_s_over_n(n):
    others = [_update(_unit(1)) for _ in range(n - 1)]
    a = _release(others + [_update(_unit(0))])
    b = _release(others + [_update(_unit(0, -1.0))])
    shift = torch.linalg.vector_norm(a - b).item()
    assert shift == pytest.approx(2 * S / n)
    # Relative to the calibrated noise std z*S/n, this neighbour pair sees a multiplier of z/2.
    z = 4.0
    assert (z * S / n) / shift == pytest.approx(z / 2)


def test_an_oversized_delta_is_clipped_so_replace_one_never_exceeds_twice_s_over_n():
    n = 10
    others = [_update(_unit(1)) for _ in range(n - 1)]
    a = _release(others + [_update(_unit(0, 1e6))])
    b = _release(others + [_update(_unit(0, -1e6))])
    assert torch.linalg.vector_norm(a - b).item() == pytest.approx(2 * S / n)


@pytest.mark.parametrize("n", [3, 10, 80])
def test_removing_a_client_with_a_data_dependent_denominator_moves_the_mean_by_twice_s_over_n(n):
    # n-1 clients push +v and the removed client pushes -v: mean_n = (n-2)v/n, mean_{n-1} = v.
    others = [_update(_unit(0)) for _ in range(n - 1)]
    full = _release(others + [_update(_unit(0, -1.0))])
    removed = _release(others)
    assert torch.linalg.vector_norm(full - removed).item() == pytest.approx(2 * S / n)


# --------------------------------------------------------------------------------------------------
# The adjacency option and the fixed public denominator
# --------------------------------------------------------------------------------------------------
def _noise_std(adjacency, n=4, z=1.5, d=200_000):
    glob = OrderedDict([(KEY, torch.zeros(d))])
    updates = [(OrderedDict([(KEY, torch.zeros(d))]), 1) for _ in range(n)]
    gen = torch.Generator().manual_seed(3)
    out = dp_aggregate(updates, glob, [KEY], clip_norm=S, noise_multiplier=z, generator=gen, adjacency=adjacency)
    return float(out[KEY].std())


def test_replace_one_doubles_the_noise_for_the_same_multiplier():
    assert _noise_std("add_remove") == pytest.approx(1.5 * S / 4, rel=0.02)
    assert _noise_std("replace_one") == pytest.approx(2 * 1.5 * S / 4, rel=0.02)


def test_unknown_adjacency_is_refused():
    with pytest.raises(ValueError, match="adjacency"):
        _noise_std("swap")


def test_a_cohort_other_than_the_public_one_is_refused():
    updates = [_update(_unit(0)) for _ in range(3)]
    glob = OrderedDict([(KEY, torch.zeros(D))])
    with pytest.raises(ValueError, match="public cohort is 4"):
        dp_aggregate(updates, glob, [KEY], clip_norm=S, noise_multiplier=1.0, public_cohort=4)
    dp_aggregate(updates, glob, [KEY], clip_norm=S, noise_multiplier=1.0, public_cohort=3)


def test_dp_strategies_fix_the_denominator_and_carry_the_adjacency():
    from fedlearn.server.strategy import DPFedAvg

    strategy = DPFedAvg(OrderedDict([(KEY, torch.zeros(D))]), clients_per_round=3, dp_clip_norm=S,
                        dp_noise_multiplier=1.0, dp_seed=0, dp_adjacency="replace_one")
    assert strategy.dp_adjacency == "replace_one"
    with pytest.raises(ValueError, match="public cohort is 3"):
        strategy.aggregate_fit(1, [(f"c{i}", OrderedDict([(KEY, torch.zeros(D))]), 1) for i in range(2)])


def test_dp_strategies_refuse_an_unknown_adjacency_at_construction():
    from fedlearn.server.strategy import DPFedAvg

    with pytest.raises(ValueError, match="dp_adjacency"):
        DPFedAvg(OrderedDict([(KEY, torch.zeros(D))]), dp_clip_norm=S, dp_noise_multiplier=1.0,
                 dp_adjacency="swap")
