# framework/tests/test_dp_adjacency.py
"""Which neighbour relation does ``dp_aggregate``'s noise calibrate?

The accountant treats ``z`` as noise std / L2 sensitivity. ``dp_aggregate`` adds std ``z*S/n``, so its
epsilon is right exactly when the released mean moves by at most ``S/n`` between neighbours. These
tests pin how far it moves under each neighbour relation; they describe behaviour, they do not change it.

- Fixed cohort, one client's delta swapped for zero (add/remove with a fixed public denominator):
  shift ``S/n``, matching the calibration.
- Fixed cohort, one client's delta swapped for any other clipped delta (replace-one): shift up to
  ``2S/n``, so the effective multiplier is ``z/2``.
- Cohort size allowed to change (add/remove with a data-dependent denominator): shift up to about
  ``2S/n`` as well. Live DP runs never release on a changed cohort (``require_live_dp_accounting``
  forces ``min_clients == clients_per_round``), so this case only applies to direct framework calls.
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
