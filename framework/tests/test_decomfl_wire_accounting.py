"""Total-wire accounting for the DeComFL-vs-FedAvg comparison.

The dimension sweep compared the two arms on UPLINK ONLY — both
``decomfl_bytes_per_round`` and ``fedavg_bytes_per_round`` are documented "Uplink per
client-round". A federated round is not one-directional, and the three omitted terms do not
cancel:

* FedAvg's server broadcasts the full model to every selected client, every round. Excluding it
  understates FedAvg by ~2x.
* DeComFL downloads the round's K*P seeds. Small, but not zero.
* DeComFL's REBUILD path re-sends seeds AND averaged gradients for every round a client missed.
  This is the one that matters: it scales with partial participation, so the dimension-free
  claim (which is about the per-round upload) does not cover it. At 10-of-20 clients per round
  over 12,500 rounds it is not a rounding error.

These pin the added downlink accounting. The uplink functions keep their existing semantics —
they are correct, just incomplete on their own.
"""

import pytest

import benchmarks.decomfl_vs_fedavg_dim as H


K, P_ = 1, 10


def test_fedavg_downlink_matches_the_broadcast_it_represents():
    """The server sends every selected client the full model — same payload it receives back."""
    for d in (1026, 10302, 103002):
        assert H.fedavg_downlink_bytes_per_round(d) == H.fedavg_bytes_per_round(d)


def test_decomfl_per_round_downlink_is_dimension_free():
    """The per-round downlink is K*P seeds — independent of d, like the uplink."""
    a = H.decomfl_downlink_bytes_per_round(K=K, P_=P_, missed_rounds=0)
    b = H.decomfl_downlink_bytes_per_round(K=K, P_=P_, missed_rounds=0)
    assert a == b
    assert a > 0


def test_decomfl_downlink_grows_with_missed_rounds():
    """The rebuild chain is the term that breaks dimension-freedom's coverage of the downlink."""
    base = H.decomfl_downlink_bytes_per_round(K=K, P_=P_, missed_rounds=0)
    one = H.decomfl_downlink_bytes_per_round(K=K, P_=P_, missed_rounds=1)
    five = H.decomfl_downlink_bytes_per_round(K=K, P_=P_, missed_rounds=5)
    assert one > base
    assert five - one == pytest.approx(4 * (one - base)), "rebuild cost must be linear in missed rounds"


def test_decomfl_oneshot_download_is_the_full_model():
    """The O(d) cost DeComFL pays once at join — the paper reports it separately, so do we."""
    for d in (1026, 103002):
        assert H.decomfl_oneshot_download_bytes(d) == d * H.FLOAT_BYTES


def test_a_fully_participating_client_still_pays_a_downlink():
    """Even with zero missed rounds DeComFL is not downlink-free; the seeds still travel."""
    assert H.decomfl_downlink_bytes_per_round(K=K, P_=P_, missed_rounds=0) > 0


def test_uplink_functions_keep_their_existing_meaning():
    """Regression guard: the added accounting must not silently redefine the uplink figures
    that every committed result was produced with."""
    assert H.decomfl_bytes_per_round(K=1, P_=10, d=1026) == 80
    assert H.fedavg_bytes_per_round(1026) == 4104
