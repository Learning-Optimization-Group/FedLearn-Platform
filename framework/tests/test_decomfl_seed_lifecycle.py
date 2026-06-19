"""Regression tests for the DeComFL seed lifecycle (audit bugs #28 / #29).

The invariant under test: for a given round, EVERY client must receive the
*identical* seed set (so their zeroth-order g-scalars are computed against the
same perturbation direction z and are therefore summable on the server), and
the server's seed_history must hold exactly ONE seed set per round, indexable by
the (1-based) round number that aggregate_fit/get_rebuild_history use.

Before the fix, grpc_servicer.GetDeComFLConfig regenerated fresh seeds and
appended them to a LIST on every per-client RPC -> N different seed sets per
round + off-by-one indexing. These tests fail on that code and pass after the
generate-once-per-round, dict-keyed-by-round fix.
"""
import threading
from collections import OrderedDict

import torch

from fedlearn.server.decomfl_strategy import DeComFL


def _make(K=2, P=3):
    return DeComFL(
        initial_parameters=OrderedDict({"w": torch.zeros(8)}),
        num_local_steps=K,
        num_perturbations=P,
        seed=123,
    )


def test_all_clients_in_a_round_get_identical_seeds():
    s = _make()
    seeds_client_a = s.get_or_create_seeds(1)
    seeds_client_b = s.get_or_create_seeds(1)
    assert seeds_client_a == seeds_client_b, (
        "All clients in a round MUST get identical seeds — the shared-perturbation "
        "invariant of DeComFL. Different seeds make g-scalars non-summable."
    )


def test_seed_history_is_one_entry_per_round_keyed_by_round():
    s = _make()
    # three clients request the same round, then one client requests the next round
    s.get_or_create_seeds(1)
    s.get_or_create_seeds(1)
    s.get_or_create_seeds(1)
    s.get_or_create_seeds(2)
    assert set(s.seed_history.keys()) == {1, 2}, (
        "seed_history must have exactly one entry per round (not N-per-round)"
    )
    # indexable by the round number aggregate_fit() uses
    assert s.seed_history[1] == s.get_or_create_seeds(1)
    assert s.seed_history[1] != s.seed_history[2]


def test_seed_shape_is_K_by_P():
    s = _make(K=2, P=3)
    seeds = s.get_or_create_seeds(5)
    assert len(seeds) == 2
    assert all(len(step) == 3 for step in seeds)


def test_concurrent_same_round_requests_are_consistent():
    s = _make()
    out = []

    def worker():
        out.append(s.get_or_create_seeds(7))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(o == out[0] for o in out), (
        "concurrent client requests for the same round must still get identical seeds"
    )
    assert list(s.seed_history.keys()).count(7) == 1
