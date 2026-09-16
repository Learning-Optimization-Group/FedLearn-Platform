"""DeComFL server history must not grow without bound.

`seed_history` and `gradient_history` are keyed by round and were never pruned, so a server
accumulated K*P seeds + K*P floats for every round it ever ran — O(rounds) memory in an algorithm
whose entire selling point is O(1) communication. The production DeComFL runs go to 20,000 rounds.

The retention rule is the one that is provably safe: a client synced through round L needs rounds
L+1 onward (see `get_rebuild_history`), so every round at or below `min(client_last_round)` is
dead weight and can be dropped. Nothing a client can still legitimately ask for is ever discarded
— which is what separates this from a fixed-size ring buffer.

A client that stops participating pins that floor and blocks pruning. That is correct behaviour
(its rebuild chain must survive), but it is also worth saying out loud, so it warns rather than
growing silently.
"""

from collections import OrderedDict

import logging

import torch

from fedlearn.server.decomfl_strategy import DeComFL


K, P, ETA = 1, 2, 0.01


def _strategy(d: int = 8) -> DeComFL:
    return DeComFL(
        initial_parameters=OrderedDict([("w", torch.zeros(d))]),
        min_fit_clients=1,
        num_local_steps=K,
        num_perturbations=P,
        learning_rate=ETA,
    )


def _scalars() -> list:
    return [[0.1 * (p + 1) for p in range(P)] for _ in range(K)]


def _run_round(strat: DeComFL, rnd: int, client_ids) -> None:
    """One full server round, in the order the coordinator actually does it."""
    strat.get_or_create_seeds(rnd)
    results = [(cid, _scalars(), 10) for cid in client_ids]
    strat.aggregate_fit(rnd, results)
    # the coordinator writes gradient_history AFTER aggregate_fit returns
    strat.gradient_history[rnd] = _scalars()


def test_history_does_not_grow_with_rounds_when_all_clients_participate():
    strat = _strategy()
    for r in range(1, 41):
        _run_round(strat, r, ["A", "B"])

    assert len(strat.seed_history) <= 2, (
        f"seed_history holds {len(strat.seed_history)} rounds after 40 rounds with every client "
        f"fully synced; rounds at or below min(client_last_round) are unreachable and must be "
        f"pruned. Unpruned history is O(rounds) memory."
    )
    assert len(strat.gradient_history) <= 2, (
        f"gradient_history holds {len(strat.gradient_history)} rounds after 40 rounds"
    )


def test_pruning_never_discards_a_round_a_lagging_client_still_needs():
    """The safety property. B stops participating after round 1, so its rebuild chain from round 2
    onward must survive intact — pruning must not turn a legitimate rejoin into a torn history."""
    strat = _strategy()
    _run_round(strat, 1, ["A", "B"])
    for r in range(2, 16):
        _run_round(strat, r, ["A"])  # B is gone

    # B is synced through round 0, so it must still be handed rounds 1..15.
    history = strat.get_rebuild_history("B", current_round=16)
    assert [h["round_number"] for h in history] == list(range(1, 16)), (
        "a lagging client's rebuild chain was pruned away"
    )
    for h in history:
        assert h["seeds"] is not None and h["gradients"] is not None


def test_fully_synced_client_still_rebuilds_correctly_after_pruning():
    strat = _strategy()
    for r in range(1, 21):
        _run_round(strat, r, ["A"])
    # A is synced through round 19; at round 21 it owes exactly round 20.
    assert [h["round_number"] for h in strat.get_rebuild_history("A", 21)] == [20]


def test_warns_when_an_absent_client_pins_the_history_floor(caplog):
    strat = _strategy()
    _run_round(strat, 1, ["A", "B"])
    with caplog.at_level(logging.WARNING, logger="fedlearn.server.decomfl_strategy"):
        for r in range(2, 40):
            _run_round(strat, r, ["A"])

    pinned = [rec for rec in caplog.records if "pinn" in rec.getMessage().lower()]
    assert pinned, (
        "an absent client pinned the history floor for 38 rounds and nothing was logged; "
        "unbounded growth must be attributable, not silent"
    )
    assert "B" in pinned[-1].getMessage(), "the warning should name the client pinning the floor"


def test_pruning_is_a_noop_before_any_client_has_been_seen():
    strat = _strategy()
    strat.get_or_create_seeds(1)
    assert 1 in strat.seed_history
    strat.aggregate_fit(1, [])  # no results -> no client watermarks
    assert 1 in strat.seed_history, "the in-flight round must never be pruned"
