"""P2-2 — resolving a secure round whose cohort will not complete.

The all-present trigger (every key-publisher submitted) is the fast path. A dropout never
reaches that count, so without a deadline the round freezes forever -- which would make
LightSecAgg's headline property, dropout resilience, unreachable in deployment.

These tests drive the MECHANISM directly, the way the simulator does, so no test sleeps out a
120s deadline. The wall-clock POLICY stays in _handle_round_timeout.
"""
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest
import torch

from fedlearn.client.secure_agg_client import FrozenSurvivors, SecureAggregationClient
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.decomfl_strategy import DeComFL
from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer
from tests.test_secure_agg_client import _DirectStub


def _servicer(threshold=2, K=1, P=2, clients_per_round=3, min_clients=2):
    strategy = DeComFL(
        initial_parameters=OrderedDict({"w": torch.zeros(4)}),
        num_local_steps=K, num_perturbations=P,
    )
    coord = FLCoordinator(
        strategy,
        min_clients_for_aggregation=min_clients,
        clients_per_round=clients_per_round,
    )
    coord.bind_or_check_identity = MagicMock(return_value=True)
    strategy.get_or_create_seeds(coord.current_round)
    # NOT registering the provider by hand: the servicer does it at construction, and these
    # tests are only meaningful if they run against that same wiring.
    return FederatedLearningServiceServicer(coord, secure_agg_threshold=threshold)


def _mask_in(servicer, clients, values, round_num):
    session = servicer._secure_session(round_num)
    for p, v in values.items():
        session.submit_masked(partition=p, elements=clients[p].mask(v))
    return session


def _cohort(servicer, partitions, round_num, threshold, n):
    clients = {
        p: SecureAggregationClient(_DirectStub(servicer, p), client_id=f"c{p}")
        for p in partitions
    }
    for p in partitions:
        clients[p].begin_round(round_num=round_num, threshold=threshold, num_scalars=2,
                               cohort_size=n)
    for p in partitions:
        clients[p].distribute_shares(round_num=round_num)
    for p in partitions:
        clients[p].collect_shares(round_num=round_num)
    return clients


def test_a_deadline_freezes_an_open_secure_round_instead_of_force_aggregating():
    """The plaintext deadline force-aggregates whatever arrived. A secure round cannot: holders
    have not produced summed shares yet, so there is nothing to recover from. The deadline's job
    here is to FREEZE the set, which is what unblocks the holders."""
    servicer = _servicer()
    coord = servicer.coordinator
    clients = _cohort(servicer, [1, 2, 3], round_num=1, threshold=2, n=3)
    session = _mask_in(servicer, clients, {1: [0.4, -0.2], 2: [0.1, 0.3]}, round_num=1)
    assert not session.is_closed

    coord.resolve_round_incomplete("deadline")

    assert session.is_closed, "the deadline did not freeze the set; holders stay blocked"
    assert session.survivors == [1, 2], "the dropped client was admitted"
    assert coord.current_round == 1, "the round advanced before any summed share arrived"
    assert not coord.stop_requested


def test_after_the_freeze_the_surviving_holders_complete_the_round():
    """The end the freeze exists to serve: a real dropout completing."""
    servicer = _servicer()
    coord = servicer.coordinator
    strategy = coord.strategy
    before = strategy.global_params_flat.clone()

    clients = _cohort(servicer, [1, 2, 3], round_num=1, threshold=2, n=3)
    session = _mask_in(servicer, clients, {1: [0.4, -0.2], 2: [0.1, 0.3]}, round_num=1)
    coord.resolve_round_incomplete("deadline")

    frozen = FrozenSurvivors(tuple(session.survivors))
    for p in session.survivors:
        clients[p].finish_round(round_num=1, survivors=frozen)

    assert not torch.equal(strategy.global_params_flat, before)
    assert coord.current_round == 2
    assert 1 in strategy.gradient_history


def test_a_second_deadline_on_a_frozen_round_that_cannot_recover_stops_the_run():
    """Freezing is not a cure. If the holders never return enough summed shares, the round is
    unrecoverable and must fail loudly rather than hang -- the same call that force-aggregates a
    plaintext round has to reach a terminal state here too."""
    servicer = _servicer()
    coord = servicer.coordinator
    clients = _cohort(servicer, [1, 2, 3], round_num=1, threshold=2, n=3)
    session = _mask_in(servicer, clients, {1: [0.4, -0.2], 2: [0.1, 0.3]}, round_num=1)

    coord.resolve_round_incomplete("deadline")      # freezes
    assert not coord.stop_requested
    coord.resolve_round_incomplete("deadline")      # still no summed shares

    assert coord.stop_requested, "an unrecoverable secure round hung instead of failing"
    assert coord.last_round_failed
    assert "summed share" in (coord.last_round_message or "")


def test_a_secure_round_below_min_clients_stops_rather_than_aggregating_a_thin_cohort():
    """Mirrors the plaintext rule. One survivor is not a federation, and a threshold of 2 could
    not reconstruct anyway."""
    servicer = _servicer(min_clients=2)
    coord = servicer.coordinator
    clients = _cohort(servicer, [1, 2, 3], round_num=1, threshold=2, n=3)
    _mask_in(servicer, clients, {1: [0.4, -0.2]}, round_num=1)

    coord.resolve_round_incomplete("deadline")

    assert coord.stop_requested
    assert coord.last_round_failed


def test_a_round_with_no_secure_session_still_takes_the_plaintext_path():
    """The secure branch must not capture a plaintext deployment."""
    servicer = _servicer()
    coord = servicer.coordinator
    # min_clients=2, so two updates make the round force-aggregatable rather than a stop.
    coord.submit_decomfl_update("c1", [[0.1, 0.2]], 10, coord.current_round)
    coord.submit_decomfl_update("c2", [[0.2, 0.1]], 10, coord.current_round)

    coord.resolve_round_incomplete("deadline")

    assert coord.current_round == 2, "the plaintext force-aggregation did not run"


def test_constructing_the_servicer_registers_the_provider_with_the_coordinator():
    """Otherwise every test above passes on a wiring the deployment never performs.

    The helper in this file registers the provider by hand. Production constructs the servicer
    and nothing else, so if registration does not happen there, the deadline branch is dead code
    and a real dropout still hangs -- with a green suite.
    """
    strategy = DeComFL(
        initial_parameters=OrderedDict({"w": torch.zeros(4)}),
        num_local_steps=1, num_perturbations=2,
    )
    coord = FLCoordinator(strategy, min_clients_for_aggregation=2, clients_per_round=3)
    coord.bind_or_check_identity = MagicMock(return_value=True)

    servicer = FederatedLearningServiceServicer(coord)   # nothing else

    assert coord._current_secure_session() is None, "asking created a session"
    created = servicer._secure_session(coord.current_round)
    assert coord._current_secure_session() is created, (
        "the coordinator cannot reach the round's secure session; the deadline branch is dead"
    )


def test_a_round_that_could_be_recovered_but_errored_is_not_blamed_on_missing_shares():
    """Distinguish "the holders never came back" from "recovery was possible and failed".

    Found live: a server-supplied evaluate_fn that returned None instead of (loss, metrics) made
    complete_secure_decomfl_round raise. The servicer swallows that -- correctly, since the
    holder's share WAS accepted and a retry would not help -- so the round then sat until its
    deadline and was reported as

        frozen at 3 survivor(s) but only 0/2 summed share(s) returned

    which is false and sends an operator looking at the network instead of at their callback.
    """
    servicer = _servicer()
    coord = servicer.coordinator
    clients = _cohort(servicer, [1, 2, 3], round_num=1, threshold=2, n=3)
    session = _mask_in(servicer, clients, {1: [0.4, -0.2], 2: [0.1, 0.3]}, round_num=1)
    coord.resolve_round_incomplete("deadline")          # freeze

    # Exactly the live failure: a server-supplied callback that blows up during completion.
    coord.strategy.evaluate = MagicMock(
        side_effect=TypeError("cannot unpack non-iterable NoneType object")
    )

    # Both holders return summed shares, so the round IS recoverable...
    frozen = FrozenSurvivors(tuple(session.survivors))
    for p in session.survivors:
        clients[p].finish_round(round_num=1, survivors=frozen)
    assert session.ready(), "the round should be recoverable"
    assert coord.current_round == 1, "completion should have failed, leaving the round open"

    # ...so when the next deadline lands, the diagnosis must not blame the holders.
    coord.resolve_round_incomplete("deadline")

    assert coord.stop_requested
    assert "summed share" not in (coord.last_round_message or ""), (
        "an errored recovery was reported as missing shares"
    )
    assert "recover" in (coord.last_round_message or "").lower()


def test_a_late_summed_share_cannot_complete_a_DIFFERENT_round():
    """Found by the live run: rounds 1-3 all reported "completing", but only rounds 1 and 2 ever
    received a masked submission.

    A holder's summed share for round r can arrive after the server has already moved to r+1 --
    the client polls on its own schedule. complete_secure_decomfl_round took the session it was
    handed but read the round number off the COORDINATOR, so round r's recovered scalars were
    applied as round r+1's update, against round r+1's perturbation seeds. The result is a
    well-formed model that is simply wrong, and nothing downstream can detect it.

    The round-complete event does not protect against this: start_round clears it at the top of
    every round, which is exactly when the late share lands.
    """
    servicer = _servicer()
    coord = servicer.coordinator
    strategy = coord.strategy
    round_one = coord.current_round

    clients = _cohort(servicer, [1, 2, 3], round_num=round_one, threshold=2, n=3)
    session = _mask_in(
        servicer, clients,
        {1: [0.4, -0.2], 2: [0.1, 0.3], 3: [-0.2, 0.1]}, round_num=round_one,
    )
    session.close_submissions()
    frozen = FrozenSurvivors(tuple(session.survivors))
    for p in session.survivors[:2]:
        clients[p].finish_round(round_num=round_one, survivors=frozen)
    assert coord.current_round == round_one + 1, "round one did not complete"

    # The server opens the next round; the third holder's share for round ONE arrives now.
    coord.start_round()
    before = strategy.global_params_flat.clone()
    strategy.get_or_create_seeds(coord.current_round)
    clients[session.survivors[2]].finish_round(round_num=round_one, survivors=frozen)

    assert torch.equal(strategy.global_params_flat, before), (
        "a stale round's scalars were applied as the current round's update"
    )
    assert coord.current_round == round_one + 1, "a stale share advanced the round"
