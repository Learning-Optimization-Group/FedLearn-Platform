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
    s = FederatedLearningServiceServicer(coord, secure_agg_threshold=threshold)
    coord.set_secure_session_provider(s.secure_session_if_present)
    return s


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
