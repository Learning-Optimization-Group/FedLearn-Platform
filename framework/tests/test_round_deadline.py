"""The server reports each round's real deadline, not one that moves with every status poll.

GetServerStatus used to report now + round_timeout_s on every call, so a client's countdown never advanced (a phone
showed "closes in 14m 59s" for minutes). The deadline is the round's start plus its timeout, which is what the
coordinator enforces.
"""
from __future__ import annotations

import time
from collections import OrderedDict

import torch

from fedlearn.communication.generated import fedlearn_pb2
from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.decomfl_strategy import DeComFL
from fedlearn.server.grpc_servicer import FederatedLearningServiceServicer

TIMEOUT_S = 900


def _coordinator() -> FLCoordinator:
    init = OrderedDict(w=torch.zeros(4))
    strategy = DeComFL(init, evaluate_fn=None, min_fit_clients=1, clients_per_round=1, num_local_steps=1,
                       num_perturbations=1, learning_rate=0.01, smoothing_param=0.01, seed=42)
    return FLCoordinator(strategy, min_clients_for_aggregation=1, clients_per_round=1, round_timeout_s=TIMEOUT_S)


def test_the_deadline_is_the_rounds_start_plus_its_timeout():
    coordinator = _coordinator()
    before = time.time()
    coordinator.start_round()
    after = time.time()
    deadline_s = coordinator.round_deadline_unix_ms() / 1000
    assert before + TIMEOUT_S - 0.002 <= deadline_s <= after + TIMEOUT_S + 0.002


def test_the_deadline_does_not_move_between_polls():
    coordinator = _coordinator()
    coordinator.start_round()
    first = coordinator.round_deadline_unix_ms()
    time.sleep(0.05)
    assert coordinator.round_deadline_unix_ms() == first


def test_a_new_round_has_a_new_deadline():
    coordinator = _coordinator()
    coordinator.start_round()
    first = coordinator.round_deadline_unix_ms()
    time.sleep(0.05)
    coordinator.start_round()
    assert coordinator.round_deadline_unix_ms() >= first + 40


def test_server_status_reports_the_coordinators_deadline():
    coordinator = _coordinator()
    coordinator.start_round()
    servicer = FederatedLearningServiceServicer(coordinator)
    time.sleep(0.05)
    status = servicer.GetServerStatus(fedlearn_pb2.GetServerStatusRequest(), None)
    assert status.round_deadline_unix_ms == coordinator.round_deadline_unix_ms()
