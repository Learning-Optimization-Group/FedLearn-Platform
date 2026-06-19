import time
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest
import torch

from fedlearn.server.coordinator import (
    DEFAULT_ROUND_TIMEOUT_S,
    FLCoordinator,
    _round_timeout_from_env,
)
from fedlearn.server.strategy import Strategy


def make_params(val: float) -> OrderedDict:
    return OrderedDict([("w", torch.tensor([val]))])


def make_mock_strategy():
    strategy = MagicMock(spec=Strategy)
    strategy.aggregate_fit.return_value = make_params(1.0)
    strategy.evaluate.return_value = (0.5, {"accuracy": 0.9})
    return strategy


def make_coordinator(round_timeout_s, min_clients, clients_per_round=2):
    strategy = make_mock_strategy()
    coord = FLCoordinator(
        strategy=strategy,
        min_clients_for_aggregation=min_clients,
        clients_per_round=clients_per_round,
        round_timeout_s=round_timeout_s,
    )
    coord.set_initial_parameters(make_params(0.0))
    return coord


# --- config resolution ----------------------------------------------------

def test_constructor_arg_takes_precedence_over_env(monkeypatch):
    monkeypatch.setenv("FEDLEARN_ROUND_TIMEOUT_S", "999")
    coord = make_coordinator(round_timeout_s=0.5, min_clients=1)
    assert coord.round_timeout_s == 0.5


def test_env_overrides_default_when_no_constructor_arg(monkeypatch):
    monkeypatch.setenv("FEDLEARN_ROUND_TIMEOUT_S", "7")
    strategy = make_mock_strategy()
    coord = FLCoordinator(
        strategy=strategy, min_clients_for_aggregation=1, clients_per_round=2
    )
    assert coord.round_timeout_s == 7.0


def test_default_when_no_env_and_no_arg(monkeypatch):
    monkeypatch.delenv("FEDLEARN_ROUND_TIMEOUT_S", raising=False)
    strategy = make_mock_strategy()
    coord = FLCoordinator(
        strategy=strategy, min_clients_for_aggregation=1, clients_per_round=2
    )
    assert coord.round_timeout_s == DEFAULT_ROUND_TIMEOUT_S


def test_bad_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("FEDLEARN_ROUND_TIMEOUT_S", "not-a-number")
    assert _round_timeout_from_env(42.0) == 42.0
    monkeypatch.setenv("FEDLEARN_ROUND_TIMEOUT_S", "-5")
    assert _round_timeout_from_env(42.0) == 42.0


# --- timeout behaviour -----------------------------------------------------

def test_force_aggregates_when_enough_clients_reported():
    # min_clients=1, clients_per_round=2: one client reports, the other drops.
    coord = make_coordinator(round_timeout_s=0.5, min_clients=1, clients_per_round=2)
    coord.start_round()
    coord.submit_client_update("c1", make_params(1.0), 100, trained_on_round=1)
    # Only 1 of 2 -> no aggregation yet.
    assert coord.strategy.aggregate_fit.call_count == 0

    start = time.monotonic()
    coord.wait_for_round_to_complete()
    elapsed = time.monotonic() - start

    # Should not hang: returns shortly after the 0.5s deadline, never near forever.
    assert elapsed < 5.0
    # Force-aggregated the single received client.
    assert coord.strategy.aggregate_fit.call_count == 1
    assert coord.current_round == 2  # round advanced
    assert coord.stop_requested is False
    assert coord.last_round_failed is True
    assert coord.last_round_message is not None
    assert "1/2" in coord.last_round_message


def test_aborts_and_stops_when_not_enough_clients_reported():
    # min_clients=2 but nobody reports -> below minimum -> stop, do not aggregate.
    coord = make_coordinator(round_timeout_s=0.5, min_clients=2, clients_per_round=2)
    coord.start_round()

    start = time.monotonic()
    coord.wait_for_round_to_complete()
    elapsed = time.monotonic() - start

    assert elapsed < 5.0  # did not hang forever
    assert coord.strategy.aggregate_fit.call_count == 0  # no aggregation
    assert coord.stop_requested is True  # server signalled to stop
    assert coord.current_round == 1  # round did NOT advance
    assert coord.last_round_failed is True
    assert coord.last_round_message is not None
    assert "0/2" in coord.last_round_message


def test_common_path_unchanged_when_all_clients_report_in_time():
    # Generous timeout; both clients report immediately -> normal completion.
    coord = make_coordinator(round_timeout_s=60.0, min_clients=2, clients_per_round=2)
    coord.start_round()
    coord.submit_client_update("c1", make_params(1.0), 100, trained_on_round=1)
    coord.submit_client_update("c2", make_params(2.0), 100, trained_on_round=1)

    # Event already set by the normal aggregation path -> returns immediately.
    start = time.monotonic()
    coord.wait_for_round_to_complete()
    elapsed = time.monotonic() - start

    assert elapsed < 1.0
    assert coord.strategy.aggregate_fit.call_count == 1
    assert coord.current_round == 2
    assert coord.stop_requested is False
    assert coord.last_round_failed is False  # untouched on the happy path
