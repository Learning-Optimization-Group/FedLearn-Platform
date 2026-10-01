"""The research-only per-client update recorder.

A multi-device run could only be checked through the aggregate, so a larger per-client disagreement between
compute backends could average out unseen. With FEDLEARN_RECORD_CLIENT_UPDATES set, the coordinator writes every
DeComFL update it accepts, exactly as it will aggregate it, to a JSONL file. Recording individual updates is a
privacy regression, so it is off by default and refused outright on a run with secure aggregation or central DP.
"""
import json
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest
import torch

from fedlearn.server.coordinator import FLCoordinator
from fedlearn.server.server import ServerConfig, build_coordinator
from fedlearn.server.strategy import Strategy
from fedlearn.server.update_recorder import ClientUpdateRecorder, update_recorder_from_env

ENV = "FEDLEARN_RECORD_CLIENT_UPDATES"


def _coordinator(tmp_path, clients=3):
    strategy = MagicMock(spec=Strategy)
    c = FLCoordinator(strategy=strategy, min_clients_for_aggregation=clients, clients_per_round=clients)
    c.set_initial_parameters(OrderedDict([("w", torch.tensor([0.0]))]))
    path = tmp_path / "updates.jsonl"
    c.update_recorder = ClientUpdateRecorder(path)
    return c, path


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_each_accepted_decomfl_update_is_recorded_as_one_line(tmp_path):
    c, path = _coordinator(tmp_path)
    c.submit_decomfl_update("mac", [[0.25, -0.5]], 8, trained_on_round=1)
    c.submit_decomfl_update("phone", [[0.125, 1.0]], 8, trained_on_round=1)

    assert _lines(path) == [
        {"round": 1, "client_id": "mac", "num_examples": 8, "gradient_scalars": [[0.25, -0.5]]},
        {"round": 1, "client_id": "phone", "num_examples": 8, "gradient_scalars": [[0.125, 1.0]]},
    ]


def test_updates_the_coordinator_does_not_accept_are_not_recorded(tmp_path):
    c, path = _coordinator(tmp_path)
    c.submit_decomfl_update("mac", [[0.25]], 8, trained_on_round=1)
    c.submit_decomfl_update("mac", [[0.75]], 8, trained_on_round=1)             # duplicate
    c.submit_decomfl_update("stale", [[0.5]], 8, trained_on_round=0)            # stale round
    c.submit_decomfl_update("bad", [[float("nan")]], 8, trained_on_round=1)     # non-finite

    assert [(r["client_id"], r["gradient_scalars"]) for r in _lines(path)] == [("mac", [[0.25]])]


def test_the_record_holds_the_value_aggregated_after_the_clamp(tmp_path):
    c, path = _coordinator(tmp_path)
    c.submit_decomfl_update("big", [[5e6]], 8, trained_on_round=1)

    assert _lines(path)[0]["gradient_scalars"] == [[c.grad_clip_threshold]]


def test_without_the_env_var_nothing_is_recorded(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert update_recorder_from_env(secure_aggregation=False, dp_enabled=False) is None


def test_the_env_var_turns_recording_on(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path / "u.jsonl"))
    recorder = update_recorder_from_env(secure_aggregation=False, dp_enabled=False)
    assert isinstance(recorder, ClientUpdateRecorder)


@pytest.mark.parametrize("secure_aggregation, dp_enabled", [(True, False), (False, True), (True, True)])
def test_recording_is_refused_on_a_private_run(monkeypatch, tmp_path, secure_aggregation, dp_enabled):
    monkeypatch.setenv(ENV, str(tmp_path / "u.jsonl"))
    with pytest.raises(ValueError, match="individual client updates"):
        update_recorder_from_env(secure_aggregation=secure_aggregation, dp_enabled=dp_enabled)
    assert not (tmp_path / "u.jsonl").exists()


def test_the_run_coordinator_records_when_asked_and_refuses_before_serving(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path / "u.jsonl"))
    strategy = MagicMock(spec=Strategy)
    strategy.min_fit_clients = 1
    strategy.clients_per_round = 1
    strategy.dp_enabled = False
    assert isinstance(build_coordinator(strategy, ServerConfig()).update_recorder, ClientUpdateRecorder)

    with pytest.raises(ValueError):
        build_coordinator(strategy, ServerConfig(secure_aggregation=True))


def test_by_default_the_run_coordinator_records_nothing(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    strategy = MagicMock(spec=Strategy)
    strategy.min_fit_clients = 1
    strategy.clients_per_round = 1
    assert build_coordinator(strategy, ServerConfig()).update_recorder is None
