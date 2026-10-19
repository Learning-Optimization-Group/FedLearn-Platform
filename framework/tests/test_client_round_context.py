"""The streamed round must reach local training's seeded batching and dropout."""
from importlib import import_module
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("config", [{}, {"server_round": 999, "learning_rate": "0.001"}])
def test_fit_receives_the_streamed_round_without_mutating_server_config(monkeypatch, config):
    runtime = import_module("fedlearn.client.client")
    original = dict(config)
    transport = Mock()
    transport.register.return_value = True
    transport.should_stop_training.return_value = False
    transport.get_global_model.side_effect = [({}, 7, config), ({}, -1, {})]
    transport.submit_update.return_value = True
    monkeypatch.setattr(runtime, "GrpcClient", lambda **kw: transport)
    participant = Mock()
    participant.fit.return_value = ({}, 8)

    runtime.start_client("localhost:50000", participant, "client")

    received = participant.fit.call_args.args[1]
    assert received["server_round"] == 7
    assert config == original
    transport.submit_update.assert_called_once_with({}, 8, 7)
