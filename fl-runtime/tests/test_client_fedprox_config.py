"""FR-20: the production client (fl-runtime/client.py, ZOSLClient) must (a) not crash on a
stringified gRPC config, and (b) refuse a FedProx config it cannot honor rather than silently
training plain local steps mislabeled as FedProx.

Root cause (audited 2026-07-17): the gRPC config map is ``map<string,string>``, so
``config["local_epochs"]`` arrives as a string ('1'); ``train()`` then does ``range(epochs)`` and
raises ``TypeError`` at round 1 (crashing every FedProx/FedOpt run). And even without the crash the
client never reads ``proximal_mu`` — it trains with local Adam and applies NO proximal term — so a
'FedProx' run through this client is bit-identical FedAvg. FedProx's correct implementation lives in
the framework ``LocalTrainer``; the production client must fail loud instead of fabricating a result.
"""
import pytest

import client


def test_local_epochs_coerced_from_proto_string():
    # str->str proto map: local_epochs arrives as a string; range() would crash without coercion.
    assert client._coerce_local_epochs({"local_epochs": "3"}, 1) == 3


def test_local_epochs_default_and_int_passthrough():
    assert client._coerce_local_epochs({}, 5) == 5
    assert client._coerce_local_epochs({"local_epochs": 2}, 1) == 2


def test_local_epochs_rejects_non_numeric():
    with pytest.raises(ValueError):
        client._coerce_local_epochs({"local_epochs": "not-a-number"}, 1)


def test_fedprox_config_is_refused_not_silently_run_as_fedavg():
    # proximal_mu present => FedProx. This client applies no proximal term, so running it would
    # fabricate a 'FedProx' result identical to FedAvg. It must refuse loudly.
    with pytest.raises(NotImplementedError, match="proximal"):
        client._assert_strategy_honored({"proximal_mu": "0.1", "local_epochs": "1"})


def test_plain_fedavg_and_fedopt_configs_are_allowed():
    client._assert_strategy_honored({})  # plain FedAvg: no raise
    # FedOpt ships learning_rate/local_epochs but no proximal term — a valid local run, allowed.
    client._assert_strategy_honored({"local_epochs": "1", "learning_rate": "0.01"})
