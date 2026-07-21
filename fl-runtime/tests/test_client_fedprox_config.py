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


def test_real_fedopt_server_config_is_accepted_by_client():
    """The client must accept the EXACT config FedOpt.get_client_config() ships. FedOpt does its
    adaptive step SERVER-side and sends ``proximal_mu='0.0'`` (i.e. NO proximal term) — the client
    guard must not reject that. Regression: ``'0.0' is not None`` tripped the FedProx refusal and
    crashed every FedOpt run at round 1. Bind to the real server output so a drift on either side
    (server stops shipping '0.0', or the guard tightens again) fails here."""
    from collections import OrderedDict

    import torch

    from fedlearn.server.strategy import FedOpt

    cfg = FedOpt(initial_parameters=OrderedDict([("w", torch.zeros(2))])).get_client_config()
    assert cfg.get("proximal_mu") == "0.0", "test premise: FedOpt ships proximal_mu='0.0'"
    client._assert_strategy_honored(cfg)  # must NOT raise


def test_nonzero_proximal_mu_is_refused_but_zero_is_allowed():
    """The distinction is mu>0 (a real FedProx proximal term this client cannot honor) vs mu==0
    (no term). Reject only the former; a literal '0.0'/'0' means FedAvg-equivalent locally."""
    with pytest.raises(NotImplementedError, match="proximal"):
        client._assert_strategy_honored({"proximal_mu": "0.5"})
    client._assert_strategy_honored({"proximal_mu": "0"})    # no raise
    client._assert_strategy_honored({"proximal_mu": "0.0"})  # no raise
