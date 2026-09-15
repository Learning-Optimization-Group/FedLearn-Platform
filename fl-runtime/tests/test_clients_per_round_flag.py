"""--clients-per-round: the round size fl_server.py hands the coordinator.

A round completes as soon as ``clients_per_round`` updates arrive, and the deadline resolves it with as few as
``min_clients``. fl_server.py used to pass only ``--min-clients``, so the two were always equal: every round
needed every client, and one client missing one round stopped the run.
"""
import logging
import os
import sys
from argparse import Namespace
from collections import OrderedDict

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fl_server  # noqa: E402

STRATEGIES = ["fedavg", "decomfl", "fedlora", "fedprox", "fedopt", "robust"]


def _parse(*extra):
    base = ["--project-id", "p1", "--model-type", "MLP", "--strategy", "DeComFL",
            "--model-path", "/tmp/m.pt", "--model-name", "m"]
    return fl_server.build_arg_parser().parse_args(base + list(extra))


def _args(strategy, min_clients=3, clients_per_round=None, **extra):
    return Namespace(strategy=strategy, min_clients=min_clients, clients_per_round=clients_per_round,
                     dataset="cb", aggregation="FFA_LORA", **extra)


def _initial_parameters():
    return OrderedDict([
        ("layer.weight", torch.zeros(2, 2)),
        ("base_model.model.layer.lora_A.weight", torch.zeros(2, 2)),
        ("base_model.model.layer.lora_B.weight", torch.zeros(2, 2)),
    ])


def _dp_args(clients_per_round):
    return _args("fedlora", min_clients=2, clients_per_round=clients_per_round, dp_enabled=True,
                 dp_clip_norm=1.0, dp_target_epsilon=4.0, dp_delta=1e-5, dp_num_clients=2,
                 dp_rounds=5, num_rounds=5)


def test_the_flag_is_unset_by_default():
    assert _parse().clients_per_round is None


def test_the_flag_is_settable():
    assert _parse("--clients-per-round", "5").clients_per_round == 5


@pytest.mark.parametrize("name", STRATEGIES)
def test_every_strategy_waits_for_the_round_size_and_keeps_the_minimum(name):
    strategy = fl_server.select_strategy(_args(name, clients_per_round=5), _initial_parameters(), None)
    assert strategy.clients_per_round == 5
    assert strategy.min_fit_clients == 3


@pytest.mark.parametrize("name", STRATEGIES)
def test_without_the_flag_the_round_size_is_the_minimum_as_before(name):
    strategy = fl_server.select_strategy(_args(name), _initial_parameters(), None)
    assert strategy.clients_per_round == 3


def test_a_namespace_without_the_field_still_works():
    """select_strategy is also called with hand-built namespaces that predate the flag."""
    args = Namespace(strategy="decomfl", min_clients=2, dataset="cb", aggregation="FFA_LORA")
    assert fl_server.select_strategy(args, _initial_parameters(), None).clients_per_round == 2


def test_a_round_size_below_the_minimum_is_refused():
    """Rounds would complete with fewer updates than min_clients allows, so the minimum would mean nothing."""
    with pytest.raises(ValueError, match="clients-per-round"):
        fl_server.select_strategy(_args("decomfl", clients_per_round=2), _initial_parameters(), None)


def test_dp_with_the_round_size_equal_to_the_minimum_still_constructs():
    """The control for the refusal below: the same DP config is valid when the two are equal."""
    strategy = fl_server.select_strategy(_dp_args(2), _initial_parameters(), None)
    assert strategy.clients_per_round == 2


def test_dp_refuses_a_round_size_above_the_minimum(caplog):
    """The accountant assumes every round aggregates the whole cohort (q = 1). A round allowed to finish short
    would release an average over fewer clients than the budget was computed for."""
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit):
        fl_server.select_strategy(_dp_args(4), _initial_parameters(), None)
    assert "clients-per-round" in caplog.text
