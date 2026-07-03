import os
import sys
from argparse import Namespace
from collections import OrderedDict

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fl_server  # noqa: E402

import fedlearn as fl  # noqa: E402
from fedlearn.server import (  # noqa: E402
    DeComFL,
    FedLoRA,
    FedProx,
    FedOpt,
    RobustAggregator,
)


def _args(strategy: str) -> Namespace:
    """Minimal args namespace carrying only the fields select_strategy reads."""
    return Namespace(
        strategy=strategy,
        min_clients=1,
        dataset="cb",          # non-ecg => decomfl uses the 'default' decomfl config
        aggregation="FFA_LORA",
    )


def _initial_parameters() -> "OrderedDict[str, torch.Tensor]":
    """A tiny state dict that also carries a lora_A key so FedLoRA's default FFA_LORA
    mode (which requires the frozen shared A) can be constructed."""
    return OrderedDict(
        [
            ("layer.weight", torch.zeros(2, 2)),
            ("base_model.model.layer.lora_A.weight", torch.zeros(2, 2)),
            ("base_model.model.layer.lora_B.weight", torch.zeros(2, 2)),
        ]
    )


@pytest.mark.parametrize(
    "strategy_name,expected_cls",
    [
        ("decomfl", DeComFL),
        ("fedlora", FedLoRA),
        ("fedprox", FedProx),
        ("fedopt", FedOpt),
        ("robust", RobustAggregator),
        ("fedavg", fl.FedAvg),
        ("FedAvg", fl.FedAvg),          # case-insensitive
        ("does-not-exist", fl.FedAvg),  # unrecognized => FedAvg fallback (unchanged behavior)
    ],
)
def test_select_strategy_maps_name_to_class(strategy_name, expected_cls):
    strategy = fl_server.select_strategy(
        _args(strategy_name), _initial_parameters(), None
    )
    assert isinstance(strategy, expected_cls)


def test_select_strategy_passes_through_min_fit_clients():
    args = _args("fedprox")
    args.min_clients = 3
    strategy = fl_server.select_strategy(args, _initial_parameters(), None)
    assert strategy.min_fit_clients == 3


def test_select_strategy_wires_evaluate_fn():
    sentinel = object()
    strategy = fl_server.select_strategy(_args("robust"), _initial_parameters(), sentinel)
    assert strategy.evaluate_fn is sentinel
