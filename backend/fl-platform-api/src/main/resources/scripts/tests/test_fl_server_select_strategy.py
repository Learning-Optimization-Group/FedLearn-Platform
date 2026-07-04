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


# --- SE-11: DP ε-budget passthrough ------------------------------------------------------------


def test_select_strategy_passes_dp_epsilon_budget_to_fedlora():
    """The four ε-budget fields reach FedLoRA, which solves z and commits the accounted trace."""
    args = _args("fedlora")
    args.dp_enabled = True
    args.dp_clip_norm = 1.0
    args.dp_target_epsilon = 8.0
    args.dp_delta = 1e-5
    args.dp_num_clients = 10
    args.dp_rounds = 5
    strategy = fl_server.select_strategy(args, _initial_parameters(), None)
    assert isinstance(strategy, FedLoRA)
    assert strategy.dp_enabled is True
    assert strategy.dp_target_epsilon == 8.0
    assert strategy.dp_delta == 1e-5
    assert strategy.dp_num_clients == 10
    assert strategy.dp_rounds == 5
    # FedLoRA owns the ε→z solve: z materialises even though --dp-noise-multiplier was not given,
    # and the accountant's committed ε trace is exposed for the eval card.
    assert strategy.dp_noise_multiplier is not None
    assert strategy.dp_noise_multiplier > 0
    assert strategy.dp_accounted_epsilon is not None
    # q = clients_per_round / N (min_clients=1, N=10)
    assert strategy.dp_q == pytest.approx(0.1)


def test_select_strategy_passes_raw_noise_multiplier_unchanged():
    """The pre-SE-11 raw-z path still works: z taken verbatim, budget fields stay None."""
    args = _args("fedlora")
    args.dp_enabled = True
    args.dp_clip_norm = 0.5
    args.dp_noise_multiplier = 1.1
    strategy = fl_server.select_strategy(args, _initial_parameters(), None)
    assert strategy.dp_enabled is True
    assert strategy.dp_noise_multiplier == 1.1
    assert strategy.dp_target_epsilon is None
    assert strategy.dp_accounted_epsilon is None  # no δ/rounds => no accounted trace


def test_select_strategy_bare_namespace_still_constructs_fedlora_dp_off():
    """Tests build bare Namespaces without any dp_* fields — the getattr guards must hold."""
    strategy = fl_server.select_strategy(_args("fedlora"), _initial_parameters(), None)
    assert isinstance(strategy, FedLoRA)
    assert strategy.dp_enabled is False
    assert strategy.dp_target_epsilon is None
    assert strategy.dp_accounted_epsilon is None


@pytest.mark.parametrize(
    "bad_fields",
    [
        # target ε without δ/rounds — accountant cannot solve z
        {"dp_target_epsilon": 8.0},
        # both z and target ε — ambiguous budget
        {"dp_target_epsilon": 8.0, "dp_delta": 1e-5, "dp_rounds": 5, "dp_noise_multiplier": 1.0},
        # DP on with neither z nor target ε
        {},
    ],
)
def test_select_strategy_bad_dp_config_is_fatal_startup_error(bad_fields):
    """FedLoRA's ValueError surfaces as a fatal startup exit so the backend's 3-second
    spawn window catches a bad DP config instead of it detonating mid-run."""
    args = _args("fedlora")
    args.dp_enabled = True
    args.dp_clip_norm = 1.0
    for k, v in bad_fields.items():
        setattr(args, k, v)
    with pytest.raises(SystemExit) as excinfo:
        fl_server.select_strategy(args, _initial_parameters(), None)
    assert excinfo.value.code == 1
