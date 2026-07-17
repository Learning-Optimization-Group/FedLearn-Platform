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
        # FR-28: an unrecognized name no longer falls back to FedAvg — it fails loud. See
        # test_select_strategy_rejects_an_unknown_strategy_name.
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
    """The four ε-budget fields reach FedLoRA, which solves z and commits the accounted trace.

    FR-24: on a live run the accounting must use q=1 (the orchestrator does not subsample), so
    dp_num_clients equals the cohort (min_clients). dp_rounds >= num_rounds is required (FR-25);
    here num_rounds is unset in the minimal args, so that cross-check is inert.
    """
    args = _args("fedlora")
    args.dp_enabled = True
    args.dp_clip_norm = 1.0
    args.dp_target_epsilon = 8.0
    args.dp_delta = 1e-5
    args.dp_num_clients = 1   # == cohort (min_clients=1) => q=1, the only honest live-run value
    args.dp_rounds = 5
    strategy = fl_server.select_strategy(args, _initial_parameters(), None)
    assert isinstance(strategy, FedLoRA)
    assert strategy.dp_enabled is True
    assert strategy.dp_target_epsilon == 8.0
    assert strategy.dp_delta == 1e-5
    assert strategy.dp_num_clients == 1
    assert strategy.dp_rounds == 5
    # FedLoRA owns the ε→z solve: z materialises even though --dp-noise-multiplier was not given,
    # and the accountant's committed ε trace is exposed for the eval card.
    assert strategy.dp_noise_multiplier is not None
    assert strategy.dp_noise_multiplier > 0
    assert strategy.dp_accounted_epsilon is not None
    # q = clients_per_round / N = 1 / 1 = 1 (no subsampling amplification on a live run).
    assert strategy.dp_q == pytest.approx(1.0)


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


def test_select_strategy_rejects_subsampling_q_below_1_on_a_live_run():
    """FR-24: the orchestrator performs no Poisson client subsampling (it aggregates whichever
    clients submit), so a live run configured with dp_num_clients > the cohort (q<1) would claim a
    subsampling amplification it never realizes — stamping a falsely-low ε on the eval card. The
    live path must refuse it. (Offline analysis can still drive the framework accountant at q<1.)"""
    args = _args("fedlora")
    args.dp_enabled = True
    args.dp_clip_norm = 1.0
    args.dp_target_epsilon = 8.0
    args.dp_delta = 1e-5
    args.dp_num_clients = 100   # >> min_clients (1) => q = 1/100 << 1
    args.dp_rounds = 50
    with pytest.raises(SystemExit) as ei:
        fl_server.select_strategy(args, _initial_parameters(), None)
    assert ei.value.code == 1


def test_select_strategy_rejects_dp_rounds_below_num_rounds():
    """FR-25: the accounted ε is composed over dp_rounds, but the server executes num_rounds (one
    noised release each). If the budget covers FEWER rounds than run, the eval card understates the
    true privacy loss. The live path must refuse dp_rounds < num_rounds."""
    args = _args("fedlora")
    args.dp_enabled = True
    args.dp_clip_norm = 1.0
    args.dp_target_epsilon = 8.0
    args.dp_delta = 1e-5
    args.dp_rounds = 10
    args.num_rounds = 100   # server will run 100 releases but the budget covers only 10
    with pytest.raises(SystemExit) as ei:
        fl_server.select_strategy(args, _initial_parameters(), None)
    assert ei.value.code == 1


def test_select_strategy_rejects_an_unknown_strategy_name():
    """FR-28: an unrecognized --strategy must fail loud, not silently train a DIFFERENT algorithm.

    The old else-branch logged one warning and constructed FedAvg, so a typo (or a factory-style
    name like 'fed_lora') trained plain FedAvg while every strategy-specific flag was silently
    ignored — the opposite of the framework factory's fail-fast contract.
    """
    with pytest.raises(ValueError, match="[Uu]nrecognized|[Uu]nknown|[Ss]upported"):
        fl_server.select_strategy(_args("fed_lora"), _initial_parameters(), None)
    with pytest.raises(ValueError):
        fl_server.select_strategy(_args("does-not-exist"), _initial_parameters(), None)


def test_select_strategy_still_accepts_plain_fedavg():
    """Regression guard: the explicit FedAvg default path must keep working after FR-28."""
    for name in ("fedavg", "FedAvg"):
        strat = fl_server.select_strategy(_args(name), _initial_parameters(), None)
        assert isinstance(strat, fl.FedAvg)
