"""SE-11: eval-card builder — the accounted-(ε, δ) DP trace committed to the artifact registry.

The privacy contract under test:
  * DP-on run  => the eval card carries a ``dp`` object whose ``accounted_epsilon`` is the
    strategy's committed value, verbatim (never recomputed, never rounded).
  * DP-off run => the eval card has NO ``dp`` key at all (the backend upload gate treats a
    missing trace on a DP-claimed artifact as a 400; absence == non-DP artifact).
"""
import json
import os
import sys
from argparse import Namespace
from collections import OrderedDict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fl_server  # noqa: E402

from fedlearn.server import FedLoRA  # noqa: E402


def _initial_parameters() -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict(
        [
            ("base_model.model.layer.lora_A.weight", torch.zeros(2, 2)),
            ("base_model.model.layer.lora_B.weight", torch.zeros(2, 2)),
        ]
    )


def _args() -> Namespace:
    """Bare namespace with only the fields build_eval_card reads (mirrors main()'s args)."""
    return Namespace(model_type="LLM_LORA", strategy="fedlora", num_rounds=5)


_HISTORY = [(1, {"loss": 0.9, "accuracy": 40.0}), (5, {"loss": 0.31, "accuracy": 72.5})]


def _dp_strategy(**overrides) -> FedLoRA:
    kwargs = dict(
        initial_parameters=_initial_parameters(),
        min_fit_clients=2,
        dp_enabled=True,
        dp_clip_norm=0.5,
        dp_target_epsilon=8.0,
        dp_delta=1e-5,
        dp_num_clients=10,
        dp_rounds=5,
    )
    kwargs.update(overrides)
    return FedLoRA(**kwargs)


def test_eval_card_base_fields_unchanged():
    card = json.loads(fl_server.build_eval_card(_args(), _HISTORY, None))
    assert card["recipe_key"] == "LLM_LORA"
    assert card["strategy"] == "fedlora"
    assert card["rounds"] == 5
    assert card["final_loss"] == 0.31
    assert card["final_accuracy"] == 72.5
    assert card["framework"] == "fedlearn"


def test_eval_card_dp_off_has_no_dp_key():
    strategy = FedLoRA(initial_parameters=_initial_parameters())  # dp_enabled defaults False
    card = json.loads(fl_server.build_eval_card(_args(), _HISTORY, strategy))
    assert "dp" not in card


def test_eval_card_no_strategy_has_no_dp_key():
    card = json.loads(fl_server.build_eval_card(_args(), _HISTORY, None))
    assert "dp" not in card


def test_eval_card_dp_on_commits_strategy_accounted_trace_verbatim():
    strategy = _dp_strategy()
    card = json.loads(fl_server.build_eval_card(_args(), _HISTORY, strategy))
    dp = card["dp"]
    assert dp["enabled"] is True
    # Precision rule: the strategy's committed value, bit-exact through JSON — no recompute/round.
    assert dp["accounted_epsilon"] == strategy.dp_accounted_epsilon
    assert dp["delta"] == strategy.dp_delta
    assert dp["clip_norm"] == 0.5
    assert dp["noise_multiplier"] == strategy.dp_noise_multiplier
    assert dp["q"] == strategy.dp_q
    assert dp["rounds"] == 5
    assert dp["target_epsilon"] == 8.0


def test_eval_card_dp_raw_z_without_accounting_emits_null_accounted_epsilon():
    """Raw-z path with no δ/rounds: enabled=true + accounted_epsilon=null. The backend gate
    rejects such an upload by design — the platform refuses unaccounted DP claims."""
    strategy = _dp_strategy(
        dp_target_epsilon=None, dp_delta=None, dp_num_clients=None, dp_rounds=None,
        dp_noise_multiplier=1.1,
    )
    card = json.loads(fl_server.build_eval_card(_args(), _HISTORY, strategy))
    dp = card["dp"]
    assert dp["enabled"] is True
    assert dp["accounted_epsilon"] is None
    assert dp["target_epsilon"] is None
    assert dp["noise_multiplier"] == 1.1


def test_eval_card_empty_history_yields_null_finals():
    card = json.loads(fl_server.build_eval_card(_args(), [], None))
    assert card["final_loss"] is None
    assert card["final_accuracy"] is None


# --- CLI contract: the backend spawner emits exactly these flag names --------------------------


def test_arg_parser_accepts_dp_epsilon_budget_flags():
    parser = fl_server.build_arg_parser()
    args = parser.parse_args(
        [
            "--model-path", "/tmp/m.npz",
            "--project-id", "p1",
            "--model-type", "LLM_LORA",
            "--model-name", "qwen",
            "--strategy", "fedlora",
            "--dp-enabled",
            "--dp-clip-norm", "0.5",
            "--dp-target-epsilon", "8.0",
            "--dp-delta", "1e-5",
            "--dp-num-clients", "10",
            "--dp-rounds", "5",
        ]
    )
    assert args.dp_enabled is True
    assert args.dp_clip_norm == 0.5
    assert args.dp_target_epsilon == 8.0
    assert args.dp_delta == 1e-5
    assert args.dp_num_clients == 10
    assert args.dp_rounds == 5


def test_arg_parser_dp_budget_flags_default_to_none():
    parser = fl_server.build_arg_parser()
    args = parser.parse_args(
        [
            "--model-path", "/tmp/m.npz",
            "--project-id", "p1",
            "--model-type", "CNN",
            "--model-name", "cnn",
        ]
    )
    assert args.dp_target_epsilon is None
    assert args.dp_delta is None
    assert args.dp_num_clients is None
    assert args.dp_rounds is None
