"""Explicit state adapters for completed-round CPU training recovery."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import random

import numpy as np
import torch

from .strategy import FedAvg, FedOpt, FedProx


@dataclass
class RoundSnapshot:
    completed_round: int
    total_rounds: int
    parameters: OrderedDict
    strategy_type: str
    strategy_config: dict
    strategy_tensors: dict
    history: list
    partition_to_client: dict
    torch_rng_state: torch.Tensor
    python_rng_state: tuple
    numpy_rng_state: tuple


def _clone(values):
    return OrderedDict((key, value.detach().cpu().contiguous().clone()) for key, value in values.items())


def _validate_parameters(values):
    if (not isinstance(values, dict) or not values
            or any(not isinstance(key, str) or not key or not isinstance(value, torch.Tensor)
                   or value.dtype != torch.float32 or not torch.isfinite(value).all()
                   for key, value in values.items())):
        raise ValueError("checkpoint model and optimizer state must contain finite float32 tensors")


def _configuration(strategy):
    if type(strategy) not in (FedAvg, FedProx, FedOpt):
        raise ValueError(f"unsupported checkpoint strategy: {type(strategy).__name__}")
    _validate_parameters(strategy.initial_parameters)
    fields = ["min_fit_clients", "clients_per_round"]
    if type(strategy) is FedProx:
        fields += ["mu", "learning_rate", "local_epochs"]
    if type(strategy) is FedOpt:
        fields += ["eta", "beta1", "beta2", "tau", "variant", "bias_correction",
                   "learning_rate", "local_epochs"]
    result = {name: getattr(strategy, name) for name in fields}
    result["parameter_layout"] = [[name, list(value.shape)]
                                  for name, value in strategy.initial_parameters.items()]
    if type(strategy) is FedOpt:
        result["buffer_names"] = sorted(strategy.buffer_names)
    digest = hashlib.sha256()
    for name, value in strategy.initial_parameters.items():
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    result["initial_parameters_sha256"] = digest.hexdigest()
    return result


def validate_checkpoint_strategy(strategy):
    """Refuse an unqualified strategy/device before a network server listens."""
    _configuration(strategy)
    if any(value.device.type != "cpu" for value in strategy.initial_parameters.values()):
        raise ValueError("checkpoint recovery currently requires CPU training")


def _tuple_tree(value):
    return tuple(_tuple_tree(item) for item in value) if isinstance(value, (list, tuple)) else value


def validate_snapshot(snapshot):
    if (type(snapshot.completed_round) is not int or type(snapshot.total_rounds) is not int
            or not 1 <= snapshot.completed_round <= snapshot.total_rounds <= 999_999_999_999):
        raise ValueError("checkpoint round bounds are invalid")
    _validate_parameters(snapshot.parameters)
    if snapshot.strategy_type not in ("FedAvg", "FedProx", "FedOpt"):
        raise ValueError("unsupported checkpoint strategy")
    expected_groups = {"_global", "_m", "_v"} if snapshot.strategy_type == "FedOpt" else set()
    if set(snapshot.strategy_tensors) != expected_groups:
        raise ValueError("checkpoint strategy state is incomplete")
    for values in snapshot.strategy_tensors.values():
        _validate_parameters(values)
    if not isinstance(snapshot.strategy_config, dict):
        raise ValueError("checkpoint strategy configuration is invalid")
    layout = [[name, list(value.shape)] for name, value in snapshot.parameters.items()]
    if layout != snapshot.strategy_config.get("parameter_layout"):
        raise ValueError("checkpoint model layout differs from the original strategy")
    for group, values in snapshot.strategy_tensors.items():
        expected = [entry for entry in layout
                    if group not in ("_m", "_v") or entry[0] not in snapshot.strategy_config["buffer_names"]]
        if [[name, list(value.shape)] for name, value in values.items()] != expected:
            raise ValueError("checkpoint optimizer layout differs from the original strategy")
    if snapshot.strategy_type == "FedOpt" and any(
            not torch.equal(snapshot.parameters[key], snapshot.strategy_tensors["_global"][key])
            for key in snapshot.parameters):
        raise ValueError("checkpoint optimizer global state differs from the completed model")
    rounds = []
    for entry in snapshot.history:
        if (not isinstance(entry, (list, tuple)) or len(entry) != 2 or type(entry[0]) is not int
                or not 1 <= entry[0] <= snapshot.completed_round
                or (entry[1] is not None and not isinstance(entry[1], dict))):
            raise ValueError("checkpoint history is invalid")
        rounds.append(entry[0])
    if rounds != sorted(set(rounds)):
        raise ValueError("checkpoint history rounds must be unique and ordered")
    if len(rounds) != snapshot.completed_round or any(number != index for index, number in enumerate(rounds, 1)):
        raise ValueError("checkpoint history must cover every completed round")
    if (not isinstance(snapshot.partition_to_client, dict)
            or any(type(key) is not int or key < 0 or not isinstance(value, str) or not value
                   for key, value in snapshot.partition_to_client.items())
            or len(set(snapshot.partition_to_client.values())) != len(snapshot.partition_to_client)):
        raise ValueError("checkpoint identity bindings are invalid")
    if (not isinstance(snapshot.torch_rng_state, torch.Tensor)
            or snapshot.torch_rng_state.dtype != torch.uint8
            or snapshot.torch_rng_state.ndim != 1):
        raise ValueError("checkpoint random state is invalid")
    try:
        torch.Generator().set_state(snapshot.torch_rng_state)
        random.Random().setstate(_tuple_tree(snapshot.python_rng_state))
        np.random.RandomState().set_state(snapshot.numpy_rng_state)
        json.dumps([snapshot.strategy_config, snapshot.history], allow_nan=False)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("checkpoint metadata or random state is invalid") from error


def capture_round(strategy, *, completed_round, total_rounds, parameters, history, partition_to_client=None):
    """Copy one evaluated boundary; callers must freeze aggregation while capturing."""
    config = _configuration(strategy)
    if any(value.device.type != "cpu" for value in parameters.values()):
        raise ValueError("checkpoint recovery currently requires CPU training")
    groups = {}
    if type(strategy) is FedOpt:
        groups = {name: _clone(getattr(strategy, name)) for name in ("_global", "_m", "_v")}
    snapshot = RoundSnapshot(
        completed_round, total_rounds, _clone(parameters), type(strategy).__name__, config, groups,
        json.loads(json.dumps(history, allow_nan=False)), dict(partition_to_client or {}),
        torch.get_rng_state().clone(), random.getstate(), np.random.get_state(),
    )
    snapshot.history = [(number, metrics) for number, metrics in snapshot.history]
    validate_snapshot(snapshot)
    return snapshot


def restore_strategy(strategy, snapshot):
    """Validate the entire target before restoring mutable strategy/random state."""
    validate_snapshot(snapshot)
    if snapshot.strategy_type != type(strategy).__name__ or snapshot.strategy_config != _configuration(strategy):
        raise ValueError("checkpoint strategy configuration does not match the target")
    expected = strategy.initial_parameters
    if list(snapshot.parameters) != list(expected):
        raise ValueError("checkpoint model layout does not match the target")
    groups = {"parameters": snapshot.parameters, **snapshot.strategy_tensors}
    for group, values in groups.items():
        names = [key for key in expected if group not in ("_m", "_v") or key not in strategy.buffer_names]
        if list(values) != names or any(values[key].shape != expected[key].shape for key in names):
            raise ValueError("checkpoint optimizer/model layout does not match the target")
    if any(value.device.type != "cpu" for value in expected.values()):
        raise ValueError("checkpoint recovery currently requires CPU training")
    restored = {name: _clone(values) for name, values in snapshot.strategy_tensors.items()}
    for name, values in restored.items():
        setattr(strategy, name, values)
    torch.set_rng_state(snapshot.torch_rng_state)
    random.setstate(_tuple_tree(snapshot.python_rng_state))
    np.random.set_state(snapshot.numpy_rng_state)
