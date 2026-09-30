"""An MLP (ECG) run trains with the strategy it was started with.

Both entry points used to overwrite the run's strategy with a hardcoded ECG_STRATEGY = "DeComFL", so every MLP run
trained zeroth-order DeComFL whatever the user chose. A laptop client even checked a FedAvg execution contract, accepted
it, and then trained DeComFL. A live phone run exposed it: the phone uploaded FedAvg weights to a server aggregating
DeComFL scalars, and every upload failed in DecomflStrategy.aggregate_fit (KeyError: 0).

The ECG settings that are genuinely the recipe's stay: its dataset and file. And the laptop's first-order path, which
the override had always bypassed, must load the ECG shard (it fell through to the default text dataset).
"""
import os
import shutil
import sys
from argparse import Namespace
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recipes  # noqa: E402

STRATEGIES = ["FedAvg", "FedOpt", "Robust", "FedProx", "DeComFL"]


def _run_args(strategy, model_type="MLP"):
    return Namespace(model_type=model_type, strategy=strategy, dataset="cifar10")


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_the_server_keeps_an_mlp_runs_strategy(strategy):
    import fl_server
    args = _run_args(strategy)
    dataset_path, _ = fl_server.apply_ecg_run_settings(args)
    assert args.strategy == strategy
    assert args.dataset == "ecg"
    assert dataset_path == fl_server.ECG_DATASET_PATH


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_the_client_keeps_an_mlp_runs_strategy(strategy):
    import client
    args = _run_args(strategy)
    dataset_path, _ = client.apply_ecg_run_settings(args)
    assert args.strategy == strategy
    assert args.dataset == "ecg"
    assert dataset_path == client.ECG_DATASET_PATH


def test_the_ecg_settings_leave_other_recipes_alone():
    import client
    import fl_server
    for module in (client, fl_server):
        args = _run_args("FedAvg", model_type="CNN")
        assert module.apply_ecg_run_settings(args) == (None, None)
        assert (args.strategy, args.dataset) == ("FedAvg", "cifar10")


def test_an_mlp_fedavg_server_aggregates_with_fedavg():
    import fl_server
    from fedlearn.server import DeComFL
    args = Namespace(model_type="MLP", strategy="FedAvg", dataset="cifar10", min_clients=1, aggregation="FFA_LORA")
    fl_server.apply_ecg_run_settings(args)
    net = recipes.get_recipe("MLP").build_model("cpu")
    strategy = fl_server.select_strategy(args, OrderedDict(net.state_dict()), None)
    assert not isinstance(strategy, DeComFL)


def _synthetic_ecg_csv(path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 140)).astype(np.float32)
    y = np.array([0, 1] * 100, dtype=np.int64)
    pd.DataFrame(np.column_stack([x, y])).to_csv(path, header=False, index=False)
    return str(path)


def _as_mlp_client(monkeypatch, client):
    """The module globals client.main() sets for --model-type MLP."""
    monkeypatch.setattr(client, "USE_LLM_LORA", False, raising=False)
    monkeypatch.setattr(client, "USE_LLM", False, raising=False)
    monkeypatch.setattr(client, "USE_MLP", True, raising=False)
    monkeypatch.setattr(client, "USE_PNEUMONIA", False, raising=False)
    monkeypatch.setattr(client, "USE_DERIVED", False, raising=False)
    monkeypatch.setattr(client, "USE_WIRE_SUBSET", False, raising=False)
    monkeypatch.setattr(client, "MODEL_TYPE", "MLP", raising=False)
    monkeypatch.setattr(client, "TRAINING_ARM", "FULL", raising=False)
    monkeypatch.setattr(client, "DEVICE", "cpu", raising=False)
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", None, raising=False)


def test_the_first_order_client_loads_its_ecg_shard(tmp_path, monkeypatch):
    import client
    monkeypatch.chdir(tmp_path)  # the ECG split cache is written under the working directory
    _as_mlp_client(monkeypatch, client)
    csv = _synthetic_ecg_csv(tmp_path / "ecg.csv")
    shutil.rmtree("data_splits", ignore_errors=True)

    train, _ = client.load_data(partition_id=0, dataset_name="ecg", dataset_path=csv, num_clients=5)

    x, y = next(iter(train))
    assert x.shape[1:] == (140,) and x.dtype == torch.float32
    assert set(y.tolist()) <= {0, 1}


def test_a_first_order_mlp_round_trains_and_returns_the_whole_model(tmp_path, monkeypatch):
    import client
    monkeypatch.chdir(tmp_path)
    _as_mlp_client(monkeypatch, client)
    csv = _synthetic_ecg_csv(tmp_path / "ecg.csv")

    trainer = client.ZOSLClient(partition_id=0, dataset_name="ecg", dataset_path=csv, num_clients=5)
    torch.manual_seed(0)
    start = OrderedDict((k, v.clone()) for k, v in recipes.get_recipe("MLP").build_model("cpu").state_dict().items())
    result = trainer.fit(OrderedDict((k, v.clone()) for k, v in start.items()), {"server_round": 1})
    params, examples = result[0], result[1]

    assert list(params) == list(start)
    assert examples == len(trainer.trainloader.dataset) > 0
    assert any(not torch.equal(params[k].cpu(), start[k]) for k in start), "the round changed nothing"
