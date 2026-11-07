"""Tests verifying that frozen-head execution plans and on-device exported bundles reload into a live client without mutating frozen backbone parameters (Area 1).

Covers:
1. Frozen-head execution plan resolution across recipes (CNN, CIFAR_RESNET18, PNEUMONIA_CNN, FROZEN_DEMO).
2. Wire subset contract validation: get_parameters() returns head-only, load_federated_state() accepts head-only.
3. Preservation of backbone weights: live training and wire reloads mutate head parameters while preserving
   backbone weights bitwise.
4. Fail-loud protection: load_federated_state() rejects missing or unexpected keys.
5. Server-side model merge: merge_non_federated() reconstitutes the complete model with the trained head and
   the exact frozen backbone.
"""
from __future__ import annotations

import os
import sys
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
FL_RUNTIME = os.path.join(HERE, "..")
if FL_RUNTIME not in sys.path:
    sys.path.insert(0, FL_RUNTIME)

import client  # noqa: E402
import execution_plan  # noqa: E402
import fl_server  # noqa: E402
import recipes  # noqa: E402
from fedlearn.communication.generated import execution_contract_pb2 as pb  # noqa: E402
from fedlearn.estimators.params import trainable_state  # noqa: E402


def _configure_client_for_recipe(monkeypatch, recipe_key: str, arm: str = "FROZEN_HEAD"):
    for flag in ("USE_LLM", "USE_LLM_LORA", "USE_MLP", "USE_PNEUMONIA"):
        monkeypatch.setattr(client, flag, False, raising=False)
    monkeypatch.setattr(client, "USE_DERIVED", True, raising=False)
    monkeypatch.setattr(client, "USE_WIRE_SUBSET", True, raising=False)
    monkeypatch.setattr(client, "MODEL_TYPE", recipe_key, raising=False)
    monkeypatch.setattr(client, "TRAINING_ARM", arm, raising=False)
    monkeypatch.setattr(client, "DEVICE", torch.device("cpu"), raising=False)
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", None, raising=False)


@pytest.mark.parametrize("recipe, expected_head_keys", [
    ("CNN", ["fc3.weight", "fc3.bias"]),
    ("CIFAR_RESNET18", ["fc.weight", "fc.bias"]),
    ("PNEUMONIA_CNN", ["classifier.1.weight", "classifier.1.bias", "classifier.4.weight", "classifier.4.bias"]),
    ("FROZEN_DEMO", ["head.weight", "head.bias"]),
])
def test_execution_plan_frozen_head_trainable_layout(recipe, expected_head_keys):
    plan = execution_plan.resolve_model_training(recipe, "FedAvg", "FROZEN_HEAD")
    assert plan.arm == pb.ARM_FROZEN_HEAD
    assert [spec.name for spec in plan.trainable] == expected_head_keys
    assert len(plan.frozen_state_sha256) == 64
    assert plan.update_protocol == pb.UPDATE_TRAINABLE_STATE_F32


@pytest.mark.parametrize("recipe, expected_head_keys", [
    ("CNN", ["fc3.weight", "fc3.bias"]),
    ("FROZEN_DEMO", ["head.weight", "head.bias"]),
])
def test_client_loads_head_without_mutating_backbone(monkeypatch, recipe, expected_head_keys):
    _configure_client_for_recipe(monkeypatch, recipe, "FROZEN_HEAD")
    recipe_obj = recipes.get_recipe(recipe)
    model = recipe_obj.build_model("cpu")
    prefixes = recipes.trainable_prefixes(recipe, "FROZEN_HEAD")
    recipes.apply_arm(model, "FROZEN_HEAD", prefixes)
    recipes.freeze_untrained_modules(model, prefixes)

    # Snapshot backbone weights
    backbone_snapshots = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if not any(name.startswith(p) for p in prefixes)
    }
    assert len(backbone_snapshots) > 0, "model must contain non-trainable backbone parameters"

    # Verify initial trainable state matches expected head keys
    initial_head = trainable_state(model)
    assert list(initial_head.keys()) == expected_head_keys

    # Simulate arrival of an aggregated global head over the wire
    new_head = OrderedDict()
    for k, v in initial_head.items():
        new_head[k] = v + 1.2345

    # Reload via load_federated_state
    client.load_federated_state(model, new_head)

    # Verify head moved to the new weights
    for k, v in new_head.items():
        loaded_tensor = dict(model.named_parameters())[k]
        assert torch.equal(loaded_tensor, v), f"head parameter {k} failed to reload"

    # Verify backbone parameters remained strictly bitwise identical
    for name, orig_tensor in backbone_snapshots.items():
        current_tensor = dict(model.named_parameters())[name]
        assert torch.equal(current_tensor, orig_tensor), f"frozen backbone parameter {name} was mutated"


def test_load_federated_state_rejects_missing_or_extra_keys(monkeypatch):
    _configure_client_for_recipe(monkeypatch, "CNN", "FROZEN_HEAD")
    model = recipes.get_recipe("CNN").build_model("cpu")
    prefixes = recipes.trainable_prefixes("CNN", "FROZEN_HEAD")
    recipes.apply_arm(model, "FROZEN_HEAD", prefixes)

    # 1. Missing head key should fail loudly
    with pytest.raises(RuntimeError, match="missing"):
        client.load_federated_state(model, OrderedDict([("fc3.weight", torch.zeros(10, 84))]))

    # 2. Unexpected backbone key should fail loudly
    with pytest.raises(RuntimeError, match="unexpected"):
        client.load_federated_state(
            model,
            OrderedDict([
                ("fc3.weight", torch.zeros(10, 84)),
                ("fc3.bias", torch.zeros(10)),
                ("conv1.weight", torch.zeros(6, 1, 5, 5)),
            ]),
        )


def test_frozen_head_training_updates_only_head_parameters(monkeypatch):
    _configure_client_for_recipe(monkeypatch, "CNN", "FROZEN_HEAD")
    model = recipes.get_recipe("CNN").build_model("cpu")
    prefixes = recipes.trainable_prefixes("CNN", "FROZEN_HEAD")
    recipes.apply_arm(model, "FROZEN_HEAD", prefixes)
    recipes.freeze_untrained_modules(model, prefixes)

    backbone_before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if not any(name.startswith(p) for p in prefixes)
    }
    head_before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if any(name.startswith(p) for p in prefixes)
    }

    class _DictDataset(torch.utils.data.Dataset):
        def __init__(self, imgs, labels):
            self.imgs = imgs
            self.labels = labels

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            return {"img": self.imgs[idx], "label": self.labels[idx]}

    ds = _DictDataset(torch.randn(8, 3, 32, 32), torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long))
    loader = DataLoader(ds, batch_size=4)

    client.train(model, loader, epochs=1, dataset_name="mnist", learning_rate_override=0.05)

    # 1. Head parameters must have updated
    for name, orig_v in head_before.items():
        curr_v = dict(model.named_parameters())[name]
        assert not torch.equal(curr_v, orig_v), f"head parameter {name} did not train"

    # 2. Backbone parameters must not have moved at all
    for name, orig_v in backbone_before.items():
        curr_v = dict(model.named_parameters())[name]
        assert torch.equal(curr_v, orig_v), f"backbone parameter {name} mutated during training"


def test_server_merge_non_federated_with_frozen_head():
    full_init = OrderedDict([
        ("conv1.weight", torch.ones(6, 1, 5, 5)),
        ("conv1.bias", torch.zeros(6)),
        ("fc3.weight", torch.full((10, 84), 0.5)),
        ("fc3.bias", torch.full((10,), 0.1)),
    ])

    trained_head = OrderedDict([
        ("fc3.weight", torch.full((10, 84), 2.5)),
        ("fc3.bias", torch.full((10,), 0.9)),
    ])

    merged = fl_server.merge_non_federated(trained_head, full_init)

    assert set(merged.keys()) == set(full_init.keys())
    assert torch.equal(merged["fc3.weight"], trained_head["fc3.weight"])
    assert torch.equal(merged["fc3.bias"], trained_head["fc3.bias"])
    assert torch.equal(merged["conv1.weight"], full_init["conv1.weight"])
    assert torch.equal(merged["conv1.bias"], full_init["conv1.bias"])
