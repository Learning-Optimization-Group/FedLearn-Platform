"""TINYNET_GOLDEN under the FedAvg strategy: the client and the server must agree on the wire.

TINYNET_GOLDEN freezes fc2 by construction, so it federates only its 25 trainable fc1 params even
on the FULL arm. The server already knew this (``init_model.py`` extracts ``trainable_state`` and
``fl_server.evaluation_load_is_strict`` special-cased the recipe), but the CLI client derived its
wire from ``trainable_prefixes(recipe, arm)``, which is None for this recipe on FULL. So a FedAvg
client uploaded the full 43-param state and then crashed loading the server's 25-param global model:

    RuntimeError: Error(s) in loading state_dict for TinyNet:
        Missing key(s) in state_dict: "fc2.weight", "fc2.bias".

Every CLI client in a four-client FedAvg run died on round 1 before submitting anything. The
DeComFL path never hit this because it does not go through load_state_dict.
"""
import os
import sys
import types
from collections import OrderedDict

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import recipes  # noqa: E402

TINYNET_WIRE = {"fc1.weight", "fc1.bias"}


def _as_fedavg_tinynet_client(monkeypatch, client):
    """Configure the module globals the way __main__ does for --model-type TINYNET_GOLDEN."""
    monkeypatch.setattr(client, "USE_LLM_LORA", False, raising=False)
    monkeypatch.setattr(client, "USE_LLM", False, raising=False)
    monkeypatch.setattr(client, "USE_MLP", False, raising=False)
    monkeypatch.setattr(client, "USE_PNEUMONIA", False, raising=False)
    monkeypatch.setattr(client, "USE_DERIVED", False, raising=False)
    monkeypatch.setattr(client, "MODEL_TYPE", "TINYNET_GOLDEN", raising=False)
    monkeypatch.setattr(client, "TRAINING_ARM", "FULL", raising=False)
    monkeypatch.setattr(client, "USE_WIRE_SUBSET",
                        recipes.federates_trainable_subset("TINYNET_GOLDEN", "FULL"), raising=False)


def test_the_recipe_declares_a_trainable_subset_wire_on_the_full_arm():
    assert recipes.federates_trainable_subset("TINYNET_GOLDEN", "FULL") is True


def test_a_full_arm_whole_model_recipe_still_federates_everything():
    # The guard must not widen: a whole-model recipe on FULL keeps the full state on the wire.
    assert recipes.federates_trainable_subset("CNN", "FULL") is False


def test_a_frozen_arm_still_counts_as_a_subset_wire():
    assert recipes.federates_trainable_subset("CIFAR_RESNET18", "FROZEN_HEAD") is True


def test_client_uploads_exactly_the_trainable_subset(monkeypatch):
    import client  # noqa: E402
    _as_fedavg_tinynet_client(monkeypatch, client)
    net = recipes.get_recipe("TINYNET_GOLDEN").build_model("cpu")

    params = client.ZOSLClient.get_parameters(types.SimpleNamespace(net=net))

    assert set(params.keys()) == TINYNET_WIRE
    assert sum(t.numel() for t in params.values()) == 25


def test_client_loads_the_servers_subset_global_model(monkeypatch):
    """The exact crash from the live run: loading a 25-param global model must not raise."""
    import torch
    import client  # noqa: E402
    _as_fedavg_tinynet_client(monkeypatch, client)
    net = recipes.get_recipe("TINYNET_GOLDEN").build_model("cpu")
    fc2_before = net.fc2.weight.detach().clone()

    global_model = OrderedDict(
        (k, torch.full_like(v, 0.5)) for k, v in net.state_dict().items() if k in TINYNET_WIRE)
    client.load_federated_state(net, global_model)

    assert torch.equal(net.fc1.weight, torch.full_like(net.fc1.weight, 0.5))
    assert torch.equal(net.fc2.weight, fc2_before), "the frozen layer must not be touched"


def test_client_still_rejects_a_payload_missing_trainable_keys(monkeypatch):
    """Non-strict must mean 'the frozen keys may be absent', not 'anything goes'."""
    import torch
    import client  # noqa: E402
    _as_fedavg_tinynet_client(monkeypatch, client)
    net = recipes.get_recipe("TINYNET_GOLDEN").build_model("cpu")

    truncated = OrderedDict([("fc1.weight", torch.zeros_like(net.fc1.weight))])
    with pytest.raises(RuntimeError, match="fc1.bias"):
        client.load_federated_state(net, truncated)


def test_server_and_client_share_one_definition():
    import fl_server  # noqa: E402
    for key in recipes.catalog_keys():
        for arm in recipes._METADATA_BY_KEY[key].get("supported_arms", [recipes.DEFAULT_ARM]):
            assert fl_server.evaluation_load_is_strict(key, arm) is (
                not recipes.federates_trainable_subset(key, arm)), (key, arm)


def test_fedavg_client_trains_a_full_round_on_the_golden_data(monkeypatch):
    """The second defect on this path, found in the same live run once the load was fixed.

    The TinyNet golden data loader was wired only into the DeComFL branch, so a FedAvg client's
    load_data() fell through to the default text dataset ("cb": 4,000 samples, dict batches) and
    died on the first forward pass. The dataset is a property of the recipe, not the strategy.
    """
    import torch
    import client  # noqa: E402
    _as_fedavg_tinynet_client(monkeypatch, client)
    monkeypatch.setattr(client, "DEVICE", torch.device("cpu"), raising=False)

    c = client.ZOSLClient(partition_id=1, dataset_name="cb", num_clients=4)
    fc2_before = c.net.fc2.weight.detach().clone()

    # The data must be the golden 4-dim vector task, whatever --dataset defaulted to.
    x, y = next(iter(c.trainloader))
    assert x.shape[-1] == 4 and x.dtype == torch.float32
    assert y.dtype == torch.int64

    initial = c.get_parameters()
    assert set(initial.keys()) == TINYNET_WIRE
    new_params, num_examples = c.fit(initial, {})

    assert set(new_params.keys()) == TINYNET_WIRE
    assert num_examples > 0
    assert not torch.equal(new_params["fc1.weight"].cpu(), initial["fc1.weight"].cpu()), \
        "fc1 should have trained"
    assert torch.equal(c.net.fc2.weight, fc2_before), "the frozen layer must not move"


def test_tinynet_optimizer_matches_one_step_native_sgd(monkeypatch):
    """Catch Adam or a different learning rate on the desktop TinyNet path."""
    import copy
    import torch
    from torch.utils.data import DataLoader
    import client  # noqa: E402

    _as_fedavg_tinynet_client(monkeypatch, client)
    monkeypatch.setattr(client, "DEVICE", torch.device("cpu"), raising=False)
    actual = recipes.get_recipe("TINYNET_GOLDEN").build_model("cpu")
    expected = copy.deepcopy(actual)
    golden = client.build_tinynet_golden_decomfl_loader(partition_id=0)
    loader = DataLoader(golden.dataset, batch_size=8, shuffle=False)
    inputs, targets = next(iter(loader))

    reference_optimizer = torch.optim.SGD(
        (p for p in expected.parameters() if p.requires_grad), lr=0.001,
    )
    reference_optimizer.zero_grad()
    torch.nn.functional.cross_entropy(expected(inputs), targets).backward()
    reference_optimizer.step()

    client.train(actual, loader, epochs=1, dataset_name="cb")

    for name, reference_param in expected.named_parameters():
        observed = dict(actual.named_parameters())[name]
        torch.testing.assert_close(observed, reference_param, rtol=0, atol=1e-6)
