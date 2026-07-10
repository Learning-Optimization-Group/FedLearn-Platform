# SCRIPTS/tests/test_client_tinynet_golden_decomfl.py
"""A desktop client must be able to join a TINYNET_GOLDEN DeComFL federation (the phone is the
other client). This pins the recipe-driven, single-source-of-truth contract the desktop path
relies on:

  * client.py builds the golden TinyNet from the SHARED recipe (never an inline redefinition), so
    its state_dict keys are byte-identical to the server-built net (init_model.get_model). fc2 is
    frozen (25 trainable); the frozen fc2 is DETERMINISTIC and matches the committed golden the
    mobile ExecuTorch .pte encodes, so the desktop trains the *same* function as the phone.
  * client.py builds a REAL 4-dim DataLoader from the committed canonical golden fixture
    (framework/tests/fixtures/decomfl_golden), yielding (inputs[B,4] float32, targets[B] int64).
  * DeComFLClient.load_global_model tolerates the server's trainable-only (fc1) global sync — the
    frozen fc2 keys are legitimately absent — WITHOUT weakening the strict invariant that every
    *trainable* param and no unexpected key must be present. A naive strict load crashes on the
    missing fc2.* keys before a single round can complete.
"""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))


def _server_built_net():
    """The net the SERVER builds for this model type (init_model.get_model → recipes)."""
    import init_model
    return init_model.get_model("TINYNET_GOLDEN", "tinynet_golden", "cpu")


def test_client_builds_golden_model_from_shared_recipe():
    import client
    net = client.build_tinynet_golden_model("cpu")

    # Same state_dict keys as the server-built net (single source of truth = the recipe).
    assert list(net.state_dict().keys()) == list(_server_built_net().state_dict().keys())

    # fc2 is frozen; exactly the 25 trainable fc1 params.
    trainable = {n for n, p in net.named_parameters() if p.requires_grad}
    frozen = {n for n, p in net.named_parameters() if not p.requires_grad}
    assert trainable == {"fc1.weight", "fc1.bias"}
    assert frozen == {"fc2.weight", "fc2.bias"}
    assert sum(p.numel() for p in net.parameters() if p.requires_grad) == 25


def test_golden_model_frozen_fc2_is_deterministic_matches_phone():
    """The frozen fc2 must be byte-identical every build (it is NEVER synced, so all federation
    peers — the phone's .pte and every desktop client — must share it), and must equal the
    committed golden the mobile ExecuTorch bundle encodes."""
    import client
    a = client.build_tinynet_golden_model("cpu").state_dict()
    b = client.build_tinynet_golden_model("cpu").state_dict()
    for k in ("fc2.weight", "fc2.bias"):
        assert torch.equal(a[k], b[k]), f"{k} is non-deterministic across builds"

    fx = client._resolve_decomfl_golden_fixture_dir()
    golden = dict(torch.jit.load(os.path.join(fx, "zo_model_tiny.pt")).state_dict())
    for k in ("fc2.weight", "fc2.bias"):
        assert torch.allclose(a[k], golden[k], atol=1e-6), f"{k} does not match the golden .pt"


def test_client_builds_real_4dim_decomfl_loader():
    import client
    loader = client.build_tinynet_golden_decomfl_loader(partition_id=0)

    assert len(loader.dataset) > 0
    inputs, targets = next(iter(loader))
    assert inputs.dim() == 2 and inputs.shape[1] == 4, inputs.shape
    assert inputs.dtype == torch.float32
    assert targets.dtype == torch.int64
    assert targets.min().item() >= 0 and targets.max().item() < 3


def test_loader_partition_id_gives_distinct_batch_order():
    """Distinct partition_ids form a genuine (non-identical-order) federation off the shared batch."""
    import client
    b0 = next(iter(client.build_tinynet_golden_decomfl_loader(partition_id=0, batch_size=4)))[0]
    b1 = next(iter(client.build_tinynet_golden_decomfl_loader(partition_id=7, batch_size=4)))[0]
    assert not torch.equal(b0, b1)


def test_load_global_model_accepts_trainable_only_sync():
    """The server's DeComFL sync carries only the 25 trainable fc1 params; the client must adopt
    them and keep its (deterministic) frozen fc2 — a strict load would crash on missing fc2.*."""
    import client
    from fedlearn.client import DeComFLClient
    from fedlearn.estimators.params import trainable_state

    net = client.build_tinynet_golden_model("cpu")
    fc2_before = net.state_dict()["fc2.weight"].clone()
    loader = client.build_tinynet_golden_decomfl_loader(partition_id=0)
    c = DeComFLClient(model=net, train_loader=loader, smoothing_param=0.001, device="cpu")

    # Server global = requires_grad-filtered trainable layout (fc1 only), with fresh values.
    sync = trainable_state(net)
    sync["fc1.weight"] = torch.full_like(sync["fc1.weight"], 0.25)
    sync["fc1.bias"] = torch.full_like(sync["fc1.bias"], -0.5)

    c.load_global_model(sync)  # must not raise

    assert c.x_current.numel() == 25
    assert torch.allclose(net.state_dict()["fc1.weight"], torch.full((5, 4), 0.25))
    # Frozen fc2 untouched by the trainable-only sync.
    assert torch.equal(net.state_dict()["fc2.weight"], fc2_before)


def test_load_global_model_rejects_missing_trainable_key():
    import client
    from fedlearn.client import DeComFLClient
    from fedlearn.estimators.params import trainable_state

    net = client.build_tinynet_golden_model("cpu")
    loader = client.build_tinynet_golden_decomfl_loader(partition_id=0)
    c = DeComFLClient(model=net, train_loader=loader, smoothing_param=0.001, device="cpu")

    sync = trainable_state(net)
    del sync["fc1.bias"]  # a MISSING trainable key would silently misalign the shared-seed z
    with pytest.raises(Exception):
        c.load_global_model(sync)


def test_load_global_model_rejects_unexpected_key():
    import client
    from fedlearn.client import DeComFLClient
    from fedlearn.estimators.params import trainable_state

    net = client.build_tinynet_golden_model("cpu")
    loader = client.build_tinynet_golden_decomfl_loader(partition_id=0)
    c = DeComFLClient(model=net, train_loader=loader, smoothing_param=0.001, device="cpu")

    sync = trainable_state(net)
    sync["fc9.weight"] = torch.zeros(1)  # not a key of this model
    with pytest.raises(Exception):
        c.load_global_model(sync)
