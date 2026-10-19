"""Stage 4 S0: under a contract that states dropout, the laptop draws the same seeded masks the device does.

The contract says every participant's dropout masks come from DROPOUT_MASKS_SEEDED_V1; torch's own dropout RNG would
break that. SeededDropoutMasks swaps the model's nn.Dropout layers for seeded ones for the duration of a round.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

import recipes
from contracted_dropout import SeededDropoutMasks
from fedlearn.contract.batch_order import batches, seeded_permutation
from fedlearn.contract.dropout_masks import dropout_mask

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MLP_GOLDEN = os.path.join(REPO, "framework", "tests", "fixtures", "mlp_golden")
LAYERS = [("dropout1", 0.3), ("dropout2", 0.3)]


def _mlp():
    torch.manual_seed(0)
    return recipes.get_recipe("MLP").build_model("cpu").train()


def test_each_layer_multiplies_by_its_seeded_mask_for_the_current_step():
    net = _mlp()
    x = torch.randn(4, 64)
    with SeededDropoutMasks(net, LAYERS, seed=42, round_=1) as masks:
        for layer, name in enumerate(("dropout1", "dropout2")):
            expected = x * torch.from_numpy(dropout_mask(4 * 64, 0.3, 42, 1, 0, layer).reshape(4, 64))
            assert torch.equal(net.get_submodule(name)(x), expected)
        masks.advance()
        expected = x * torch.from_numpy(dropout_mask(4 * 64, 0.3, 42, 1, 1, 0).reshape(4, 64))
        assert torch.equal(net.get_submodule("dropout1")(x), expected)


def test_evaluation_is_untouched_and_the_original_layers_come_back():
    net = _mlp()
    x = torch.randn(4, 64)
    with SeededDropoutMasks(net, LAYERS, seed=42, round_=1):
        net.eval()
        assert torch.equal(net.get_submodule("dropout1")(x), x)
        net.train()
    assert isinstance(net.get_submodule("dropout1"), nn.Dropout)


@pytest.mark.parametrize("layers", [
    [("dropout1", 0.3)],                           # a layer missing
    [("dropout1", 0.3), ("dropout2", 0.5)],        # another rate
    [("dropout2", 0.3), ("dropout1", 0.3)],        # another order
])
def test_a_contract_stating_other_dropout_than_the_model_has_is_refused(layers):
    with pytest.raises(ValueError):
        with SeededDropoutMasks(_mlp(), layers, seed=42, round_=1):
            pass


def test_training_with_it_reproduces_the_devices_masked_golden():
    """Adam over the golden's seeded batches, masks from this helper: exactly the endpoint the device lands on."""
    sys.path.insert(0, MLP_GOLDEN)
    sys.path.insert(0, os.path.join(REPO, "mobile_client", "scripts"))
    import generate_mlp_golden as g
    net = g.build_net()
    x, y = g.dataset()
    opt = torch.optim.Adam(net.parameters(), lr=g.LR)
    with SeededDropoutMasks(net, LAYERS, seed=g.SEED, round_=g.ROUND) as masks:
        for epoch in range(g.EPOCHS):
            for idx in batches(seeded_permutation(g.EXAMPLES, g.SEED, g.ROUND, epoch), g.BATCH):
                opt.zero_grad()
                torch.nn.functional.cross_entropy(net(x[idx]), y[idx]).backward()
                opt.step()
                masks.advance()
    got = torch.cat([p.detach().reshape(-1) for p in net.parameters()]).numpy().astype("<f4")
    golden = np.fromfile(os.path.join(MLP_GOLDEN, "mlp_masked_adam_final.f32"), dtype="<f4")
    assert np.abs(got - golden).max() < 1e-6


def test_the_client_installs_the_contracts_masks_for_the_round_and_nothing_otherwise(monkeypatch):
    import contextlib
    import client
    from fedlearn.communication.generated import execution_contract_pb2 as pb
    net = _mlp()
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", None)
    assert isinstance(client._contracted_dropout(net, 3), contextlib.nullcontext)
    contract = pb.ExecutionContract(seed=7)
    contract.model_training.dropout.extend(
        [pb.DropoutLayer(module=n, rate=r, masks=pb.DROPOUT_MASKS_SEEDED_V1) for n, r in LAYERS])
    monkeypatch.setattr(client, "EXECUTION_CONTRACT", contract)
    masks = client._contracted_dropout(net, 3)
    assert isinstance(masks, SeededDropoutMasks)
    assert (masks.seed, masks.round, masks.step) == (7, 3, 0)
