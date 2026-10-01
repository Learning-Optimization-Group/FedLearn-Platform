"""Training-mode BatchNorm on a device (Stage 4 S7): rebuilt from primitive ops, with its running statistics.

ExecuTorch cannot lower training-mode BatchNorm (aten._native_batch_norm_legit_functional is not in its op set), so a
mobile training program swaps each BatchNorm for pte_export.TrainBatchNorm2d: the same normalisation built from
mean, multiply and rsqrt, which also returns the batch statistics it used. The device then updates the running
statistics with the contract's rule (BATCH_NORM_RUNNING_STATS_V1), and laptops keep training with torch's own BatchNorm.
These tests hold the rebuilt layer and the rule to torch's.
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "mobile_client", "scripts"))

import pte_export  # noqa: E402
from fedlearn.contract.batch_norm import running_stats_update  # noqa: E402


def _bn(channels, seed):
    torch.manual_seed(seed)
    bn = nn.BatchNorm2d(channels).train()
    nn.init.normal_(bn.weight)
    nn.init.normal_(bn.bias)
    bn.running_mean.normal_()
    bn.running_var.uniform_(0.5, 2.0)
    return bn


@pytest.mark.parametrize("shape", [(16, 8, 12, 12), (2, 3, 5, 7), (32, 64, 4, 4)])
def test_the_rebuilt_layer_normalises_and_differentiates_as_torchs_batch_norm(shape):
    bn = _bn(shape[1], seed=1)
    rebuilt = pte_export.TrainBatchNorm2d(copy.deepcopy(bn))
    x = torch.randn(*shape, requires_grad=True)
    y_torch, y_rebuilt = bn(x), rebuilt(x)
    assert (y_torch - y_rebuilt).abs().max() < 1e-5
    g = torch.randn_like(y_torch)
    gx1, gw1, gb1 = torch.autograd.grad(y_torch, (x, bn.weight, bn.bias), g)
    gx2, gw2, gb2 = torch.autograd.grad(y_rebuilt, (x, rebuilt.weight, rebuilt.bias), g)
    assert (gx1 - gx2).abs().max() < 1e-4
    assert torch.allclose(gw1, gw2, rtol=1e-4, atol=1e-4) and torch.allclose(gb1, gb2, rtol=1e-4, atol=1e-4)


def test_the_rebuilt_layer_reports_the_batch_statistics_it_normalised_with():
    x = torch.randn(4, 3, 6, 6)
    rebuilt = pte_export.TrainBatchNorm2d(_bn(3, seed=2))
    rebuilt(x)
    mean, var = rebuilt.stats
    assert torch.allclose(mean, x.mean(dim=(0, 2, 3)), atol=1e-6)
    assert torch.allclose(var, x.var(dim=(0, 2, 3), unbiased=False), atol=1e-5)


def test_the_running_statistics_rule_is_torchs_update():
    """BATCH_NORM_RUNNING_STATS_V1 from the batch statistics reproduces torch's buffers after one training step."""
    bn = _bn(5, seed=3)
    before_mean, before_var = bn.running_mean.clone(), bn.running_var.clone()
    x = torch.randn(8, 5, 7, 7)
    bn(x)
    x64 = x.double()
    mean = x64.mean(dim=(0, 2, 3))
    var = x64.var(dim=(0, 2, 3), unbiased=False)
    new_mean, new_var = running_stats_update(before_mean.numpy(), before_var.numpy(), mean.numpy(), var.numpy(),
                                             count=8 * 7 * 7, momentum=0.1)
    assert np.abs(new_mean - bn.running_mean.numpy()).max() < 1e-7
    assert np.abs(new_var - bn.running_var.numpy()).max() < 1e-6


def test_one_example_per_channel_cannot_update_the_variance():
    with pytest.raises(ValueError):
        running_stats_update(np.zeros(2, np.float32), np.ones(2, np.float32), np.zeros(2), np.zeros(2), count=1,
                             momentum=0.1)


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        return self.fc(torch.relu(self.bn2(torch.relu(self.bn1(self.conv(x))))).mean(dim=(2, 3)))


def test_the_batch_norm_layers_are_listed_in_the_order_their_statistics_are_returned():
    assert [(name, n) for name, n, _momentum, _eps in pte_export.batch_norm_layers(_Net())] == [("bn1", 8), ("bn2", 8)]


def test_the_training_program_returns_each_layers_batch_mean_and_variance():
    from executorch.exir._serialize._program import deserialize_pte_binary
    torch.manual_seed(0)
    pte = pte_export.export_bn_trainable_pte(_Net(), (torch.randn(4, 3, 16, 16), torch.zeros(4, dtype=torch.int64)),
                                             max_batch=8)
    program = deserialize_pte_binary(pte)
    program = getattr(program, "program", program)
    ops = {o.name for p in program.execution_plan for o in p.operators}
    assert not any("batch_norm" in op for op in ops), "training-mode BatchNorm must not reach the program"


def test_the_evaluation_programs_take_the_running_statistics_as_inputs():
    """A device's running statistics change as it trains, so the loss and infer programs read them from the state."""
    from executorch.runtime import Runtime
    import tempfile
    torch.manual_seed(0)
    net = _Net().eval()
    x = torch.randn(4, 3, 16, 16)
    y = torch.tensor([1, 2, 3, 4])
    pte = pte_export.export_functional_pte(net, (x, y), max_batch=8, include_buffers=True)
    with tempfile.NamedTemporaryFile(suffix=".pte", delete=False) as fh:
        fh.write(pte)
    method = Runtime.get().load_program(fh.name).load_method("forward")
    state = pte_export.federated_flat(net)
    assert state.numel() == pte_export.trainable_flat(net).numel() + 4 * 8
    loss_at = lambda s: float(method.execute([s, x, y])[0])
    shifted = state.clone()
    shifted[-8:] += 1.0                                           # bn2.running_var
    params = {n: p.detach() for n, p in net.named_parameters()}
    buffers = {n: b.clone() for n, b in net.named_buffers()}
    buffers["bn2.running_var"] += 1.0
    expected = torch.nn.functional.cross_entropy(torch.func.functional_call(net, {**params, **buffers}, (x,)), y)
    assert abs(loss_at(shifted) - float(expected)) < 1e-5
    assert abs(loss_at(shifted) - loss_at(state)) > 1e-4
    os.unlink(fh.name)
