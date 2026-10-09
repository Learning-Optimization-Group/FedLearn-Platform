"""Contracted training must perform every finite batch and refuse non-finite loss."""
import copy
from types import SimpleNamespace

import pytest
import torch

import client


def configure(monkeypatch):
    for flag in ('USE_LLM', 'USE_LLM_LORA', 'USE_PNEUMONIA', 'USE_DERIVED'):
        monkeypatch.setattr(client, flag, False)
    monkeypatch.setattr(client, 'USE_MLP', True)
    monkeypatch.setattr(client, 'MODEL_TYPE', 'MLP')
    monkeypatch.setattr(client, 'TRAINING_ARM', 'FULL')
    monkeypatch.setattr(client, 'DEVICE', torch.device('cpu'))
    monkeypatch.setattr(client, 'EXECUTION_CONTRACT', SimpleNamespace())


def test_large_finite_loss_performs_all_declared_adam_steps(monkeypatch):
    configure(monkeypatch)
    net = torch.nn.Linear(2, 2)
    with torch.no_grad():
        net.weight.zero_()
        net.bias.copy_(torch.tensor([1000., -1000.]))
    reference = copy.deepcopy(net)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.tensor([[1., 0.], [0., 1.]]), torch.ones(2, dtype=torch.long)), batch_size=1)
    optimizer = torch.optim.Adam(reference.parameters(), lr=0.01)
    for x, y in loader:
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(reference(x), y).backward()
        optimizer.step()
    progress = []
    client.train(net, loader, 1, 'ecg', learning_rate_override=0.01,
                 progress_callback=lambda step, total: progress.append((step, total)))
    assert progress == [(1, 2), (2, 2)]
    for actual, expected in zip(net.parameters(), reference.parameters()):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_contracted_loss_refuses_update(monkeypatch, value):
    configure(monkeypatch)
    net = torch.nn.Linear(2, 2)
    with torch.no_grad():
        net.bias.fill_(value)
    before = [p.detach().clone() for p in net.parameters()]
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.ones(1, 2), torch.ones(1, dtype=torch.long)), batch_size=1)
    with pytest.raises(RuntimeError, match='non-finite loss'):
        client.train(net, loader, 1, 'ecg', learning_rate_override=0.01)
    for actual, expected in zip(net.parameters(), before):
        torch.testing.assert_close(actual, expected, equal_nan=True)
