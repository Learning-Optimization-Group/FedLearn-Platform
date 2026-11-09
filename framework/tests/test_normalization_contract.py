"""Unit and regression tests for normalization layers contract and privacy preservation."""
import pytest
import torch
import torch.nn as nn

from fedlearn.contract.normalization import (
    convert_bn_to_gn,
    freeze_batch_norm_buffers,
    has_batch_norm,
)


class DummyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Dummy1DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.bn1 = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        return self.fc2(self.bn1(self.fc1(x)))


def test_has_batch_norm_identifies_batchnorm():
    net = DummyCNN()
    assert has_batch_norm(net) is True

    linear = nn.Linear(10, 2)
    assert has_batch_norm(linear) is False


def test_freeze_batch_norm_buffers_sets_eval_mode():
    net = DummyCNN()
    net.train()
    assert net.bn1.training is True
    assert net.bn2.training is True

    freeze_batch_norm_buffers(net)
    assert net.bn1.training is False
    assert net.bn2.training is False
    # Convolutions remain in whatever mode they were in
    assert net.conv1.training is True


def test_convert_bn_to_gn_replaces_all_batchnorm():
    net = DummyCNN()
    assert has_batch_norm(net) is True

    convert_bn_to_gn(net, max_groups=8, copy_affine=True)
    assert has_batch_norm(net) is False

    # Check that bn1 and bn2 were replaced by GroupNorm
    assert isinstance(net.bn1, nn.GroupNorm)
    assert isinstance(net.bn2, nn.GroupNorm)
    assert net.bn1.num_channels == 16
    assert net.bn1.num_groups == 8
    assert net.bn2.num_channels == 32
    assert net.bn2.num_groups == 8

    # Forward pass works cleanly
    x = torch.randn(2, 3, 4, 4)
    out = net(x)
    assert out.shape == (2, 10)


def test_convert_bn_to_gn_1d_model():
    net = Dummy1DModel()
    assert has_batch_norm(net) is True
    convert_bn_to_gn(net, max_groups=4, copy_affine=False)
    assert has_batch_norm(net) is False
    assert isinstance(net.bn1, nn.GroupNorm)
    assert net.bn1.num_groups == 4
    assert net.bn1.num_channels == 20


def test_convert_bn_to_gn_rejects_invalid_max_groups():
    net = DummyCNN()
    with pytest.raises(ValueError, match="max_groups"):
        convert_bn_to_gn(net, max_groups=0)
