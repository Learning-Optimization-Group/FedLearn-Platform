import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('nih_pretrained', Path(__file__).resolve().parents[2] / 'research/benchmarks/nih_pretrained.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_rgb_normalization_uses_all_three_imagenet_channels():
    x = torch.zeros(2, 1, 224, 224)
    got = m.imagenet_input(x)
    assert got.shape == (2, 3, 224, 224)
    for channel, (mean, std) in enumerate(zip(m.MEAN, m.STD)):
        torch.testing.assert_close(got[:, channel], torch.full((2, 224, 224), (.5 - mean) / std))


def test_feature_extraction_freezes_parameters_and_batchnorm_state():
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.BatchNorm2d(2), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    original = {k: v.clone() for k, v in net.state_dict().items()}
    features = m.extract_features(net, torch.rand(4, 1, 224, 224), batch_size=2)
    assert features.shape == (4, 2)
    assert not features.requires_grad
    assert not net.training
    assert all(not p.requires_grad for p in net.parameters())
    for name, value in net.state_dict().items():
        assert torch.equal(value, original[name])


def test_partial_fine_tuning_updates_only_last_block_and_head_with_frozen_bn():
    net = torch.nn.Module()
    net.stem = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4))
    net.layer4 = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 1), torch.nn.BatchNorm2d(4))
    net.fc = torch.nn.Linear(4, 2)
    m.configure_fine_tuning(net)
    m.fine_tuning_mode(net)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=.001)
    x = torch.rand(4, 3, 8, 8)
    out = net.fc(net.layer4(net.stem(x)).mean((2, 3)))
    loss = torch.nn.functional.cross_entropy(out, torch.tensor([0, 1, 0, 1]))
    loss.backward(); optimizer.step()
    assert net.layer4[0].weight.grad is not None and net.fc.weight.grad is not None
    assert net.stem[0].weight.grad is None
    for name, value in net.state_dict().items():
        if name.startswith('stem.') or 'running_' in name or name.endswith('num_batches_tracked'):
            assert torch.equal(value, before[name]), name
    assert not torch.equal(net.layer4[0].weight, before['layer4.0.weight'])
    assert not torch.equal(net.fc.weight, before['fc.weight'])
