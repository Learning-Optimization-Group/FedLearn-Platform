"""Local DP: an honest client clips its own update to C and adds noise of std z * 2C."""
from collections import OrderedDict

import pytest
import torch

from fedlearn.privacy.local_dp import privatize


def test_noise_std_is_z_times_2c():
    d = OrderedDict(w=torch.zeros(40_000))
    out = privatize(d, clip_norm=0.5, noise_multiplier=1.5, generator=torch.Generator().manual_seed(1))
    assert out["w"].std().item() == pytest.approx(1.5, rel=0.03)   # 1.5 * 2 * 0.5


def test_update_is_clipped_jointly_before_noise():
    d = OrderedDict(a=torch.tensor([3.0, 4.0]), b=torch.tensor([0.0]))   # norm 5
    out = privatize(d, clip_norm=1.0, noise_multiplier=0.0, generator=None)
    assert torch.allclose(out["a"], torch.tensor([0.6, 0.8])) and torch.equal(out["b"], torch.tensor([0.0]))


def test_noise_is_reproducible_from_the_generator():
    d = OrderedDict(w=torch.zeros(10))
    a = privatize(d, 1.0, 1.0, torch.Generator().manual_seed(3))
    b = privatize(d, 1.0, 1.0, torch.Generator().manual_seed(3))
    assert torch.equal(a["w"], b["w"])


def test_rejects_invalid_arguments():
    d = OrderedDict(w=torch.zeros(2))
    with pytest.raises(ValueError):
        privatize(d, 0.0, 1.0, None)
    with pytest.raises(ValueError):
        privatize(d, 1.0, -1.0, None)
