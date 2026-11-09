"""Unit and algorithmic tests for Differentially Private Adaptive Clipping."""
import math
from collections import OrderedDict

import pytest
import torch

from fedlearn.privacy.adaptive_clipping import DPAdaptiveClipping


def test_adaptive_clipping_validation():
    with pytest.raises(ValueError, match="initial_clip_norm"):
        DPAdaptiveClipping(initial_clip_norm=-1.0)
    with pytest.raises(ValueError, match="target_quantile"):
        DPAdaptiveClipping(target_quantile=1.5)
    with pytest.raises(ValueError, match="clip_learning_rate"):
        DPAdaptiveClipping(clip_learning_rate=0.0)
    with pytest.raises(ValueError, match="noise_multiplier"):
        DPAdaptiveClipping(noise_multiplier=-0.1)
    with pytest.raises(ValueError, match="min_clip_norm"):
        DPAdaptiveClipping(min_clip_norm=-0.5)
    with pytest.raises(ValueError, match="max_clip_norm"):
        DPAdaptiveClipping(min_clip_norm=2.0, max_clip_norm=1.0)


def test_adaptive_clipping_increases_when_clients_exceed_budget():
    """If client norms consistently exceed C, C must adaptively increase."""
    ac = DPAdaptiveClipping(
        initial_clip_norm=1.0,
        target_quantile=0.5,
        clip_learning_rate=0.2,
        noise_multiplier=0.0,  # Zero noise to isolate gradient direction
    )
    # 10 clients with norm 5.0 (all clipped -> raw unclipped fraction = 0.0 < 0.5)
    large_norms = [5.0] * 10
    new_c = ac.step(large_norms)

    # factor = exp(-0.2 * (0.0 - 0.5)) = exp(0.1) > 1.0
    expected = 1.0 * math.exp(0.1)
    assert new_c == pytest.approx(expected)
    assert new_c > 1.0


def test_adaptive_clipping_decreases_when_clients_within_budget():
    """If client norms are well within C, C must adaptively decrease."""
    ac = DPAdaptiveClipping(
        initial_clip_norm=2.0,
        target_quantile=0.5,
        clip_learning_rate=0.2,
        noise_multiplier=0.0,
    )
    # 10 clients with norm 0.1 (all unclipped -> raw unclipped fraction = 1.0 > 0.5)
    small_norms = [0.1] * 10
    new_c = ac.step(small_norms)

    # factor = exp(-0.2 * (1.0 - 0.5)) = exp(-0.1) < 1.0
    expected = 2.0 * math.exp(-0.1)
    assert new_c == pytest.approx(expected)
    assert new_c < 2.0


def test_adaptive_clipping_respects_clamps():
    ac = DPAdaptiveClipping(
        initial_clip_norm=1.0,
        min_clip_norm=0.5,
        max_clip_norm=1.5,
        clip_learning_rate=5.0,
        noise_multiplier=0.0,
    )
    # Huge step up
    c_high = ac.step([10.0] * 5)
    assert c_high == pytest.approx(1.5)

    # Huge step down
    c_low = ac.step([0.01] * 5)
    assert c_low == pytest.approx(0.5)


def test_adaptive_clipping_clip_deltas():
    ac = DPAdaptiveClipping(initial_clip_norm=1.0)
    d1 = OrderedDict(w=torch.tensor([0.6, 0.8]))  # norm = 1.0 -> within budget
    d2 = OrderedDict(w=torch.tensor([3.0, 4.0]))  # norm = 5.0 -> scaled by 1/5
    clipped, norms = ac.clip_deltas([d1, d2])

    assert norms[0] == pytest.approx(1.0)
    assert norms[1] == pytest.approx(5.0)
    assert torch.allclose(clipped[0]["w"], torch.tensor([0.6, 0.8]))
    assert torch.allclose(clipped[1]["w"], torch.tensor([0.6, 0.8]))


def test_adaptive_clipping_isolated_generator():
    """Verify that omitting seed uses fresh OS entropy and is independent of torch.manual_seed."""
    ac1 = DPAdaptiveClipping(initial_clip_norm=1.0, noise_multiplier=2.0)
    ac2 = DPAdaptiveClipping(initial_clip_norm=1.0, noise_multiplier=2.0)

    torch.manual_seed(42)
    step1 = ac1.step([1.0] * 10)

    torch.manual_seed(42)
    step2 = ac2.step([1.0] * 10)

    # Two independent instances with fresh entropy should produce different steps
    assert step1 != step2
