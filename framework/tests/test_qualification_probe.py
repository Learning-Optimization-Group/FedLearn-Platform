"""The qualification probe is a stable number to compare a device against (Stage 4 S7).

A device proves it runs a trainable program correctly by reproducing the exporter's two probe losses within a
tolerance. That only works if the reference does not move under float rounding. The first probe batch (shifted copies
of one period-17 pattern) made ResNet-18's step-1 gradient a cancellation of nearly equal terms in training-mode
BatchNorm: torch disagreed with itself, float32 against float64, by 1.3e-2 at step 2, and the vivo was refused at
qualification while running the program correctly. Independent rows from an integer hash remove the cancellation, and
the exporter now refuses to record a probe that rounding alone moves by a tenth of its tolerance.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "mobile_client", "scripts"))
sys.path.insert(0, os.path.join(REPO, "fl-runtime"))

import pte_export  # noqa: E402
import recipes  # noqa: E402


def _reference_value(step: int, n: int) -> float:
    """The probe's value at flat index n (row i, column j: n = i * width + j), in 32-bit unsigned arithmetic."""
    h = (n * 2654435761 + step * 40503) % 2**32
    h ^= h >> 15
    h = (h * 2246822519) % 2**32
    h ^= h >> 13
    return (h % 17 - 8) / 8


def test_the_probe_batch_is_the_hashed_pattern_the_native_probe_builds():
    x, y = pte_export.probe_batch(2, rows=3, width=5, classes=2)
    expected = [[_reference_value(2, i * 5 + j) for j in range(5)] for i in range(3)]
    assert x.dtype == torch.float32 and x.tolist() == expected
    assert y.tolist() == [0, 1, 0]


def test_the_probe_batch_values_are_pinned():
    """The C++ probe (Qualification.cpp) builds the same values; these are what both must produce."""
    x, _ = pte_export.probe_batch(1, rows=2, width=4, classes=3)
    assert x.flatten().tolist() == [0.0, 0.625, -0.25, 0.875, 0.5, -1.0, 0.625, -0.5]


def test_probe_rows_are_independent_not_shifted_copies():
    """The old batch's rows were one pattern shifted; a training-mode BatchNorm subtracts what rows share."""
    x, _ = pte_export.probe_batch(1, rows=8, width=3 * 112 * 112, classes=10)
    rows = x.double().numpy()
    corr = np.corrcoef(rows)
    assert np.abs(corr[~np.eye(8, dtype=bool)]).max() < 0.02


def test_the_resnet_recipe_probe_is_stable_under_rounding():
    torch.manual_seed(0)
    model = recipes.get_recipe("CIFAR_RESNET18").build_model("cpu")
    f32 = pte_export.probe_reference(model, rows=8, width=3 * 112 * 112, classes=10, input_shape=(3, 112, 112))
    f64 = pte_export.probe_losses(model, rows=8, width=3 * 112 * 112, classes=10, input_shape=(3, 112, 112),
                                  dtype=torch.float64)
    for step, key in enumerate(("loss_step1", "loss_step2")):
        assert abs(f32[key] - f64[step]) < f32["loss_tolerance"] / 10


def test_a_probe_rounding_alone_can_move_is_refused():
    """ResNet-18 on 32 x 32 inputs normalises its last stage over 8 values per channel: an unstable probe."""
    torch.manual_seed(0)
    model = recipes.get_recipe("CIFAR_RESNET18").build_model("cpu")
    with pytest.raises(ValueError, match="rounding"):
        pte_export.probe_reference(model, rows=8, width=3 * 32 * 32, classes=10, input_shape=(3, 32, 32))
