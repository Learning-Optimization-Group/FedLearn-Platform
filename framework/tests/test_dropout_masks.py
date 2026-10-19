"""DROPOUT_MASKS_SEEDED_V1: dropout masks every runtime reproduces (execution_contract.proto)."""
from __future__ import annotations

import os
import struct

import numpy as np
import pytest

from fedlearn.contract.dropout_masks import dropout_keep, dropout_mask, dropout_scale, dropout_state

GOLDEN = os.path.join(os.path.dirname(__file__), "fixtures", "execution_contract_v1", "dropout_masks_v1.golden")


def _lines(kind):
    with open(GOLDEN) as fh:
        for line in fh:
            if line.startswith(kind + " "):
                head, _, tail = line.partition(":")
                yield head.split()[1:], tail.strip()


@pytest.mark.parametrize("case", list(_lines("mask")), ids=lambda c: "-".join(c[0]))
def test_the_golden_masks_are_reproduced(case):
    (n, rate, seed, round_, step, layer), bits = case
    keep = dropout_keep(int(n), float(rate), int(seed), int(round_), int(step), int(layer))
    assert "".join("1" if k else "0" for k in keep) == bits


@pytest.mark.parametrize("case", list(_lines("scale")), ids=lambda c: c[0][0])
def test_the_golden_scales_are_float32_inverted_dropout(case):
    (rate,), hexbits = case
    assert struct.pack("<f", dropout_scale(float(rate))).hex() == hexbits
    assert dropout_scale(float(rate)) == np.float32(1) / np.float32(1 - float(rate))


def test_the_golden_covers_the_rate_range_and_every_stream_input():
    cases = [c for c, _ in _lines("mask")]
    assert {float(c[1]) for c in cases} >= {0.0, 0.3, 0.5, 0.9}
    assert {int(c[2]) for c in cases} >= {0, (1 << 64) - 1}
    assert len({tuple(c[2:]) for c in cases}) >= 6


def test_a_zero_rate_keeps_everything_and_masks_are_kept_elements_times_the_scale():
    assert all(dropout_keep(100, 0.0, 1, 1, 0, 0))
    mask = dropout_mask(64, 0.3, 42, 1, 0, 0)
    keep = dropout_keep(64, 0.3, 42, 1, 0, 0)
    assert mask.dtype == np.float32
    assert [m == dropout_scale(0.3) if k else m == 0 for m, k in zip(mask, keep)] == [True] * 64


def test_the_kept_fraction_follows_the_rate():
    keep = dropout_keep(100_000, 0.3, 7, 1, 0, 0)
    assert abs(sum(keep) / len(keep) - 0.7) < 0.01


def test_step_layer_round_and_seed_each_select_another_stream():
    base = dropout_state(42, 1, 0, 0)
    assert len({base, dropout_state(42, 1, 0, 1), dropout_state(42, 1, 1, 0), dropout_state(42, 2, 0, 0),
                dropout_state(43, 1, 0, 0)}) == 5


def test_a_rate_outside_zero_to_one_is_refused():
    for rate in (-0.1, 1.0, float("nan")):
        with pytest.raises(ValueError):
            dropout_keep(4, rate, 1, 1, 0, 0)
