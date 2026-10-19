"""The execution contract's reproducible dropout masks (DROPOUT_MASKS_SEEDED_V1).

Every runtime draws the same mask for a given run seed, round, local step and dropout layer, so a training step that
uses dropout is reproducible from the contract and the data. The algorithm is specified in execution_contract.proto:
the SplitMix64 stream of BATCH_ORDER_SEEDED_PERMUTATION_V1, one draw per activation element in row-major order, a
53-bit uniform, drop iff below the rate, and the float32 inverted-dropout scale. The native trainer implements the
same; both are pinned by framework/tests/fixtures/execution_contract_v1/dropout_masks_v1.golden.
"""
from __future__ import annotations

import math

import numpy as np

from fedlearn.contract.batch_order import SplitMix64, mix

_MASK64 = (1 << 64) - 1


def dropout_state(seed: int, round_: int, step: int, layer: int) -> int:
    for value in (seed, round_, step, layer):
        if not 0 <= value <= _MASK64:
            raise ValueError("seed, round, step and layer must be unsigned 64-bit integers")
    return mix(mix(mix(mix(seed) ^ round_) ^ step) ^ layer)


def _check_rate(rate: float) -> None:
    if not (math.isfinite(rate) and 0 <= rate < 1):
        raise ValueError(f"a dropout rate must be in [0, 1), not {rate}")


def dropout_keep(n: int, rate: float, seed: int, round_: int, step: int, layer: int) -> list[bool]:
    """Which of a layer's ``n`` activation elements (row-major) this step keeps."""
    _check_rate(rate)
    rng = SplitMix64(dropout_state(seed, round_, step, layer))
    return [(rng.next() >> 11) * 2.0 ** -53 >= rate for _ in range(n)]


def dropout_scale(rate: float) -> np.float32:
    """The kept elements' multiplier: float32(1) / float32(1 - rate), as torch's inverted dropout computes it."""
    _check_rate(rate)
    return np.float32(1) / np.float32(1 - rate)


def dropout_mask(n: int, rate: float, seed: int, round_: int, step: int, layer: int) -> np.ndarray:
    """The float32 mask a layer's activations are multiplied by: the scale where kept, 0 where dropped."""
    scale = dropout_scale(rate)
    return np.array([scale if k else np.float32(0) for k in dropout_keep(n, rate, seed, round_, step, layer)],
                    dtype=np.float32)
