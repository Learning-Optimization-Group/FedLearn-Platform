"""TE-15: the wire-byte accounting must reflect the real serialized payload sizes and pin the
communication contrast that is the whole point of the FedAvg-vs-DeComFL comparison.
"""
from collections import OrderedDict

import torch

from benchmarks.wire_bytes import (
    first_order_model_bytes,
    decomfl_upload_bytes,
    decomfl_download_config_bytes,
)


def test_first_order_bytes_track_model_size():
    small = first_order_model_bytes(OrderedDict([("w", torch.zeros(10, dtype=torch.float32))]))
    big = first_order_model_bytes(OrderedDict([("w", torch.zeros(1000, dtype=torch.float32))]))
    assert big > small
    # ~4 bytes per float32 param, plus a small safetensors header.
    assert big >= 1000 * 4
    assert (big - small) >= (1000 - 10) * 4


def test_decomfl_upload_is_K_times_P_scalars_scale():
    b1 = decomfl_upload_bytes(num_local_steps=1, num_perturbations=10)
    b2 = decomfl_upload_bytes(num_local_steps=1, num_perturbations=20)
    assert b2 > b1
    # 8 bytes per float64 scalar + ~8 bytes per int64 seed => growth ~ 16 bytes per extra (k,p).
    assert (b2 - b1) >= 10 * 16 * 0.5  # allow for varint/int64 protobuf encoding


def test_decomfl_beats_first_order_by_orders_of_magnitude():
    """The paper's thesis on the communication axis: for a ~100k-param model, one DeComFL round is
    kilobytes-or-less while one first-order round is hundreds of KB — a >100x per-round win."""
    fo = first_order_model_bytes(OrderedDict([("w", torch.zeros(100_000, dtype=torch.float32))]))
    zo_up = decomfl_upload_bytes(num_local_steps=1, num_perturbations=10)
    zo_down = decomfl_download_config_bytes(num_local_steps=1, num_perturbations=10)
    assert fo >= 100_000 * 4                 # ~400 KB
    assert zo_up < 1000                      # tiny absolute
    assert zo_up < fo / 100                  # >100x smaller per round
    assert (zo_up + zo_down) < fo / 100
