"""C4 — adversarial hardening for the deterministic safetensors wire.

New edges not covered by the existing wire tests, from the adversarial re-audit:
  * FLOAT-BIT fidelity — NaN / +Inf / -Inf / -0.0 / denormals round-trip BIT-identical (the wire is
    a raw float32 buffer, so the exact bit pattern must survive; -0.0 must NOT become +0.0).
  * ENCODE fail-loud guards (audit fixes): a 0-dim scalar (the wire carries rank>=1 model params;
    a scalar silently round-tripped with the wrong shape) and the RESERVED name '__metadata__' (a
    param so named collided with the safetensors metadata block and was silently dropped) are now
    REJECTED loudly, consistent with the existing float32-only rejection.
"""
import struct
from collections import OrderedDict

import numpy as np
import pytest
import torch

from fedlearn.communication.safetensors_codec import load_safetensors, save_safetensors
from fedlearn.communication.serializer import state_dict_to_safetensors


def _bits(arr):
    return np.frombuffer(np.ascontiguousarray(arr, dtype="<f4").tobytes(), dtype="<u4")


@pytest.mark.parametrize("val", [float("nan"), float("inf"), float("-inf"), -0.0, 1e-40, 1.4e-45])
def test_float_edge_values_round_trip_bit_identical(val):
    src = np.array([val, 1.0, -2.5], dtype="<f4")
    blob = save_safetensors([("t", src)])
    (_, out), = load_safetensors(blob)[0]
    # Bit-exact, not just numerically close — -0.0 must stay -0.0, NaN payload preserved.
    assert np.array_equal(_bits(src), _bits(out))


def test_encode_rejects_a_zero_dim_scalar():
    # The wire carries model parameters (rank >= 1); a 0-dim scalar silently round-tripped as shape
    # [1]. Reject it loudly instead (audit fix).
    with pytest.raises(ValueError):
        state_dict_to_safetensors(OrderedDict([("s", torch.tensor(3.5, dtype=torch.float32))]))


def test_encode_rejects_the_reserved_metadata_name():
    # A parameter literally named '__metadata__' collided with the safetensors metadata block and was
    # silently dropped. Reject it loudly (audit fix).
    with pytest.raises(ValueError):
        state_dict_to_safetensors(OrderedDict([
            ("real.weight", torch.tensor([1.0, 2.0], dtype=torch.float32)),
            ("__metadata__", torch.tensor([9.0], dtype=torch.float32)),
        ]))


def test_encode_still_rejects_non_float32_and_accepts_a_normal_state_dict():
    # Regression: the float32-only guard is intact, and an ordinary rank>=1 float32 state_dict encodes.
    with pytest.raises(ValueError):
        state_dict_to_safetensors(OrderedDict([("w", torch.zeros(2, dtype=torch.float64))]))
    blob = state_dict_to_safetensors(OrderedDict([("w", torch.zeros(2, 3, dtype=torch.float32))]))
    assert isinstance(blob, (bytes, bytearray)) and len(blob) > 0
