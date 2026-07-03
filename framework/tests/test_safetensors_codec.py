"""FR-8 — load_safetensors validates the header against the actual blob.

load_safetensors runs on the UNTRUSTED gRPC receive path (serializer.chunks_to_parameters) and in
bundle loading, so a malformed or malicious blob must fail loudly here — not silently mis-read bytes
(lenient slicing on out-of-range offsets), silently ignore a wrong dtype, or abuse reshape(-1).
"""
import json
import struct

import numpy as np
import pytest

from fedlearn.communication.safetensors_codec import load_safetensors, save_safetensors


def _blob(header: dict, data: bytes) -> bytes:
    """Hand-build a raw safetensors blob from a header dict + tensor-data region."""
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(hjson)) + hjson + data


def test_round_trip_preserves_tensors_and_metadata():
    tensors = [("w", np.array([1.0, 2.0, 3.0], dtype="<f4")),
               ("b", np.array([[1.0], [2.0]], dtype="<f4"))]
    out, meta = load_safetensors(save_safetensors(tensors, {"num_examples": "8"}))
    assert meta == {"num_examples": "8"}
    assert [n for n, _ in out] == ["w", "b"]
    assert np.array_equal(out[0][1], tensors[0][1])
    assert np.array_equal(out[1][1], tensors[1][1])


def test_data_offsets_out_of_range_is_rejected():
    # Data region is 8 bytes; offsets claim 1000. Lenient slicing would silently return 8 bytes.
    blob = _blob({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 1000]}}, b"\x00" * 8)
    with pytest.raises(ValueError):
        load_safetensors(blob)


def test_byte_count_shape_mismatch_is_rejected():
    # 8 bytes = 2 float32, but shape [4] implies 16 bytes.
    blob = _blob({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]}}, b"\x00" * 8)
    with pytest.raises(ValueError):
        load_safetensors(blob)


def test_unsupported_dtype_is_rejected():
    # dtype F64 is silently read as F32 today.
    blob = _blob({"w": {"dtype": "F64", "shape": [2], "data_offsets": [0, 8]}}, b"\x00" * 8)
    with pytest.raises(ValueError):
        load_safetensors(blob)


def test_negative_shape_dim_is_rejected():
    # shape [-1] would abuse numpy reshape's infer-dimension instead of failing.
    blob = _blob({"w": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 8]}}, b"\x00" * 8)
    with pytest.raises(ValueError):
        load_safetensors(blob)
