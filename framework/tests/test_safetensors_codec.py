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


def test_shape_product_int64_overflow_is_rejected():
    """A shape whose element product overflows int64 must still be rejected, not mis-read. The
    byte-count guard computes expected = 4*int(np.prod(shape, dtype=int64)); shape [2**32, 2**32]
    overflows that product to 0, so a matching data_offsets [0,0] slips PAST the explicit byte-count
    check — numpy's reshape (which uses the true Python-int product) is the backstop that rejects it.
    Pin that the overflow blob is rejected so this untrusted-input parser can't be regressed into a
    silent mis-read if the reshape backstop is ever weakened."""
    blob = _blob({"w": {"dtype": "F32", "shape": [2 ** 32, 2 ** 32], "data_offsets": [0, 0]}}, b"")
    with pytest.raises(ValueError):
        load_safetensors(blob)


@pytest.mark.parametrize("bad", [
    b"\x00\x03",                                             # truncated: fewer than 8 header-length bytes
    struct.pack("<Q", 10 ** 9) + b"{}",                     # header length far exceeds the blob (legacy-pickle guard)
    struct.pack("<Q", 0),                                   # zero-length header -> empty JSON -> JSONDecodeError (a ValueError)
])
def test_malformed_framing_is_rejected(bad):
    with pytest.raises(ValueError):
        load_safetensors(bad)


@pytest.mark.parametrize("header", [
    [1, 2, 3],                                                       # header is a JSON list, not an object
    5,                                                               # header is a JSON int
    {"__metadata__": "not-a-dict"},                                 # metadata block is not an object
    {"w": {"dtype": "F32", "shape": [2]}},                          # entry missing data_offsets
    {"w": {"dtype": "F32", "shape": [1], "data_offsets": [8, 0]}},  # inverted offsets (s > e)
])
def test_malformed_header_entries_are_rejected(header):
    # Every structurally-invalid header must raise ValueError (the correct client-error status), never
    # crash with a non-ValueError or silently mis-read.
    data = b"\x00" * 8
    with pytest.raises(ValueError):
        load_safetensors(_blob(header, data))


def test_overlapping_offsets_are_accepted_and_safe():
    # Non-overlapping/contiguous layout is NOT required on decode: two tensors that share the same
    # data range decode to independent copies (safe — bounds are still enforced). Documents the
    # lenient-but-safe contract so a future "tighten" doesn't mistake it for a bug.
    blob = _blob({
        "a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "b": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
    }, struct.pack("<f", 2.5))
    out, _ = load_safetensors(blob)
    assert {n for n, _ in out} == {"a", "b"}
    assert all(float(arr[0]) == 2.5 for _, arr in out)
