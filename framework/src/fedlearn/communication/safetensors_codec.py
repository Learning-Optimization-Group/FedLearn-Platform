"""Self-contained, deterministic safetensors-shaped codec for the FedAvg state-dict.

This is the cross-language wire format for model state-dicts (replacing the legacy
torch.save/pickle blob). It is intentionally minimal and DETERMINISTIC so the C++ mobile
core (mobile_client/shared/src/ModelManager.cpp, Phase 3c T5) can produce byte-identical
output and a golden fixture can pin the contract:

  bytes = u64_le(header_len) ++ header_json_utf8 ++ raw_tensor_data

The header JSON is compact (no spaces) with tensor entries emitted in the order given
(insertion order, preserved by dict), each {"dtype":"F32","shape":[...],"data_offsets":[s,e]},
followed by an optional "__metadata__" object of string->string. Tensor data is concatenated
in the same order; float32 little-endian only (the mobile FL models are float32).

A magic-prefix sniff (the 8-byte length is always < a pickle's first bytes) plus the JSON
parse means a legacy pickle blob fails loudly here rather than being silently mis-read.
"""
from __future__ import annotations

import json
import struct
from typing import Dict, List, Tuple

import numpy as np

_DTYPE = "F32"


def save_safetensors(tensors: List[Tuple[str, np.ndarray]],
                     metadata: Dict[str, str] | None = None) -> bytes:
    """Serialize named float32 tensors (in the given order) + optional string metadata."""
    header: Dict[str, object] = {}
    data = bytearray()
    for name, arr in tensors:
        a = np.ascontiguousarray(arr, dtype="<f4")
        start = len(data)
        data += a.tobytes()
        header[name] = {"dtype": _DTYPE, "shape": list(a.shape), "data_offsets": [start, len(data)]}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(hjson)) + hjson + bytes(data)


def load_safetensors(blob: bytes) -> Tuple[List[Tuple[str, np.ndarray]], Dict[str, str]]:
    """Inverse of save_safetensors. Returns (named tensors in stored order, metadata).

    Validates the header against the actual blob (FR-8) so a malformed or malicious blob fails loudly
    here instead of silently mis-reading bytes or raising a cryptic numpy error deeper in the
    pipeline. This runs on the untrusted gRPC receive path (serializer.chunks_to_parameters), so it
    is a hardening boundary alongside the non-finite reject (SE-3): out-of-range offsets would
    otherwise slice leniently and return whatever bytes exist, a wrong dtype would be read as F32,
    and a negative shape dim would abuse numpy's reshape infer-dimension.
    """
    if len(blob) < 8:
        raise ValueError("safetensors blob too short")
    (hlen,) = struct.unpack("<Q", blob[:8])
    if 8 + hlen > len(blob):
        raise ValueError("safetensors header length exceeds blob (corrupt or legacy pickle blob)")
    header = json.loads(blob[8:8 + hlen].decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")
    meta = header.pop("__metadata__", {})
    if not isinstance(meta, dict):
        raise ValueError("safetensors __metadata__ is not an object")
    base = 8 + hlen
    data_len = len(blob) - base
    out: List[Tuple[str, np.ndarray]] = []
    for name, info in header.items():
        if not isinstance(info, dict) or not {"dtype", "shape", "data_offsets"} <= info.keys():
            raise ValueError(f"safetensors: malformed entry for tensor {name!r}")
        if info["dtype"] != _DTYPE:
            raise ValueError(f"safetensors: unsupported dtype {info['dtype']!r} for {name!r} (only {_DTYPE})")
        shape = info["shape"]
        if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
            raise ValueError(f"safetensors: invalid shape {shape!r} for {name!r}")
        offsets = info["data_offsets"]
        if (not isinstance(offsets, list) or len(offsets) != 2
                or not all(isinstance(o, int) for o in offsets)):
            raise ValueError(f"safetensors: invalid data_offsets {offsets!r} for {name!r}")
        s, e = offsets
        if not (0 <= s <= e <= data_len):
            raise ValueError(
                f"safetensors: data_offsets [{s},{e}] out of range (data length {data_len}) for {name!r}")
        expected = 4 * int(np.prod(shape, dtype=np.int64))  # F32 = 4 bytes/element; scalar shape [] -> 1 elt
        if (e - s) != expected:
            raise ValueError(
                f"safetensors: byte count {e - s} != shape {shape} size {expected} for {name!r}")
        arr = np.frombuffer(blob[base + s:base + e], dtype="<f4").reshape(shape)
        out.append((name, arr.copy()))
    return out, meta
