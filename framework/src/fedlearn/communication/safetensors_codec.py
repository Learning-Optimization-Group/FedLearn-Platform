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
    """Inverse of save_safetensors. Returns (named tensors in stored order, metadata)."""
    if len(blob) < 8:
        raise ValueError("safetensors blob too short")
    (hlen,) = struct.unpack("<Q", blob[:8])
    if 8 + hlen > len(blob):
        raise ValueError("safetensors header length exceeds blob (corrupt or legacy pickle blob)")
    header = json.loads(blob[8:8 + hlen].decode("utf-8"))
    meta = header.pop("__metadata__", {})
    base = 8 + hlen
    out: List[Tuple[str, np.ndarray]] = []
    for name, info in header.items():
        s, e = info["data_offsets"]
        arr = np.frombuffer(blob[base + s:base + e], dtype="<f4").reshape(info["shape"])
        out.append((name, arr.copy()))
    return out, meta
