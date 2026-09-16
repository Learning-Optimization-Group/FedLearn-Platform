import logging
import os
from collections import OrderedDict
from typing import Dict, Generator, Optional, Tuple

import numpy as np
import torch

from fedlearn.communication.generated import fedlearn_pb2
from fedlearn.communication.safetensors_codec import load_safetensors, save_safetensors

log = logging.getLogger(__name__)

try:
    import lz4.frame  # noqa: F401
    LZ4_AVAILABLE = True
except ImportError:
    log.warning("lz4 not installed. Install with: pip install lz4 — compression disabled.")
    LZ4_AVAILABLE = False

# Compression is opt-in via env var; off by default for parity with existing deployments.
USE_COMPRESSION = LZ4_AVAILABLE and os.environ.get("FEDLEARN_USE_COMPRESSION", "0") == "1"

# Default chunk size. 50MB is appropriate for LAN; tune down for WAN / AWS ALB.
# Override with FEDLEARN_CHUNK_SIZE_MB.
_DEFAULT_CHUNK_SIZE_MB = int(os.environ.get("FEDLEARN_CHUNK_SIZE_MB", "4"))
CHUNK_SIZE = _DEFAULT_CHUNK_SIZE_MB * 1024 * 1024

ModelParameters = fedlearn_pb2.ModelParameters
Tensor = fedlearn_pb2.Tensor


def parameters_to_proto(parameters: OrderedDict[str, torch.Tensor], num_examples: int) -> ModelParameters:
    """Serialize a PyTorch state_dict to a proto message."""
    tensors = {}
    for name, tensor in parameters.items():
        np_array = tensor.cpu().detach().numpy()
        tensors[name] = Tensor(
            data=np_array.tobytes(),
            dims=list(np_array.shape),
            dtype=str(np_array.dtype),
        )
    return ModelParameters(tensors=tensors, num_examples_trained=num_examples)


# Whitelist of safe numpy dtypes to prevent arbitrary dtype injection.
_SAFE_DTYPES = {
    'float16', 'float32', 'float64',
    'int8', 'int16', 'int32', 'int64',
    'uint8',
    'bool',
    'bfloat16',
}


def _reject_non_finite(name: str, np_array: np.ndarray) -> None:
    """Reject a deserialized tensor carrying NaN/Inf. A single malicious or buggy client could
    otherwise push non-finite weights that propagate into the averaged global model and destroy it
    for every honest client in a round (SE-3 poisoning defense)."""
    if not np.isfinite(np_array).all():
        raise ValueError(
            f"Tensor '{name}' contains non-finite values (NaN/Inf); rejecting poisoned update."
        )


def proto_to_parameters(proto: ModelParameters) -> Tuple[OrderedDict[str, torch.Tensor], int]:
    """Deserialize a proto message to a PyTorch state_dict."""
    parameters: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, tensor_proto in proto.tensors.items():
        dtype_str = tensor_proto.dtype
        if dtype_str not in _SAFE_DTYPES:
            raise ValueError(f"Unsafe dtype '{dtype_str}' for tensor '{name}'. Allowed: {_SAFE_DTYPES}")

        np_array = np.frombuffer(tensor_proto.data, dtype=np.dtype(dtype_str))

        expected_size = 1
        for d in tensor_proto.dims:
            if d <= 0:
                raise ValueError(f"Invalid dimension {d} for tensor '{name}'")
            expected_size *= d
        if expected_size != len(np_array):
            raise ValueError(
                f"Shape mismatch for tensor '{name}': dims product {expected_size} != data length {len(np_array)}")

        np_array = np_array.reshape(tensor_proto.dims).copy()
        _reject_non_finite(name, np_array)
        parameters[name] = torch.tensor(np_array)
    return parameters, proto.num_examples_trained


def state_dict_to_safetensors(params: OrderedDict[str, torch.Tensor], num_examples: int = 0) -> bytes:
    """Serialize a full state_dict to a single deterministic safetensors blob — the inverse of
    chunks_to_parameters. float32-only and FAIL-LOUD: a non-float tensor raises here rather than
    being silently coerced to F32 (save_safetensors would otherwise cast int/bool and corrupt the
    model). Shared by the upload chunk stream AND the global-model DOWNLOAD stream so the wire is
    safetensors — never torch.save/pickle — and decodable by the libtorch-free, F32-only mobile
    C++ core."""
    for name, tensor in params.items():
        if tensor.dtype != torch.float32:
            raise ValueError(
                f"Tensor '{name}' has dtype {tensor.dtype}; only float32 is supported "
                "on the safetensors wire format. Cast to float32 before training."
            )
        # Fail-loud on two edges the wire mishandled silently (adversarial-audit fixes): a 0-dim
        # scalar round-trips with the wrong shape (the wire carries rank>=1 model params), and a
        # parameter literally named '__metadata__' collides with the safetensors metadata block and
        # is dropped. Neither occurs for a real state_dict; reject rather than corrupt.
        if tensor.dim() == 0:
            raise ValueError(
                f"Tensor '{name}' is 0-dim (scalar); the safetensors wire carries rank>=1 model "
                "parameters — reshape or exclude a scalar rather than send it."
            )
        if name == "__metadata__":
            raise ValueError(
                "Parameter name '__metadata__' is reserved (it names the safetensors metadata "
                "block); rename the parameter."
            )
    named_arrays = [(name, tensor.detach().cpu().numpy()) for name, tensor in params.items()]
    return save_safetensors(named_arrays, metadata={"num_examples": str(num_examples)})


def parameters_to_chunks(
        params: OrderedDict[str, torch.Tensor],
        num_examples: int,
        chunk_size: int = CHUNK_SIZE,
        compress: Optional[bool] = None,
) -> Generator[Dict, None, None]:
    """Memory-efficient serialization using the safetensors wire format."""
    if compress is None:
        compress = USE_COMPRESSION

    try:
        log.debug("Serializing %d tensors with safetensors", len(params))

        serialized = state_dict_to_safetensors(params, num_examples)

        original_size = len(serialized)
        log.debug("Serialized size: %.2f MB", original_size / (1024 ** 2))

        if compress and LZ4_AVAILABLE:
            import lz4.frame
            log.debug("Compressing with lz4")
            compressed = lz4.frame.compress(serialized, compression_level=lz4.frame.COMPRESSIONLEVEL_MIN)
            data_to_send = compressed
            ratio = original_size / len(compressed) if compressed else 1.0
            log.debug("Compressed size: %.2f MB (ratio %.2fx)", len(compressed) / (1024 ** 2), ratio)
        else:
            data_to_send = serialized

        del serialized

        total_size = len(data_to_send)
        num_chunks = (total_size + chunk_size - 1) // chunk_size
        log.debug("Emitting %d chunks of ~%d bytes", num_chunks, chunk_size)

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total_size)

            yield {
                'chunk_index': i,
                'total_chunks': num_chunks,
                'chunk_data': data_to_send[start:end],
                'is_final_chunk': (i == num_chunks - 1),
                'num_examples': num_examples,
            }

    except Exception:
        log.exception("parameters_to_chunks failed")
        raise


def _looks_like_safetensors(data: bytes) -> bool:
    """FR-27: positively identify a safetensors blob to avoid false-rejecting it as legacy pickle/zip.

    The safetensors format is: an 8-byte little-endian u64 header length N, then N bytes of a JSON
    header object (always starting with ``{``), then the tensor data. Requiring a sane N (fits in the
    blob, non-empty) plus a ``{`` at the header start distinguishes real safetensors from a pickle
    (``0x80``) or zip (``PK``) blob whose bytes only coincidentally collide with the magic sniff.
    """
    if len(data) < 9:
        return False
    header_len = int.from_bytes(data[:8], "little")
    return 0 < header_len <= len(data) - 8 and data[8:9] == b"{"


def chunks_to_parameters(
        chunks_data: bytes,
        compressed: Optional[bool] = None,
) -> Tuple[OrderedDict[str, torch.Tensor], int]:
    """Reconstruct a state_dict from a safetensors blob (optionally lz4-compressed)."""
    if compressed is None:
        compressed = USE_COMPRESSION

    try:
        if compressed and LZ4_AVAILABLE:
            import lz4.frame
            log.debug("Decompressing lz4 blob")
            data = lz4.frame.decompress(chunks_data)
        else:
            data = chunks_data

        # Sniff for legacy pickle/zip blobs and fail loudly rather than silently mis-reading.
        # torch.save produces a zip archive starting with PK\x03\x04; raw pickle starts with 0x80.
        # FR-27: gate the sniff behind a POSITIVE safetensors check first. A valid safetensors blob
        # begins with a little-endian u64 header length whose low byte is legitimately 0x80 (header
        # 128/384/... bytes) or spells b"PK" (header_len ≡ 19280 mod 65536) — the bare magic-byte
        # sniff false-rejected those well-formed payloads. When the blob positively parses as
        # safetensors (sane header length followed by a JSON object), skip the legacy guard.
        if not _looks_like_safetensors(data) and len(data) >= 2 and (data[:2] == b"PK" or data[0] == 0x80):
            raise ValueError(
                "Received a legacy pickle/zip blob (torch.save format). "
                "Only safetensors wire format is accepted. Re-upload with an updated client."
            )

        named_arrays, meta = load_safetensors(data)

        params: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, arr in named_arrays:
            _reject_non_finite(name, arr)
            params[name] = torch.tensor(arr)

        num_examples = int(meta["num_examples"])
        return params, num_examples

    except Exception:
        log.exception("chunks_to_parameters failed")
        raise
