import io
import logging
import os
from collections import OrderedDict
from typing import Dict, Generator, Optional, Tuple

import numpy as np
import torch

from fedlearn.communication.generated import fedlearn_pb2

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
        parameters[name] = torch.tensor(np_array)
    return parameters, proto.num_examples_trained


def parameters_to_chunks(
        params: OrderedDict[str, torch.Tensor],
        num_examples: int,
        chunk_size: int = CHUNK_SIZE,
        compress: Optional[bool] = None,
) -> Generator[Dict, None, None]:
    """Memory-efficient serialization using torch.save with memoryview (streaming)."""
    if compress is None:
        compress = USE_COMPRESSION

    try:
        log.debug("Serializing %d tensors with torch.save", len(params))

        # Don't use `with io.BytesIO()`: closing the buffer while a memoryview
        # export from getbuffer() is still alive raises BufferError. The buffer
        # is freed when the local refs die at function exit.
        buffer = io.BytesIO()
        torch.save(params, buffer)
        view = memoryview(buffer.getbuffer())
        serialized = view.tobytes() if compress else view

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


def chunks_to_parameters(
        chunks_data: bytes,
        compressed: Optional[bool] = None,
) -> Tuple[OrderedDict[str, torch.Tensor], int]:
    """Reconstruct a state_dict from a serialized blob using torch.load."""
    if compressed is None:
        compressed = USE_COMPRESSION

    try:
        if compressed and LZ4_AVAILABLE:
            import lz4.frame
            log.debug("Decompressing lz4 blob")
            data = lz4.frame.decompress(chunks_data)
        else:
            data = chunks_data

        # weights_only=True prevents arbitrary pickle execution.
        with io.BytesIO(data) as buffer:
            model_data = torch.load(buffer, map_location='cpu', weights_only=True)

        return model_data['parameters'], model_data['num_examples']

    except Exception:
        log.exception("chunks_to_parameters failed")
        raise
