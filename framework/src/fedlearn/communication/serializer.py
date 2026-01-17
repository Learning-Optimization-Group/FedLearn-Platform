import numpy as np
import torch
import pickle
from collections import OrderedDict
from fedlearn.communication.generated import fedlearn_pb2
from typing import Generator, Dict, Tuple, Optional
try:
    import lz4.frame
    LZ4_AVAILABLE = False
    USE_COMPRESSION = False
except ImportError:
    print("WARNING: lz4 not installed. Install with: pip install lz4")
    print("WARNING: Compression disabled - transfers will be slower")
    LZ4_AVAILABLE = False
    USE_COMPRESSION = False

ModelParameters = fedlearn_pb2.ModelParameters
Tensor = fedlearn_pb2.Tensor
import io

CHUNK_SIZE = 50 * 1024 * 1024  # 50 MB per chunk
USE_COMPRESSION = False

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

def proto_to_parameters(proto: ModelParameters) -> tuple[OrderedDict[str, torch.Tensor], int]:
    """Deserialize a proto message to a PyTorch state_dict."""
    parameters = OrderedDict()
    for name, tensor_proto in proto.tensors.items():
        np_array = np.frombuffer(tensor_proto.data, dtype=np.dtype(tensor_proto.dtype))
        np_array = np_array.reshape(tensor_proto.dims).copy()
        parameters[name] = torch.tensor(np_array)
    return parameters, proto.num_examples_trained


def parameters_to_chunks(
        params: OrderedDict[str, torch.Tensor],
        num_examples: int,
        chunk_size: int = CHUNK_SIZE,
        compress: bool = False
) -> Generator[Dict, None, None]:
    """Memory-efficient serialization using torch.save."""
    try:
        print(f"[Serializer] Using torch.save for {len(params)} tensors...")

        # Use BytesIO buffer
        buffer = io.BytesIO()

        # Save using torch (more memory efficient than pickle)
        model_data = {
            'parameters': params,  # torch.save handles OrderedDict efficiently
            'num_examples': num_examples
        }

        torch.save(model_data, buffer)

        # Get bytes
        serialized = buffer.getvalue()
        buffer.close()

        original_size = len(serialized)
        print(f"[Serializer] Serialized: {original_size / (1024 ** 2):.2f} MB")

        # Compress
        if compress and LZ4_AVAILABLE:
            print(f"[Serializer] Compressing...")
            compressed = lz4.frame.compress(serialized, compression_level=lz4.frame.COMPRESSIONLEVEL_MIN)
            data_to_send = compressed
            ratio = original_size / len(compressed)
            print(f"[Serializer] Compressed: {len(compressed) / (1024 ** 2):.2f} MB (ratio: {ratio:.2f}x)")
        else:
            data_to_send = serialized

        del serialized

        # Chunk
        total_size = len(data_to_send)
        num_chunks = (total_size + chunk_size - 1) // chunk_size
        print(f"[Serializer] Creating {num_chunks} chunk(s)")

        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total_size)

            yield {
                'chunk_index': i,
                'total_chunks': num_chunks,
                'chunk_data': data_to_send[start:end],
                'is_final_chunk': (i == num_chunks - 1),
                'num_examples': num_examples
            }

    except Exception as e:
        print(f"[Serializer] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


def chunks_to_parameters(chunks_data: bytes, compressed: bool = USE_COMPRESSION) -> Tuple[
    OrderedDict[str, torch.Tensor], int]:
    """Reconstruct using torch.load."""
    try:
        # Decompress
        if compressed and LZ4_AVAILABLE:
            print(f"[Serializer] Decompressing...")
            data = lz4.frame.decompress(chunks_data)
        else:
            data = chunks_data

        # Load using torch
        buffer = io.BytesIO(data)
        model_data = torch.load(buffer, map_location='cpu')
        buffer.close()

        return model_data['parameters'], model_data['num_examples']

    except Exception as e:
        print(f"[Serializer] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise