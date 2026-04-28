import pytest
import torch
import numpy as np
from collections import OrderedDict
from fedlearn.communication.serializer import (
    parameters_to_proto,
    proto_to_parameters,
    parameters_to_chunks,
    chunks_to_parameters,
)
from fedlearn.communication.generated import fedlearn_pb2


def make_state_dict(seed: int = 0) -> OrderedDict:
    torch.manual_seed(seed)
    return OrderedDict([
        ("layer.weight", torch.randn(3, 4, dtype=torch.float32)),
        ("layer.bias",   torch.randn(3,    dtype=torch.float32)),
    ])


class TestProtoRoundtrip:

    def test_parameters_to_proto_then_back_preserves_values(self):
        original = make_state_dict()
        proto = parameters_to_proto(original, num_examples=100)
        recovered, num_examples = proto_to_parameters(proto)

        assert num_examples == 100
        for key in original:
            assert key in recovered
            assert torch.allclose(original[key], recovered[key], atol=1e-6)

    def test_proto_roundtrip_preserves_shape(self):
        original = make_state_dict()
        proto = parameters_to_proto(original, num_examples=50)
        recovered, _ = proto_to_parameters(proto)
        for key in original:
            assert original[key].shape == recovered[key].shape

    def test_proto_to_parameters_rejects_unsafe_dtype(self):
        # Inject a malicious dtype string into the proto
        proto = fedlearn_pb2.ModelParameters()
        t = proto.tensors["evil"]
        t.data = np.zeros(4, dtype=np.float32).tobytes()
        t.dims.extend([4])
        t.dtype = "object"  # Not in the safe whitelist

        with pytest.raises(ValueError, match="Unsafe dtype"):
            proto_to_parameters(proto)

    def test_proto_to_parameters_rejects_shape_mismatch(self):
        proto = fedlearn_pb2.ModelParameters()
        t = proto.tensors["bad"]
        t.data = np.zeros(4, dtype=np.float32).tobytes()  # 4 floats of data
        t.dims.extend([2, 5])  # But dims say 2*5=10 elements — mismatch!
        t.dtype = "float32"

        with pytest.raises(ValueError, match="Shape mismatch"):
            proto_to_parameters(proto)

    def test_proto_to_parameters_rejects_zero_dimension(self):
        proto = fedlearn_pb2.ModelParameters()
        t = proto.tensors["zero_dim"]
        t.data = b""
        t.dims.extend([0])
        t.dtype = "float32"

        with pytest.raises(ValueError, match="Invalid dimension"):
            proto_to_parameters(proto)


class TestChunkedRoundtrip:

    def test_chunks_roundtrip_single_chunk(self):
        original = make_state_dict(seed=1)
        # Use a very large chunk size so everything fits in one chunk
        chunks = list(parameters_to_chunks(original, num_examples=200, chunk_size=10 * 1024 * 1024))
        assert len(chunks) == 1
        assert chunks[0]["is_final_chunk"] is True

        # Reassemble and recover
        raw_bytes = chunks[0]["chunk_data"]
        recovered, num_examples = chunks_to_parameters(raw_bytes, compressed=False)

        assert num_examples == 200
        for key in original:
            assert torch.allclose(original[key], recovered[key], atol=1e-6)

    def test_chunks_roundtrip_multiple_chunks(self):
        original = make_state_dict(seed=2)
        # Force multiple chunks with a tiny chunk size (1 byte per chunk)
        chunks = list(parameters_to_chunks(original, num_examples=42, chunk_size=1))
        assert len(chunks) > 1
        assert chunks[-1]["is_final_chunk"] is True
        # Verify chunk indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

        # Reassemble raw bytes from all chunks
        raw_bytes = b"".join(c["chunk_data"] for c in chunks)
        recovered, num_examples = chunks_to_parameters(raw_bytes, compressed=False)

        assert num_examples == 42
        for key in original:
            assert torch.allclose(original[key], recovered[key], atol=1e-6)

    def test_chunks_metadata_is_consistent(self):
        original = make_state_dict()
        chunks = list(parameters_to_chunks(original, num_examples=10, chunk_size=64))
        total = chunks[0]["total_chunks"]
        for chunk in chunks:
            assert chunk["total_chunks"] == total
            assert chunk["num_examples"] == 10
