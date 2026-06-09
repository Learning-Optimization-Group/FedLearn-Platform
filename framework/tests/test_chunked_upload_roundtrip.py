"""Regression test for the chunked (LLM-scale) model-upload path.

The client streams a model update as ``ModelUpdateChunk`` messages produced by
``GrpcClient._generate_model_chunks``; the server reassembles the chunk bytes and
deserializes them with ``serializer.chunks_to_parameters``. Those two halves must
agree on the on-the-wire payload format.

Before the fix, the client serialized a *bare* ``state_dict`` while the server
expected a *wrapped* ``{'parameters', 'num_examples'}`` dict, so every chunked
upload (i.e. every transformer / LLM-scale model — the case DeComFL exists for)
aborted with ``KeyError: 'parameters'`` on the server. This test exercises the
real producer against the real consumer and would have caught that.
"""
from collections import OrderedDict
from types import SimpleNamespace

import torch

from fedlearn.client.grpc_client import GrpcClient
from fedlearn.communication.serializer import chunks_to_parameters


def _make_params() -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict(
        [
            ("layer.weight", torch.randn(8, 4)),
            ("layer.bias", torch.randn(8)),
        ]
    )


def test_chunked_upload_roundtrips_through_server_deserializer():
    params = _make_params()
    num_examples = 123
    # _generate_model_chunks only touches self.client_id, so a stand-in avoids
    # constructing a real channel-backed client.
    fake_self = SimpleNamespace(client_id="c0")

    # Small chunk size forces multi-chunk reassembly.
    chunks = list(
        GrpcClient._generate_model_chunks(
            fake_self, params, num_examples, round_number=7, chunk_size=1024
        )
    )

    assert len(chunks) >= 1
    assert chunks[-1].is_final_chunk
    assert chunks[-1].total_chunks == len(chunks)
    assert chunks[-1].num_examples == num_examples

    blob = b"".join(c.chunk_data for c in chunks)

    # Server-side reassembly. This raised KeyError('parameters') before the fix.
    recovered, recovered_n = chunks_to_parameters(blob)

    assert recovered_n == num_examples
    assert set(recovered.keys()) == set(params.keys())
    for name, tensor in params.items():
        assert torch.allclose(recovered[name], tensor)
