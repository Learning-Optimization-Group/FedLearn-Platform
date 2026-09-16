"""FR-27: a VALID safetensors blob must not be misclassified as a legacy pickle/zip blob.

chunks_to_parameters sniffs data[0]==0x80 (pickle PROTO opcode) / data[:2]==b"PK" (zip) to reject
legacy torch.save blobs. But a valid safetensors blob begins with a little-endian u64 header length,
whose low byte is legitimately 0x80 whenever the JSON header is 128/384/640/... bytes (and spells
b"PK" for header_len ≡ 19280 mod 65536). The sniff runs before the real parse, so it false-rejects a
well-formed payload — breaking the global-model download / update-upload for those key sets.
"""
import struct
from collections import OrderedDict

import pytest
import torch

from fedlearn.communication.serializer import state_dict_to_safetensors, chunks_to_parameters


def _safetensors_blob_starting_with(first_byte: int):
    """Build a VALID safetensors blob whose first byte == first_byte by padding a key name until
    the little-endian u64 header length lands on the matching residue (mod 256)."""
    base = OrderedDict([("w", torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))])
    for pad in range(400):
        state = OrderedDict(base)
        state[f"k{'x' * pad}"] = torch.tensor([7.0], dtype=torch.float32)
        blob = state_dict_to_safetensors(state, num_examples=100)
        if blob[0] == first_byte:
            return blob, state
    raise RuntimeError(f"could not construct a safetensors blob starting with {first_byte:#x}")


def test_valid_safetensors_with_pickle_magic_first_byte_is_not_rejected():
    # header_len ≡ 128 (mod 256) => blob[0] == 0x80, colliding with the pickle PROTO opcode.
    blob, state = _safetensors_blob_starting_with(0x80)
    assert blob[0] == 0x80
    # Confirm it is genuinely safetensors: u64 header length then a JSON object.
    header_len = struct.unpack("<Q", blob[:8])[0]
    assert 0 < header_len <= len(blob) - 8 and blob[8:9] == b"{"
    # Must PARSE, not raise "legacy pickle/zip blob".
    params, num_examples = chunks_to_parameters(blob, compressed=False)
    assert num_examples == 100
    assert set(params.keys()) == set(state.keys())
    assert torch.allclose(params["w"], torch.tensor([1.0, 2.0, 3.0]))


def test_genuine_pickle_blob_is_still_rejected():
    # A real raw-pickle blob (0x80 PROTO-4 + junk) is NOT valid safetensors and must still be refused
    # — the fix must not weaken the legacy-format guard.
    fake_pickle = b"\x80\x04" + b"\x00" * 64  # header_len decodes huge => not safetensors
    with pytest.raises(ValueError, match="legacy pickle|safetensors"):
        chunks_to_parameters(fake_pickle, compressed=False)
