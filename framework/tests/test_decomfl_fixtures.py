"""Phase 3c T0 — assert the DeComFL golden fixtures carry what the torch-free C++ core needs:
an ordered param_layout summing to flat_dim, and a safetensors state-dict golden that round-trips
and matches the trainable flat vector. This pins the cross-language codec + load-layout contract
that ModelManager (T4/T5) is built against.
"""
from __future__ import annotations

import json
import os

import numpy as np

from fedlearn.communication.safetensors_codec import load_safetensors, save_safetensors

HERE = os.path.join(os.path.dirname(__file__), "fixtures", "decomfl_golden")


def _manifest():
    with open(os.path.join(HERE, "zo_manifest.json")) as fh:
        return json.load(fh)


def test_param_layout_sums_to_flat_dim():
    m = _manifest()
    layout = m["param_layout"]
    assert layout == [
        {"name": "fc1.weight", "shape": [5, 4], "numel": 20},
        {"name": "fc1.bias", "shape": [5], "numel": 5},
    ]
    assert sum(e["numel"] for e in layout) == m["flat_dim"] == 25
    # numel must equal the product of shape for every entry.
    for e in layout:
        assert int(np.prod(e["shape"])) == e["numel"]


def test_state_safetensors_matches_flat_and_roundtrips():
    m = _manifest()
    blob = open(os.path.join(HERE, m["state_file"]), "rb").read()

    # sha256 of the golden matches the manifest.
    import hashlib
    assert hashlib.sha256(blob).hexdigest() == m["state_sha256"]

    tensors, meta = load_safetensors(blob)
    assert meta == {"num_examples": "8"}
    assert [n for n, _ in tensors] == ["fc1.weight", "fc1.bias"]

    # The concatenated tensors (in layout order) reconstruct the committed trainable flat vector.
    flat = np.fromfile(os.path.join(HERE, m["flat_file"]), dtype="<f4")
    recon = np.concatenate([a.reshape(-1) for _, a in tensors])
    assert recon.shape == flat.shape == (25,)
    np.testing.assert_array_equal(recon, flat)

    # Re-saving the loaded tensors is byte-identical (deterministic codec).
    assert save_safetensors(tensors, meta) == blob
