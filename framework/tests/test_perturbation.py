"""Parity + determinism tests for ``canonical_perturbation`` (the DeComFL RNG contract).

Run::

    cd framework && PYTHONPATH=src pytest tests/test_perturbation.py -v

These assert the Python side reproduces the committed golden fixture bit-for-bit on the
pinned ``torch`` version. The C++ mobile port is gated separately and to a ULP tolerance
by ``mobile_client/shared/tests/rng_parity_test.cpp``.
"""
import hashlib
import json
import os

import numpy as np
import pytest
import torch

from fedlearn.estimators.perturbation import canonical_perturbation

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "decomfl_golden")
with open(os.path.join(FIXTURE_DIR, "manifest.json")) as _f:
    MANIFEST = json.load(_f)
CASES = MANIFEST["cases"]
_IDS = [c["file"] for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_matches_golden_fixture(case):
    golden = np.fromfile(os.path.join(FIXTURE_DIR, case["file"]), dtype="<f4")
    z = canonical_perturbation(case["seed"], case["num_params"]).numpy()
    assert z.shape == (case["num_params"],)
    # Same machine + pinned torch => bit-exact reproduction of the frozen contract.
    np.testing.assert_array_equal(z, golden)


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_sha256_matches_manifest(case):
    raw = np.fromfile(os.path.join(FIXTURE_DIR, case["file"]), dtype="<f4").tobytes()
    assert hashlib.sha256(raw).hexdigest() == case["sha256"]


def test_torch_version_matches_manifest():
    # Guards against an unintentional torch bump silently changing the contract. Compares the BASE
    # version (strip +cpu/+cuXXX) since the CPU RNG kernel is identical across build variants.
    assert torch.__version__.split("+")[0] == MANIFEST["torch_version"], (
        "torch version drifted from the frozen golden fixture; re-freeze deliberately "
        "(run tests/fixtures/decomfl_golden/generate.py) and re-validate the C++ parity test."
    )


def test_deterministic():
    assert torch.equal(canonical_perturbation(42, 256), canonical_perturbation(42, 256))


def test_is_cpu_float32():
    z = canonical_perturbation(7, 8)
    assert z.device.type == "cpu"
    assert z.dtype == torch.float32


def test_rejects_nonpositive():
    with pytest.raises(ValueError):
        canonical_perturbation(0, 0)


@pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="no non-CPU device available to check cross-device determinism",
)
def test_device_independent():
    # The point of CPU-canonical generation: moving to a device must not change the values.
    z_cpu = canonical_perturbation(123, 512)
    dev = "mps" if torch.backends.mps.is_available() else "cuda"
    z_dev = canonical_perturbation(123, 512).to(dev)
    assert torch.equal(z_cpu, z_dev.cpu())
