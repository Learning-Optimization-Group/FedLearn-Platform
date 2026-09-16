"""Parity + determinism tests for ``canonical_perturbation`` (the DeComFL RNG contract).

Run::

    cd framework && PYTHONPATH=src pytest tests/test_perturbation.py -v

These assert the Python side reproduces the committed golden fixture on the pinned
``torch`` version.

**Cross-architecture caveat.** ``torch.randn``'s CPU kernel is bit-reproducible on the
*same* CPU architecture, but only ~1-ULP reproducible *across* architectures (its
vectorized Box–Muller transcendentals differ by a last bit between x86-64 and
Apple-Silicon arm64). The fixtures are frozen on one arch (``manifest.platform_machine``,
currently the x86-64 CI runner), so:

* On the freeze arch → assert **bit-exact** (the strongest guarantee; this is the CI gate).
* On any other arch → assert **ULP-tolerance** parity, matching the C++ mobile gate
  ``mobile_client/shared/tests/rng_parity_test.cpp`` (``ASSERT_NEAR(..., 1e-6f)``).
"""
import hashlib
import json
import os
import platform

import numpy as np
import pytest
import torch

from fedlearn.estimators.perturbation import canonical_perturbation

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "decomfl_golden")
with open(os.path.join(FIXTURE_DIR, "manifest.json")) as _f:
    MANIFEST = json.load(_f)
CASES = MANIFEST["cases"]
_IDS = [c["file"] for c in CASES]

# Arch the golden vectors were frozen on. None for legacy manifests → always tolerance.
FREEZE_ARCH = MANIFEST.get("platform_machine")
# Cross-arch absolute tolerance. Empirically the x86↔arm64 spread of canonical_perturbation
# on the pinned torch peaks at ~1.4e-6 (a handful of ULPs on O(1) float32 randn values); 2e-6
# gives margin. Real RNG drift is still caught bit-exact on the freeze arch in CI.
CROSS_ARCH_ATOL = 2e-6


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_matches_golden_fixture(case):
    golden = np.fromfile(os.path.join(FIXTURE_DIR, case["file"]), dtype="<f4")
    z = canonical_perturbation(case["seed"], case["num_params"]).numpy()
    assert z.shape == (case["num_params"],)
    if platform.machine() == FREEZE_ARCH:
        # Same arch as the freeze → bit-exact reproduction of the frozen contract.
        np.testing.assert_array_equal(z, golden)
    else:
        # Different CPU arch → torch.randn diverges by ~1 ULP; assert ULP-tolerance parity.
        # Bit-exactness for this contract is validated on the freeze arch (CI).
        np.testing.assert_allclose(z, golden, atol=CROSS_ARCH_ATOL, rtol=0)


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
