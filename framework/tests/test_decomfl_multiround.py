"""Python side of the multi-round DeComFL trajectory golden (DA-5 / D2b).

Asserts the framework's DeComFL primitives (canonical_perturbation +
ZerothOrderEstimator.compute_gradient_scalar) reproduce the frozen N-round
trajectory in framework/tests/fixtures/decomfl_golden/zo_multiround_*, and that
the committed fixture is internally consistent (sha256 + seeds). The C++ mobile
core replays the SAME trajectory and asserts a tolerance-bounded endpoint match
(mobile_client/shared/tests/et_multiround_test.cpp) — together they close the one
conformance claim the single-step goldens did not make.

Tolerance-based by design: forward-backend + float32-vs-double g division + z
arch-ULP drift are absorbed by the manifest's *_atol (never bit-exact cross-arch).
See research/notes/2026-07-15-cross-language-conformance-contract.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "fixtures", "decomfl_golden")
sys.path.insert(0, GOLDEN)

from generate_zo import TinyNet  # noqa: E402
from generate_zo_multiround import MU, compute_multiround_trajectory  # noqa: E402

from fedlearn.estimators.zeroth_order import ZerothOrderEstimator  # noqa: E402


def _manifest() -> dict:
    with open(os.path.join(GOLDEN, "zo_multiround_manifest.json")) as fh:
        return json.load(fh)


def _recompute() -> dict:
    torch.manual_seed(0)
    net = TinyNet().eval()
    zo = ZerothOrderEstimator(smoothing_param=MU, device="cpu")
    x0 = np.fromfile(os.path.join(GOLDEN, "zo_flat.f32"), dtype="<f4")
    inputs = torch.from_numpy(np.fromfile(os.path.join(GOLDEN, "zo_inputs.f32"), dtype="<f4").reshape(8, 4).copy())
    targets = torch.from_numpy(np.fromfile(os.path.join(GOLDEN, "zo_targets.i64"), dtype="<i8").reshape(8).copy())
    return compute_multiround_trajectory(net, zo, x0, inputs, targets)


def test_endpoint_reproduces_frozen_trajectory():
    man = _manifest()
    frozen = np.fromfile(os.path.join(GOLDEN, "zo_multiround_final.f32"), dtype="<f4")
    traj = _recompute()
    assert traj["final_flat"].shape == (man["flat_dim"],)
    np.testing.assert_allclose(traj["final_flat"], frozen, atol=man["endpoint_atol"], rtol=0)


def test_per_round_g_reproduces():
    man = _manifest()
    traj = _recompute()
    got = np.array(traj["per_round_g"])
    exp = np.array(man["per_round_g"])
    assert got.shape == (man["num_rounds"], man["K"], man["P"])
    np.testing.assert_allclose(got, exp, atol=man["g_atol"], rtol=0)


def test_final_flat_sha256_matches_manifest():
    man = _manifest()
    raw = np.fromfile(os.path.join(GOLDEN, "zo_multiround_final.f32"), dtype="<f4").tobytes()
    assert hashlib.sha256(raw).hexdigest() == man["final_flat_sha256"]


def test_seeds_file_matches_manifest():
    man = _manifest()
    seeds_file = np.fromfile(os.path.join(GOLDEN, "zo_multiround_seeds.i64"), dtype="<i8")
    seeds_manifest = np.array([s for rd in man["seeds"] for st in rd for s in st], dtype="<i8")
    assert seeds_file.size == man["num_rounds"] * man["K"] * man["P"]
    np.testing.assert_array_equal(seeds_file, seeds_manifest)


def test_state_safetensors_sha256_matches_manifest():
    man = _manifest()
    with open(os.path.join(GOLDEN, "zo_multiround_state.safetensors"), "rb") as fh:
        blob = fh.read()
    assert hashlib.sha256(blob).hexdigest() == man["state_sha256"]
