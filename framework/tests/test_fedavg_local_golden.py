"""First-order (FedAvg) local-update golden — Python self-consistency + the cross-language parity
contract the native C++ TrainableExecutorchModel replays (Phase B M1a).

``LocalTrainer.fit(mu=0)`` IS the FedAvg client (local_trainer.py:78-84). This freezes its endpoint
on the committed TinyNet fixture so (a) a change to the first-order update is caught in the framework
CI gate — pure torch, no executorch — and (b) the native first-order primitive has a reference to
match within tolerance, mirroring how the DeComFL multiround golden pins the zeroth-order path.

Bit-exact on the freeze arch, tolerance cross-arch — the same discipline as test_perturbation
(transcendentals in log_softmax drift ~1 ULP x86<->arm64, compounded over the SGD steps).
"""
import json
import os
import platform
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN_DIR = os.path.join(HERE, "fixtures", "decomfl_golden")
sys.path.insert(0, GOLDEN_DIR)

from generate_fedavg_golden import build_initial_net, compute_fedavg_endpoint  # noqa: E402

MANIFEST = os.path.join(GOLDEN_DIR, "fedavg_local_manifest.json")
CROSS_ARCH_ATOL = 2e-3  # == manifest endpoint_atol; matches the ZO endpoint golden's family


def _manifest() -> dict:
    with open(MANIFEST) as fh:
        return json.load(fh)


def test_fedavg_local_endpoint_reproduces_golden():
    man = _manifest()
    final = compute_fedavg_endpoint(lr=man["learning_rate"], local_epochs=man["local_epochs"])
    golden = np.fromfile(os.path.join(GOLDEN_DIR, man["final_flat_file"]), dtype="<f4")
    assert final.shape == golden.shape == (man["flat_dim"],)
    if platform.machine() == man["platform_machine"]:
        np.testing.assert_array_equal(final, golden)  # bit-exact on the freeze arch
    else:
        np.testing.assert_allclose(final, golden, atol=CROSS_ARCH_ATOL, rtol=0)


def test_fedavg_initial_flat_matches_committed_zo_flat():
    # the FedAvg golden must start from the SAME committed init as the ZO goldens (byte-identical),
    # so the native side loads ONE initial fixture (zo_flat.f32) for both the ZO and FO paths.
    from fedlearn.estimators.params import flat_params

    init = flat_params(build_initial_net()).detach().numpy().astype("<f4")
    zo_flat = np.fromfile(os.path.join(GOLDEN_DIR, "zo_flat.f32"), dtype="<f4")
    np.testing.assert_array_equal(init, zo_flat)


def test_fedavg_torch_version_matches_manifest():
    # mirrors test_perturbation: the golden is frozen under the pinned torch (2.12.0). A mismatch
    # asserts loudly (not a silent skip) — the intended "stale local pin" signal; CI runs 2.12.0.
    import torch

    assert torch.__version__.split("+")[0] == _manifest()["torch_version"], (
        f"golden frozen on torch {_manifest()['torch_version']}, running {torch.__version__}; "
        "regenerate under the pinned torch or align the environment."
    )


def test_fedavg_param_layout_is_canonical_named_parameters_order():
    # guards the ordering gotcha: the frozen layout must equal params.param_layout (named_parameters
    # order, trainable-only) — the order the native side re-maps ET's alphabetical map into.
    from fedlearn.estimators.params import param_layout

    man = _manifest()
    expected = [[name, list(shape), k] for name, shape, k in param_layout(build_initial_net())]
    assert man["param_layout"] == expected
