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
import torch

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


# --- FedProx: the same real LocalTrainer.fit, with the proximal term on ------------------------------------

FEDPROX_MANIFEST = os.path.join(GOLDEN_DIR, "fedprox_local_manifest.json")


def _fedprox_manifest() -> dict:
    with open(FEDPROX_MANIFEST) as fh:
        return json.load(fh)


def test_fedprox_local_endpoint_reproduces_golden():
    man = _fedprox_manifest()
    final = compute_fedavg_endpoint(lr=man["learning_rate"], local_epochs=man["local_epochs"],
                                    proximal_mu=man["proximal_mu"])
    golden = np.fromfile(os.path.join(GOLDEN_DIR, man["final_flat_file"]), dtype="<f4")
    assert final.shape == golden.shape == (man["flat_dim"],)
    if platform.machine() == man["platform_machine"]:
        np.testing.assert_array_equal(final, golden)
    else:
        np.testing.assert_allclose(final, golden, atol=man["endpoint_atol"], rtol=0)


def test_the_fedprox_golden_is_far_from_the_fedavg_golden_at_its_own_tolerance():
    """At one step the proximal term is zero, and at mu 0.1 over five steps it moves the endpoint by ~1e-3 -- inside
    the FedAvg golden's 2e-3. So the FedProx golden carries its own tolerance, and must sit well outside it from the
    FedAvg endpoint, or a native trainer with no proximal term would pass the FedProx test."""
    man = _fedprox_manifest()
    fedavg = _manifest()
    assert (man["learning_rate"], man["local_epochs"]) == (fedavg["learning_rate"], fedavg["local_epochs"])
    assert man["proximal_mu"] > 0.0
    prox = np.fromfile(os.path.join(GOLDEN_DIR, man["final_flat_file"]), dtype="<f4")
    plain = np.fromfile(os.path.join(GOLDEN_DIR, fedavg["final_flat_file"]), dtype="<f4")
    separation = float(np.abs(prox - plain).max())
    assert separation == man["separation_from_fedavg"]
    assert separation >= 10 * man["endpoint_atol"]


# Stage 3 C2: the minibatch golden the native minibatch loop replays.

def _minibatch_manifest():
    with open(os.path.join(GOLDEN_DIR, "fedavg_minibatch_manifest.json")) as fh:
        return json.load(fh)


def test_the_minibatch_endpoint_reproduces_its_golden():
    """LocalTrainer.fit in the seeded batch order over the committed 20-example dataset lands on the golden."""
    import generate_fedavg_golden as g
    # The committed dataset is the ground truth; the native test trains from these same bytes. Regenerating it is
    # checked only to a tolerance: torch.randn fills 16+ values through a vectorised path whose math differs by a few
    # ulps between CPUs (x86 AVX vs ARM), so the regenerated inputs are not bit-identical on every machine.
    x = torch.from_numpy(np.fromfile(os.path.join(GOLDEN_DIR, "minibatch_inputs.f32"), dtype="<f4").reshape(20, 4).copy())
    y = torch.from_numpy(np.fromfile(os.path.join(GOLDEN_DIR, "minibatch_targets.i64"), dtype="<i8").copy())
    regenerated_x, regenerated_y = g.minibatch_dataset()
    np.testing.assert_allclose(regenerated_x.numpy(), x.numpy(), rtol=0, atol=1e-5)
    np.testing.assert_array_equal(regenerated_y.numpy(), y.numpy())
    got = g.compute_minibatch_endpoint(g._SeededMinibatchLoader(x, y, g.MINIBATCH_SIZE, g.MINIBATCH_SEED,
                                                                g.MINIBATCH_ROUND))
    golden = np.fromfile(os.path.join(GOLDEN_DIR, "fedavg_minibatch_final.f32"), dtype="<f4")
    assert np.abs(got - golden).max() < 1e-6


def test_the_minibatch_golden_is_far_from_every_wrong_batch_order():
    """The native test's 1e-5 tolerance must reject each plausible wrong order by at least tenfold."""
    separations = _minibatch_manifest()["control_separation"]
    assert set(separations) == {"sequential_order", "one_full_batch", "epoch_zero_order_every_epoch"}
    assert min(separations.values()) > 10 * 1e-5


# Stage 4 S2: the Adam goldens the native Adam replays.

def test_the_adam_goldens_are_torch_adam_and_far_from_every_control():
    import generate_fedavg_golden as g
    with open(os.path.join(GOLDEN_DIR, "adam_local_manifest.json")) as fh:
        manifest = json.load(fh)
    nearest = {}
    for name, entry in manifest["goldens"].items():
        golden = np.fromfile(os.path.join(GOLDEN_DIR, name), dtype="<f4")
        assert np.abs(g.compute_adam_endpoint(entry["epsilon"]) - golden).max() < 1e-6
        for control, separation in entry["control_separation"].items():
            nearest[control] = max(nearest.get(control, 0.0), separation)
    # Each plausible bug is visible, far above the native test's 1e-5 tolerance, in at least one golden.
    assert set(nearest) == {"no_bias_correction", "eps_inside_sqrt", "betas_swapped"}
    assert min(nearest.values()) > 10 * 1e-5
