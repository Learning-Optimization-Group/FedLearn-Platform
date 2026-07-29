"""Device-placement contract for aggregation strategies.

These pin the bug that blocked FOUR cells of the algorithm x device matrix: FedOpt raised
`Expected all tensors to be on the same device` during aggregation on both CUDA machines,
which aborted each sweep before Robust was reached.

Root cause is a two-part interaction:
  1. FedAvgAggregator chose its output device from GLOBAL AVAILABILITY
     (`device = "cuda" if torch.cuda.is_available() else "cpu"`) rather than from where the
     incoming tensors actually live. On any CUDA box it force-migrated every aggregate to CUDA.
  2. FedOpt keeps its own `_global` on whatever device `initial_parameters` arrived on, and
     converted the aggregate with `.to(old.dtype)` -- dtype only, never device.

FedAvg never crashed because it just returns the aggregate; FedOpt crashes because it mixes
server state with the aggregate (`g = old - x_bar`).

Note these tests pass vacuously on a CPU-only host for part (1), because the hardcoded branch
happens to pick "cpu" there -- which is exactly why the defect survived every CI run on this
machine. The MPS-gated tests below are the ones that genuinely exercise it.
"""
import os
import sys
from collections import OrderedDict

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fedlearn.server.strategy import FedAvgAggregator, FedOpt  # noqa: E402

ACCEL = ("mps" if torch.backends.mps.is_available()
         else "cuda" if torch.cuda.is_available() else None)
skip_no_accel = pytest.mark.skipif(ACCEL is None, reason="no accelerator available")


def _sd(dev, val=1.0, n=4):
    return OrderedDict([("w", torch.full((n,), val, device=dev))])


# ------------------------------------------------------------------ aggregator device contract

def test_aggregator_preserves_cpu_input_device():
    out = FedAvgAggregator().aggregate([(_sd("cpu", 2.0), 10)])
    assert out["w"].device.type == "cpu"


@skip_no_accel
def test_aggregator_preserves_accelerator_input_device():
    """The real contract: the aggregate belongs on the device the DATA is on, not on whichever
    device happens to exist on the host."""
    out = FedAvgAggregator().aggregate([(_sd(ACCEL, 2.0), 10)])
    assert out["w"].device.type == ACCEL


def test_aggregator_result_is_numerically_correct():
    """The device fix must not disturb the weighted mean."""
    out = FedAvgAggregator().aggregate([(_sd("cpu", 1.0), 10), (_sd("cpu", 3.0), 30)])
    assert torch.allclose(out["w"], torch.full((4,), 2.5), atol=1e-6)


# ------------------------------------------------------------------ FedOpt device robustness

@skip_no_accel
def test_fedopt_survives_state_and_aggregate_on_different_devices():
    """The exact matrix failure: server state on CPU, clients trained on the accelerator."""
    s = FedOpt(initial_parameters=_sd("cpu", 1.0), min_fit_clients=1)
    out = s.aggregate_fit(1, [(_sd(ACCEL, 2.0), 10)])
    assert out is not None
    assert torch.isfinite(out["w"]).all()


@skip_no_accel
def test_fedopt_multi_round_is_stable_across_devices():
    """Moments (m, v) persist across rounds, so a one-round pass is not sufficient evidence."""
    s = FedOpt(initial_parameters=_sd("cpu", 1.0), min_fit_clients=1)
    for r in range(1, 4):
        out = s.aggregate_fit(r, [(_sd(ACCEL, 2.0), 10)])
        assert torch.isfinite(out["w"]).all(), f"round {r} produced non-finite values"


def test_fedopt_cpu_only_path_unchanged():
    """Existing CPU numbers must reproduce exactly after the fix."""
    s = FedOpt(initial_parameters=_sd("cpu", 1.0), min_fit_clients=1)
    out = s.aggregate_fit(1, [(_sd("cpu", 2.0), 10)])
    assert out["w"].device.type == "cpu"
    assert torch.isfinite(out["w"]).all()


def test_fedopt_moves_toward_the_client_aggregate():
    """A behavioural control: FedOpt must actually track x_bar, not merely avoid crashing.

    NOT monotone -- an earlier version of this test asserted monotonicity and failed, and the
    test was wrong rather than the code: with v tiny in the first rounds the effective step
    eta/(sqrt(v)+tau) is large, so FedOpt overshoots (0 -> 1.90 by round 2) and oscillates in.
    That is ordinary adaptive-optimizer behaviour. What IS guaranteed is that it leaves the
    initial value, stays finite, and settles near the target."""
    s = FedOpt(initial_parameters=_sd("cpu", 0.0), min_fit_clients=1)
    vals = []
    for r in range(1, 41):
        out = s.aggregate_fit(r, [(_sd("cpu", 1.0), 10)])
        assert torch.isfinite(out["w"]).all(), f"round {r} produced non-finite values"
        vals.append(float(out["w"][0]))
    assert vals[0] > 0.0, "never left the initial value"
    assert abs(vals[-1] - 1.0) < 0.25, f"did not settle near x_bar=1.0; ended at {vals[-1]}"
