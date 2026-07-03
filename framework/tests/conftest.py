# framework/tests/conftest.py
import os
import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def disable_cuda(monkeypatch):
    """Force CPU for all tests to avoid device mismatch issues."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Make all tests deterministic."""
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    yield


# --- TE-10: skip-integrity guard ---------------------------------------------------------------
# A silently-skipped test is a false green: the suite reports success while the behaviour under
# test never ran (the PTE-export parity test skipping when executorch wasn't installed is the
# canonical example). When active, any test that ends up SKIPPED for a reason not explicitly
# allowlisted below fails the run with a summary of the offenders.
#
# Activation: on when $CI is truthy (GitHub Actions always sets CI=true); force with
# FEDLEARN_FAIL_ON_UNEXPECTED_SKIP=1, suppress with =0. Local default-off keeps dev machines
# usable when their env legitimately lacks CI-only deps.
#
# What does NOT trip the guard:
#   * `-m "not slow"` (pytest.ini addopts) — deselection, not a skip; deselected tests emit no
#     skip report at all, so the slow-marker workflow is untouched.
#   * xfail/xpass — an expected failure is not a silent skip.
#   * reasons listed in _ALLOWED_SKIP_REASONS (exact match) — each entry needs a justification.

_ALLOWED_SKIP_REASONS = frozenset(
    {
        # test_perturbation.py::test_device_independent — hardware-capability gate. CI runners
        # have no CUDA/MPS device, so cross-device determinism is only checkable on dev machines
        # that carry a second device; there is nothing to fix by "making it run" on a CPU-only box.
        "no non-CPU device available to check cross-device determinism",
    }
)

_unexpected_skips: dict[str, str] = {}


def _skip_guard_active() -> bool:
    override = os.environ.get("FEDLEARN_FAIL_ON_UNEXPECTED_SKIP")
    if override is not None:
        return override == "1"
    return os.environ.get("CI", "").lower() in ("1", "true", "yes")


def _skip_reason(report) -> str:
    # Skipped reports carry longrepr = (path, lineno, "Skipped: <reason>").
    longrepr = report.longrepr
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        message = str(longrepr[2])
    else:
        message = str(longrepr)
    prefix = "Skipped: "
    return message[len(prefix):] if message.startswith(prefix) else message


def _record_skip(report) -> None:
    reason = _skip_reason(report)
    if reason not in _ALLOWED_SKIP_REASONS:
        _unexpected_skips.setdefault(report.nodeid, reason)


def pytest_runtest_logreport(report):
    # Marker skips surface in the setup phase, runtime pytest.skip()/importorskip in call.
    if report.skipped and not hasattr(report, "wasxfail"):
        _record_skip(report)


def pytest_collectreport(report):
    # Module-level skips (pytest.skip(allow_module_level=True), module importorskip).
    if report.skipped:
        _record_skip(report)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if _unexpected_skips and _skip_guard_active():
        terminalreporter.section("unexpected skipped tests (TE-10)")
        for nodeid, reason in sorted(_unexpected_skips.items()):
            terminalreporter.line(f"  {nodeid} — {reason}")
        terminalreporter.line(
            "A skipped test is a false green: make it run, or allowlist its exact reason in "
            "framework/tests/conftest.py with a written justification."
        )


def pytest_sessionfinish(session, exitstatus):
    if _unexpected_skips and _skip_guard_active() and session.exitstatus == 0:
        session.exitstatus = 1
