import os
import sys

# The FL trainer scripts (fl_server.py, client.py, …) `import fedlearn` — the custom framework
# that, in a real deploy, is pip-installed alongside them. The backend-scripts CI job and a bare
# local checkout don't install it, so any test that imports one of those scripts would fail at
# collection with ModuleNotFoundError. Put the in-repo framework source on sys.path here (walking
# up to the repo root that carries framework/src) so `import fedlearn` resolves to the checked-out
# framework without requiring a separate editable install.
_here = os.path.dirname(os.path.abspath(__file__))
_dir = _here
for _ in range(12):
    _candidate = os.path.join(_dir, "framework", "src")
    if os.path.isdir(_candidate):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break
    _parent = os.path.dirname(_dir)
    if _parent == _dir:
        break
    _dir = _parent


# --- TE-10: skip-integrity guard ---------------------------------------------------------------
# A silently-skipped test is a false green: the suite reports success while the behaviour under
# test never ran. When active, any test that ends up SKIPPED for a reason not explicitly
# allowlisted below fails the run with a summary of the offenders. Mirrors the same guard in
# framework/tests/conftest.py (kept local so this scripts dir stays self-contained in deploys).
#
# Activation: on when $CI is truthy (GitHub Actions always sets CI=true); force with
# FEDLEARN_FAIL_ON_UNEXPECTED_SKIP=1, suppress with =0.
#
# What does NOT trip the guard:
#   * `-m "not slow"` (pytest.ini addopts) — deselection, not a skip; deselected tests emit no
#     skip report at all, so the slow-marker workflow is untouched.
#   * xfail/xpass — an expected failure is not a silent skip.
#   * reasons listed in _ALLOWED_SKIP_REASONS (exact match) — each entry needs a justification.

_ALLOWED_SKIP_REASONS = frozenset()  # this suite has no legitimate skips today

_unexpected_skips: dict = {}


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
            "tests/conftest.py with a written justification."
        )


def pytest_sessionfinish(session, exitstatus):
    if _unexpected_skips and _skip_guard_active() and session.exitstatus == 0:
        session.exitstatus = 1
