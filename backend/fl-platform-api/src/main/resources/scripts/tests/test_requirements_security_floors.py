# SCRIPTS/tests/test_requirements_security_floors.py
"""SE-22: the backend requirements.txt must not pin packages below the repo's own security floors.

`backend/fl-platform-api/requirements.txt` is NOT dead — `.github/workflows/ci.yml` installs it for
the backend-scripts pytest job. It was pinning aiohttp/pillow/requests BELOW the patched floors that
`framework/requirements.txt` documents (incl. an aiohttp RCE). This guard reads the floors straight
from `framework/requirements.txt` (single source of truth) and fails if the backend lockfile allows
an install below any of them, so the two can't drift back apart.

`cryptography` is the ONE exception and is deliberately excluded from the floor check: this lockfile
uses `flwr-datasets` (`FederatedDataset` in `fl_server.py`/`client.py`) -> `flwr` 1.20.0, which pins
`cryptography<45.0.0`. The framework's `>=46.0.6` floor is therefore unreachable here without dropping
flwr-datasets; a separate check pins it to the newest flwr-compatible version instead (SE-22 residual).
"""
import os
import re

# Floor-checked packages. cryptography is intentionally NOT here — see the module docstring (flwr cap).
_FLAGGED = ("aiohttp", "pillow", "requests")


def _repo_root():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(12):
        if os.path.isfile(os.path.join(d, "framework", "requirements.txt")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    raise RuntimeError("repo root (with framework/requirements.txt) not found")


def _min_version(requirements_path, pkg):
    """The lowest version the requirement for `pkg` permits (from a `==` or `>=` pin), or None."""
    pat = re.compile(rf"^\s*{re.escape(pkg)}\s*(==|>=)\s*([0-9][0-9A-Za-z.\-]*)", re.IGNORECASE)
    with open(requirements_path, encoding="utf-8") as f:
        for line in f:
            m = pat.match(line)
            if m:
                return _tuple(m.group(2))
    return None


def _tuple(v):
    parts = []
    for chunk in v.split("."):
        num = re.match(r"\d+", chunk)
        parts.append(int(num.group()) if num else 0)
    return tuple(parts)


def test_backend_requirements_meet_framework_security_floors():
    root = _repo_root()
    backend = os.path.join(root, "backend", "fl-platform-api", "requirements.txt")
    framework = os.path.join(root, "framework", "requirements.txt")
    violations = []
    for pkg in _FLAGGED:
        floor = _min_version(framework, pkg)
        got = _min_version(backend, pkg)
        assert floor is not None, f"no floor for {pkg} in framework/requirements.txt"
        assert got is not None, f"{pkg} not pinned in backend/fl-platform-api/requirements.txt"
        if got < floor:
            violations.append(f"{pkg}: backend allows {got} < floor {floor}")
    assert violations == [], "SE-22: backend pins below the security floor: " + "; ".join(violations)


def test_cryptography_stays_flwr_compatible_below_45():
    """cryptography can't reach the framework's >=46.0.6 floor here: flwr-datasets -> flwr 1.20.0 pins
    cryptography<45.0.0, so a >=45 bump makes `pip install -r requirements.txt` unresolvable (the
    backend-scripts CI job fails with ResolutionImpossible). Pin it to the newest flwr-compatible
    patched version instead, and pin THAT invariant here so a future security bump doesn't silently
    re-break the install."""
    root = _repo_root()
    backend = os.path.join(root, "backend", "fl-platform-api", "requirements.txt")
    got = _min_version(backend, "cryptography")
    assert got is not None, "cryptography not pinned in backend/fl-platform-api/requirements.txt"
    assert (44, 0, 1) <= got < (45, 0, 0), (
        f"SE-22: cryptography must stay in flwr's >=44.0.1,<45.0.0 range (flwr-datasets caps it); "
        f"got {got}. A bump to >=45 breaks pip resolution in the backend-scripts CI job."
    )
