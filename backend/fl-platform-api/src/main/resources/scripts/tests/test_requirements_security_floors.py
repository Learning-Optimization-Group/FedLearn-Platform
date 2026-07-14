# SCRIPTS/tests/test_requirements_security_floors.py
"""SE-22: the backend requirements.txt must not pin packages below the repo's own security floors.

`backend/fl-platform-api/requirements.txt` is NOT dead — `.github/workflows/ci.yml` installs it for
the backend-scripts pytest job. It was pinning aiohttp/cryptography/pillow/requests BELOW the
patched floors that `framework/requirements.txt` documents (incl. an aiohttp RCE). This guard reads
the floors straight from `framework/requirements.txt` (single source of truth) and fails if the
backend lockfile allows an install below any of them, so the two can't drift back apart.
"""
import os
import re

_FLAGGED = ("aiohttp", "cryptography", "pillow", "requests")


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
