# SCRIPTS/tests/test_requirements_security_floors.py
"""SE-22: the backend requirements.txt must not pin packages below the repo's own security floors.

`backend/fl-platform-api/requirements.txt` is NOT dead — `.github/workflows/ci.yml` installs it for
the backend-scripts pytest job. It was pinning aiohttp/pillow/requests BELOW the patched floors that
`framework/requirements.txt` documents (incl. an aiohttp RCE). This guard reads the floors straight
from `framework/requirements.txt` (single source of truth) and fails if the backend lockfile allows
an install below any of them, so the two can't drift back apart.

`cryptography` used to be excluded from this check: the lockfile pulled in `flwr-datasets` ->
`flwr` 1.20.0, which caps `cryptography<45.0.0`, so the framework's `>=46.0.6` floor was
unreachable here. P0-2b dropped flwr-datasets (its only use was one CIFAR-10 IID shard, now taken
directly from `datasets` and verified byte-identical), so cryptography is now floor-checked like
everything else and the exception is gone. `test_flwr_stays_out_of_the_lockfiles` below is the
regression guard: re-adding flwr silently re-caps BOTH cryptography and protobuf.
"""
import os
import re

# Floor-checked packages. cryptography joined the list in P0-2b when the flwr cap was removed.
_FLAGGED = ("aiohttp", "pillow", "requests", "cryptography")


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


def test_flwr_stays_out_of_the_lockfiles():
    """P0-2b regression guard: re-adding flwr silently re-caps two floors at once.

    `flwr` 1.20.0 caps `cryptography<45.0.0` (SE-22) AND `protobuf<5.0.0`. The protobuf cap is
    the quieter of the two: `fot_pb2.py` is generated at gencode 5.29.0 and protobuf requires
    runtime >= gencode, so a capped lockfile yields a clean install whose FoT servicer cannot
    import — and nothing notices, because the FoT path imports it lazily and the gradient path
    never does.

    flwr was only ever used for one CIFAR-10 IID shard. That shard now comes straight from
    `datasets` and is byte-identical to what flwr produced, verified per-partition by
    research/benchmarks/verify_flwr_shard_equivalence.py.
    """
    root = _repo_root()
    offenders = []
    for rel in (
        os.path.join("backend", "fl-platform-api", "requirements.txt"),
        os.path.join("client-docker", "requirements.txt"),
        os.path.join("client-docker", "packaging", "requirements-client.txt"),
        os.path.join("framework", "requirements.txt"),
    ):
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            continue
        for i, line in enumerate(open(path), 1):
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if re.match(r"^flwr(-[\w]+)?\s*[><=~!]", stripped):
                offenders.append(f"{rel}:{i}: {stripped}")
    assert offenders == [], (
        "flwr is back in a lockfile, which re-caps cryptography<45.0.0 and protobuf<5.0.0: "
        + "; ".join(offenders)
    )


def test_protobuf_reaches_the_committed_gencode_floor():
    """The FoT path must be installable from the backend lockfile.

    protobuf requires runtime >= gencode, and the newest committed stub (`fot_pb2.py`) is
    generated at 5.29.0. A lockfile below that installs cleanly and then fails at import time
    inside the FoT servicer only.
    """
    root = _repo_root()
    backend = os.path.join(root, "backend", "fl-platform-api", "requirements.txt")
    got = _min_version(backend, "protobuf")
    assert got is not None, "protobuf not pinned in backend/fl-platform-api/requirements.txt"
    assert got >= (5, 29, 0), (
        f"backend pins protobuf {got}, below the committed gencode floor 5.29.0; "
        f"the FoT servicer would not import."
    )
