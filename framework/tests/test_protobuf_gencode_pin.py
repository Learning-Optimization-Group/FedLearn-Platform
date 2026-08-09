"""The installed protobuf runtime must satisfy every committed generated stub.

Protobuf's cross-version guarantee is one-directional: the **runtime version must be >= the gencode
version** that produced a ``*_pb2.py``. Violate it and the import fails at load time with
``cannot import name 'runtime_version'`` or a ``VersionError`` — not at build time, and not in CI if
CI happens to install a newer protobuf than the pin allows.

This repo hit exactly that. Two stubs were regenerated at different times with different protoc
versions:

    fedlearn_pb2.py   gencode 4.25.1
    fot_pb2.py        gencode 5.29.0

while ``framework/requirements.txt`` pinned ``protobuf>=4.21.6,<5.0.0``. A clean install from that
file therefore produces an environment that **cannot import fot_pb2** — the FoT servicer, the FoT
server, and the ``fl_fot_server.py`` entry point the Spring backend spawns. It went unnoticed because
``fot_pb2`` is imported lazily by the FoT path only; the gradient path never touches it.

These tests pin the invariant rather than the version, so regenerating a stub with a newer protoc
fails loudly here instead of silently at deploy time.
"""
import glob
import os
import re

import pytest

google_protobuf = pytest.importorskip("google.protobuf")

_GENCODE_RE = re.compile(r"Protobuf Python Version:\s*([0-9]+(?:\.[0-9]+)*)")
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _version_tuple(text):
    return tuple(int(p) for p in text.strip().split("."))


def _generated_stubs():
    """Every committed *_pb2.py with a declared gencode version."""
    found = []
    for path in glob.glob(os.path.join(_SRC, "**", "*_pb2.py"), recursive=True):
        m = _GENCODE_RE.search(open(path, encoding="utf-8").read())
        if m:
            found.append((os.path.relpath(path, _SRC), m.group(1)))
    return sorted(found)


def test_there_are_generated_stubs_to_check():
    """Guard against the scan silently finding nothing and the suite passing vacuously."""
    stubs = _generated_stubs()
    assert stubs, "no *_pb2.py with a gencode marker found — the scan is broken, not the repo clean"


def test_runtime_satisfies_every_stubs_gencode():
    """The invariant: installed runtime >= every stub's gencode."""
    runtime = _version_tuple(google_protobuf.__version__)
    violations = [
        (path, gencode) for path, gencode in _generated_stubs()
        if _version_tuple(gencode) > runtime
    ]
    assert not violations, (
        f"protobuf runtime {google_protobuf.__version__} is OLDER than the gencode of: "
        f"{violations}. Protobuf requires runtime >= gencode; these stubs will fail to import. "
        f"Either regenerate them with an older protoc or raise the protobuf pin."
    )


def test_every_stub_actually_imports():
    """The invariant above is necessary but not sufficient — prove the modules really load.

    A stub can also fail on a descriptor-pool conflict or a missing transitive dependency, neither of
    which a version comparison catches.
    """
    import importlib

    failures = []
    for path, _gencode in _generated_stubs():
        module = path.replace(os.sep, ".")[: -len(".py")]
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 — reporting every failure is the point
            failures.append((module, f"{type(exc).__name__}: {exc}"))
    assert not failures, f"generated stubs failed to import: {failures}"


def test_the_declared_pin_can_satisfy_every_stub():
    """The defect this file exists for: a requirements pin whose UPPER bound excludes the gencode a
    committed stub needs. A clean install from that file is broken on arrival, and no amount of
    testing the *current* environment would reveal it — the venv in use may legitimately sit above
    the pin.
    """
    req = os.path.join(os.path.dirname(_SRC), "requirements.txt")
    if not os.path.exists(req):
        pytest.skip("no framework/requirements.txt")

    line = next((ln.strip() for ln in open(req, encoding="utf-8")
                 if ln.strip().startswith("protobuf")), None)
    assert line, "framework/requirements.txt declares no protobuf pin"

    upper = re.search(r"<\s*([0-9]+(?:\.[0-9]+)*)", line)
    if not upper:
        return  # unbounded above — cannot exclude any gencode

    upper_v = _version_tuple(upper.group(1))
    needed = max((_version_tuple(g) for _p, g in _generated_stubs()), default=(0,))
    assert needed < upper_v, (
        f"framework/requirements.txt pins '{line}', whose upper bound {upper.group(1)} excludes the "
        f"gencode {'.'.join(map(str, needed))} required by a committed stub. A clean install from "
        f"this file cannot import it."
    )
