#!/usr/bin/env python3
"""Spawn entrypoint for the standalone FoT (Federation over Text) server.

The Spring control plane spawns this exactly like fl_server.py (via run_fot_server.sh), so its
stdout JSON events stream to the dashboard. FoT is a SEPARATE, local-LLM-only, non-PHI research
mode; it does not touch the DeComFL/FedAvg gradient path.
"""
import argparse
import os
import sys

# Make sibling scripts (and, if not installed, the framework) importable when spawned from the
# backend working dir.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _can_import_fedlearn() -> bool:
    try:
        import fedlearn  # noqa: F401
        return True
    except Exception:
        return False


def _ensure_framework_on_path() -> None:
    if _can_import_fedlearn():
        return
    cur = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):  # walk up looking for framework/src/fedlearn
        candidate = os.path.join(cur, "framework", "src")
        if os.path.isdir(os.path.join(candidate, "fedlearn")):
            sys.path.insert(0, candidate)
            return
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent


_ensure_framework_on_path()

from fedlearn.fot.fot_server import start_fot_server  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="FoT (Federation over Text) text-federation server")
    parser.add_argument("--project-id", default="")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--round-seconds", type=float, default=5.0)
    parser.add_argument("--quorum", type=int, default=2)
    parser.add_argument(
        "--backend",
        default="stub",
        help="AgentBackend name; 'stub' (offline) by default. A real LOCAL-LLM adapter must be "
        "wired in fedlearn.fot.backend.get_backend — FoT does not call hosted APIs.",
    )
    args = parser.parse_args()
    start_fot_server(
        f"[::]:{args.port}",
        num_rounds=args.num_rounds,
        round_seconds=args.round_seconds,
        backend_name=args.backend,
        quorum=args.quorum,
    )


if __name__ == "__main__":
    main()
