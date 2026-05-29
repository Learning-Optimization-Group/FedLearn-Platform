"""Freeze the DeComFL canonical-perturbation golden vectors (the cross-language contract).

These vectors are the SOURCE OF TRUTH that both the Python framework and the native
C++ (libtorch) mobile core must reproduce. They pin the exact output of
``fedlearn.estimators.perturbation.canonical_perturbation`` for a fixed ``torch``
version, so any drift (a torch bump, a device change, a buggy C++ port) is caught by:

  * Python: ``framework/tests/test_perturbation.py``
  * C++:    ``mobile_client/shared/tests/rng_parity_test.cpp`` (release gate, 15-LLD §13.4)

Re-run ONLY on an intentional torch bump:

    cd framework && PYTHONPATH=src python tests/fixtures/decomfl_golden/generate.py

It rewrites ``manifest.json`` and one ``z_<seed>_<n>.f32`` per case. The ``.f32`` files
are raw little-endian float32 bytes (np.float32 C-order) so both Python (np.fromfile)
and C++ (std::ifstream) read them with no format dependency. Bumping torch changes the
contract — review the C++ parity test afterwards.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import torch

from fedlearn.estimators.perturbation import canonical_perturbation

# (seed, num_params) cases: small + a 31-bit-max seed + sizes spanning the chunked path.
CASES = [
    {"seed": 0, "num_params": 16},
    {"seed": 1, "num_params": 100},
    {"seed": 1234567, "num_params": 1000},
    {"seed": 2147483646, "num_params": 4096},  # near int32 max — matches generate_seeds() range
]

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    entries = []
    for case in CASES:
        seed, n = case["seed"], case["num_params"]
        z = canonical_perturbation(seed, n)  # CPU, float32
        arr = z.numpy().astype("<f4", copy=False)  # little-endian float32
        fname = f"z_{seed}_{n}.f32"
        raw = arr.tobytes()
        with open(os.path.join(HERE, fname), "wb") as fh:
            fh.write(raw)
        entries.append(
            {
                "seed": seed,
                "num_params": n,
                "dtype": "float32",
                "byte_order": "little-endian",
                "file": fname,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "first8": [float(x) for x in arr[:8]],
            }
        )

    manifest = {
        "description": (
            "DeComFL canonical_perturbation golden vectors. The single source of truth "
            "for Python<->C++ perturbation RNG parity. Frozen from "
            "fedlearn.estimators.perturbation.canonical_perturbation."
        ),
        "generator_recipe": (
            "torch.Generator(device='cpu').manual_seed(seed); "
            "torch.randn(num_params, generator=g, dtype=float32, device='cpu')"
        ),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "file_format": "raw little-endian float32, C-order, length == num_params",
        "cases": entries,
    }
    with open(os.path.join(HERE, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    print(f"Froze {len(entries)} golden vectors at torch {torch.__version__}")


if __name__ == "__main__":
    main()
