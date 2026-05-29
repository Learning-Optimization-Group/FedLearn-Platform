"""Canonical, device-independent perturbation generation for DeComFL.

The DeComFL protocol requires the server and *every* client — Python on
CPU/CUDA/MPS, and the native C++ (libtorch) mobile core — to regenerate
bit-identical perturbation vectors ``z ~ N(0, I_d)`` from the same integer seed.
A seeded ``torch.randn`` is **not** identical across devices, so this module
fixes one canonical path: generate on the CPU with a *local* generator, then let
callers move the result to their compute device.

This module is the SOURCE OF TRUTH for that contract:

* The Python server (``decomfl_strategy.py``) and client
  (``estimators/zeroth_order.py``) both delegate here (DeComFL correctness spec,
  Bug 2 — see ``docs/v2/specs/2026-05-29-decomfl-correctness-design.md``).
* The golden-vector fixture in ``tests/fixtures/decomfl_golden/`` is frozen from
  this function (run ``tests/fixtures/decomfl_golden/generate.py``).
* The C++ mobile port ``mobile_client/shared/src/Perturbation.cpp`` must
  reproduce these vectors; the gtest ``rng_parity_test.cpp`` is the release gate
  (15-LLD-mobile.md §13 task 4).

Re-freeze the fixture only on an *intentional* ``torch`` version bump, and record
the new version in the fixture manifest; the parity gate then re-validates the
C++ port against the new contract.
"""
from __future__ import annotations

import torch

__all__ = ["canonical_perturbation"]


def canonical_perturbation(
    seed: int,
    num_params: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a device-independent ``N(0, I_d)`` sample of length ``num_params``.

    Always generated on the CPU with a *local* :class:`torch.Generator` (never the
    process-global RNG), so the output is bit-stable across compute devices for a
    pinned ``torch`` version. Callers move it to their device at the use site::

        z = canonical_perturbation(seed, d).to(device)

    Args:
        seed: Non-negative integer seed shared between server and client for one
            ``(local_step, perturbation)`` index. Coerced with ``int()``.
        num_params: Dimension ``d`` (number of trainable parameters). Must be > 0.
        dtype: Output floating dtype. **Fixed to float32 for parity** — do not pass
            a model's dtype; that would break the golden-vector contract.

    Returns:
        A 1-D CPU tensor of shape ``(num_params,)`` and the requested ``dtype``.

    Raises:
        ValueError: If ``num_params`` is not positive.
    """
    if num_params <= 0:
        raise ValueError(f"num_params must be positive, got {num_params}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(num_params, generator=generator, dtype=dtype, device="cpu")
