# DeComFL Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Fix the three independent correctness bugs in the platform's DeComFL (Decentralized Communication-efficient Federated Learning) path, plus the small adjacent defects that block the tests, plus the two opted-in correctness cleanups — and pin every fix with a TDD (Test-Driven Development) suite that becomes the acceptance contract. DeComFL is the platform's only paper-backed differentiator: ZO (Zeroth-Order) optimization with communication that is independent of model dimension (roughly 1,000,000x less bandwidth than FedAvg (Federated Averaging) for LLMs (Large Language Models); see `docs/wikis/framework/06_decomfl.md:495`). All three bugs are on its live path.

**Architecture:** The relevant unit is `framework/` — the custom Python FL (Federated Learning) framework (no Flower / `flwr` dependency). The pieces that interact for DeComFL:
- **Server strategy** `framework/src/fedlearn/server/decomfl_strategy.py` (`DeComFL`): holds the global model as a flat tensor, generates per-round seeds, and in `aggregate_fit` reconstructs each perturbation `z` from its seed and applies the averaged ZO update.
- **Coordinator** `framework/src/fedlearn/server/coordinator.py` (`FLCoordinator`): owns rounds; `submit_decomfl_update` collects client gradient scalars and calls `strategy.aggregate_fit`; it stores per-round averaged gradients into `strategy.gradient_history`.
- **Client** `framework/src/fedlearn/client/decomfl_client.py` (`DeComFLClient`): runs ZO `fit`, and on reconnect replays missed rounds via `rebuild_model` using the server-supplied seed + gradient history.
- **Estimator** `framework/src/fedlearn/estimators/zeroth_order.py` (`ZerothOrderEstimator`): regenerates `z` from a seed and computes the gradient scalar `g = (f(x+μz) - f(x)) / μ`.
- **Serializer** `framework/src/fedlearn/communication/serializer.py`: chunked gRPC (gRPC Remote Procedure Call) save/load of model parameters; the reassembly call site is `framework/src/fedlearn/server/grpc_servicer.py:213`.

The correctness invariant binding these together: the server's `global_params_flat` trajectory and every client's `rebuild_model` trajectory must follow the **same** update for the **same** seeds and gradients. Bugs 1 and 2 break that invariant; Bug 3 breaks the chunked-upload transport entirely.

**Tech Stack:** Python 3.10+, PyTorch (`torch` 2.12.0+cu130 in this environment), NumPy (`numpy` 2.1.2), gRPC, protobuf (package `fedlearn.v1`). Tests run with `pytest` (config in `framework/pyproject.toml:83`; `addopts = "-v --tb=short"`). The autouse fixtures in `framework/tests/conftest.py` force CPU (Central Processing Unit) (`torch.cuda.is_available` is monkeypatched to `False`) and seed `torch` / `numpy` / `random` to `0` before every test.

---

## Methodology

This plan is strict TDD. Every task follows the same cycle:
1. **RED** — write the full failing test, run the exact `pytest` command, confirm the exact failure.
2. **GREEN** — make the smallest real code change, rerun, confirm pass.
3. **COMMIT** — exact `git add` + `git commit`.

No AI attribution in any commit message (repo policy). Authorship is human-only. Do not add `Co-Authored-By` trailers.

All commands assume you start each task from the repo root `/home/anurag/codebase/FedLearn-Platform`. The agent's working directory resets between shell calls, so every command below `cd`s explicitly.

---

## File Structure

| File | Created / Modified | One responsibility |
|---|---|---|
| `framework/src/fedlearn/communication/serializer.py` | Modified | Bug 3: make `parameters_to_chunks` save symmetric with `chunks_to_parameters` load (wrap `parameters` + `num_examples`). |
| `framework/src/fedlearn/estimators/perturbation.py` | **Created** | Single source of truth for `canonical_perturbation(seed, num_params, dtype)` — CPU-canonical, device-independent `N(0, I_d)`. |
| `framework/src/fedlearn/estimators/zeroth_order.py` | Modified | Bug 2: `generate_perturbation` delegates to `canonical_perturbation`, then `.to(self.device)`. |
| `framework/src/fedlearn/server/decomfl_strategy.py` | Modified | Bug 1 (`1/P` fix), Bug 2 (delegate `_generate_perturbation`), B-2 (local RNG (Random Number Generator)), C-1 (hoisted `z`), C-2 (bounded history). |
| `framework/tests/fixtures/decomfl_golden/` | **Created** | Golden-vector fixtures (`.npy` (NumPy array file) + JSON (JavaScript Object Notation) manifest recording torch/numpy versions) and a generator script for `refreeze`. |
| `framework/tests/test_serializer.py` | Modified | T3: multi-chunk + transformer-shaped dict roundtrip (extends existing `TestChunkedRoundtrip`). |
| `framework/tests/test_perturbation.py` | **Created** | T2: golden bit-exact (CPU, always runs) + cross-device parity (skip-guarded). |
| `framework/tests/test_decomfl_strategy.py` | Modified | T1 (rebuild-trajectory), T4 (optimized ≡ corrected-naive), T5 (bounded history), B-1 (fix stale `seed_history` test), B-2 (no global RNG mutation). |
| `docs/wikis/framework/06_decomfl.md` | Modified | Correct the "P factor cancels in derivation" note (two spots: pseudocode `:97`, code listing `:333`). |

---

## Task 0: Environment setup & baseline (verification only, no code change)

**Why:** Establish the editable install and see exactly which tests are red today, so later GREEN steps are provable.

- [ ] Install the framework editable and capture the baseline test run.

```bash
cd /home/anurag/codebase/FedLearn-Platform/framework && pip install -e . && pytest
```

**Expected:** The suite runs (it does not require a GPU (Graphics Processing Unit)). Record which tests fail. Per prior verification, expect **at least 4 failures**:
- `tests/test_serializer.py::TestChunkedRoundtrip::test_chunks_roundtrip_single_chunk` — `KeyError: 'parameters'` (Bug 3).
- `tests/test_serializer.py::TestChunkedRoundtrip::test_chunks_roundtrip_multiple_chunks` — `KeyError: 'parameters'` (Bug 3).
- `tests/test_serializer.py::TestChunkedRoundtrip::test_chunks_metadata_is_consistent` — this one only reads chunk dict metadata so it may pass; confirm from the actual run and record it either way.
- `tests/test_decomfl_strategy.py::TestDeComFLStrategy::test_aggregate_fit_updates_global_params` — `AttributeError: 'dict' object has no attribute 'append'` (B-1: the test calls `self.strategy.seed_history.append(seeds)` but `seed_history` is a `Dict` keyed by round, see `framework/src/fedlearn/server/decomfl_strategy.py:66`).

> If `pip install -e .` errors on missing `[project]` dependencies (`framework/pyproject.toml` has no `[project]` table; deps like `torch`, `numpy`, `grpcio` are assumed pre-installed in the environment), do **not** add a `[project]` table in this plan — that is out of scope. Confirm `python3 -c "import torch, numpy, grpc"` succeeds, then run `pytest` directly from `framework/` (the editable install only puts `src/fedlearn` on the path). Flag the missing `[project]` table as a known gap and move on.

- [ ] Write the failing-test list into the task tracker / scratch notes. This is the canary set. Do not edit any source in Task 0.

---

## Task 1: Bug 3 — serializer save/load symmetry (test T3)

**Why first:** Lowest risk, self-contained, flips the currently-red serializer tests, and unblocks the chunked/LLM upload path that DeComFL's whole reason-for-existing depends on.

**Root cause:** `parameters_to_chunks` does `torch.save(params, buffer)` (a bare `OrderedDict`) at `framework/src/fedlearn/communication/serializer.py:97`, but `chunks_to_parameters` returns `model_data['parameters'], model_data['num_examples']` at `:155` — expecting a wrapped dict. Any model without a tensor literally named `parameters` → `KeyError`.

### RED — extend the existing chunk roundtrip tests

The existing `tests/test_serializer.py::TestChunkedRoundtrip` already asserts `num_examples` comes back (lines 86, 104, 114) and already fails. Add two more cases that mirror the real DeComFL path: a model larger than `CHUNK_SIZE` forced into many chunks, and a transformer-shaped `state_dict` whose tensor names are NOT `parameters`.

- [ ] Append the following test methods to the `TestChunkedRoundtrip` class in `framework/tests/test_serializer.py` (after `test_chunks_metadata_is_consistent`, line 114). Keep the existing import block; it already imports `parameters_to_chunks`, `chunks_to_parameters`, `OrderedDict`, `torch`.

```python
    def test_chunks_roundtrip_forced_multichunk_large_model(self):
        # A model deliberately larger than a small chunk_size so it takes the
        # multi-chunk path that every transformer/LLM takes in production.
        torch.manual_seed(3)
        original = OrderedDict([
            ("encoder.weight", torch.randn(512, 512, dtype=torch.float32)),
            ("encoder.bias", torch.randn(512, dtype=torch.float32)),
        ])
        # 512*512*4 bytes ~= 1 MB of weights; force chunks at 64 KB.
        chunks = list(parameters_to_chunks(original, num_examples=777, chunk_size=64 * 1024))
        assert len(chunks) > 1, "large model must span multiple chunks"
        assert chunks[-1]["is_final_chunk"] is True

        raw_bytes = b"".join(c["chunk_data"] for c in chunks)
        recovered, num_examples = chunks_to_parameters(raw_bytes, compressed=False)

        assert num_examples == 777
        for key in original:
            assert torch.allclose(original[key], recovered[key], atol=1e-6)

    def test_chunks_roundtrip_transformer_shaped_state_dict(self):
        # Tensor names are realistic transformer keys, NONE of which is the
        # literal string "parameters". This is the exact shape that triggers
        # the KeyError when save/load are asymmetric.
        torch.manual_seed(4)
        original = OrderedDict([
            ("transformer.h.0.attn.c_attn.weight", torch.randn(8, 8, dtype=torch.float32)),
            ("transformer.h.0.attn.c_attn.bias", torch.randn(8, dtype=torch.float32)),
            ("transformer.ln_f.weight", torch.randn(8, dtype=torch.float32)),
            ("lm_head.weight", torch.randn(4, 8, dtype=torch.float32)),
        ])
        chunks = list(parameters_to_chunks(original, num_examples=1234, chunk_size=1))
        assert len(chunks) > 1

        raw_bytes = b"".join(c["chunk_data"] for c in chunks)
        recovered, num_examples = chunks_to_parameters(raw_bytes, compressed=False)

        assert num_examples == 1234
        assert set(recovered.keys()) == set(original.keys())
        for key in original:
            assert torch.allclose(original[key], recovered[key], atol=1e-6)
```

- [ ] Run only the chunk tests and confirm they fail.

```bash
cd /home/anurag/codebase/FedLearn-Platform/framework && pytest tests/test_serializer.py::TestChunkedRoundtrip -v
```

**Expected FAIL:** Every `TestChunkedRoundtrip` test that calls `chunks_to_parameters` errors with `KeyError: 'parameters'` (raised inside `chunks_to_parameters` at `serializer.py:155`, re-raised after the `log.exception("chunks_to_parameters failed")`). The two new tests fail with the same `KeyError`.

### GREEN — wrap the save payload

- [ ] In `framework/src/fedlearn/communication/serializer.py`, change line 97 from the bare save to the wrapped save so it matches the load at line 155.

Replace:
```python
        torch.save(params, buffer)
```
with:
```python
        torch.save({'parameters': params, 'num_examples': num_examples}, buffer)
```

> Do not touch `chunks_to_parameters` (line 155) — the load is already correct. Do not touch the per-chunk `num_examples` field (line 127); it stays as early-read metadata, now redundant but harmless. Do not touch `grpc_servicer.py:213` — its `chunks_to_parameters(full_data, ...)` call already unpacks `(parameters, num_examples)` and now gets a correctly-wrapped blob.

- [ ] Rerun the chunk tests and confirm pass.

```bash
cd /home/anurag/codebase/FedLearn-Platform/framework && pytest tests/test_serializer.py -v
```

**Expected PASS:** All of `TestChunkedRoundtrip` passes, including the two new tests and the previously-red `test_chunks_roundtrip_single_chunk` / `test_chunks_roundtrip_multiple_chunks`. `TestProtoRoundtrip` is unaffected.

### COMMIT

- [ ] Commit.

```bash
cd /home/anurag/codebase/FedLearn-Platform && git add framework/src/fedlearn/communication/serializer.py framework/tests/test_serializer.py && git commit -m "fix(framework): make chunked serializer save/load symmetric

parameters_to_chunks saved a bare OrderedDict while chunks_to_parameters
expected a wrapped {parameters, num_examples} dict, raising KeyError on
every model whose state_dict has no tensor named 'parameters' (i.e. every
transformer/LLM, which always takes the chunked path). Wrap the save to
match the load. Adds multi-chunk + transformer-shaped roundtrip tests."
```

---

## Task 2: Shared `canonical_perturbation` helper + golden-vector fixtures (test T2)

**Why now:** Bug 2's fix is a new shared helper. Build and pin it in isolation (with frozen golden vectors that double as the C++ port's later conformance test) before wiring the server and client to it in Task 3.

**Root cause (Bug 2):** Both the server (`decomfl_strategy.py:210-219`) and the client (`zeroth_order.py:45-48`) generate `z` with `torch.Generator(device=self.device)` + `torch.randn(..., device=self.device)`. Seeded `torch.randn` is **not** bit-identical across CPU/CUDA (Compute Unified Device Architecture)/MPS (Metal Performance Shaders), so a CUDA server reconstructs a different `z` than a CPU/MPS client for the same seed and `Σ g·z` aggregation is corrupted on any GPU server or mixed-device fleet. The fix (locked decision "Approach A") is to generate on CPU with a local `torch.Generator`, then `.to(device)` at the use site.

### Create the shared helper

- [ ] Create `framework/src/fedlearn/estimators/perturbation.py` with exactly this content:

```python
# src/fedlearn/estimators/perturbation.py
"""
Device-independent perturbation generation for DeComFL.

Single source of truth for the zeroth-order perturbation vector z ~ N(0, I_d).
Generated on CPU with a local torch.Generator so the output is bit-stable
across CPU / CUDA / MPS for a given (seed, num_params, dtype). Callers move the
result to their working device at the use site. This is the RNG contract that
the server aggregation path and the client fit/rebuild path MUST share, and
that the future C++ mobile port must conform to (see the golden fixtures in
tests/fixtures/decomfl_golden/).
"""

import torch

# The perturbation dtype is pinned explicitly so it never silently follows a
# model's dtype (which would break golden-vector parity).
CANONICAL_DTYPE = torch.float32


def canonical_perturbation(
    seed: int,
    num_params: int,
    dtype: torch.dtype = CANONICAL_DTYPE,
) -> torch.Tensor:
    """Return a device-independent N(0, I_d) perturbation of shape (num_params,).

    Generated on CPU with a local torch.Generator for bit-stable output across
    devices. Callers should move to their device at the use site: z.to(device).

    Args:
        seed: Random seed for reproducibility.
        num_params: Dimension d.
        dtype: Floating dtype of the output (default torch.float32).

    Returns:
        CPU tensor of shape (num_params,).
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(num_params, generator=generator, dtype=dtype, device="cpu")
```

### Create the golden-fixture generator + committed fixture

- [ ] Create the fixtures directory and a generator script `framework/tests/fixtures/decomfl_golden/generate.py`. The script docstring below references CI (Continuous Integration); the `sha256` calls are the standard `hashlib.sha256` (Secure Hash Algorithm 256-bit) API and stay verbatim as code identifiers:

```python
# framework/tests/fixtures/decomfl_golden/generate.py
"""Regenerate the DeComFL golden perturbation vectors.

Language-neutral RNG-contract artifact: each case stores z as a .npy plus a
sha256 in manifest.json, alongside the torch/numpy versions used to freeze it.
Re-run ONLY on an intentional torch bump (see `make refreeze-golden`); CI
compares committed vectors against freshly generated ones and fails on drift
without a version bump.

Usage:
    cd framework && python tests/fixtures/decomfl_golden/generate.py
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fedlearn.estimators.perturbation import canonical_perturbation

HERE = Path(__file__).parent

# (seed, num_params) cases. dtype is float32 (the pinned canonical dtype).
CASES = [
    {"name": "seed42_d23", "seed": 42, "num_params": 23},
    {"name": "seed7_d100", "seed": 7, "num_params": 100},
    {"name": "seed12345_d1024", "seed": 12345, "num_params": 1024},
]


def main() -> None:
    manifest = {
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "dtype": "float32",
        "cases": [],
    }
    for case in CASES:
        z = canonical_perturbation(case["seed"], case["num_params"]).numpy()
        npy_path = HERE / f"{case['name']}.npy"
        np.save(npy_path, z)
        sha = hashlib.sha256(z.tobytes()).hexdigest()
        manifest["cases"].append({
            "name": case["name"],
            "seed": case["seed"],
            "num_params": case["num_params"],
            "npy": f"{case['name']}.npy",
            "sha256": sha,
        })
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Froze {len(CASES)} golden vectors for torch {torch.__version__}")


if __name__ == "__main__":
    main()
```

- [ ] Generate and commit the fixture artifacts now (this freezes them against the current torch `2.12.0+cu130` / numpy `2.1.2`):

```bash
cd /home/anurag/codebase/FedLearn-Platform/framework && python tests/fixtures/decomfl_golden/generate.py && ls tests/fixtures/decomfl_golden/
```

**Expected:** Prints `Froze 3 golden vectors for torch 2.12.0+cu130` and the directory lists `generate.py`, `manifest.json`, `seed42_d23.npy`, `seed7_d100.npy`, `seed12345_d1024.npy`. Open `manifest.json` and confirm `torch_version` is recorded.

### RED — write the golden + cross-device parity test (T2)

- [ ] Create `framework/tests/test_perturbation.py`:

```python
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fedlearn.estimators.perturbation import canonical_perturbation

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "decomfl_golden"


def _load_manifest() -> dict:
    return json.loads((GOLDEN_DIR / "manifest.json").read_text())


class TestGoldenVectors:

    def test_canonical_perturbation_matches_committed_golden(self):
        manifest = _load_manifest()
        for case in manifest["cases"]:
            z = canonical_perturbation(case["seed"], case["num_params"]).numpy()
            # Bit-exact: recompute the sha256 over the raw bytes.
            sha = hashlib.sha256(z.tobytes()).hexdigest()
            assert sha == case["sha256"], (
                f"golden drift for {case['name']}: regenerate via "
                f"`python tests/fixtures/decomfl_golden/generate.py` only on an "
                f"intentional torch bump (manifest pins {manifest['torch_version']})"
            )
            # And the stored .npy still matches what we just generated.
            stored = np.load(GOLDEN_DIR / case["npy"])
            assert np.array_equal(z, stored)

    def test_canonical_perturbation_is_float32_on_cpu(self):
        z = canonical_perturbation(seed=1, num_params=16)
        assert z.dtype == torch.float32
        assert z.device.type == "cpu"
        assert z.shape == (16,)

    def test_same_seed_same_vector(self):
        a = canonical_perturbation(seed=99, num_params=50)
        b = canonical_perturbation(seed=99, num_params=50)
        assert torch.equal(a, b)


class TestCrossDeviceParity:
    """Server-path z and client-path z must be identical for the same seed.

    The CPU generation is asserted unconditionally; the GPU comparison is
    skip-guarded so CI stays GPU-free.
    """

    def test_cpu_generation_then_move_is_value_stable(self):
        z_cpu = canonical_perturbation(seed=2024, num_params=64)
        moved = z_cpu.to("cpu")  # no-op move; value must be unchanged
        assert torch.equal(z_cpu, moved)

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_move_preserves_values(self):
        z_cpu = canonical_perturbation(seed=2024, num_params=64)
        z_cuda = z_cpu.to("cuda")
        assert torch.equal(z_cpu, z_cuda.cpu())

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS not available"
    )
    def test_mps_move_preserves_values(self):
        z_cpu = canonical_perturbation(seed=2024, num_params=64)
        z_mps = z_cpu.to("mps")
        assert torch.equal(z_cpu, z_mps.cpu())
```

> Note: `conftest.py`'s autouse `disable_cuda` fixture monkeypatches `torch.cuda.is_available` to `False`, so `test_cuda_move_preserves_values` always skips under the test suite. That is intentional and correct — the golden CPU test is the always-on guard; the GPU tests are documentation of intent that only run on a real GPU host with the fixture disabled.

- [ ] Run the perturbation tests.

```bash
cd /home/anurag/codebase/FedLearn-Platform/framework && pytest tests/test_perturbation.py -v
```

**Expected PASS:** All non-skipped tests pass (the helper and fixtures were just created against the same torch). The CUDA/MPS tests report `SKIPPED`. If `test_canonical_perturbation_matches_committed_golden` fails, the fixtures were not regenerated against the current torch — rerun `generate.py` first. (This task's "RED" is the prior absence of the file; the helper + fixture are authored together, so the test goes green immediately. The real Bug-2 regression guard lands in Task 3 where the server/client paths are forced to agree.)

### COMMIT

- [ ] Commit.

```bash
cd /home/anurag/codebase/FedLearn-Platform && git add framework/src/fedlearn/estimators/perturbation.py framework/tests/fixtures/decomfl_golden/ framework/tests/test_perturbation.py && git commit -m "feat(framework): add CPU-canonical perturbation helper + golden vectors

Introduces canonical_perturbation(seed, num_params, dtype=float32): the
single, device-independent source of truth for the DeComFL N(0, I_d)
perturbation. Generated on CPU with a local torch.Generator for bit-stable
output across CPU/CUDA/MPS. Adds version-pinned golden fixtures (npy + sha256
+ torch/numpy versions) as the language-neutral RNG contract for the later
C++ mobile port, plus T2 golden + cross-device parity tests."
```

---

## Task 3: Wire server and client to the shared helper (Bug 2)

**Why now:** The helper exists and is pinned. Now eliminate the two duplicated, device-dependent RNG copies by delegating both to it.

### RED — assert server-path and client-path z agree

- [ ] Add a test class to `framework/tests/test_perturbation.py` that exercises the two real call sites and asserts they produce the identical vector for the same seed. Append after `TestCrossDeviceParity`:

```python
class TestServerClientPerturbationAgree:
    """The server aggregation path and the client fit/rebuild path must
    reconstruct the exact same z for a given seed (the DeComFL invariant)."""

    def test_estimator_path_matches_canonical(self):
        from fedlearn.estimators.zeroth_order import ZerothOrderEstimator

        est = ZerothOrderEstimator(smoothing_param=0.001, device="cpu")
        seed, d = 314159, 128
        client_z = est.generate_perturbation(seed=seed, num_params=d)
        ref = canonical_perturbation(seed=seed, num_params=d)
        assert torch.equal(client_z, ref.to(client_z.device))

    def test_strategy_path_matches_canonical(self):
        from collections import OrderedDict

        from fedlearn.server.decomfl_strategy import DeComFL

        # 128 params total so _generate_perturbation produces a length-128 z.
        init = OrderedDict([("w", torch.zeros(128, dtype=torch.float32))])
        strategy = DeComFL(
            initial_parameters=init,
            evaluate_fn=None,
            min_fit_clients=1,
            clients_per_round=1,
            num_local_steps=1,
            num_perturbations=1,
            learning_rate=0.01,
            smoothing_param=0.001,
            seed=42,
        )
        seed = 314159
        server_z = strategy._generate_perturbation(seed)
        ref = canonical_perturbation(seed=seed, num_params=128)
        assert torch.equal(server_z.cpu(), ref)

    def test_server_and_client_agree_for_same_seed(self):
        from collections import OrderedDict

        from fedlearn.estimators.zeroth_order import ZerothOrderEstimator
        from fedlearn.server.decomfl_strategy import DeComFL

        init = OrderedDict([("w", torch.zeros(64, dtype=torch.float32))])
        strategy = DeComFL(
            initial_parameters=init,
            evaluate_fn=None,
            min_fit_clients=1,
            clients_per_round=1,
            num_local_steps=1,
            num_perturbations=1,
            learning_rate=0.01,
            smoothing_param=0.001,
            seed=42,
        )
        est = ZerothOrderEstimator(smoothing_param=0.001, device="cpu")
        seed = 271828
        server_z = strategy._generate_perturbation(seed).cpu()
        client_z = est.generate_perturbation(seed=seed, num_params=64)
        assert torch.equal(server_z, client_z)
```

- [ ] Run the new class.

```bash
cd /home/anurag/codebase/FedLearn-Platform/framework && pytest tests/test_perturbation.py::TestServerClientPerturbationAgree -v
```

**Expected RED before the wiring:** On CPU these may already pass because `torch.randn(..., device='cpu')` already matches `canonical_perturbation`. The point of this task is to make the agreement **device-independent and structural** (single code path), not coincidental: the wiring below removes the device-dependent divergence that would make them fail on a CUDA/MPS host, and the golden test in Task 2 plus T1 in Task 4 are the bit-exact guards. Record the CPU result either way.

### GREEN — delegate both call sites

- [ ] In `framework/src/fedlearn/estimators/zeroth_order.py`, add the import after line 10 (`from collections import OrderedDict`):

```python
from fedlearn.estimators.perturbation import canonical_perturbation
```

- [ ] Replace the body of `generate_perturbation` (lines 45-48) so it delegates to the shared helper and moves to the working device.

Replace:
```python
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        z = torch.randn(num_params, generator=generator, device=self.device)
        return z
```
with:
```python
        z = canonical_perturbation(seed, num_params)
        return z.to(self.device)
```

- [ ] In `framework/src/fedlearn/server/decomfl_strategy.py`, add the import after line 12 (`from .strategy import Strategy`):

```python
from fedlearn.estimators.perturbation import canonical_perturbation
```

- [ ] Replace the body of `_generate_perturbation` (lines 211-219) so it delegates to the shared helper and moves to `self.device`.

Replace:
```python
        """Generate perturbation vector from seed."""
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        z = torch.randn(
            len(self.global_params_flat),
            generator=generator,
            device=self.device
        )
        return z
```
with:
```python
        """Generate perturbation vector from seed (device-independent)."""
        z = canonical_perturbation(seed, len(self.global_params_flat))
        return z.to(self.device)
```

- [ ] Rerun the parity tests and the estimator/strategy suites.

```bash
cd /home/anurag/codebase/FedLearn-Platform/framework && pytest tests/test_perturbation.py tests/test_zeroth_order.py tests/test_decomfl_strategy.py -v
```

**Expected PASS:** `TestServerClientPerturbationAgree` passes; `test_zeroth_order.py` still passes (shape, same-seed, different-seed assertions hold because `canonical_perturbation` is deterministic). `test_decomfl_strategy.py` still has the one B-1 failure (`test_aggregate_fit_updates_global_params`) — that is fixed in Task 8; everything else passes.

### COMMIT

- [ ] Commit.

```bash
cd /home/anurag/codebase/FedLearn-Platform && git add framework/src/fedlearn/estimators/zeroth_order.py framework/src/fedlearn/server/decomfl_strategy.py framework/tests/test_perturbation.py && git commit -m "fix(framework): route server + client perturbation through shared helper

Both DeComFL._generate_perturbation and ZerothOrderEstimator.generate_perturbation
generated z on self.device, and seeded torch.randn is not bit-identical across
CPU/CUDA/MPS, so a GPU server reconstructed a different z than a CPU client for
the same seed and corrupted the sum-of g*z aggregation. Delegate both to
canonical_perturbation (CPU-canonical) and .to(device) at the use site,
killing the duplicated RNG logic."
```

---

## Task 4: Bug 1 — the `1/P` fix (test T1, the canary)

**Why now:** With perturbations now device-stable (Task 3), the rebuild-trajectory equivalence test can isolate the `1/P` error cleanly.

**Root cause:** `decomfl_strategy.py:197` computes `delta = delta / (num_clients * self.P)`, then `:200` multiplies it back: `x_current = x_current - self.eta * delta * self.P`. The `* self.P` cancels the `1/P`, so the server steps **P times too far** (P defaults to 10). The client at `decomfl_client.py:208` correctly applies `(eta / P) * delta`, and `rebuild_model` (`decomfl_client.py:115`) replays the `(1/P)` step. So whenever P>1 the server's `global_params_flat` and every client's rebuild trajectory diverge by construction — violating the paper's central seed/gradient replay guarantee. **Fix = delete `* self.P` on line 200.** Do not touch the client (it is correct).

### RED — rebuild-trajectory equivalence (T1)

This test drives `aggregate_fit` for several rounds (server trajectory) and replays the same seeds + server-averaged gradients through `rebuild_model` (client trajectory), then asserts they land on the same parameters. Today the server takes P-sized steps and the rebuild takes (1/P)-sized steps, so they diverge.

- [ ] Append this test class to `framework/tests/test_decomfl_strategy.py` (after the existing `TestDeComFLStrategy` class). It uses the public seed API (`get_or_generate_seeds`) and a dataloader-free rebuild via `DeComFLClient.rebuild_model`.

```python
class TestRebuildTrajectoryEquivalence:
    """T1 (canary): a client that reconstructs every round via rebuild_model
    must land on the same parameters as the server's aggregated trajectory.
    Fails while the server step is P x too large (Bug 1)."""

    def _make_strategy(self, P):
        return DeComFL(
            initial_parameters=make_params(0.0),
            evaluate_fn=None,
            min_fit_clients=1,
            clients_per_round=1,
            num_local_steps=2,   # K=2
            num_perturbations=P,
            learning_rate=0.05,
            smoothing_param=0.001,
            seed=123,
        )

    def test_server_trajectory_matches_client_rebuild(self):
        import torch as _torch
        from fedlearn.client.decomfl_client import DeComFLClient

        P = 4
        K = 2
        n_rounds = 3
        strategy = self._make_strategy(P)

        # Minimal nn.Module whose flat param count matches make_params
        # (fc.weight 1x3 + fc.bias 1 = 4 params).
        class _Tiny(_torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = _torch.nn.Linear(3, 1)  # 3 weights + 1 bias = 4 params

            def forward(self, x):
                return self.fc(x)

        model = _Tiny()
        # Zero the model so its flat params match the strategy's initial 0.0.
        with _torch.no_grad():
            for p in model.parameters():
                p.zero_()
        client = DeComFLClient(model=model, train_loader=None, smoothing_param=0.001, device="cpu")

        rebuild_history = []
        for r in range(1, n_rounds + 1):
            seeds = strategy.get_or_generate_seeds(r)  # stores into seed_history[r]
            # Deterministic fake gradient scalars for the single client.
            grads = [[0.1 * (k + 1) + 0.01 * p for p in range(P)] for k in range(K)]
            # Server aggregates (single client => average == grads).
            strategy.aggregate_fit(server_round=r, results=[("c1", grads, 100)])
            rebuild_history.append({
                "round_number": r,
                "seeds": seeds,
                "gradients": grads,
            })

        # Client reconstructs the whole trajectory from history.
        client.rebuild_model(rebuild_history, learning_rate=0.05)

        server_flat = strategy.global_params_flat.cpu()
        client_flat = client.x_current.cpu()
        assert _torch.allclose(server_flat, client_flat, atol=1e-6), (
            f"server {server_flat} != client rebuild {client_flat}"
        )
```

- [ ] Run T1 and confirm it fails.

```bash
cd /home/anurag/codebase/FedLearn-Platform/framework && pytest tests/test_decomfl_strategy.py::TestRebuildTrajectoryEquivalence -v
```

**Expected FAIL:** `AssertionError: server [...] != client rebuild [...]` — the server vector is ~P times further from zero than the client's, because the server applies `η·delta·P` while the rebuild applies `(η/P)·delta`.

### GREEN — drop the `* self.P`

- [ ] In `framework/src/fedlearn/server/decomfl_strategy.py`, change line 200.

Replace:
```python
            # Update model parameters
            x_current = x_current - self.eta * delta * self.P
```
with:
```python
            # Update model parameters (delta is already averaged by 1/(N*P))
            x_current = x_current - self.eta * delta
```

- [ ] Rerun T1 and confirm pass.

```bash
cd /home/anurag/codebase/FedLearn-Platform/framework && pytest tests/test_decomfl_strategy.py::TestRebuildTrajectoryEquivalence -v
```

**Expected PASS:** server trajectory now equals the client rebuild trajectory (`torch.allclose` at `atol=1e-6`).

### Correct the docs (no behavior change, but the spec requires it)

- [ ] Fix the misleading note in `docs/wikis/framework/06_decomfl.md`. There are **two** spots.

Spot 1 — pseudocode at `docs/wikis/framework/06_decomfl.md:97`. Replace:
```
       x_t = x_{t-1} - η × P × Δ   (note: P factor cancels in full derivation)
```
with:
```
       x_t = x_{t-1} - η × Δ   (Δ already averaged by 1/(N×P); matches client (η/P)×δ)
```

Spot 2 — code listing at `docs/wikis/framework/06_decomfl.md:333`. Replace:
```
        x_current = x_current - self.eta * delta * self.P  # ×P cancels in derivation
```
with:
```
        x_current = x_current - self.eta * delta  # delta already averaged by 1/(N*P)
```

### COMMIT

- [ ] Commit.

```bash
cd /home/anurag/codebase/FedLearn-Platform && git add framework/src/fedlearn/server/decomfl_strategy.py framework/tests/test_decomfl_strategy.py docs/wikis/framework/06_decomfl.md && git commit -m "fix(framework): drop spurious *P in DeComFL server update

aggregate_fit divided delta by (num_clients*P) then multiplied the step back
out by *P, so the global model stepped P x too far (P=10 default) and the
server trajectory diverged from every client's rebuild_model trajectory,
violating the paper's seed/gradient replay guarantee. Delete the *P; delta is
already 1/(N*P)-averaged. Corrects the 'P cancels in derivation' wiki note."
```

---

## Task 5: B-2 — remove process-global RNG mutation

**Why now:** Isolated hygiene fix on the strategy constructor; independent of the aggregation math fixed above.

**Root cause:** `decomfl_strategy.py:82-83` calls `np.random.seed(seed)` and `torch.manual_seed(seed)` in `__init__`, mutating **process-global** RNG state and corrupting reproducibility for anything else in-process (audit M5). `generate_seeds` (line 107) then draws from the global numpy RNG via `np.random.randint`. Fix: hold a local `np.random.Generator` on the strategy and draw from it; drop the global `torch.manual_seed` (nothing in the strategy reads the global torch RNG after perturbations route through `canonical_perturbation`, which uses its own local generator).

### RED — assert constructing the strategy does not move global RNG

- [ ] Append this test class to `framework/tests/test_decomfl_strategy.py`:

```python
class TestNoGlobalRNGMutation:
    """B-2: constructing DeComFL must not mutate process-global torch/numpy RNG."""

    def test_constructor_leaves_global_rng_untouched(self):
        import numpy as _np
        import torch as _torch

        # Snapshot global RNG state.
        torch_state_before = _torch.random.get_rng_state()
        np_state_before = _np.random.get_state()

        DeComFL(
            initial_parameters=make_params(0.0),
            evaluate_fn=None,
            min_fit_clients=1,
            clients_per_round=1,
            num_local_steps=2,
            num_perturbations=3,
            learning_rate=0.01,
            smoothing_param=0.001,
            seed=777,
        )

        torch_state_after = _torch.random.get_rng_state()
        np_state_after = _np.random.get_state()

        assert _torch.equal(torch_state_before, torch_state_after), (
            "DeComFL.__init__ mutated the global torch RNG"
        )
        # numpy get_state returns a tuple; the key array is index 1.
        assert _np.array_equal(np_state_before[1], np_state_after[1]), (
            "DeComFL.__init__ mutated the global numpy RNG"
        )

    def test_generate_seeds_is_deterministic_per_strategy(self):
        s1 = DeComFL(
            initial_parameters=make_params(0.0), evaluate_fn=None,
            min_fit_clients=1, clients_per_round=1, num_local_steps=2,
            num_perturbations=3, learning_rate=0.01, smoothing_param=0.001, seed=555,
        )
        s2 = DeComFL(
            initial_parameters=make_params(0.0), evaluate_fn=None,
            min_fit_clients=1, clients_per_round=1, num_local_steps=2,
            num_perturbations=3, learning_rate=0.01, smoothing_param=0.001, seed=555,
        )
        assert s1.generate_seeds(0) == s2.generate_seeds(0), (
            "same strategy seed must yield the same seed schedule"
        )
```

- [ ] Run it and confirm the first test fails.

```bash
cd /home/anurag/codebase/FedLearn-Platform/framework && pytest tests/test_decomfl_strategy.py::TestNoGlobalRNGMutation -v
```

**Expected FAIL:** `test_constructor_leaves_global_rng_untouched` fails with `AssertionError: DeComFL.__init__ mutated the global torch RNG` (and/or the numpy assertion), because lines 82-83 call the global seeders.

### GREEN — local generators

- [ ] In `framework/src/fedlearn/server/decomfl_strategy.py`, replace the global seed calls (lines 81-83; note lines 79-80 are both blank — leave one blank line in place). Match on the three non-blank lines below:

Replace:
```python
        # Random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
```
with:
```python
        # Local RNG only — never mutate process-global torch/numpy RNG (audit M5).
        self._np_rng = np.random.default_rng(seed)
```

- [ ] In the same file, update `generate_seeds` (line 107) to draw from the local generator.

Replace:
```python
                seed = np.random.randint(0, 2 ** 31 - 1)
```
with:
```python
                seed = self._np_rng.integers(0, 2 ** 31 - 1)
```

- [ ] Rerun B-2 tests and confirm pass.

```bash
cd /home/anurag/codebase/FedLearn-Platform/framework && pytest tests/test_decomfl_strategy.py::TestNoGlobalRNGMutation -v
```

**Expected PASS:** both tests pass — global RNG state is byte-identical before/after construction, and two strategies with the same `seed` produce identical seed schedules.

> Note: `np.random.Generator.integers` returns a numpy integer; `generate_seeds` already wraps it `int(seed)` at line 108, so the `List[List[int]]` contract is preserved. Confirm the existing `test_generate_seeds_values_are_non_negative_integers` (which asserts `isinstance(s, int)`) still passes in the rerun.

### COMMIT

- [ ] Commit.

```bash
cd /home/anurag/codebase/FedLearn-Platform && git add framework/src/fedlearn/server/decomfl_strategy.py framework/tests/test_decomfl_strategy.py && git commit -m "fix(framework): use local RNG in DeComFL strategy, not global seeders

__init__ called np.random.seed + torch.manual_seed, mutating process-global
RNG state and corrupting reproducibility for everything else in-process
(audit M5). Hold a local np.random.default_rng on the strategy and draw seeds
from it; drop the global torch seeding (perturbations already use a local
generator via canonical_perturbation)."
```

---

## Task 6: C-1 — hoist z-generation to O(K*P) (test T4)

**Why now:** A pure performance refactor of `aggregate_fit` that must be numerically identical to the (now Bug-1-corrected) naive loop.

**Root cause / opportunity:** the loop at `decomfl_strategy.py:180-200` regenerates the d-dimensional `z` once per `(client, k, p)` — O(K*P*N) `randn`+`mul` over the full d-vector. But `z` depends only on `(k, p)`, not on the client. Hoist: for each `(k, p)` generate `z` once, sum `g` across clients, then `delta += (Σ_c g_c) · z`. This is O(K*P) generations — N-fold fewer — and numerically identical to the corrected naive loop.

### RED — optimized ≡ corrected-naive (T4)

This test builds a standalone reference implementation of the *corrected* naive loop (the one that includes the Bug-1 `1/P` fix) and asserts the production `aggregate_fit` matches it on multi-client inputs.

- [ ] Append this test class to `framework/tests/test_decomfl_strategy.py`:

```python
class TestOptimizedEqualsNaiveAggregate:
    """T4: the hoisted O(K*P) aggregate_fit must equal a reference naive
    O(K*P*N) loop that ALSO includes the Bug-1 1/P fix (corrected naive)."""

    def _corrected_naive_aggregate(self, strategy, server_round, results):
        import torch as _torch

        x = strategy.global_params_flat.clone()
        client_grads = {cid: g for cid, g, _ in results}
        num_clients = len(client_grads)
        for k in range(strategy.K):
            delta = _torch.zeros_like(x)
            for _cid, grads in client_grads.items():
                for p in range(strategy.P):
                    z = strategy._generate_perturbation(
                        strategy.seed_history[server_round][k][p]
                    )
                    delta += grads[k][p] * z
            delta = delta / (num_clients * strategy.P)
            x = x - strategy.eta * delta  # corrected: no *P
        return x

    def test_aggregate_fit_matches_corrected_naive(self):
        import torch as _torch

        K, P = 2, 3
        strategy = DeComFL(
            initial_parameters=make_params(0.0), evaluate_fn=None,
            min_fit_clients=1, clients_per_round=2, num_local_steps=K,
            num_perturbations=P, learning_rate=0.03, smoothing_param=0.001, seed=2026,
        )
        strategy.get_or_generate_seeds(1)  # populate seed_history[1]

        g1 = [[0.1, -0.2, 0.3], [0.05, 0.4, -0.15]]
        g2 = [[-0.3, 0.1, 0.25], [0.2, -0.1, 0.35]]
        results = [("c1", g1, 100), ("c2", g2, 200)]

        # Reference, computed against the SAME pre-update global params.
        expected = self._corrected_naive_aggregate(strategy, 1, results)

        # Production path mutates global_params_flat in place.
        strategy.aggregate_fit(server_round=1, results=results)
        actual = strategy.global_params_flat

        assert _torch.allclose(actual, expected, atol=1e-6), (
            f"hoisted aggregate {actual} != corrected naive {expected}"
        )
```

- [ ] Run T4 against the still-naive (but Bug-1-fixed) `aggregate_fit`.

```bash
cd /home/anurag/codebase/FedLearn-Platform/framework && pytest tests/test_decomfl_strategy.py::TestOptimizedEqualsNaiveAggregate -v
```

**Expected:** This **passes already** — after Task 4 the production loop is the corrected naive loop, so it equals the reference. That is intentional: T4 is the pre-registered equivalence guard, written before the refactor so the refactor cannot silently change the math. Record the green result, then perform the refactor and confirm it stays green.

### GREEN — hoist the z generation

- [ ] In `framework/src/fedlearn/server/decomfl_strategy.py`, replace the per-step loop body (lines 179-200) with the hoisted form.

Replace:
```python
        # For each local step
        for k in range(self.K):
            delta = torch.zeros_like(x_current)

            # Average gradients across clients
            num_clients = len(client_gradients)
            for client_id, grad_scalars in client_gradients.items():
                for p in range(self.P):
                    # Regenerate perturbation from seed
                    z = self._generate_perturbation(self.seed_history[server_round][k][p])

                    # Get gradient scalar for this client
                    g = grad_scalars[k][p]

                    # Accumulate gradient direction
                    delta += g * z

            # Average across clients and perturbations
            delta = delta / (num_clients * self.P)

            # Update model parameters (delta is already averaged by 1/(N*P))
            x_current = x_current - self.eta * delta
```
with:
```python
        # For each local step
        num_clients = len(client_gradients)
        for k in range(self.K):
            delta = torch.zeros_like(x_current)

            # z depends only on (k, p), not on the client, so generate it once
            # per (k, p) and weight by the summed gradient across clients.
            # O(K*P) generations instead of O(K*P*N); numerically identical.
            for p in range(self.P):
                z = self._generate_perturbation(self.seed_history[server_round][k][p])
                g_sum = 0.0
                for client_id, grad_scalars in client_gradients.items():
                    g_sum += grad_scalars[k][p]
                delta += g_sum * z

            # Average across clients and perturbations
            delta = delta / (num_clients * self.P)

            # Update model parameters (delta is already averaged by 1/(N*P))
            x_current = x_current - self.eta * delta
```

- [ ] Rerun T4 and the rebuild-trajectory canary (T1) to confirm the refactor preserved the math.

```bash
cd /home/anurag/codebase/FedLearn-Platform/framework && pytest tests/test_decomfl_strategy.py::TestOptimizedEqualsNaiveAggregate tests/test_decomfl_strategy.py::TestRebuildTrajectoryEquivalence -v
```

**Expected PASS:** both pass. T4 still matches the reference (which still uses the per-client inner loop), and T1 still equals the client rebuild — the hoist is a pure reassociation of the same sum.

### COMMIT

- [ ] Commit.

```bash
cd /home/anurag/codebase/FedLearn-Platform && git add framework/src/fedlearn/server/decomfl_strategy.py framework/tests/test_decomfl_strategy.py && git commit -m "perf(framework): hoist DeComFL z-generation to O(K*P)

aggregate_fit regenerated the d-dim perturbation once per (client,k,p) even
though z depends only on (k,p). Generate z once per (k,p), sum gradient
scalars across clients, then accumulate g_sum*z: O(K*P) randn/mul over the
d-vector instead of O(K*P*N). Numerically identical (asserted by T4 against a
reference corrected-naive loop)."
```

---

## Task 7: C-2 — bounded history eviction (test T5)

**Why now:** Builds on the corrected aggregation. `seed_history` / `gradient_history` (dicts keyed by round, `decomfl_strategy.py:66-67`) grow per-round forever. Add bounded eviction without breaking the rebuild path within the retention window.

**Design (from spec §5):** after each aggregation, evict entries with `round < min(client_last_round.values())` (the oldest round any known client could still need), behind a configurable `max_retained_rounds` cap. Clients absent longer than the window resync from a checkpoint — explicitly **out of scope** here (owned by the C1 reliability item); this task must only avoid breaking rebuild within the window.

### RED — bounded-history rebuild + bounded size (T5)

- [ ] Append this test class to `framework/tests/test_decomfl_strategy.py`:

```python
class TestBoundedHistory:
    """T5: a client missing N <= max_retained_rounds rounds reconnects and
    rebuilds correctly, and history size stays bounded over many rounds."""

    def _strategy(self, max_retained_rounds):
        return DeComFL(
            initial_parameters=make_params(0.0), evaluate_fn=None,
            min_fit_clients=1, clients_per_round=1, num_local_steps=1,
            num_perturbations=2, learning_rate=0.02, smoothing_param=0.001, seed=9,
            max_retained_rounds=max_retained_rounds,
        )

    def test_history_stays_bounded_across_many_rounds(self):
        strategy = self._strategy(max_retained_rounds=3)
        K, P = strategy.K, strategy.P
        for r in range(1, 21):
            strategy.get_or_generate_seeds(r)
            grads = [[0.1 for _ in range(P)] for _ in range(K)]
            strategy.aggregate_fit(server_round=r, results=[("c1", grads, 100)])
            # Mimic coordinator storing averaged gradients for the round.
            strategy.gradient_history[r] = grads
            strategy.evict_old_history()
        assert len(strategy.seed_history) <= 3, (
            f"seed_history unbounded: {sorted(strategy.seed_history)}"
        )
        assert len(strategy.gradient_history) <= 3

    def test_rebuild_within_window_still_works(self):
        import torch as _torch
        from fedlearn.client.decomfl_client import DeComFLClient

        strategy = self._strategy(max_retained_rounds=5)
        K, P = strategy.K, strategy.P

        class _Tiny(_torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = _torch.nn.Linear(3, 1)

            def forward(self, x):
                return self.fc(x)

        model = _Tiny()
        with _torch.no_grad():
            for p in model.parameters():
                p.zero_()
        client = DeComFLClient(model=model, train_loader=None, smoothing_param=0.001, device="cpu")

        # Client c1 participates in rounds 1..4; an absent client missed 1..3.
        rebuild_history = []
        for r in range(1, 5):
            seeds = strategy.get_or_generate_seeds(r)
            grads = [[0.05 * (p + 1) for p in range(P)] for _ in range(K)]
            strategy.aggregate_fit(server_round=r, results=[("c1", grads, 100)])
            strategy.gradient_history[r] = grads
            strategy.evict_old_history()  # c1 is the only known client
            rebuild_history.append({"round_number": r, "seeds": seeds, "gradients": grads})

        # Within max_retained_rounds=5 and c1 last_round=4, nothing is evicted.
        assert set(strategy.seed_history.keys()) == {1, 2, 3, 4}

        # The absent client rebuilds the full window and matches the server.
        client.rebuild_model(rebuild_history, learning_rate=0.02)
        assert _torch.allclose(
            strategy.global_params_flat.cpu(), client.x_current.cpu(), atol=1e-6
        )
```

- [ ] Run T5 and confirm it fails (no `max_retained_rounds` param, no `evict_old_history`).

```bash
cd /home/anurag/codebase/FedLearn-Platform/framework && pytest tests/test_decomfl_strategy.py::TestBoundedHistory -v
```

**Expected FAIL:** `TypeError: __init__() got an unexpected keyword argument 'max_retained_rounds'` (and, once that is added, `AttributeError: 'DeComFL' object has no attribute 'evict_old_history'`).

### GREEN — add the cap param + eviction method

- [ ] In `framework/src/fedlearn/server/decomfl_strategy.py`, add the `max_retained_rounds` parameter to `__init__`. Change the signature (lines 35-38).

Replace:
```python
            learning_rate: float = 0.001,
            smoothing_param: float = 0.001,
            seed: int = 42
    ):
```
with:
```python
            learning_rate: float = 0.001,
            smoothing_param: float = 0.001,
            seed: int = 42,
            max_retained_rounds: int = 100,
    ):
```

- [ ] Store it. After the hyperparameter block (after line 60, `self.mu = smoothing_param`), add:
```python

        # Cap on how many past rounds of seed/gradient history to retain.
        # Clients absent longer than this must resync from a checkpoint
        # (out of scope here; owned by the C1 reliability item).
        self.max_retained_rounds = max_retained_rounds
```

- [ ] Add the eviction method. Insert it after `aggregate_fit` returns (after line 208, before `_generate_perturbation` at line 210):
```python
    def evict_old_history(self) -> None:
        """Evict seed/gradient history older than any known client still needs.

        Keeps every round >= min(client_last_round) (the oldest round any known
        client could still rebuild from), and additionally caps total retained
        rounds at max_retained_rounds. Safe no-op when no history exists.
        """
        all_rounds = set(self.seed_history) | set(self.gradient_history)
        if not all_rounds:
            return

        # Floor 1: the oldest round any known client could still need.
        if self.client_last_round:
            client_floor = min(self.client_last_round.values())
        else:
            client_floor = min(all_rounds)

        # Floor 2: the max_retained_rounds cap, measured from the newest round.
        newest = max(all_rounds)
        cap_floor = newest - self.max_retained_rounds + 1

        floor = max(client_floor, cap_floor)
        for r in list(self.seed_history):
            if r < floor:
                del self.seed_history[r]
        for r in list(self.gradient_history):
            if r < floor:
                del self.gradient_history[r]
```

- [ ] Rerun T5 and confirm pass.

```bash
cd /home/anurag/codebase/FedLearn-Platform/framework && pytest tests/test_decomfl_strategy.py::TestBoundedHistory -v
```

**Expected PASS:** both tests pass — over 20 rounds with `max_retained_rounds=3` the dicts hold at most 3 entries, and the within-window rebuild still equals the server trajectory.

> Wiring note (not a code change in this task): `evict_old_history` is a method on the strategy. The natural call site is the coordinator's `_trigger_decomfl_aggregation_and_evaluation` (`framework/src/fedlearn/server/coordinator.py:286`) right after it stores `self.strategy.gradient_history[self.current_round] = avg_gradients`. Wiring that call is left to the C1 reliability item per spec §5; this task delivers the bounded mechanism and proves it does not break in-window rebuild. Do **not** call `evict_old_history` automatically inside `aggregate_fit` here — the coordinator stores gradients *after* `aggregate_fit` returns (`coordinator.py:282-286`), so evicting inside `aggregate_fit` could drop the just-aggregated round's gradients before they are recorded.

### COMMIT

- [ ] Commit.

```bash
cd /home/anurag/codebase/FedLearn-Platform && git add framework/src/fedlearn/server/decomfl_strategy.py framework/tests/test_decomfl_strategy.py && git commit -m "feat(framework): bound DeComFL seed/gradient history

seed_history and gradient_history grew per-round forever. Add evict_old_history:
retains every round >= min(client_last_round), capped at a configurable
max_retained_rounds. Does not break in-window rebuild (T5). Auto-resync for
clients absent beyond the window is deferred to the C1 reliability item."
```

---

## Task 8: B-1 — fix the stale `seed_history` test; full suite green

**Why last:** This is the final red test from the Task-0 baseline. `seed_history` is a `Dict[int, List[List[int]]]` keyed by round (`decomfl_strategy.py:66`), but `test_aggregate_fit_updates_global_params` (`tests/test_decomfl_strategy.py:63-77`) treats it as a list and calls `.append(seeds)`. Update the test to the round-keyed API.

### RED — confirm the stale test still fails

- [ ] Run the stale test.

```bash
cd /home/anurag/codebase/FedLearn-Platform/framework && pytest tests/test_decomfl_strategy.py::TestDeComFLStrategy::test_aggregate_fit_updates_global_params -v
```

**Expected FAIL:** `AttributeError: 'dict' object has no attribute 'append'` at the `self.strategy.seed_history.append(seeds)` line.

### GREEN — use the round-keyed API

- [ ] In `framework/tests/test_decomfl_strategy.py`, fix `test_aggregate_fit_updates_global_params` (lines 63-77).

Replace:
```python
    def test_aggregate_fit_updates_global_params(self):
        # Populate seed history so aggregate_fit can look up seeds
        seeds = self.strategy.generate_seeds(round_idx=0)
        self.strategy.seed_history.append(seeds)

        # Build fake gradient scalars: shape [K][P] = [2][3]
        grads = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        results = [
            ("c1", grads, 100),
            ("c2", grads, 100),
        ]
        result = self.strategy.aggregate_fit(server_round=0, results=results)
        assert result is not None
        assert isinstance(result, OrderedDict)
        assert "fc.weight" in result
```
with:
```python
    def test_aggregate_fit_updates_global_params(self):
        # seed_history is a dict keyed by round; use the public accessor that
        # generates and caches seeds for the round aggregate_fit will look up.
        self.strategy.get_or_generate_seeds(round_idx=0)

        # Build fake gradient scalars: shape [K][P] = [2][3]
        grads = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        results = [
            ("c1", grads, 100),
            ("c2", grads, 100),
        ]
        result = self.strategy.aggregate_fit(server_round=0, results=results)
        assert result is not None
        assert isinstance(result, OrderedDict)
        assert "fc.weight" in result
```

- [ ] Rerun the fixed test, then the entire framework suite.

```bash
cd /home/anurag/codebase/FedLearn-Platform/framework && pytest tests/test_decomfl_strategy.py::TestDeComFLStrategy::test_aggregate_fit_updates_global_params -v && pytest
```

**Expected PASS:** the fixed test passes, and the **full suite is green** — all previously-red tests (3 serializer + this one) plus the new T1–T5 and B-2 tests pass; CUDA/MPS parity tests report `SKIPPED`.

### COMMIT

- [ ] Commit.

```bash
cd /home/anurag/codebase/FedLearn-Platform && git add framework/tests/test_decomfl_strategy.py && git commit -m "test(framework): fix stale seed_history list-vs-dict assumption

test_aggregate_fit_updates_global_params appended to seed_history as if it
were a list, but it is a Dict keyed by round. Use get_or_generate_seeds to
populate the round entry aggregate_fit looks up."
```

---

## Self-review checklist

Run from `/home/anurag/codebase/FedLearn-Platform`. All must hold before declaring done.

- [ ] **Full suite green, GPU-free:** `cd framework && pytest` — every test passes; CUDA/MPS parity tests `SKIPPED`, not failed. No `KeyError`, no `AttributeError`.
- [ ] **The three bug-pinning tests pass:** T1 `TestRebuildTrajectoryEquivalence`, T2 `TestGoldenVectors` + `TestServerClientPerturbationAgree`, T3 `TestChunkedRoundtrip` (multi-chunk + transformer-shaped).
- [ ] **Bug 1 surgical:** `git diff` shows the deleted `* self.P` in `aggregate_fit`'s update step plus the C-1 hoist; the client `decomfl_client.py:208` (`(eta / P) * delta`) is **untouched**.
- [ ] **Bug 2 single source of truth:** `grep -rn "torch.Generator(device=" framework/src` returns only `framework/src/fedlearn/estimators/perturbation.py` (CPU). Neither `decomfl_strategy.py` nor `zeroth_order.py` constructs a device-bound generator anymore.
- [ ] **Bug 3 symmetric:** `grep -n "torch.save" framework/src/fedlearn/communication/serializer.py` shows the wrapped `{'parameters': ..., 'num_examples': ...}` save; `chunks_to_parameters` (line ~155) reads the same two keys.
- [ ] **No global RNG mutation:** `grep -n "np.random.seed\|torch.manual_seed" framework/src/fedlearn/server/decomfl_strategy.py` returns nothing.
- [ ] **Golden fixtures committed with version record:** `framework/tests/fixtures/decomfl_golden/manifest.json` exists, records `torch_version` (`2.12.0+cu130`) and `numpy_version` (`2.1.2`), lists 3 cases each with a `sha256`; the 3 `.npy` files are committed.
- [ ] **Docs corrected:** `grep -n "cancels in derivation" docs/wikis/framework/06_decomfl.md` returns nothing (both spots fixed).
- [ ] **History bounded:** `DeComFL` accepts `max_retained_rounds` and exposes `evict_old_history`; T5 proves the dicts stay bounded and in-window rebuild still matches.
- [ ] **Lint clean:** `cd framework && ruff check src tests` passes (the repo gates `framework/` with ruff + mypy via pre-commit). The new `perturbation.py` is fully typed for mypy `strict`.
- [ ] **No AI attribution** in any commit message or doc edit (repo policy).
- [ ] Eight focused commits exist (Task 0 has no commit), each a red→green cycle.

---

## What this plan deliberately does NOT touch (deferred per spec §9)

- **C++ mobile `ZerothOrderEstimator.cpp`** — the P3 mobile-lift item. It is *contract-gated* by T2's golden fixtures (the C++ port must later pass them), but no C++ is changed here.
- **Mobile mono-vs-poly repo decision** — the B7 monorepo/CI (Continuous Integration) brainstorm. This plan only authors the language-neutral RNG contract (helper + golden vectors + pinned versions) so either topology works later.
- **Checkpoint/resume + resync-after-long-absence** — the C1 reliability item. `evict_old_history` is delivered, but its coordinator wiring and the long-absence resync path are left to C1. Clients absent beyond `max_retained_rounds` are explicitly not handled here.
- **The false "Byzantine-robust" README/docs claim, DP (Differential Privacy) / robust aggregation** — the B4/B1 robustness item.
- **Full per-run determinism manifest** (seed/hyperparam/model/dataset hashes) — the C3 reproducibility item. This plan authors only the RNG-contract slice.
- **`async_coordinator.py` dead RabbitMQ code** — B3.
- **`framework/pyproject.toml` missing `[project]` dependencies table** — flagged in Task 0, but adding a dependency table is out of scope for this correctness plan.
- **A `framework/Makefile` `refreeze-golden` target** — spec §8 mentions it; this plan ships the equivalent `python tests/fixtures/decomfl_golden/generate.py` invocation and documents it in the golden test's drift message. Adding the Makefile target is optional polish, not required for green.
