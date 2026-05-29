# DeComFL Correctness — v2 Spec

**Date:** 2026-05-29
**Branch:** `main-clean`
**Queue item:** P0 — "DeComFL correctness trifecta" (from `docs/audit/2026-05-29/README.md` §5)
**Status:** Design — awaiting user review before writing-plans

---

## 1. Goal

DeComFL is the platform's only paper-backed differentiator, and it is currently
**broken on its live path in three independent ways**. This spec fixes those
three bugs, the small adjacent defects that block the tests from going green,
and the two correctness-adjacent cleanups the user opted into — and pins the
fixes with a test suite that becomes the acceptance contract.

The work is **correctness-first and test-driven**: the failing/new tests are
written first and define "done."

## 2. Locked decisions

| Decision | Value |
|---|---|
| RNG determinism | **Approach A — CPU-canonical torch RNG + frozen golden vectors.** Generate every perturbation on CPU with a local `torch.Generator`, then `.to(device)`. Bit-exact across Python devices immediately; the shared ATen backend gives best-case Python↔C++ parity for a pinned libtorch version. A custom counter-based PRNG (Approach B) was rejected: it cannot beat A cross-language (platform `libm` differs) and adds hand-rolled-RNG risk. |
| Mobile repo (mono vs poly) | **Deferred** to the monorepo/CI brainstorm (audit item B7). This spec only commits to authoring the determinism contract (proto + golden vectors + pinned torch version) as a **language-neutral source of truth** so either topology works later. |
| C++ mobile RNG change | **Deferred** to the P3 mobile-lift item, but **contract-gated now**: the golden vectors frozen here are the conformance test the C++ port must pass. |
| Scope boundary | **Trifecta + blocking fixes + correctness cleanups** (user selection). |
| Methodology | TDD (`superpowers:test-driven-development`). The 5 tests below are written first. |

## 3. The three correctness bugs

### Bug 1 — Server update drops the `1/P` averaging factor
- **Where:** `framework/src/fedlearn/server/decomfl_strategy.py:197,200`.
- **Root cause:** line 197 computes `delta = delta / (num_clients * self.P)`, then line
  200 multiplies it back out: `x_current = x_current - self.eta * delta * self.P`.
  The `* self.P` cancels the `1/P`, so the net step is
  `x ← x − η·(1/N)·Σ_p g·z` — **P× (10× at the `P=10` default) too large**.
- **Why it's worse than a tuning quirk:** the client at `decomfl_client.py:208`
  correctly applies `(eta / P) * delta`, and the rebuild path replays the
  `(1/P)` step. So the server's `global_params_flat` and every client's
  rebuild trajectory **diverge by construction whenever P>1** — the seed/
  gradient-history replay that is the paper's central correctness guarantee is
  violated. (Reference `cezo_fl` averages over P: `random_gradient_estimator.py:176`
  `return grad.div_(self.num_pert)`.)
- **Fix:** drop the `* self.P`:
  `x_current = x_current - self.eta * delta` (delta is already `/(N·P)`).
  Remove the misleading inline comment, and correct the rationalising note in
  `docs/wikis/framework/06_decomfl.md` (the "×P cancels in derivation" line).

### Bug 2 — Perturbation RNG is device-dependent → heterogeneous fleets corrupt silently
- **Where:** server `decomfl_strategy.py:77` (`self.device = 'cuda' if … else 'cpu'`)
  + `:210-219` (`_generate_perturbation`); client
  `framework/src/fedlearn/estimators/zeroth_order.py:45-48` (`generate_perturbation`).
- **Root cause:** both generate `z` via `torch.Generator(device=self.device)` +
  `torch.randn(..., device=self.device)`. Seeded `torch.randn` is **not
  bit-identical across CPU/CUDA/MPS**. A CUDA server reconstructs a different
  `z` than a CPU/MPS client for the same seed → `Σ g·z` aggregation is garbage
  on any GPU server or mixed-device fleet. The two RNG copies having drifted is
  itself the latent hazard.
- **Fix (Approach A):** introduce **one shared perturbation helper** used by
  both server and client — `framework/src/fedlearn/estimators/perturbation.py`:
  ```python
  def canonical_perturbation(seed: int, num_params: int,
                             dtype: torch.dtype = torch.float32) -> torch.Tensor:
      """Device-independent N(0, I_d). Generated on CPU for bit-stable output."""
      g = torch.Generator(device="cpu")
      g.manual_seed(seed)
      return torch.randn(num_params, generator=g, dtype=dtype, device="cpu")
  ```
  Callers move to the working device at the use site (`z.to(self.device)`).
  `decomfl_strategy._generate_perturbation` and
  `zeroth_order.ZerothOrderEstimator.generate_perturbation` both delegate to it,
  eliminating the duplicate RNG logic that allowed the drift.
- **dtype note:** fix the perturbation dtype explicitly (`float32`) so it does
  not silently follow a model's dtype and break golden-vector parity.

### Bug 3 — Serializer save/load asymmetry → `KeyError` on every chunked/LLM upload
- **Where:** `framework/src/fedlearn/communication/serializer.py:97` (save) vs `:155` (load).
- **Root cause:** `parameters_to_chunks` does `torch.save(params, buffer)` — a
  **bare** state-dict `OrderedDict` — but `chunks_to_parameters` returns
  `model_data['parameters'], model_data['num_examples']`, expecting a **wrapped**
  dict. Any model without a tensor literally named `parameters` → `KeyError`.
  Every model >`CHUNK_SIZE` (i.e. every transformer/LLM, the path DeComFL
  exists for) takes the chunked path. 3 framework tests already fail here.
- **Fix:** make save symmetric with the load (and with the unary
  `parameters_to_proto` contract):
  `torch.save({'parameters': params, 'num_examples': num_examples}, buffer)`.
  `num_examples` then lives in the blob; the per-chunk `num_examples` field
  becomes redundant metadata (kept for early-read, not authoritative).

## 4. Blocking / hygiene fixes (needed for green tests)

- **B-1 — `seed_history` stale test:** a test appends to `seed_history` as if it
  were a list, but it is `Dict[int, List[List[int]]]` keyed by round
  (`decomfl_strategy.py:66`). Update the test to the round-keyed API.
- **B-2 — Global RNG mutation (audit M5):** `decomfl_strategy.py:82-83` calls
  `np.random.seed(seed)` + `torch.manual_seed(seed)`, mutating **process-global**
  RNG state (corrupts reproducibility for anything else in-process). Replace
  with local `torch.Generator` / `np.random.Generator` instances held on the
  strategy; nothing reads global RNG after the fix.

## 5. Correctness-adjacent cleanups (user opted in)

- **C-1 — O(KP·N) → O(KP) aggregation loop:** `aggregate_fit` regenerates the
  d-dim `z` once per `(client, k, p)`, but `z` depends only on `(k, p)`. Hoist:
  for each `(k, p)` generate `z` once, sum `g` over clients, then
  `delta += (Σ_c g_c) · z`. **Numerically identical** to the corrected naive
  loop (asserted by T4); N× fewer `randn`+`mul` ops over the d-dim vector.
- **C-2 — Bounded history eviction:** `seed_history` / `gradient_history` grow
  per-round forever. After each round, evict entries with
  `round < min(client_last_round.values())` (the oldest round any known client
  could still need to rebuild from), behind a configurable
  `max_retained_rounds` cap. Clients absent longer than the window must resync
  from a checkpoint — **out of scope here, owned by the C1 reliability item**;
  this spec just must not break the existing rebuild path within the window.

## 6. Determinism contract (the language-neutral artifact)

So the mobile repo decision stays open and the C++ port is gated later:

- **Golden-vector fixtures** at `framework/tests/fixtures/decomfl_golden/` —
  a small, version-controlled set of `(seed, num_params, dtype) → z` cases
  (store `z` plus a `sha256`), generated from `canonical_perturbation`. Format
  is JSON + `.npy` (language-neutral; loadable from Python now and C++ later).
- **Pinned-version record:** the fixture file records `torch_version` (and
  numpy version). Golden vectors are **only re-frozen on an intentional torch
  bump**, via a documented `make refreeze-golden` step; CI fails if generated
  vectors drift from the committed fixture without a version bump. (The broader
  per-run determinism manifest — seed/hyperparams/model/dataset hashes — is
  owned by the C3 reproducibility item; this spec authors only the RNG-contract
  slice.)

## 7. Test plan (TDD — these are the acceptance criteria)

Written first; all must pass; the 3 currently-red tests must flip green.

| ID | Pins | Test |
|---|---|---|
| **T1** | Bug 1 | **Rebuild-trajectory equivalence.** A client that trains every round must end bit-close (`torch.allclose`, tight atol) to a client that misses every round and reconstructs via `rebuild_model`. Fails today; passes after the `1/P` fix. |
| **T2** | Bug 2 | **Golden-vector + cross-device parity.** `canonical_perturbation` matches the committed golden fixture (bit-exact); and on a host with MPS/CUDA, server-path and client-path `z` for the same seed are identical. |
| **T3** | Bug 3 | **Streaming-chunk roundtrip.** `params → parameters_to_chunks → reassemble → chunks_to_parameters → params` for (a) a small model, (b) a model larger than `CHUNK_SIZE` (multi-chunk), (c) a transformer-shaped dict-input state-dict. Flips the 3 red tests. |
| **T4** | Cleanup C-1 | **Optimized ≡ naive aggregate.** The hoisted O(KP) `aggregate_fit` produces a model `allclose` to a reference naive O(KPN) loop that **also includes the Bug-1 `1/P` fix** (i.e. equivalence is asserted against the *corrected* naive form, not the current buggy one), on the same inputs. |
| **T5** | Cleanup C-2 | **Bounded-history rebuild.** A client missing N rounds (N ≤ `max_retained_rounds`) reconnects and rebuilds correctly; history size stays bounded across many rounds. |

Run via `cd framework && pytest`. Determinism tests must run GPU-free in CI
(cross-device assertions guard on `torch.backends.mps.is_available()` /
`torch.cuda.is_available()` and skip when absent, but the golden-vector
bit-exact CPU test always runs).

## 8. Files touched

- `framework/src/fedlearn/server/decomfl_strategy.py` — Bug 1, Bug 2 (delegate), B-2, C-1, C-2.
- `framework/src/fedlearn/estimators/zeroth_order.py` — Bug 2 (delegate to shared helper).
- `framework/src/fedlearn/estimators/perturbation.py` — **new** shared `canonical_perturbation`.
- `framework/src/fedlearn/communication/serializer.py` — Bug 3.
- `framework/tests/…` — T1–T5 + fix the stale `seed_history` test (B-1).
- `framework/tests/fixtures/decomfl_golden/` — **new** golden fixtures + version record.
- `docs/wikis/framework/06_decomfl.md` — correct the "×P cancels" note.
- `framework/Makefile` (or `pyproject` script) — `refreeze-golden` target.

## 9. Out of scope (deferred to their queue items)

- C++ mobile `ZerothOrderEstimator.cpp` change (P3 mobile lift — contract-gated by T2's fixtures).
- Mobile mono-vs-poly repo decision (B7 monorepo/CI item).
- Checkpoint/resume + resync-after-long-absence (C1 reliability item).
- The false "Byzantine-robust" README/docs claim, DP/robust aggregation (B4/B1 robustness item).
- Full per-run determinism manifest (C3 reproducibility item).
- `async_coordinator.py` dead RabbitMQ code (B3).

## 10. Risks & notes

- **Cross-language FP parity is tolerance-level, not guaranteed bit-level.**
  Within Python (shared code path, CPU) it is bit-exact. For C++ (later), the
  golden-vector test documents an FP tolerance and validates it does not perturb
  convergence; if a shared pinned libtorch version yields bit-exact CPU `randn`
  (same ATen), tighten to bit-exact.
- **Perf cost of CPU-canonical generation** for large `d` (LLMs): generating
  randn on CPU then copying to GPU adds latency vs on-device generation. This is
  the price of cross-device determinism and is accepted; the O(KP) hoist (C-1)
  offsets it by cutting generations N-fold. Flag for later profiling, not a
  blocker.
- **T1 is the canary:** it currently fails and is the single highest-signal
  guard that Bugs 1 + 2 are both fixed and stay fixed.

## 11. Next step

On user approval of this spec → `superpowers:writing-plans` to produce the
TDD implementation plan (tests-first, one fix per red→green cycle).
