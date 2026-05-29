# DeComFL Correctness — Critical Decisions Record

**Date:** 2026-05-29
**Branch:** `main-clean`
**Companion spec:** [`docs/v2/specs/2026-05-29-decomfl-correctness-design.md`](../specs/2026-05-29-decomfl-correctness-design.md)
**Evidence base:** [`docs/audit/2026-05-29/B1-paper-alignment.md`](../../audit/2026-05-29/B1-paper-alignment.md), [`A3-framework.md`](../../audit/2026-05-29/A3-framework.md), [`C3-reproducibility.md`](../../audit/2026-05-29/C3-reproducibility.md)

---

## How to read this document

This is a set of ADRs (Architecture Decision Records). Each records one decision: the problem that forced it, the alternatives weighed, why one alternative won, and the cost accepted. The records are **ranked by how much is at stake** — the first three are labelled **MOST CRITICAL** because getting them wrong silently breaks the platform's only paper-backed differentiator.

### Abbreviations (full form on first use, per house style)

| Short | Full |
|---|---|
| ADR | Architecture Decision Record |
| FL | Federated Learning |
| DeComFL | Decomposed Federated Learning (the in-repo wiki's expansion, followed here for consistency; the underlying ICLR 2025 paper is titled "Achieving Dimension-Free Communication in Federated Learning via Zeroth-Order Optimization", [arXiv:2405.15861](https://arxiv.org/abs/2405.15861)) |
| ZO | Zeroth-Order (optimization) |
| RNG | Random Number Generator |
| PRNG | Pseudo-Random Number Generator |
| TDD | Test-Driven Development |
| DRY | Don't Repeat Yourself |
| FP | Floating-Point |
| CPU | Central Processing Unit |
| GPU | Graphics Processing Unit |
| CUDA | Compute Unified Device Architecture (NVIDIA GPU compute) |
| MPS | Metal Performance Shaders (Apple GPU backend) |
| ATen | A Tensor library (PyTorch's C++ tensor backend) |
| API | Application Programming Interface |
| CI | Continuous Integration |
| LLM | Large Language Model |
| DP | Differential Privacy |
| SHA-256 | Secure Hash Algorithm, 256-bit |
| FedAvg | Federated Averaging (the baseline FL aggregation strategy) |
| ARM | Advanced RISC Machines (the CPU instruction-set architecture used by Jetson and Apple Silicon) |
| RISC | Reduced Instruction Set Computer |
| PR | Pull Request |
| proto | Protocol Buffers (the gRPC interface-definition format) |
| ulp | Unit in the Last Place (the spacing between adjacent floating-point numbers) |
| JSON | JavaScript Object Notation |

### One-paragraph background (plain language)

DeComFL (Decomposed Federated Learning) is an FL (Federated Learning) algorithm that avoids shipping whole models over the network. Instead of sending gradients, every client and the server each generate the *same* random perturbation vector `z` from a *shared integer seed*, and the client sends back only a handful of **scalar** numbers describing how much the loss changed along `z`. The server reconstructs the model update as a sum of `scalar × z` terms. This is why DeComFL needs roughly 1,000,000× less bandwidth than FedAvg (Federated Averaging) for LLMs (Large Language Models) (see [`docs/wikis/framework/06_decomfl.md`](../../wikis/framework/06_decomfl.md)). The catch: **the whole scheme collapses silently if the server and a client ever reconstruct a different `z` from the same seed, or if the server scales the reconstructed update differently than the client.** Three independent bugs do exactly that today.

---

## Decision index

| # | Decision | Status | Tier |
|---|---|---|---|
| 1 | RNG determinism approach (CPU-canonical torch RNG) | Accepted | **MOST CRITICAL** |
| 2 | The `1/P` fix location (fix the server, not the client) | Accepted | **MOST CRITICAL** |
| 3 | Serializer symmetry (wrap-on-save, not unwrap-on-load) | Accepted | **MOST CRITICAL** |
| 4 | One shared perturbation helper (DRY) vs two parallel fixes | Accepted | High |
| 5 | Defer the C++ mobile RNG change (contract-gate it instead) | Deferred | High |
| 6 | Defer the mobile monorepo-vs-polyrepo topology decision | Deferred | Medium |
| 7 | Scope = trifecta + correctness cleanups (not strict 3 bugs) | Accepted | Medium |
| 8 | TDD with golden vectors as a frozen cross-language contract | Accepted | Medium |
| 9 | Bounded history eviction policy (not unbounded, not checkpoint-now) | Accepted | Medium |

---

## ADR-1 — RNG determinism: CPU-canonical torch RNG + frozen golden vectors  **[MOST CRITICAL]**

**Decision.** Generate every perturbation `z` on the CPU with a local `torch.Generator`, in a fixed `float32` dtype, via one shared helper `canonical_perturbation(seed, num_params, dtype)`, then `.to(device)` at the use site; freeze a set of golden vectors as the cross-language conformance contract.

### Context / the problem

DeComFL's correctness rests on one invariant: **server and every client must regenerate the identical `z` from the same seed.** Today the server generates `z` on `'cuda'` when a GPU is present (`framework/src/fedlearn/server/decomfl_strategy.py:77`, `:210-219`) and each client generates on its own device (`framework/src/fedlearn/estimators/zeroth_order.py:45-48`). Seeded `torch.randn` is **not bit-identical across CPU/CUDA/MPS** — PyTorch documents this explicitly ([reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html); divergence reproduced in [pytorch/pytorch#79496](https://github.com/pytorch/pytorch/issues/79496)). So a GPU server and a CPU client reconstruct *different* `z` for the same seed. The server then reconstructs `scalar × z_server` from a scalar the client measured along `z_client`. There is no error and no crash — the model walks in a near-random direction and **silently does not learn**. This only manifests on heterogeneous hardware, which is exactly the federated setting the platform targets (Mac MPS + Jetson + Docker CPU + GPU server). The bug is masked today only because local smoke tests run server and clients on the same CPU host.

### Options considered

| Option | Pros | Cons |
|---|---|---|
| **A — CPU-canonical torch RNG + golden vectors** | Bit-exact across all Python devices *immediately* (single code path, CPU only). Reuses the shared, audited ATen normal-sampler — no hand-rolled randomness risk. Best-case Python↔C++ parity for a pinned libtorch (same ATen kernels). Near-zero cost: the RNG draw is trivial next to the two forward passes per perturbation. | CPU generation then host→device copy adds latency for very large `d` (LLMs). Cross-*version* / cross-*language* parity is only tolerance-level, not guaranteed bit-level (platform `libm` differs). |
| **B — Custom counter-based PRNG (Philox/Threefry) in Python + C++** | Counter-based generators (Random123 family, used by JAX) are reproducible by construction, independent of any framework RNG and of device. | **Does not actually beat A cross-language:** the integer counter stream is portable, but turning uniforms into Gaussians needs `log`/`sqrt`/`cos` from each platform's `libm`, and those differ in the last bits — so cross-language output is *still* only tolerance-level. Adds a hand-rolled-RNG correctness surface to own and test forever. Changes the actual perturbation values vs the current torch path, so it must be re-validated against the paper (coordinate with paper-alignment) before adoption. A larger lift for no extra cross-language guarantee. |
| **Status quo — on-device generation** | No code change. | Is the bug. Silently corrupts learning on any GPU server or mixed fleet. Non-negotiably wrong. |

### Decision & why this beat the alternatives

Approach A is chosen. The honest crux is that **B's headline selling point — provable cross-language reproducibility — does not survive contact with floating-point reality.** Both A and B can make the *integer* RNG (Random Number Generator) stream portable; neither can make the *Gaussian transform* bit-identical across platforms, because each platform's `libm` rounds `log`/`sqrt`/`cos` differently in the final ulp (Unit in the Last Place). So B buys no cross-language bit-exactness that A doesn't already have, while adding a bespoke PRNG (Pseudo-Random Number Generator) that the team must implement twice (Python + C++) and defend against subtle bias forever. A, by contrast, gives **bit-exact parity across every Python device today** (the immediate live bug, C3-1/B1-C2) by collapsing to a single CPU code path, and reuses PyTorch's already-audited sampler. For the residual cross-language gap that *neither* approach closes, the answer is the same under A or B: pin one libtorch/torch version across all builds and freeze golden vectors as a tolerance-checked conformance contract. Given equal cross-language outcomes, the lower-risk, lower-effort option wins.

### Consequences / trade-offs accepted

- **Accepted cost:** CPU-then-copy latency for large `d`. This is the price of cross-device determinism; the O(K·P) aggregation hoist (ADR-7 / spec C-1) offsets it by cutting the number of generations N-fold. Flagged for later profiling on Jetson/M4, not a blocker (spec §10).
- **Accepted limit:** within Python on CPU, parity is bit-exact; **cross-language and cross-torch-version parity is tolerance-level, not bit-level** — stated openly (spec §10). The golden-vector test documents the tolerance and verifies it does not perturb convergence. If a shared pinned libtorch happens to yield bit-exact CPU `randn`, the contract is tightened to bit-exact.
- `float32` is pinned on the perturbation explicitly so `z` does not silently follow a model's dtype and break golden-vector parity (spec Bug 2).

**Status:** Accepted.

---

## ADR-2 — The `1/P` fix: correct the server, keep paper-correct averaging  **[MOST CRITICAL]**

**Decision.** Delete `* self.P` on `decomfl_strategy.py:200` so the server update is `x_current = x_current - self.eta * delta` (with `delta` already divided by `num_clients * self.P` on line 197). Leave the client and the rebuild path untouched.

### Context / the problem

The server's `aggregate_fit` divides the accumulated update by `num_clients * P` (`decomfl_strategy.py:197`) and then multiplies it back out by `P` (`decomfl_strategy.py:200`). The two `P`s cancel, so the global model takes a step **`P×` too large** — with the `P=10` default, a silent 10× learning-rate inflation on the global model only. This is not a tuning quirk: the client applies the correct `(eta / P) * delta` (`decomfl_client.py:208`) and the rebuild path replays that same `1/P` step. So the server's trajectory and every client's reconstructed trajectory **diverge by construction whenever `P > 1`** — which voids the seed/gradient-history replay that is the paper's central correctness guarantee. The reference implementation averages over `P` (`grad.div_(self.num_pert)`, [DeComFL `random_gradient_estimator.py:176`](https://github.com/ZidongLiu/DeComFL)).

### Options considered

| Option | Pros | Cons |
|---|---|---|
| **Fix the server (delete `* self.P`)** | Restores the paper's `1/P` averaging on the global update. Makes the server consistent with the *already-correct* client and rebuild paths. One-character-class deletion; lowest blast radius. Matches the reference. | None of substance. Requires correcting the misleading "×P cancels in derivation" note in the wiki. |
| **Change the client to match the server** | Also makes server and client agree. | Aligns everything on the **wrong** value: a `P×` inflated step that contradicts the paper and the reference. Requires touching the client *and* the rebuild path *and* the C++ mobile rebuild — a far larger, riskier edit that propagates the bug instead of removing it. |
| **Just lower the learning rate `η`** | No code change to the update logic. | Treats a structural bug as a hyperparameter. Cancels the inflation *only* for a hand-tuned `η` at one fixed `P`; breaks again the moment `P` changes. Does **nothing** for the server-vs-rebuild trajectory divergence — the two paths still compute different points, so late-joining clients still land on a different model. A coincidence, not a fix. |

### Decision & why this beat the alternatives

Fix the server. There is only one self-consistent target: the paper's `x ← x − η·(1/(N·P))·ΣΣ g·z`, which the client and rebuild paths already implement. The server is the lone outlier, so the minimal correct change is to bring the server to where everything else already is. Changing the client would force three coordinated edits (client + Python rebuild + C++ rebuild) to standardise on a value that is provably wrong. Lowering `η` doesn't address the real defect at all — the trajectory divergence between the global update and the rebuild replay survives any `η`, because it is a difference in *formula*, not in *step size*. Fixing the server is both the smallest edit and the only one that makes all four code paths (server, client, Python rebuild, C++ rebuild) compute the identical trajectory.

### Consequences / trade-offs accepted

- The effective global learning rate drops by `P×` (10× at default). Any previously "working" hyperparameters were compensating for the bug and will need re-tuning — but they were never paper-valid. This is correctness restored, not a regression.
- The wiki's rationalising note ("×P cancels in derivation", `docs/wikis/framework/06_decomfl.md`) must be corrected so the bug is not re-introduced as "intent."
- Guarded by test **T1** (participate-vs-rebuild trajectory equivalence), which fails today and is the highest-signal canary that this bug stays fixed.

**Status:** Accepted.

---

## ADR-3 — Serializer: wrap-on-save, not unwrap-on-load  **[MOST CRITICAL]**

**Decision.** Make the chunked save symmetric with the load: `torch.save({'parameters': params, 'num_examples': num_examples}, buffer)` at `serializer.py:97`, so `num_examples` travels inside the blob.

### Context / the problem

The chunked-upload path is asymmetric. The save side writes a **bare** state-dict `OrderedDict` (`torch.save(params, buffer)`, `serializer.py:97`), but the load side reads a **wrapped** dict (`return model_data['parameters'], model_data['num_examples']`, `serializer.py:155`). Reassembly happens at `grpc_servicer.py:213`. The result: every model without a tensor literally named `parameters` raises `KeyError: 'parameters'`. Every model larger than `CHUNK_SIZE` — i.e. **every transformer/LLM, the exact path DeComFL exists to serve** — takes this chunked path, so the LLM federation cannot complete a single round. The audit confirmed three tests already red on this in `framework/tests/test_serializer.py` (`TestChunkedRoundtrip`).

### Options considered

| Option | Pros | Cons |
|---|---|---|
| **Wrap on save** (change line 97) | Restores symmetry with the load side. Aligns the chunked path with the **unary** `parameters_to_proto` contract, which already carries `num_examples`. `num_examples` lives in the authoritative blob, so the metadata travels with the data it describes. Flips the three red tests green with one edit on the producer side. | The per-chunk `num_examples` field becomes redundant (kept as early-read metadata, not authoritative). |
| **Unwrap on load** (change line 155) | Also makes the two sides agree. | Picks the wrong direction: it would make the chunked path diverge from the unary `parameters_to_proto` contract (which wraps), creating *two* different on-wire conventions in one serializer. And `num_examples` would have to be reconstructed from the per-chunk field rather than the blob — fragile, and the existing tests assert `num_examples` comes back **from the blob**. |

### Decision & why this beat the alternatives

Wrap on save. The load side and the unary path already agree on a wrapped contract; the bare save is the one outlier, so making it conform is the minimal change that produces **one** consistent serialization convention across both the unary and chunked paths. It also keeps `num_examples` bundled with the parameters it belongs to, which is what the existing `TestChunkedRoundtrip` tests assert — so the fix turns red tests green without rewriting them. Unwrapping on load would instead create a second, contradictory wire format and force `num_examples` to be recovered from redundant chunk metadata.

### Consequences / trade-offs accepted

- The per-chunk `num_examples` field is now redundant; it is retained for early-read convenience but is no longer authoritative (spec Bug 3).
- Guarded by test **T3** (streaming-chunk roundtrip for small, multi-chunk, and transformer-shaped state-dicts), which flips the three currently-red serializer tests green.

**Status:** Accepted.

---

## ADR-4 — One shared perturbation helper (DRY — Don't Repeat Yourself) vs independently fixing the two copies

**Decision.** Introduce a single `framework/src/fedlearn/estimators/perturbation.py::canonical_perturbation(...)`; both the server (`decomfl_strategy._generate_perturbation`) and the client (`zeroth_order.ZerothOrderEstimator.generate_perturbation`) delegate to it.

### Context / the problem

The device-dependent RNG bug (ADR-1) exists in **two** places: `decomfl_strategy.py:210-219` and `zeroth_order.py:45-48`. Both implement the same logic — `torch.Generator(device)` + `torch.randn(..., device)` — independently. That duplication is *itself* the latent hazard: the two copies already drifted (different default devices), and the perturbation contract is the one thing server and client must agree on byte-for-byte.

### Options considered

| Option | Pros | Cons |
|---|---|---|
| **One shared helper (single source of truth)** | The server-must-equal-client invariant is enforced *structurally* — there is only one implementation, so they cannot disagree. Kills the duplicated logic that allowed the original drift. One place to pin CPU + `float32`. One place the golden vectors validate. | Tiny new module + an import edge from both server and client into `estimators/`. |
| **Fix the two copies independently** | No new module; smaller diff today. | Re-creates the exact condition that caused the bug: two implementations of an invariant that must match. Any future edit to one (dtype, device, generator construction) silently re-introduces divergence. The golden-vector test would have to assert against two code paths instead of one. |

### Decision & why this beat the alternatives

A single source of truth. The bug's root cause is not "the device was wrong in two files" — it is "the same critical invariant was implemented twice and the copies drifted." Fixing both copies leaves that root cause in place; the next person who edits one estimator re-opens the bug. Collapsing both callers onto one helper makes divergence *impossible by construction* and gives the golden-vector contract (ADR-8) exactly one function to certify. This is the standard DRY argument, but here it is load-bearing: the duplicated code is precisely where correctness was lost.

### Consequences / trade-offs accepted

- A new module and an import dependency from both `server/` and `client/`-side estimators into `estimators/perturbation.py`. Acceptable: `estimators/` is already the shared home for ZO (Zeroth-Order) math.
- Guarded by test **T2** (golden-vector + cross-device parity), which now certifies a single function.

**Status:** Accepted.

---

## ADR-5 — Defer the C++ mobile RNG change; contract-gate it now

**Decision.** Do **not** modify the C++ mobile `ZerothOrderEstimator.cpp` in this work. Instead, freeze the golden vectors here as the conformance test the C++ port must later pass.

### Context / the problem

The mobile C++ client asserts in a comment that its `torch::Generator` produces "bit-identical vectors to Python's torch.randn" (B1-H1, C3-2). That claim is **unverified, undocumented as to version, and brittle** — the mobile libtorch build is pinned independently of the server's torch version, and PyTorch's normal-sampler path has changed across releases. The audit could not confirm bit-identity without running both builds and flagged it as uncertain. The C++ change also lives on a different branch (`origin/fed-mobile`) and belongs to a separate P3 mobile-lift workstream.

### Options considered

| Option | Pros | Cons |
|---|---|---|
| **Defer C++, contract-gate with golden vectors** | Keeps this change focused on the live Python path (where the bug actively breaks production). Produces a language-neutral artifact (JSON + `.npy` + SHA-256) the C++ port must pass to ship — so deferral does not mean "untested later." Avoids editing/building an ARM cross-compiled toolchain inside a correctness fix. | The mobile path stays nominally broken until P3 — but it is not in production use today, so no live regression. |
| **Fix C++ now** | Mobile and Python fixed together. | Pulls a separate branch, a cross-compiled libtorch toolchain, and a `gtest` harness into a Python-framework correctness fix — large scope creep. The C++ fix *cannot be validated* without the very golden vectors this work produces, so it would be fixing blind. Bit-exact Python↔C++ parity is unproven (flagged uncertain in B1-H1); committing to it now would be asserting something not yet verified. |

### Decision & why this beat the alternatives

Defer and contract-gate. The C++ change has a hard dependency on an artifact that does not exist yet — the frozen golden vectors. Fixing C++ before those exist would be unverifiable by definition. By authoring the golden vectors as a **language-neutral source of truth** now (loadable from Python today, C++ later), deferral becomes "gated," not "dropped": the C++ port has a precise, committed acceptance test waiting for it, and any drift is a release blocker. This also respects project isolation — the correctness fix stays in `framework/`, and the mobile toolchain work stays in its own P3 item.

### Consequences / trade-offs accepted

- Mobile clients remain on the unverified RNG path until the P3 mobile lift; acceptable because mobile is not a live deployment target today.
- The golden-vector fixture format (JSON + `.npy`, with `torch_version` recorded) is chosen specifically so C++ can consume it later (spec §6).
- Honest caveat (carried from ADR-1): even when C++ is fixed against the fixture, cross-language parity will be tolerance-level unless a shared pinned libtorch yields bit-exact CPU `randn`.

**Status:** Deferred (contract authored now).

---

## ADR-6 — Defer the mobile monorepo-vs-polyrepo decision

**Decision.** Do not decide the mobile repository topology (monorepo vs polyrepo) here; defer it to the separate monorepo/CI tooling brainstorm (audit item B7).

### Context / the problem

Whether the mobile C++ client should live in the same repository as the Python framework (monorepo) or its own (polyrepo) affects how the determinism contract (proto + golden vectors + pinned torch version) is shared and CI-enforced across languages. It is a real decision — but it is a *tooling/topology* decision, not a *correctness* decision.

### Options considered

| Option | Pros | Cons |
|---|---|---|
| **Defer to the B7 tooling brainstorm** | Keeps this correctness work single-purpose. The determinism contract is authored as a **language-neutral artifact** that works under *either* topology, so the decision can be made later without rework. Topology choice needs CI/codegen/release-process inputs that belong to B7, not to a bug fix. | The topology question stays open. |
| **Decide topology now** | Resolves it in one pass. | Couples an irreversible org/tooling decision to a correctness fix, under-informed by the CI and 4-language codegen considerations that B7 owns. High risk of choosing wrong for reasons unrelated to correctness. |

### Decision & why this beat the alternatives

Defer. The correctness work has exactly one obligation toward the topology question: make the determinism contract topology-agnostic. It does — the golden vectors and pinned-version record are language-neutral files, not code in either repo. With that obligation met, forcing the topology decision now would only import unrelated tradeoffs (CI runners, codegen single-source-of-truth, release coupling across four language targets) into a bug fix. Those belong to B7. Deferring costs nothing because no choice made here is invalidated by either later outcome.

### Consequences / trade-offs accepted

- The repo topology remains undecided; this spec commits only to authoring the contract as a neutral source of truth (spec §2 locked decisions).

**Status:** Deferred (to audit item B7).

---

## ADR-7 — Scope = trifecta + blocking fixes + correctness cleanups (not strict 3 bugs)

**Decision.** Scope this work to the three correctness bugs **plus** the two blocking hygiene fixes that unblock the tests **plus** the two correctness-adjacent cleanups the user opted into (C-1 loop hoist, C-2 bounded history) — explicitly *not* a strict three-bugs-only scope, and explicitly *not* the broader v2 rebuild.

### Context / the problem

The audits surfaced a long tail: forward-vs-central ZO difference, the false "Byzantine-robust" claim, dead RabbitMQ code, dependency bloat, no checkpointing, no run-lineage entity, etc. A boundary had to be drawn. Two facts forced expansion beyond "just the 3 bugs": (B-1) a stale test treats the round-keyed `seed_history` dict as a list, and (B-2) the strategy mutates **process-global** RNG via `np.random.seed`/`torch.manual_seed` (`decomfl_strategy.py:82-83`). Neither is one of the three headline bugs, but the test suite cannot go green without B-1, and B-2 actively undermines the determinism the whole fix is about.

### Options considered

| Option | Pros | Cons |
|---|---|---|
| **Trifecta + blocking + cleanups** (chosen) | Ships a *provably green* suite (B-1 is required for that). Removes the global-RNG corruption (B-2) that would otherwise undercut ADR-1. The two cleanups are correctness-adjacent: C-1 is asserted numerically identical to the corrected loop (so it is free correctness-preserving speedup that also offsets ADR-1's CPU-copy cost), and C-2 stops unbounded memory growth on the exact replay path the fix depends on. User explicitly opted in. | Larger diff than three lines; the cleanups carry their own (test-guarded) risk. |
| **Strict 3 bugs only** | Smallest possible diff. | Cannot produce a green suite — B-1 leaves a red test, so "done" is unprovable. Leaves the global-RNG mutation (B-2) live, partially re-opening the determinism problem ADR-1 closes. Leaves the O(K·P·N) loop that makes ADR-1's CPU generation needlessly N× more expensive. A technically-narrower scope that fails the acceptance bar. |
| **Full v2 rebuild** | Addresses the entire audit tail. | Massively out of scope; weeks of work; conflates correctness with architecture. Violates minimal-surgical-change discipline. |

### Decision & why this beat the alternatives

The chosen scope is the smallest scope that can *prove* it is done. "Three bugs only" sounds tighter but is a false economy: B-1 must be fixed or the suite stays red and "done" is an assertion rather than a demonstrated fact; B-2 must be fixed or the determinism guarantee from ADR-1 is partially re-opened by process-global RNG mutation. The two cleanups are admitted because they are *correctness-adjacent and test-pinned*: C-1 is proven `allclose` to the corrected naive loop (T4), and C-2 keeps the replay path the fix relies on from growing without bound (T5). Everything genuinely orthogonal — central ZO, Byzantine claim, checkpointing, lineage, dep hygiene, dead RabbitMQ — is explicitly pushed to its owning queue item (spec §9), holding the line against scope creep.

### Consequences / trade-offs accepted

- A broader diff and two extra test cases (T4, T5) to maintain.
- Deliberately deferred, owned elsewhere (spec §9): C++ mobile change (P3), mobile repo topology (B7), checkpoint/resume + long-absence resync (C1 reliability), the false Byzantine-robust claim and DP/robust aggregation (B4/B1 robustness), full per-run determinism manifest (C3), dead `async_coordinator.py` RabbitMQ code (B3).

**Status:** Accepted.

---

## ADR-8 — TDD with golden vectors as a frozen cross-language contract

**Decision.** Write the five tests (T1–T5) first as the acceptance contract, and freeze the perturbation golden vectors (`(seed, num_params, dtype) → z` plus SHA-256, recording `torch_version`) as a version-controlled, language-neutral fixture re-frozen only on an intentional torch bump.

### Context / the problem

The audit found the suite was **red** (3 failures) yet had passed review — direct evidence there was no PR-time test gate (A3-N1). The one property the entire protocol rests on — server `z` equals client `z` for a given seed — had **zero tests** (B1-Low). An ad-hoc "fix and eyeball it" approach has no way to prove the fix is correct, no way to prevent regression, and no way to gate the future C++ port.

### Options considered

| Option | Pros | Cons |
|---|---|---|
| **TDD + frozen golden vectors** | Tests *are* the acceptance criteria — "done" is demonstrable, not asserted. T1 catches Bug 1 and fails today (a true canary). Golden vectors turn "bit-identical" from a *comment* (which is how the C++ claim was wrong) into a *CI-enforced fact*. The fixture is the conformance gate for the deferred C++ port (ADR-5). Version-pinning + `make refreeze-golden` makes an unintentional RNG drift a CI failure. | Up-front cost of writing tests and generating/committing fixtures before the fix. |
| **Ad-hoc fix, verify by inspection** | Faster to first edit. | No regression guard — the next refactor silently re-breaks the invariant (which is *exactly* how the duplicated RNG drifted and how the suite went red unnoticed). No artifact to gate the C++ port. "Done" is an opinion. |

### Decision & why this beat the alternatives

TDD with frozen golden vectors. The failure mode this whole effort exists to prevent is **silent** corruption — bugs that produce no error and degrade learning invisibly. The only defense against silent bugs is an executable contract that fails loudly, which is precisely what a test-first suite plus committed golden vectors provides. T1 is the canary (fails today, proves Bugs 1+2 are fixed and stay fixed); the golden vectors convert the determinism invariant from an unverifiable comment into a CI gate and double as the conformance test for the deferred C++ port. An ad-hoc fix would re-create the conditions that let these bugs ship in the first place.

### Consequences / trade-offs accepted

- Up-front test + fixture authoring cost, paid before the implementation edits.
- Golden vectors are re-frozen **only** on an intentional torch version bump, via a documented `make refreeze-golden` step; CI fails if generated vectors drift without a version bump (spec §6).
- Cross-device assertions guard on `torch.backends.mps.is_available()` / `torch.cuda.is_available()` and skip when absent, so the suite runs GPU-free in CI; the bit-exact CPU golden-vector test always runs (spec §7).

**Status:** Accepted.

---

## ADR-9 — Bounded history eviction (not unbounded, not full checkpoint/resume now)

**Decision.** After each round, evict `seed_history` / `gradient_history` entries older than the oldest round any known client could still need to rebuild from (`round < min(client_last_round.values())`), behind a configurable `max_retained_rounds` cap. Full checkpoint/resume and resync-after-long-absence are deferred to the reliability item.

### Context / the problem

`seed_history` and `gradient_history` are round-keyed dicts (`decomfl_strategy.py:66-67`) that grow **forever**: every round adds an entry that is never removed. For a long-running federation this is an unbounded in-memory leak. But the histories cannot simply be capped blindly — they are the data the rebuild path replays for clients that missed rounds, so anything still within a reachable client's window must be retained.

### Options considered

| Option | Pros | Cons |
|---|---|---|
| **Bounded eviction by oldest-needed round + cap** (chosen) | Bounds memory while preserving every entry any currently-known client can still replay (correctness within the window). The `max_retained_rounds` cap puts a hard ceiling on memory regardless of client behavior. Small, local change; numerically transparent to the rebuild path within the window. | A client absent longer than the window cannot rebuild from history alone — it must resync from a checkpoint (out of scope here). |
| **Leave unbounded** | No change; never loses history. | Unbounded memory growth; a long federation eventually exhausts host memory. Already flagged as a scaling defect. |
| **Full checkpoint/resume now** | Solves both memory *and* long-absence resync and server-restart recovery. | Large feature owned by the C1 reliability item; pulls persistence, restart, and resync into a correctness fix — major scope creep. Not needed to fix the three bugs or keep the within-window rebuild correct. |

### Decision & why this beat the alternatives

Bounded eviction. It is the minimal change that fixes the unbounded-growth defect *without* breaking the rebuild guarantee this whole effort protects: by evicting only rounds older than the oldest round any known client could still need, in-window replay stays bit-identical, while memory gains a hard ceiling via the cap. Leaving it unbounded is a known scaling cliff. Full checkpoint/resume is the *right* long-term answer for clients absent beyond the window and for server restarts — but that is a reliability feature with its own persistence design, explicitly owned by the C1 item; importing it here would balloon a correctness fix into a reliability epic. The boundary drawn is exactly: this spec must not break the existing rebuild path within the window, and nothing more.

### Consequences / trade-offs accepted

- Clients absent longer than `max_retained_rounds` (or longer than the oldest-needed window) **cannot** rebuild from history and will need checkpoint-based resync — explicitly out of scope, owned by the C1 reliability item (spec §5 C-2).
- Guarded by test **T5** (a client missing N ≤ `max_retained_rounds` rounds rebuilds correctly; history size stays bounded across many rounds).

**Status:** Accepted (full checkpoint/resume deferred to C1 reliability item).

---

## Summary

The three **MOST CRITICAL** decisions all share one theme: the bugs are *silent*, so every choice optimizes for **provable** correctness over apparent convenience.

- **ADR-1 (RNG):** chose the lower-risk CPU-canonical path because the fancier counter-based PRNG cannot actually beat it cross-language — `libm` differences cap both at tolerance-level — while adding hand-rolled-RNG risk.
- **ADR-2 (`1/P`):** fixed the lone outlier (the server) to converge on the paper-correct value the client and rebuild paths already use, rather than propagating the bug or papering over it with a learning-rate hack.
- **ADR-3 (serializer):** wrapped on save to produce one consistent wire contract across the unary and chunked paths, flipping the red tests without rewriting them.

The supporting decisions (ADR-4 through ADR-9) consistently favor a **single source of truth** (shared helper, golden-vector contract, TDD acceptance suite) and **disciplined scope** (defer the C++ port, the repo topology, and full checkpoint/resume to their owning items), so the correctness fix stays surgical and the determinism guarantee becomes a CI-enforced fact rather than a hopeful comment.
