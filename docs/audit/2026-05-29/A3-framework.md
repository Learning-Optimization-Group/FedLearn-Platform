# A3 — Python FL Framework Audit (v2 Greenfield)

**Target:** `framework/` (Python 3.10+, PyTorch, custom gRPC, package `fedlearn`, contract `fedlearn.v1`)
**Date:** 2026-05-29
**Builds on:** `docs/audit/2026-05-27/03-framework.md` (cited as *PRIOR* below). This report **verifies** the prior findings against the live code, **adds** newly-discovered defects, and **recommends a v2 framework architecture** calibrated to a production-grade startup.

> **Verification stance:** Every codebase claim below is `file:line`. I created a clean venv (`torch 2.12.0`, `grpcio 1.80.0`, `protobuf 4.25.9`), editable-installed the framework, and **ran the test suite**. Results are quoted verbatim where they change a prior verdict.

---

## 0. Headline

Three things matter more than everything else combined:

1. **C1 (PRIOR) is confirmed live and now demonstrably breaks two tests, not one.** The chunked-upload path is unconditionally broken; every transformer/LLM client upload raises `KeyError: 'parameters'`. **The LLM federation — the entire reason parameter chunking exists — cannot complete a single round.**
2. **A second, previously-unreported test failure exists:** `test_aggregate_fit_updates_global_params` throws `AttributeError: 'dict' object has no attribute 'append'`. A `List → Dict` refactor of `seed_history` left a test (and an unguarded indexing assumption) stranded. **This proves the test suite is not run in CI** — a green-on-paper suite is actually red.
3. **The "no `flwr`" invariant is violated inside the framework itself**, not just `client-docker`. `framework/requirements.txt:7-8` pins `flwr==1.20.0` + `flwr-datasets==0.5.0`, and `pip install -e .` propagates them (`src/fedlearn.egg-info/requires.txt:1-2`) — while the same package's `PKG-INFO:126` advertises *"Built from scratch — no Flower / `flwr` dependency."* That is a shipped contradiction.

**Live test run (verbatim):**
```
================== 3 failed, 39 passed, 15 warnings in 0.23s ===================
FAILED tests/test_decomfl_strategy.py::TestDeComFLStrategy::test_aggregate_fit_updates_global_params
FAILED tests/test_serializer.py::TestChunkedRoundtrip::test_chunks_roundtrip_single_chunk
FAILED tests/test_serializer.py::TestChunkedRoundtrip::test_chunks_roundtrip_multiple_chunks
```

The framework is a **competent research POC with correct cryptographic hygiene** (`weights_only=True`, dtype whitelist, shape validation) wrapped around a **broken transport, untested aggregation math, and a dependency surface that contradicts its own charter.** It is salvageable in pieces; the transport/lifecycle layer should be rebuilt.

---

## 1. Verified prior findings (delta vs 2026-05-27)

| PRIOR id | Status now | Evidence (this audit) |
|---|---|---|
| **C1** chunked-upload asymmetry | **CONFIRMED — escalated.** Now 2 failing tests. | `serializer.py:97` `torch.save(params,…)` (bare OrderedDict); `serializer.py:155` reads `model_data['parameters']`; `grpc_client.py:194` same bare save; server `grpc_servicer.py:213` calls `chunks_to_parameters`. Download path *does* wrap (`grpc_servicer.py:91`). Ran: both `test_chunks_roundtrip_*` fail with `KeyError: 'parameters'`. |
| **C3** `compressed` inferred from env | **CONFIRMED.** | `serializer.py:22` reads `FEDLEARN_USE_COMPRESSION`; proto descriptor (`fedlearn_pb2.py:17`) shows `ModelChunk`/`ModelUpdateChunk` carry **no** `compressed` field. `weights_only=True` confirmed at `serializer.py:153` and `grpc_client.py:155` — RCE blocked today, brittleness real. |
| **C4** gRPC plaintext over WAN | **CONFIRMED — partially mitigated.** | TLS *is* wired (`server.py:102-127`, `grpc_client.py:51-72`) behind `FEDLEARN_GRPC_USE_TLS=1`, but default is insecure with only a `logging.warning` (`server.py:127`). No profile-gated *refuse-to-boot*. |
| **H1** heartbeat death invisible | **CONFIRMED — worse than reported.** | `_heartbeat_loop` swallows all exceptions (`grpc_client.py:282`); `heartbeat_active` untouched on failure. **New:** server-side `should_stop` is hard-coded `False` (`coordinator.py:165`) and `is_client_alive` (`coordinator.py:191`) is **never called** anywhere — the liveness signal exists but is dead on both ends. |
| **H2** `MAX_SAMPLES=100_000` silent cap | **CONFIRMED.** | `strategy.py:81` and `coordinator.py:55,75`. Cap applies with **no WARN** when it fires (`strategy.py:122`, `coordinator.py:75`). |
| **H4** DeComFL aggregation is Python-loop O(K·P·N) | **CONFIRMED — two hot loops, not one.** | Line numbers drifted (file is now 255 lines). The scalar averaging loop is `coordinator.py:327-335`. The **far more expensive** loop is `decomfl_strategy.py:180-200`: it regenerates a **full model-dimension** `torch.randn` perturbation (`_generate_perturbation`, line 210) **inside `for k: for client: for p:`** → `K·N_clients·P` allocations of a length-`d` vector. For a 7B model that is `d≈7e9` floats per inner iteration. |
| **H5** streaming upload unbounded | **CONFIRMED.** | `grpc_servicer.py:177-203` writes chunks to `BytesIO` with no cumulative-size guard; `max_receive_message_length = 1GB` per chunk (`server.py:91`). |
| **H6** `flwr` runtime dep | **CONFIRMED IN FRAMEWORK (new scope).** | Prior located it only in `client-docker`. It is also in **`framework/requirements.txt:7-8`** and the installed metadata `src/fedlearn.egg-info/requires.txt:1-2`. No `import flwr` in `framework/src` (grep clean), so it is a **dead but shipped** dep here. |
| **M2** unpack-None on `evaluate()==None` | **CONFIRMED — two sites.** | `coordinator.py:107` and `:290` both do `loss, metrics = self.strategy.evaluate(...)`. `FedAvg.evaluate` / `DeComFL.evaluate` return `None` when `evaluate_fn is None` (`strategy.py:68-69`, `decomfl_strategy.py:248-249`). The guard above only checks `aggregated_parameters is not None`. **A run with no eval fn `TypeError`s on round completion.** |
| **M3** `datetime.utcnow()` deprecated | **CONFIRMED.** | `server.py:23`; emits `DeprecationWarning` on every test run (captured in suite output). |
| **M4** `GetGlobalModelStream` 2× peak memory | **CONFIRMED.** | `grpc_servicer.py:93` `buffer.getvalue()` copies; `:111` `data_to_send[start:end]` copies per chunk. Client download side was *fixed* (`grpc_client.py:132-156` streams into one `BytesIO`), but the **upload generator** still slices `view[i:i+chunk].tobytes()` (`grpc_client.py:207`) — acceptable (per-chunk copy only). Server **download** path is the remaining 2× offender. |
| **M5** global numpy RNG seed | **CONFIRMED.** | `decomfl_strategy.py:82` `np.random.seed(seed)` + `:83` `torch.manual_seed(seed)` — process-global; two DeComFL servers in one process clobber each other. (Note: backend spawns one process per project, so impact is bounded *today* — but it blocks any in-process multi-tenant server in v2.) |
| **M6** test-coverage gaps | **CONFIRMED — and the suite is RED.** | See §2. No tests for `grpc_client`, `grpc_servicer` streaming, `server` round-trip, `decomfl_client.fit`. |

**Refuted / softened from prior:**
- **PRIOR M4 client-download "3× memory"** is already fixed in code (`grpc_client.py:128-156`); the comment there documents the fix. The remaining peak-memory bug is the **server download** (`grpc_servicer.py:93,111`), which the prior report also flagged — so net: server side outstanding, client side resolved.

---

## 2. New findings (not in the 2026-05-27 report)

### N1 — (HIGH) Second red test reveals the suite never runs in CI; `seed_history` refactor stranded a test and an indexing assumption
`test_decomfl_strategy.py:66` calls `self.strategy.seed_history.append(seeds)`, but `seed_history` was changed from `List` to `Dict[int, …]` (`decomfl_strategy.py:66`). Ran it:
```
tests/test_decomfl_strategy.py:66: AttributeError: 'dict' object has no attribute 'append'
```
Beyond the stale test: `aggregate_fit` indexes `self.seed_history[server_round]` directly (`decomfl_strategy.py:188`). If a round's seeds were not first materialised via `get_or_generate_seeds` (`:113`), aggregation `KeyError`s. The happy path works only because `GetDeComFLConfig` (`grpc_servicer.py:296`) populates the cache first — i.e. **aggregation correctness depends on RPC ordering**, an implicit, untested coupling. **A 3-failure suite passing review is direct evidence there is no PR-time test gate** (corroborates B7's CI gap).

### N2 — (MEDIUM) `PKG-INFO` advertises "no Flower dependency" while `requires.txt` pins two Flower packages
`src/fedlearn.egg-info/PKG-INFO:126` ↔ `src/fedlearn.egg-info/requires.txt:1-2`. Self-contradicting shipped metadata. The README claim is *true of the source* (no `import flwr` in `framework/src`), but **`pip install -e framework` still drags in `flwr` + `flwr-datasets`** — and these are heavy, transitively pull Ray, and have their own license/CVE surface (hand to **C4** for IP/license and **B4** for CVE).

### N3 — (MEDIUM) Dependency surface is enormous and largely dead for a "thin custom framework"
`requirements.txt` pins `ray`, the full `opentelemetry-*` stack, `opencensus`, `google-api-core`/`google-auth`/`googleapis-common-protos`, `matplotlib`/`seaborn`, `nvidia-ml-py`/`pynvml`, `aiohttp(-cors)`. Grep shows **no `import ray` and no `import pika` in `src/`**. The OTel/Prometheus deps are pinned but the telemetry pipeline is empty (corroborates **B3**: dead observability deps). This bloats every client image and the Jetson ARM64 wheel-resolution surface (corroborates **A4**). A custom FL core should depend on roughly `torch`, `numpy`, `grpcio`, `protobuf`, `lz4` — single digits, not ~40 pins.

### N4 — (MEDIUM) `async_coordinator.py` is dead, half-wired RabbitMQ code that imports a package not in requirements
`async_coordinator.py:1-40` defines `get_rabbitmq_parameters()` using `pika.PlainCredentials(...)` while `import pika` is **commented out** (lines 4, 10) and `pika` is **absent from `requirements.txt`** → guaranteed `NameError` if the alternate coordinator is ever instantiated. It re-declares a *second* `class FLCoordinator` (shadowing the real one) and is referenced only by commented imports (`server.py:9-10`, `client.py:3,11`). Dead, misleading, and a maintenance trap. **Kill it.**

### N5 — (LOW/observability) CLAUDE.md "300MB chunking threshold" does not match code
The platform invariant states chunking triggers for "models over 300MB." Actual triggers: `STREAMING_THRESHOLD_MB = 100` **OR** `ALWAYS_STREAM_TRANSFORMERS = True` (`grpc_client.py:18-19`). The doc figure is stale; worth reconciling because it changes when the (currently broken) chunk path is exercised — *any* transformer hits it regardless of size.

### N6 — (LOW) `update_status()` mutates shared client state without a lock
`grpc_client.py:299-302` writes `current_status/step/round` from the training thread; `send_heartbeat` (`:257-264`) reads them from the heartbeat thread. CPython's GIL makes individual field writes atomic, so this is *benign today*, but it is an unsynchronised cross-thread read that a v2 (e.g. free-threaded 3.13t, or asyncio) must not inherit.

### N7 — (LOW) `is_transformer` keyword sniffing is a foot-gun
`grpc_client.py:243-247` decides streaming by substring-matching layer names against `['transformer','bert','gpt','opt','attention','encoder','decoder']`. A ResNet with an "encoder" block, or a custom LLM whose params lack these tokens, mis-routes. Decision should be size-based (and explicit in config), not name-sniffed.

---

## 3. DeComFL correctness (paper-alignment view — hand-off to B1)

I traced the math end-to-end and cross-checked the distributed strategy against the lab's **centralized reference** (`examples/ecg_decomfl_central/run_server.py`), which is the artifact used to validate convergence.

- **ZO estimator** (`estimators/zeroth_order.py:105`): `g = (f(x+μz) − f(x))/μ`, forward-difference, two forward passes under `no_grad`. Matches paper Eq.1 / Alg.4 L18. **Correct.** Cost: **2 full forward passes per perturbation** → `2·K·P` forwards per client per round. For a 7B model on Jetson this is the real wall-clock bottleneck, not communication.
- **Client `fit`** (`decomfl_client.py:208-218`): local step `x −= (η/P)·Σ_p g·z`; after K steps it **exactly reverts** (`:218`) so the client holds the same `x` it started with and ships only scalars. Revert-by-arithmetic is clean.
- **Server `aggregate_fit`** (`decomfl_strategy.py:197-200`): `delta /= (num_clients·P)` then `x −= η·delta·P`. The `·P` and `/P` **cancel**, so the server effectively does `x −= η·(1/num_clients)·Σ_clients Σ_p g·z`. **This is byte-for-byte identical to the centralized reference** (`run_server.py:96-100`), so the distributed port is *faithful to the lab's own reference*.

> **Uncertainty I will not paper over:** the **client local step divides by P** (`η/P`) while the **server global step does not** (the P cancels). Whether that asymmetry is intended by the DeComFL paper (Alg.2/3 vs Alg.4) or is a shared bug carried from the reference into the distributed code, I **cannot determine without the paper's exact update rule**. It is *internally consistent* (matches the reference), so it does not look like a porting error — but it is exactly the kind of factor-of-P that silently changes the effective learning rate. **Explicit B1 work item: verify η/P vs η against Alg.2/3.** I flag, I do not assert.

- **`rebuild_model`** (`decomfl_client.py:71-118`): replays missed rounds with `x −= (η/P)·Σ_p g·z` using the **client's** `/P` convention — which **diverges from the server's effective no-`/P` global update**. A client that misses rounds and rebuilds may land on a *different* point than a client that stayed online. **B1 must check whether rebuild and the global update use the same scaling.** (Higher-confidence concern than the above, because rebuild and aggregate are supposed to reconstruct the *same* trajectory.)
- **Seed determinism:** seeds are int32 from the global numpy RNG (`decomfl_strategy.py:107`), perturbations regenerated via per-call `torch.Generator(seed)` (`:210`, `zeroth_order.py:45`). Reproducible *if* device is fixed — but `_generate_perturbation` uses `device='cuda'` on server and clients may be CPU; `torch.randn` with a seeded generator is **not guaranteed bit-identical across CPU/CUDA**. **Cross-device reproducibility is unverified** (hand to **C3**).

---

## 4. Memory & performance bounds (M4 Max / Jetson Orin)

- **Chunked upload, if fixed:** `torch.save(state_dict)` materialises the full serialized blob in a `BytesIO` before chunking (`serializer.py:96-97`, `grpc_client.py:193-194`). For a 14 GB LLaMA-7B state dict that is **~14 GB peak in addition to the in-memory model** → ~28 GB, against the M4 Max 36 GB unified ceiling and far over Jetson Orin. The chunking saves *wire* memory, not *host* memory. v2 must stream `torch.save` incrementally (e.g. `safetensors` zero-copy mmap, or tensor-by-tensor framing) — see §6.
- **Server download (M4, confirmed):** `grpc_servicer.py:93` `getvalue()` + `:111` per-chunk slice → ~2× model in RAM on the server while streaming to each client. With N clients pulling concurrently the server holds 1 copy (good, shared `params`) but 2× transiently per active stream serialization. Use `memoryview` slices.
- **DeComFL aggregation (confirmed hot):** `decomfl_strategy.py:180-200` allocates a length-`d` `torch.randn` **inside the triple loop**. At `K=10, P=20, 50 clients, d=7e9` that is 10·50·20 = 10⁴ allocations of a 28 GB vector. **This is not "dimension-free" on the server** — the paper's O(K·P) communication guarantee says nothing about server *compute/memory*, which is O(K·P·N·d) here. The `·P` averaging can be vectorised: stack scalars into a `(num_clients, P)` tensor and `einsum` against a `(P, d)` perturbation batch, or accumulate per-seed once. **This is the single biggest server-side scaling cliff for large models.** (corroborates and sharpens PRIOR H4.)
- **gRPC worker model:** `server.py:69-72` sizes a `ThreadPoolExecutor(max_workers = 2·MAX_CLIENTS + 10)`. With Python's GIL, large-model (de)serialization on these threads serializes anyway; `proto_to_parameters` (`serializer.py:56-77`) and `torch.load` are CPU-bound and will not parallelize. Thread count is a connection-concurrency knob, **not** a throughput knob.

---

## 5. gRPC contract & streaming RPCs

The proto (`communication/protos/fedlearn.proto`, `package fedlearn.v1`, `option java_package="com.fedlearn.v1"`) is clean and well-structured. Observations:

- **Two upload paths, one contract, asymmetric framing.** `SubmitModelUpdate` (unary, `ModelParameters` proto, tensor-by-tensor) vs `SubmitModelUpdateStream` (`stream ModelUpdateChunk`, opaque `torch.save` blob). The unary path is **safe** (uses `proto_to_parameters` with dtype whitelist + shape validation). The streaming path is **opaque pickle-via-torch.save** and is the C1 break. v2 should **unify on one framing** — ideally the typed `Tensor` framing for *both* (stream `repeated Tensor` instead of an opaque blob), eliminating `torch.save`/`torch.load` from the wire entirely and removing the `weights_only` foot-gun (corroborates C3).
- **`compressed` is not on the wire** (proto descriptor confirms). C3 fix = add `bool compressed = N` to both chunk messages; fail closed on mismatch.
- **No request/stream size negotiation.** No max-chunks, max-bytes, or model-hash field. H5 + integrity: add a `sha256` trailer and a server-enforced `max_payload_bytes`.
- **No protocol version field.** With a 4-language codegen target (Java/Python/TS/C++) and a *known drift* on `fed-mobile` (per 00-DESIGN §3 — malformed `SubmitModelUpdate` there), the contract needs an explicit version and a single source of truth (hand to **B7** for codegen). The Java option `com.fedlearn.v1` differs from the Spring package `com.federated.fl_platform_api.flower` — confirm the backend actually consumes generated stubs or not.

---

## 6. v2 framework architecture recommendation (keep custom)

**Keep the framework custom.** The DeComFL ZO core, the dual-stub heartbeat idea, and chunking are genuine differentiators and the "no `flwr`" charter is a deliberate IP/positioning choice (C4). But the **transport + lifecycle + serialization layer should be rebuilt**, not patched. Calibrated to a startup:

**6.1 Package structure (`src/fedlearn/`):**
```
fedlearn/
  core/            # framework-agnostic: Parameters, RoundContext, Strategy ABC, typed config (pydantic)
  strategies/      # fedavg.py, decomfl.py  (pure math, no I/O, no gRPC — unit-testable in isolation)
  estimators/      # zeroth_order.py  (already clean; add CPU/CUDA determinism tests)
  transport/       # grpc/  (servicer, client, channel factory, TLS) — the ONLY place that imports grpc
    codec/         # typed Tensor (de)serialization; safetensors-based; NO torch.save on the wire
  server/          # coordinator (round state machine), lifecycle, checkpointing
  client/          # base client, decomfl client, heartbeat supervisor
  telemetry/       # OTel spans + per-round metrics emitter (wire the dead deps — B3)
  proto/           # generated, single source of truth (B7 owns codegen)
```
Rationale: today `serializer`, `grpc_client`, and `coordinator` each mix transport, math, and lifecycle. Separating `strategies/` (pure functions) from `transport/` makes the aggregation math **unit-testable without a gRPC server** (directly fixing the N1 RPC-ordering coupling and M6 coverage gap).

**6.2 Typed protocol & config:**
- Replace the `Dict`-typed `config`/`results` tuples (`OrderedDict`, `Tuple[str, List[List[float]], int]`) with **pydantic models / dataclasses**: `RoundConfig`, `ClientUpdate`, `GradientScalars`, `AggregationResult`. mypy is already `strict=true` (`pyproject.toml`) — give it real types to check.
- **Drop `torch.save`/`torch.load` from the wire.** Serialize via the typed `Tensor` proto (already exists) or `safetensors` (zero-copy, no pickle, mmap-friendly for large models). This deletes C1, C3, and the `weights_only` invariant in one move.
- Add a `[project]` table to `pyproject.toml` with **pinned, minimal** deps (`torch`, `numpy`, `grpcio`, `protobuf`, `lz4`, `safetensors`, `pydantic`, `opentelemetry-sdk`). Delete `ray`, `flwr`, `flwr-datasets`, `opencensus`, `google-*`, `matplotlib/seaborn`, `aiohttp*` from the framework's deps (N2/N3) — move dataset/plotting concerns to `examples/` extras.

**6.3 Concurrency model:**
- **Keep the dual-stub heartbeat** (it is a real solution to long-`fit()` server-side timeouts) but make it **observable and fail-loud**: heartbeat supervisor sets a `threading.Event` after N consecutive failures (H1); training thread checks it between local steps and aborts the round cleanly. Wire the server's `should_stop` (`coordinator.py:165`) to actually fire on `is_client_alive == False`, and **consume `is_client_alive` in the round wait loop** so a dead client doesn't deadlock `wait_for_round_to_complete` (`coordinator.py:42-46`) — today a client that dies mid-round can hang the round until `num_rounds` exhausts.
- For the **server**, the round coordinator is a small state machine guarded by one lock; that design is fine and the threading contract is documented (`coordinator.py:88-94`). **Do not move the server to asyncio** purely for elegance — gRPC Python sync + threadpool is adequate for the per-project process model and avoids rewriting the entire servicer. Reserve asyncio for a future multi-tenant in-process server, which also requires fixing the global-RNG issue (M5) first.
- Make `MAX_SAMPLES`/`MAX_NUM_EXAMPLES` **strategy config**, log a WARN when a cap fires with `client_id, requested, capped` (H2).

**6.4 Aggregation (perf + correctness):**
- Vectorise DeComFL aggregation: batch perturbation generation per `(k)` once, stack client scalars into a tensor, single `matmul`/`einsum` (H4 / §4). Add a `pytest-benchmark` regression at `K=10,P=20,50 clients`.
- Add the **unpack-None guard** at both `_trigger_*` sites (M2): `result = strategy.evaluate(...); if result is not None: loss, metrics = result`.
- Settle the `η/P` question with B1 before v2 freezes the update rule (§3).

**6.5 Reliability (hand-off to C1, but framework-local):**
- **Checkpoint global params + round counter + seed/gradient history every round** so `rebuild_model` and server restart can resume (today all state is in-process RAM; a crashed `python fl_server.py` loses the whole run).
- Add per-round `sha256` of the global model for integrity and reproducibility (C3).

---

## 7. Decision table (per module)

| Module | Verdict | One-line rationale |
|---|---|---|
| `communication/serializer.py` | **rebuild** | C1 break, opaque `torch.save`-on-wire, env-inferred compression (C3); replace with typed/safetensors codec. |
| `client/grpc_client.py` | **refactor** | Solid retry/keepalive/TLS, but C1 upload, H1 heartbeat-death-invisible, N6 unlocked state, N7 name-sniffing. |
| `server/grpc_servicer.py` | **refactor** | Correct unary path; streaming upload broken (C1) + unbounded (H5); download 2× memory (M4). |
| `server/coordinator.py` | **salvage** | Round state machine & locking are sound; fix M2 unpack-None, wire `is_client_alive`/`should_stop` (dead), cap-WARN (H2). |
| `server/strategy.py` (FedAvg) | **salvage** | Aggregation math correct; add cap-WARN, drop JSON-string param path (`strategy.py:99-104`, unused/legacy). |
| `server/decomfl_strategy.py` | **refactor** | Math faithful to reference but O(K·P·N·d) loop + global-RNG (M5) + RPC-ordered seed coupling (N1); η/P open (B1). |
| `client/decomfl_client.py` | **salvage** | Clean Alg.4 impl + exact revert; verify rebuild scaling vs server (B1). |
| `estimators/zeroth_order.py` | **salvage** | Correct ZO; only gap is cross-device determinism tests (C3). |
| `server/server.py` | **refactor** | Works, but `utcnow()` (M3), no checkpointing, no profile-gated TLS-required boot. |
| `server/async_coordinator.py` | **kill** | Dead RabbitMQ alt-coordinator; imports uninstalled `pika`; shadows real `FLCoordinator` (N4). |
| `communication/protos/fedlearn.proto` | **salvage** | Clean `fedlearn.v1`; add `compressed`, `sha256`, version fields; single-source codegen (B7). |
| `requirements.txt` / `setup.py` / `pyproject.toml` | **rebuild** | `flwr` violation (H6/N2), ~40 deps incl. dead `ray` (N3), no `[project]` table. |
| Test suite (`tests/`) | **rebuild** | **Currently RED (3 fail)**; no transport/lifecycle tests; not gated in CI (N1). |

**Overall framework verdict: REFACTOR** — keep the custom FL core (DeComFL, FedAvg, ZO, dual-stub heartbeat); **rebuild the serialization/transport layer and the dependency manifest; kill `async_coordinator`.**

---

## 8. Prioritized recommendations

**P0 — unbreak + prove (days):**
1. Fix C1: wrap `{'parameters': params, 'num_examples': n}` before `torch.save` at `serializer.py:97` **and** `grpc_client.py:194`; or better, fix forward by moving to typed framing. Add a **bidirectional streaming round-trip test**.
2. Fix N1: update `test_decomfl_strategy.py:66` to the `Dict` API and add a test that calls `aggregate_fit` **without** pre-seeding to prove the `KeyError` guard. **Get the suite green.**
3. Stand up a **PR-time CI gate** that runs `pytest` (the root cause of N1 surviving). `pytest-cov --cov-fail-under=60`, ratchet to 80 (corroborates B7).
4. Fix M2 unpack-None at `coordinator.py:107,290`.

**P1 — security & correctness (1-2 weeks):**
5. C3: add `compressed` to chunk protos, fail closed; document `weights_only=True` as invariant + regression test.
6. C4: profile-gate `FEDLEARN_GRPC_USE_TLS` — refuse boot insecure outside `dev` (`server.py:127`).
7. H1: heartbeat `threading.Event` failure latch + training-loop check; **wire `is_client_alive`/`should_stop`** and consume in the round wait loop.
8. H5: cumulative `max_payload_bytes` + `max_chunks` guard in `SubmitModelUpdateStream`.
9. H2: cap-WARN with client/requested/capped.
10. B1 hand-off: settle `η/P` (§3) and rebuild-vs-global scaling before any convergence claim.

**P2 — scale & maintainability (v2 build):**
11. H4/§4: vectorise DeComFL aggregation; `pytest-benchmark` regression.
12. M4: `memoryview` slices on server download; incremental/safetensors serialization to cap host memory for large models.
13. N3/N4: prune deps to single digits; add `[project]` table; **delete `async_coordinator.py`**.
14. M5: per-instance `np.random.Generator(PCG64(seed))`; remove global seed.
15. Restructure into `core/strategies/transport/telemetry` (§6.1); wire OTel + per-round metrics (B3).

---

## 9. Cross-references
- **B1 (paper-alignment):** η/P client-vs-server asymmetry (§3); rebuild-vs-global scaling; cross-device seed determinism.
- **B3 (observability):** dead OTel/Prometheus deps (N3); empty per-round telemetry; wire from `_trigger_*aggregation`.
- **B4 (security):** `flwr`/`ray` CVE surface (N2/N3); plaintext gRPC (C4); pickle-on-wire elimination (C3).
- **B7 (standards/DX):** no PR-time test gate (N1, root cause of RED suite); 4-language proto codegen; dep hygiene.
- **C1 (reliability/SRE):** no checkpointing; dead-client round deadlock (§6.3); in-process-only state.
- **C3 (reproducibility):** seed determinism across CPU/CUDA + Python/C++ (§3, mobile A6).
- **C4 (business/IP):** `flwr` + `flwr-datasets` license implications vs the "no Flower" wedge (N2).
- **A4 (client-docker):** the same `flwr-datasets` leak + ARM64 wheel surface from N3.
