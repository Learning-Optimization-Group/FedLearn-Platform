# Execution Contract v1 — Implementation Plan

**Goal:** Implement [Execution Contract v1](03-android-execution-contract-v1-design.md) as a sequence of independently tested commits, ending with a mixed-device TinyNet FedAvg run in which the laptop and Android clients both execute the same published contract.

**Status:** Stage 2 is complete (2A–2I, 2026-09-24). A vivo 1805 and three laptop clients trained one published contract to COMPLETED, and the phone's training was checked numerically against the contract. Android depends on execution contract v1 — see "Android is v1-dependent" below for what that costs.

## Rules for every slice

- Work on `mobSupp`, commit locally, do not push.
- Write the failing test first and confirm that it fails for the expected reason.
- The canonical schema is `proto/fedlearn/contract/v1/execution_contract.proto`. The validation rules and limits are in `framework/tests/fixtures/execution_contract_v1/README.md`, and `conformance.json` beside it is the shared executable specification. A rule change updates the README, the generator, and all three readers in the same commit.
- Android runtime behavior does not depend on the contract until slice 2H.

## Stage 2A — schema and readers (complete)

| Slice | Commit | Verified by |
| --- | --- | --- |
| Canonical proto, framework mirror, committed Python stubs, golden fixtures | `855f19d` | `buf lint`, `buf breaking` against `main`, `buf generate` for all pinned plugins, `check_proto_mirror.sh`, `test_execution_contract_v1_fixtures.py` (binary/ProtoJSON agreement, byte-stable round trip, explicit-presence zeros, generator drift) |
| Python reference validator and 157-case conformance corpus | `26833a9` | `test_execution_contract_v1_validation.py`; 100% line coverage of `fedlearn.contract`; framework suite 1107 passed |
| Java reader (Gradle protobuf codegen from the canonical proto) | `cb77a56` | `ExecutionContractConformanceTest` (157 cases), `ExecutionContractGoldenTest`; backend suite 908 passed, line coverage 79.8% |
| TypeScript reader (committed protobuf-es output with a CI drift check) | `676d756` | `executionContract.test.ts` (157 cases + golden tests); mobile suite 405 passed; lint and `tsc` clean |

Each reader was also checked by deliberately breaking individual rules; the corpus failed every time.

### Decisions made while implementing the schema

- **Explicit presence.** Behavioral scalars whose zero value is legal (`momentum`, `dampening`, `weight_decay`, `nesterov`, `amsgrad`, `centered`, `reset_optimizer_each_round`, `drop_last`, `max_transient_retries`) are proto3 `optional`, so a publisher that omits one produces a refusal instead of a silent default.
- **Batching.** `LocalTraining` gained `batch_size`, `drop_last` and `batch_order`. Without them the step budget cannot be stated: the laptop TinyNet path trains `DataLoader(batch_size=8, shuffle=True)` for `local_epochs` passes, while Android runs `local_epochs` full-batch steps. They agree today only because the fixture is exactly one 8-row batch.
- **Refusal vocabulary.** `ContractIssueCode` is part of the schema, so every language uses the same generated names.
- **Approved matrix.** Only TinyNet / FedAvg / FULL / vector classification / cross-entropy / trainable-state F32 is publishable. FedOpt and Robust, validated live in Stage 1, are added only with their own conformance tests.
- **Parsing.** ProtoJSON readers ignore unknown fields (a compatible addition must not break older readers), and an unknown enum name therefore reads as 0 and is refused. Java's `JsonFormat` and Python's `json_format` were each more permissive than the specification (a bare `NaN` literal, and a top-level array, respectively); both readers pre-check the document, and the corpus pins the behavior.
- **C++.** The design routes only a validated, typed projection to C++, and the host C++ build links no protobuf runtime. There is therefore no C++ generated reader; the design's C++ round-trip gate is met instead by projection conformance in slice 2H.

## Stages 2B–2H — what landed

Each slice was committed only after its own tests passed. The stage sections below are the plan as
written and are kept for the detail; this table is what actually shipped.

| Stage | Commits | Verified by |
| --- | --- | --- |
| 2B — immutable run intent | `89d68dd`, `a07d5d3` | `V27`/`V29` migration tests in the `V*MigrationTest` shape; a project edit after start changes neither the snapshot nor the spawned server |
| 2C — publication lifecycle and storage | `b7543a6`, `5a3689f` | `V28` migration test; concurrent publication yields one `READY` row (`INSERT … ON CONFLICT DO NOTHING`); a PostgreSQL trigger forbids `UPDATE`, so a `READY` contract cannot change |
| 2D — Python resolved training plan | `b344a7d`, `c5794d3` | `fl-runtime/tests/test_execution_plan.py`: the resolved plan equals what `client.py` constructs; canonical state digests; unsupported recipes raise `NotRepresentable` |
| 2E — publisher | `5e2f180`, `c450eb9`, `8c4baad`, `6708f5a` | staging-outcome and publisher tests for `PENDING`, `READY` and each `UNAVAILABLE` reason |
| 2F — dual-emitted manifest and equivalence | `c9cb0c1` | legacy-vs-contract equivalence blocks publication on disagreement; no partial contract is ever returned |
| 2G — laptop client | `5d8a6f0`, `030e46e`, `99885c3` | `client.py --execution-contract` refuses any difference from its own plan before data loading; desktop launcher and `client-docker` entrypoint tests |
| 2H — Android decision, refusal and projection | `a3e1618`, `6597e65`, `c1697cb`, `f2446a4` | Jest refusal tests before any provisioning call; one shared projection fixture consumed by the TypeScript projection test **and** the native C++ round test; C++ suite 63 passed |

The backend pipeline was also exercised once end to end against a throwaway PostgreSQL and a real
backend: the contract published `READY`, was served on the manifest, validated by the Python reader with
zero issues, and carried digests matching the phone's golden state.

### Android is v1-dependent

The design left open what Android should do for a run v1 does not cover. Decision (2026-09-22, Anurag):
**strict.** The phone refuses every run without a `READY` v1 contract it can execute exactly, and never
approximates. Each refusal names its reason (`CONTRACT_MISSING`, `CONTRACT_LEGACY_ONLY`,
`CONTRACT_UNAVAILABLE`, `CONTRACT_INVALID`, `UNSUPPORTED_STRATEGY`, `UNSUPPORTED_UPDATE_PROTOCOL`,
`UNSUPPORTED_SECURITY`, `UNSUPPORTED_OPTIMIZER`, `UNSUPPORTED_BATCHING`, `MISSING_CPU_ARTIFACT`).

**This was a deliberate capability regression and must not be described as anything else.** At first the approved v1
matrix was TinyNet / FedAvg / FULL only, so DeComFL, FedOpt and Robust stopped running on the phone even though
Stage 1 validated them live. **FedOpt and Robust are restored** (2026-09-24,
[07](07-fedopt-robust-contract-plan.md)): they are in the matrix and completed live on the phone. **DeComFL is restored
too** (2026-09-25, [08](08-decomfl-contract-design.md)), after a schema extension for zeroth-order training. Only
FedProx is still refused; it needs a native proximal term.

### Findings from 2H worth carrying into the paper

Recorded in `research/notes/on-device/2026-09-22-contract-endpoint-replay-and-tolerance-trap.md`, measured
in `research/results/on-device/contract_endpoint_replay.json`.

- **A tolerance inherited across configurations can exceed the signal.** The existing FedAvg endpoint
  golden (lr 0.1, 5 epochs) carries `endpoint_atol = 2e-3`. The contract states lr 1e-3 for one epoch,
  which moves the parameters by at most 1.44e-4 — 13.9x *smaller* than that tolerance. Reusing it would
  have produced a test that passes whether or not the trainer trains. The contract endpoint therefore has
  its own golden at 1e-6, the generator refuses to write a golden whose movement is within its own
  tolerance, and the test also asserts the parameters moved at all.
- **A fixture that tests complete before asserting cannot detect drift.** The golden contract was found to
  state training `fl-runtime` never resolves, so `check_contract` refused it — the published fixture was a
  contract no client would train under. It survived because every test overwrote its plan fields with the
  resolved plan before asserting. `test_the_golden_contract_is_one_this_client_accepts_unmodified` now
  asserts the production predicate on the fixture as published (`c1697cb`).

### Outstanding within 2H

- ~~The on-device Hermes check~~ — done in Stage 2I (2026-09-24): the phone's activity log shows the contract accepted.
- The native launcher passes `--dataset-path`, which `client.py`'s parser does not define. Unrelated to the
  contract work, noticed while testing, left untouched.

## Stage 2B — immutable run intent

- Add a Flyway migration and entity for a run-intent snapshot captured at `POST /api/projects/{id}/start`: recipe, arm, task, optimizer and hyperparameters, local epochs, central-DP settings, and effective TLS/client-auth deployment settings.
- The FL-server spawn and every later contract step read the snapshot, not the mutable `Project`.
- Tests: a project edit after start changes neither the snapshot nor the spawned server's configuration; a migration test in the `V*MigrationTest` shape (Flyway on, per-test Testcontainers database); runs predating the migration read as having no intent.

## Stage 2C — publication lifecycle and storage

- Store `contract_state` (`PENDING`, `READY`, `UNAVAILABLE`, `LEGACY_ONLY`), canonical contract bytes, `contract_id` (SHA-256 of the stored bytes) and a machine-readable unavailability reason per run.
- `PENDING → READY | UNAVAILABLE` is a single guarded update, and a `READY` contract never changes. Runs without an intent snapshot are `LEGACY_ONLY`.
- Tests: concurrent publication attempts produce one `READY` row; repeated reads return identical bytes and ID; a failed attempt cannot replace `READY`; state survives a backend restart; publication is refused when Java validation reports any issue.

## Stage 2D — Python resolved training plan

- Add a torch-free `fl-runtime` entry point that resolves, for a run intent, the plan the laptop client actually executes: optimizer and every hyperparameter, local epochs, batch size, drop-last, batch order, the ordered trainable layout from the recipe, and the objective.
- Tests: for TinyNet the resolved plan equals what `client.train()` constructs (SGD, learning rate, momentum and weight decay zero, batch 8, shuffled); the layout equals `named_parameters()` filtered to `requires_grad`; unsupported recipes resolve to "not representable" instead of a guess.

## Stage 2E — publisher

- After bundle staging, build the contract from the run intent, the resolved plan and the staged bundle (file paths, sizes, digests, required operators, frozen- and initial-state digests), validate it with `ExecutionContractValidator`, and publish `READY` or `UNAVAILABLE` with a reason.
- Tests: delayed staging stays `PENDING`; staging failure, a corrupt file, a missing CPU artifact, or a validation issue yields `UNAVAILABLE`; a file that disappears after `READY` surfaces an integrity error on provisioning without changing the contract.

## Stage 2F — dual-emitted manifest and equivalence

- `RunManifestDto` and the enrollment manifest gain `contractState`, and, only when `READY`, `contractId` and `executionContract` (ProtoJSON via `JsonFormat`). Legacy fields stay.
- Before `READY`, a server-side check compares the legacy decisions (strategy, recipe, first-order capability, secure aggregation) with the contract. Disagreement prevents publication and is reported.
- Tests: controller and service tests for each state, including that no partial contract is ever returned; an old-client-shaped read still works.

## Stage 2G — laptop client

- The desktop launcher passes the `READY` contract to `fl-runtime/client.py`. The client validates it with `fedlearn.contract`, compares it with its own resolved plan, and refuses on any difference.
- Tests: TinyNet agreement passes; a changed learning rate, optimizer, batch size or trainable layout is refused before data loading; a legacy-only run still works on the legacy path.

## Stage 2H — Android parsing, refusal and projection

- During enrollment, before provisioning, parse `executionContract` with the generated reader, validate it with `executionContract.ts` against the enrolled run and project, and refuse with a distinct, user-facing code for `PENDING`, `UNAVAILABLE`, `LEGACY_ONLY`, each `ContractIssueCode`, an unsupported strategy or optimizer, and a missing CPU artifact.
- Project the validated contract into the native round configuration (learning rate, local steps, batch semantics, strategy) and remove the hard-coded TinyNet fallbacks. Cross-check provisioned bundle files against the contract's digests.
- Tests: Jest refusal tests before any provisioning call; a golden projection fixture consumed by both the TypeScript projection test and a C++ round test; the protobuf-es runtime exercised on the device, because Hermes behavior is not proven by Jest.

## Stage 2I — mixed-device conformance

- Build and install a fresh APK, then run three TinyNet FedAvg rounds with the vivo and three desktop clients on one published contract.
- Record the run ID, contract ID, client and partition assignments, accepted contributions, byte counts and terminal state under `research/results/multiplatform/`, and update the mobile status documentation.
- Acceptance: one accepted update per participant per round; both client types report the same contract ID; the laptop and Android local plans match the contract. This does not certify other recipes, DeComFL security or GPU training.

### Stage 2I — result (2026-09-24)

Run `50d14f0a…` on contract `de3d4610f0f8…`, from a fresh debug APK built at `5452658` (vivo 1805, Android 10,
arm64-v8a) plus three laptop CLI clients, reached **COMPLETED**.

| Criterion | Outcome |
| --- | --- |
| One accepted update per participant per round | Pass. `All 4 clients reported` in rounds 1–3; 12 accepted updates (9 laptop, 3 phone); no duplicates. The coordinator counts distinct client IDs. |
| Both client types report the same contract ID | Pass. The phone's activity log shows `Execution contract de3d4610f0f8… accepted.`; the contract served to the laptops hashes to the full contract ID. |
| Laptop and Android local plans match the contract | Pass. Laptops: `check_contract` accepts the contract as served. Phone: a replay under "every client runs lr 1e-3 × 1" matches all six of the phone's reported losses, the digest the backend recorded for the aggregated model, and the saved final model (max abs diff 0.0). The retired lr 0.1 × 5 fails all three. |
| Hermes runtime check (carried over from 2H) | Pass. The accepted line can only appear after protobuf-es parsing, validation and projection have run under Hermes on the device. |

The first attempt (run `6e43e025…`) trained all three rounds but was marked FAILED. The FL server's callback URL
defaults to `http://localhost:8081` whatever port the backend really uses. The test backend ran on :8084, so the
callbacks, carrying the run's internal token, went to an unrelated backend on :8081 and were rejected. Setting
`app.backend.internal-url` fixed it for the second run. The platform default itself is **not fixed**: it should
derive from the backend's own `server.port`.

Not shown by this run: laptop clients on separate hardware, the phone on a real network rather than USB port
forwarding, the Electron launcher path, any learning signal (TinyNet has no evaluation set), and anything outside
TinyNet / FedAvg / FULL.
