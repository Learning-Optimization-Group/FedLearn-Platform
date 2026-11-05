# Capstone reliability program implementation plan

**Goal:** Implement and qualify all seven requested improvements without altering frozen research evidence.

**Architecture:** Extend the existing framework/runtime and execution contract. Use private completed-round checkpoint generations, shared generated-data conformance workloads and separate physical/deployment acceptance gates. Reuse the simulator and mobile trainer rather than creating parallel training implementations.

**Tech stack:** Python 3.12/PyTorch/safetensors, Java 21/Spring/PostgreSQL, React/TypeScript, Android/C++/ExecuTorch and external Playwright acceptance.

**Spec:** `docs/superpowers/specs/2026-11-04-capstone-reliability-program.md`.

## Constraints

- Preserve existing untracked output and frozen medical studies; do not read the closed holdout for new evaluation.
- Use the canonical protobuf under `proto/` when protocol changes are required; verify mirrors.
- Keep cookies, keys, random-state checkpoints and private data outside public artifacts.
- Write behavioral tests first; inspect a meaningful failure before implementation.
- Keep the full program open until every acceptance gate has direct evidence.

## Review focus

- Crash between tensor persistence and generation publication: readers must see a complete generation or the previous complete generation.
- Correctly formatted but wrong-run, wrong-layout or wrong-strategy checkpoints: restore must fail without mutating the target.
- Disk failure or Stop during round completion: no uncommitted next-round model may become available.
- Fast uploads/retries crossing the round boundary: no duplicate contribution or discarded accepted next-round update.
- Private randomness/accounting or security modes missing from an adapter: reject continuation rather than approximating it.

## Task 1: Private atomic checkpoint storage and strategy state

**Files:** create `framework/src/fedlearn/server/checkpoint.py`; create `framework/tests/test_round_checkpoint.py`.

**Interfaces:** `CheckpointIdentity(run_id, configuration_sha256)`; `RoundCheckpointStore(directory, identity).save(snapshot)` and `.load_latest()`; capture/restore helpers with an explicit strategy allowlist.

- [x] Write tests for round-trip model/history/state, FedOpt continuation versus uninterrupted training, mismatched configuration and layout, immutable generation conflicts, checksum/truncation refusal, non-finite state and private filesystem permissions.
- [x] Run the new test file with `--no-cov`; confirm the missing implementation fails.
- [x] Implement JSON/safetensors generations with atomic publication, integrity checks and validation before mutation.
- [x] Run the checkpoint suite and existing coordinator/strategy regressions.

## Task 2: Coordinator and gRPC-server checkpoint integration

**Files:** modify `framework/src/fedlearn/server/coordinator.py`, `framework/src/fedlearn/server/server.py`; create `framework/tests/test_coordinator_checkpoint.py` and server resume tests.

**Interfaces:** opt-in checkpoint callback invoked after successful evaluation and before next-round publication; server configuration binds a checkpoint store and explicit restore request.

- [x] Add failing tests for async/sync completed-round persistence, responsive heartbeat/Stop during blocked storage, fail-closed storage errors, stale retry rejection and preserved history after restore.
- [x] Integrate storage without moving disk I/O under the network coordinator lock.
- [x] Test a real gRPC interruption/restart against an uninterrupted generated-data control.

## Task 3: Runtime/API continuation and complete strategy coverage

**Files:** runtime CLI/configuration and tests; backend run model, migrations, orchestration, controller/service and tests; strategy checkpoint adapters.

**Runtime files:** create `fl-runtime/recovery.py` and `fl-runtime/tests/test_runtime_recovery.py`; modify `fl-runtime/fl_server.py` and add entrypoint forwarding coverage.

**Runtime interfaces:** `runtime_recovery(args, *, env=None, source_files=None)` is a context manager yielding `RuntimeRecovery | None`; its `server_options()` supplies the existing framework configuration fields and `validate_inputs()` rechecks source/initial bytes before listening. `source_sha256([(logical_name, path), ...])` binds source content independently of checkout location. An exclusive file lease lasts through the complete training entrypoint.

Runtime sequence: write and observe failing CLI/session/identity/lease tests; implement the private session context; forward options through the actual entrypoint; run runtime and framework recovery regressions; exercise a real CLI process interruption/restart; review and commit. Retain both full-source and generated-fixture scope in evidence.

Backend continuation requirements, before wiring the public resume action:

- Preserve the logical run UUID and client bindings; append an execution-attempt record for each authorized restart. Automatic terminal callbacks cannot reopen a run.
- Lock/check project and run ownership, unchanged intent, absence of a different active run and absence of a live matching training process. Never reuse a start path that unconditionally stops another active server.
- Configure checkpoint paths from a private server root plus run UUID; never accept a filesystem path from a browser request or serve checkpoint contents through artifact endpoints.
- Rotate callback authorization per attempt and reject stale-attempt callbacks even if a request was already in flight during rotation.
- Add run-scoped round-result idempotency with a database uniqueness/atomic-insert boundary. Retain first metrics, reject conflicting replay and broadcast only newly accepted results.
- Prove controller-to-worker resume and callback replay with PostgreSQL and actual processes before qualifying the dashboard action. Runtime command-line recovery is a separate completed slice, not evidence for these API guarantees.

- [x] Bind runtime checkpoints to run/source/configuration/model identities and configure private per-run storage. Backend-managed provisioning remains part of the API workflow below.
- [x] Make result ingestion run-scoped and retry-safe: verified callback identity, current authorization, active-run locks, immutable metrics, strict round parsing and database uniqueness. Persist callback authorization before child launch.
- [x] Bind completion to the authenticated active run and current authorization. Serialize lifecycle transitions using refreshed locked state and preserve Project-before-Run ordering in stop and callback paths.
- [ ] Specify continuation ownership, eligible terminal states and lineage before adding API behavior.
- [ ] Add private/robust/DeComFL state with uninterrupted-versus-resumed tests and privacy/release invariants; no silent fallback.
- [ ] Implement explicit owned continuation, preserving accepted history and truthful failure states; qualify API/worker restart with actual processes.

## Task 4: Frozen-head execution across devices (area 1)

**Files:** `fl-runtime/execution_plan.py`, export/bundle staging code, canonical contract if necessary, mobile contract projection/provisioning, shared native trainer and cross-runtime tests.

- [ ] Pin one generated-data frozen-head fixture with named trainable layout, frozen-state digest and preprocessing/objective identity.
- [ ] Add failing publication, graph-export, bundle-binding, native-gradient and full-model reconstruction tests.
- [ ] Extend existing export and training paths; independently replay native updates and verify frozen state remains unchanged.
- [ ] Qualify one supported public-data recipe through dashboard/desktop/physical Android, recording actual artifact/source identities and final-model reload.

## Task 5: Release/TLS and failure acceptance (areas 3 and 4)

**Files:** release qualification scripts/workflow tests and external `homelab/fedlearn` acceptance suite; read its instructions before edits.

- [ ] Add generated-data real-process fault cases for upload, response loss, process exit, API/server restart and deadline.
- [ ] Add a CI smoke gate that does real training and checks committed state, contribution uniqueness and cleanup.
- [ ] Build/install with persistent signing material, qualify HTTPS enrollment/TLS training, then same-key update and unchanged dataset hashes.
- [ ] Run bounded physical Wi-Fi/screen-off and fault checks with retained failed outcomes.

## Task 6: Profile and improve distributed execution (area 5)

- [ ] Instrument process startup, data staging, serialization, network waits and compute with bounded aggregate timing output.
- [ ] Record a prospective generated-data baseline on the current transport and hardware.
- [ ] Implement the measured dominant improvement, compare matched whole-run/steady-round time and recheck numerical state.

## Task 7: Android secure aggregation (area 6)

- [ ] Specify the scalar protocol binding, threat model, authenticated identity/key distribution, sealed-share layout and dropout state machine against the existing Python wire.
- [ ] Add cross-language golden vectors and adversarial/refusal tests before native implementation.
- [ ] Implement native key/share/masking and round lifecycle; qualify mixed Python/Android secure rounds, dropout and below-threshold refusal.
- [ ] Keep unverified distributed clipping and weight-update protocols refused until their separate implementations and proofs/tests exist.

## Task 8: Matched research baseline and independent verification (area 7)

- [ ] Inspect the primary baseline implementation/paper, define comparable privacy/attack/workload budgets and freeze a generated/public-data protocol.
- [ ] Qualify implementation against analytic/toy controls, then execute the bounded predeclared matrix once.
- [ ] Independently rescore outputs and uncertainty; retain failures and distinguish this work from frozen medical results.

## Execution ledger

2026-11-04: Checked both branches and selected the existing isolated capstone checkout. The baseline coordinator/run-end/adversarial regression selection passes 24 tests. Existing timestamp deprecation warnings remain. No implementation area is complete yet.

Initial recovery slice: private immutable storage, FedAvg/FedProx/FedOpt CPU adapters, coordinator/server opt-in restore, and a real gRPC process-kill check are implemented. A 121-test selection passed before final review corrections. Review reproduced oversized unreadable generations, incompatible saved layouts, incomplete histories and a Stop/publication race; all have failing reproductions and corrections. The current 92-test recovery/coordinator/heartbeat selection passes, including the real process restart. The broader framework run, collected before the final review corrections, passed 2,440 tests with 14 deselected and 79.02% coverage against the 73% floor. Retain that scope distinction; a final whole-program gate still follows remaining implementation. This is not runtime/API continuation or complete area-2 qualification.

Runtime continuation gate (5 November): 838 runtime tests pass, with 30 slow tests deselected; the final source also passes 105 framework recovery/coordinator/strategy tests. Five actual generated-data runtime/client cases cover all three CPU strategies and exits during/after final saving. CLI recovery is implemented; backend provisioning, execution-attempt transitions and API/browser qualification remain open.

Backend result callback gate (5 November): 1,231 tests pass, two optional tests are skipped, and line coverage is 84.47% against the 70% floor. Real PostgreSQL tests cover concurrent and conflicting replay, stale authorization, invalid metrics/rounds, authenticated run identity and V33 upgrade with ambiguous historical rows preserved. Review additionally reproduced and corrected JSON round coercion and authorization persistence after child launch. Notification delivery remains best-effort after commit. Finished/artifact/benchmark authorization, attempt transitions, backend checkpoint provisioning and API/browser continuation remain open; this does not complete recovery area 2.

Backend completion/lifecycle gate (5 November): 1,243 tests pass, two optional actual-worker tests are skipped, and line coverage is 84.76%. Completion now rejects old-run and retired-attempt callbacks. Real PostgreSQL controls reproduced delayed terminal-state overwrites, stale managed entities, cached-project authorization and a stop/completion deadlock; refreshed locking and consistent lock order pass all controls. Direct orchestration metadata writes, artifact/benchmark/connection-verification authorization, attempt transitions and controller-to-worker resume remain open. The optional actual-worker gates still require separate execution evidence.

Physical qualification status (5 November): the connected Android device is reachable over authorized USB debugging. The installed app is an older debug build. Preserve physical training, release-update, thermal/Wi-Fi and Android interoperability gates until direct current-source evidence exists; package inventory and host tests do not substitute for them.
