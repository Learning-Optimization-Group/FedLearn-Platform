# Capstone Multi-host Training Implementation Plan

**Goal:** Train one model across Mac, G14 and Orin, qualify canonical transport against simulation, then execute the approved 52-cell CPU systems benchmark.

**Architecture:** Keep `fl-runtime/client.py` as the executable client and reuse the existing gRPC lifecycle, coordinator and composed uniform clipped mean. Add a training-only vector helper and a narrow research host mode that multiplexes logical clients with independent state and a two-fit per-host limit. Mac owns preparation, server-side evaluation and receipts; remote hosts receive assigned training data only.

**Tech stack:** Python, Torch CPU, NumPy, existing FedLearn gRPC/safetensors transport, isolated host runtimes and private SSH/Tailscale routes.

**Spec:** `docs/superpowers/specs/2026-10-25-capstone-multihost-training-design.md`, explicitly approved by the user on25October.

## Global constraints

- Preserve completed studies and v3 source/PDF/HTML; no holdout/test reads or transfer.
- No second FL client implementation outside `fl-runtime/client.py`; reuse framework `start_client`.
- Existing generic recipes retain their behavior. The feature recipe must not select Pneumonia or Adam by accident.
- Separate logical identity/shards/RNG from physical hosts and fit concurrency. Freeze both per-placement concurrency limits.
- Full cohort, uniform client weighting, stable logical-ID reduction order, declared float32 semantics.
- CPU qualification first; never silently report accelerator fallback as accelerator execution.
- Isolated dependency directories/containers; do not stop or alter the Orin inference service.
- Private endpoints and patient payloads remain outside committed/public artifacts.
- No protected-release, hospital, GPU, clinical or untested distributed-success claims.
- Approved benchmark:3/6/24/80 clients ×2methods ×3seeds ×2placements ×5rounds =48cells; four20-round80-client anchors;52cells/320rounds total. Protocol/source checkpoint before medical fits.

## Review focus

1. Threaded RNG/model creation can perturb per-client batch order: compare real batches/heads to simulator and check global RNG preservation.
2. One host can impersonate/multiply logical submissions: manifest membership, unique identity and round/layout checks must precede aggregation.
3. Eighty clients can exhaust memory or time out while queued: one interpreter per host, shared immutable arrays, bounded fits, heartbeat/cancellation-aware queues and explicit deadlines.
4. Example-weighted FedAvg or Adam can masquerade as capstone parity: test unequal shards against uniform clipped mean and SGD.
5. Host failure/partial result can produce a successful artifact: exercise missing/duplicate/stale updates on generated data and require complete round/cell accounting.

## Task1 — Canonical training-only feature recipe

**Files:** Create `fl-runtime/effusion_runtime.py`, `fl-runtime/tests/test_effusion_runtime.py`.

**Interfaces:** `load_host_data(path: Path, expected_sha256: str) -> HostData`; `FeatureState(data, client_id: int, settings: dict)` with `get_parameters() -> OrderedDict`, `fit(parameters, round_number: int, should_stop: Callable) -> tuple[OrderedDict,int,dict]`. No transport or `fl.Client` implementation in this module.

- [x] Write generated-data tests for payload hashes/training-only flags, assigned IDs/ranges, no evaluation arrays, finite shape, head keys, partial batches and refusal of changed optimizer/settings.
- [x] Watch them fail before implementation.
- [x] Implement immutable shared host data, per-client head/SGD state, canonical `ClientRng` round seed and explicit local generator. Reproduce the RNG draws consumed by simulator model construction before each epoch permutation rather than assuming zero initialization consumes no randomness.
- [x] Compare actual one- and multi-round output against `HeadClient`/`ZeroBundleWorkload` on generated arrays, including concurrent clients and global-RNG preservation; proposed parameter tolerance1e-5.
- [x] Run focused tests and record RED/GREEN evidence.

## Task2 — Fixed membership, deterministic reduction and streamed settings

**Files:** Modify `framework/src/fedlearn/server/coordinator.py`, `framework/src/fedlearn/server/server.py` only as required; create `framework/tests/test_multihost_cohort_contract.py`.

**Interfaces:** Existing coordinator remains backward compatible. A strategy may declare `allowed_client_ids` and `get_client_config()`; coordinator rejects unregistered/out-of-manifest submissions, incompatible head layouts and sorts accepted updates by the declared IDs when this contract is present. Generic callers without a manifest retain existing behavior.

- [x] Tests first: unknown registration/submission, unregistered member, duplicates, stale/ahead rounds, nonfinite/mismatched head and unequal-shard uniform result; no partial success at deadline.
- [x] Watch failures, then add the smallest opt-in enforcement around existing locked ingress/aggregation paths.
- [x] Verify synchronous and async aggregation ordering and exact full-cohort minimum; preserve existing privacy/security gates.
- [x] Run focused coordinator/server regressions; record evidence.

## Task3 — Canonical executable host mode

**Files:** Modify `fl-runtime/client.py`; create `fl-runtime/effusion_host.py` for CLI validation, scheduling and journals without a Client subclass or a new FL loop; create `fl-runtime/tests/test_effusion_host.py`.

**Interfaces:** `python fl-runtime/client.py --effusion-host-plan <private.json> --host-plan-sha256 <hash> --server-address <endpoint>` selects a narrow early dispatch before legacy transformer/dataset imports. The Client subclass remains in `client.py` and delegates to `FeatureState`; host helpers call existing `fedlearn.client.start_client` once per logical identity. Common run settings are immutable.

- [x] Write failing tests for mode isolation, client IDs beyond9, duplicate IDs, semaphore concurrency, cancellation while queued, round/config cross-checks and no heavyweight legacy imports.
- [x] Implement shared data loading, independent per-client objects and network threads with a fit semaphore of two; no per-client environment mutation.
- [x] Bind actual device/host label/run ID in private journals. Detect inherited auth settings requiring per-client credentials and refuse rather than disable them or share one identity token silently; add an explicit token seam only if needed by the chosen qualified transport.
- [x] Run local generated-data gRPC smoke through the actual canonical executable, coordinator and codec; verify all identities and five complete rounds.

## Task4 — Preparation, frozen runner and remote qualification

**Files:** Create `docs/artifacts/capstone/repro/run_multihost_benchmark.py`, its tests, and `fl-runtime/effusion_server.py` as a research entry using the existing server/coordinator; no alternative server protocol. Private launch helpers live in the ignored study directory.

**Interfaces:** Runner modes `preflight`, `execute`, `analyze`; fresh outputs only. Exact source/runtime hashes, logical/physical assignments, settings and all declared outcomes bound in private/public manifests. Only Mac loads the verified canonical train/selection bundle. Remote host packs contain assigned training rows/labels and immutable projection metadata only.

- [ ] Failing tests for prospective cell budget, host mappings, source/runtime changes, selection leakage, changed source/shards, incomplete cohorts and export allowlist.
- [ ] Implement explicit seed IDs14001–14003 for systems cells and14004 for anchors; identify them as training-seed IDs, not dates. Pair shards and streams across placement/method cells for each client count.
- [ ] Use uniform clipped mean, C0.05, LR0.03, SGD, seven local epochs, batch256, zero head, fixed projection seed20261025. Systems cells have five rounds; anchors twenty; no noise or attacks.
- [ ] Generated-data reference with independent batch-generator implementation, then CPU one-client-per-host smoke over the qualified private transport. Verify distinct physical contributors and five complete rounds before medical preflight.
- [ ] Freeze startup, steady-state, serialization/transport scope, memory and per-host fit limits. Measure elapsed times locally; no subtraction of unsynchronized host clocks. Freeze public deadline600seconds per round after readiness; missing cohort aborts, never partial aggregate.
- [ ] Create local source/protocol/public-manifest checkpoint before the52 medical cells; retain all failures and unrun cells, no replacement fitting.

## Task5 — Execution, analysis, report and review

**Files:** New aggregate result/verification notes under `docs/artifacts/capstone/`; contribution-first reportv4 and reading copies; no changes to old measurements.

- [ ] Execute all52 declared nonprivate cells with owned process handles, progress journals and immutable source. Record all outcomes.
- [ ] Independently rescore all returned heads on selection, compare transport/placement/reference heads and execution cohorts, and report measured numerical differences without retroactive tolerance changes.
- [ ] Export aggregate completion, strong-scaling timing, memory and communication tables; keep endpoints, raw shards and models private.
- [ ] Render and inspect the approved contribution-first report. Add multi-host claims only for verified actual runs; preserve previous report versions.
- [ ] Run framework/runtime gates appropriate to code changes and a fresh whole-change review. Resolve material findings with regression tests; record remaining limitations accurately.
- [ ] Preserve reviewed work locally. No push, merge, service interruption or external publication is included.

## Current ledger

Design approved; source-level builder check found no missing user decisions. Existing isolated capstone worktree is reused. Orin is reachable through the existing Tailscale TCP proxy; G14 is reachable through its current LAN route with its previously trusted key. Remote hardware checks show12/16 CPU cores respectively, roughly12/10GiB available memory, and no Torch in host system Python. System Python versions differ, so isolated compatible runtimes are a readiness task. G14 CUDA is unverified; do not infer absence or presence from a missing command.

Ruling: Mac-only concurrency is two active fits; three-host concurrency is two per host, six total — this measures adding host resources under the same per-host limit, not a fixed-total-concurrency network-overhead experiment. Report this explicitly.

Ruling: use the existing private SSH route/tunnels if direct Mac network routing is unsuitable — no public listener or disabled security gate. Record the actual transport; SSH tunnel measurements are not native direct-TCP performance.


26 October continuation ledger:
- Tasks1–3 implemented and generated-data tested. Full framework gate previously passed2403/14deselected/80.22%; it is being rerun after the opt-in admission-ack fix. Full runtime previously799pass; updated gate includes actual gRPC failure exercises.
- Isolated compatible CPU environments installed on both remotes; system Python and Orin inference service unchanged. Actual three-host generated runs completed for both methods, five rounds each. Maximum simulator parameter difference7.450580596923828e-9 (mean512),1.862645149230957e-9 (projected128); Mac-only reference exact. Hardened source gets a new generated qualification rather than overwriting those artifacts.
- Whole-change review exposed input-commitment, acknowledgement, runtime-binding, cleanup, final-head and queue-metric defects. Corrective regressions observed failing, then passing where applicable. Prepared inputs are now sealed in the public manifest; actual interpreter runtimes and per-host summaries are reconciled; cleanup uses an owned Linux process descriptor, even after SSH exits.
- Contribution-first v4 rendered to12pages and visually inspected; previous report and scientific inputs preserved. No new distributed-execution success claim added to that report.
- Ruling: anchors replay historical seed13001, not proposed14004 — preserves the52-fit budget while comparing stored selection-stage simulator heads; not fresh confirmation. Input record hashes are frozen in the prospective26October protocol.
- Ruling: randomized prospective execution order is sorted SHA256 of20261026 plus cell name — avoids always testing Mac-only first; actual order is published before medical fits.
- Ruling: application protobuf counts exclude wire framing/SSH; serialization time is not separately observed. Report this measurement limit rather than inventing network throughput.
- Medical fits remain zero while final qualification, reviewed test gates and source/public-manifest checkpoint are pending.

Final pre-fit gates: reviewed source passed2403 framework tests,14deselected,80.23%coverage (583.06s); runtime802passed,30deselected (56.81s); final actual transport2passed (20.38s); preparation/supervisor17passed. Hardened generated snapshot completed four qualification cells/twenty rounds and independent simulator comparison, with the same7.45e-9/1.86e-9 cross-host maxima. No remaining material blocker from the fix review. The medical preflight contains52cells/320rounds and public manifest SHA256 a02da5d262fc59d054f67bdff6c5ad7ca71dd42f721d141af7cb1acebd9dc935. Medical fitting is zero before this local source/protocol checkpoint.
