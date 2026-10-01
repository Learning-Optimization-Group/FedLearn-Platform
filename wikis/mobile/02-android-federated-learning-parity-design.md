# Android Federated-Learning Parity Design

**Date:** 2026-09-18

**Status:** Approved umbrella architecture; Stage 1 mixed-device TinyNet stabilization validated, Stage 2 subdesign proposed

This document defines the target architecture, boundaries, delivery order, and acceptance criteria. It is not a single implementation specification. Stages that introduce a wire contract, artifact format, dataset format, generic native training, tokenization/LoRA, accelerator qualification, or cryptography require the focused subdesigns listed under **Implementation Readiness** before their implementation plans are written.

## Goal

Make Android a first-class federated-learning client with the same protocol and data-modality coverage as the laptop client. Android must support DeComFL and first-order federation, image/vector/text tasks, LoRA sequence classification and causal-language-model training, and the training arms exposed by the recipe catalog. Participation is capability-negotiated: a device supports a particular run only after its hardware, runtime, model artifact, and local dataset pass explicit checks.

Android is the first mobile target. iOS native-runtime work is outside this design.

## Product Contract

An Android client must be able to join the same mixed-device federation as laptop clients when the device can execute that run safely and correctly. “Supported” means all of the following:

1. The run's algorithm, model, task, local data, and security requirements are understood by the client.
2. A compatible model artifact exists for an execution backend available on the device.
3. The device passes a model-specific qualification test for that artifact and backend.
4. Local data is selected explicitly by the user, validated on-device, and kept in app-private storage.
5. The client produces an update compatible with the laptop wire contract.
6. The client submits at most one accepted update per server round.
7. Unsupported participation is refused before training with a precise reason.

This contract does not promise that every Android phone can train every model. Large transformer and LoRA runs may be rejected for insufficient memory, storage, operator coverage, runtime, or thermal headroom. That is an honest capability result, not missing protocol support.

Keeping raw examples on the device is an important privacy boundary, but it does not by itself make model updates private. First-order gradients or weights can leak information about local examples. The client and run UI must state which protections are active: transport encryption, authentication, central differential privacy, DeComFL LightSecAgg, or none beyond transport security. Server-side central DP limits information in the released aggregate/model but does not hide an individual update from the server; the UI must distinguish those threat models. Android must not describe a run as securely aggregated when the selected update protocol does not provide that property.

## Current State and Defects

The existing Android path already provides a React Native application, a C++ TurboModule core, ExecuTorch execution, gRPC communication, DeComFL, a first-order weight-upload path, model-bundle delivery, server TLS, enrollment tokens, heartbeats, and device metrics.

The current path is not at parity:

- Production bundle staging is effectively limited to the TinyNet fixture; the general exporter is not recipe-driven.
- The first-order trainer implements basic SGD but not the full client behavior needed for FedProx, recipe-specific optimizers, LoRA, or causal language modeling.
- Text tokenization and user-owned local dataset ingestion do not exist on Android.
- Secure aggregation is refused by Android even though the protocol supports LightSecAgg for DeComFL scalar updates.
- The app decides eligibility mostly from static memory tiers rather than a model-specific execution test.
- Backend-specific acceleration is not safely selected. Repository measurements show that an inference-fast delegated graph can produce incorrect multi-step training.
- The round loop immediately trains again after a successful upload, causing duplicate work while the server remains on the same round.
- React Native DevTools-only logging made a live round-3 failure undiagnosable after the fact.
- A stale native APK can execute current Metro JavaScript without a compatibility handshake.
- A child FL server that crashes after the fixed startup probe can leave its run reported as `RUNNING`.

The vivo 1805 live run also exposed two desktop TinyNet defects. Their existing uncommitted, test-first fixes are prerequisites and must be preserved: the trainable-subset wire rule and the TinyNet vector-data loader must remain shared and consistent between server and clients.

## Architectural Direction

The existing React Native, C++, gRPC, and ExecuTorch architecture remains. It is generalized into a recipe-independent Android participant rather than replaced with an embedded Python runtime.

The design has seven cooperating units:

1. **Execution contract** — an explicit, versioned description of the run and its client obligations.
2. **Local dataset store** — Android file-picker import, validation, normalization, and app-private persistence.
3. **Artifact service** — recipe-driven export and staging of portable and accelerator-specific model variants.
4. **Capability and qualification engine** — hardware discovery plus model/backend correctness and resource checks.
5. **Generic native trainer** — common model state, optimizer, objective, and round execution with strategy adapters.
6. **Round coordinator** — durable, idempotent participation state across retries, reconnects, and app restarts.
7. **Lifecycle and diagnostics** — persistent privacy-safe client logs and authoritative backend process/run status.

These units are deliberately independent. Accelerator support does not gate portable CPU training, and LightSecAgg is a separate security workstream rather than part of hardware acceleration.

## Execution Contract

The backend run manifest gains a versioned `executionContract`. The canonical schema is owned by the backend API and mirrored into generated Java, Python, TypeScript, and C++ bindings; clients do not maintain handwritten interpretations of it. The contract is authoritative and contains:

- `contractVersion` and minimum client protocol version;
- stable enum identifiers for recipe, training arm, task, update protocol, objective, optimizer, preprocessing operators, and security modes;
- the base model and immutable model revision;
- optimizer hyperparameters, local epochs/steps, gradient clipping, and the FedProx coefficient when applicable;
- ordered trainable parameter names, shapes, dtypes, and a hash identifying the frozen state;
- input modality, label schema, input schema, declarative preprocessing graph, and tokenizer artifact references;
- explicit privacy and security requirements, including transport, authentication, central-DP configuration when server-side DP is enabled, and whether DeComFL LightSecAgg is required;
- available artifact variants with backend, ABI, hash, byte size, operator set, and declared resource envelope;
- round deadline, contribution identity, and retry/idempotency policy.

The canonical schema defines required fields, defaults, numeric ranges, enum extension rules, and rejection behavior for unknown required values. Generation must fail when a recipe cannot be represented without client-side inference. The backend validates the contract when a run is created; the Android client validates it before downloading an artifact or opening local data.

The contract replaces behavior inferred from recipe names or scattered booleans. Rollout uses an explicit compatibility window:

1. The backend emits both the existing fields and execution contract v1.
2. Updated laptop and Android clients prefer and validate v1 while old laptop clients continue using the legacy fields.
3. Cross-client conformance tests prove that both representations select the same behavior.
4. A later protocol-version change makes v1 mandatory and removes legacy-field interpretation.

An Android client never guesses a missing or newer mandatory contract into a shape it understands. It reports the unsupported contract version and the minimum application version required.

Only declarative, allowlisted transforms are accepted. A manifest cannot deliver Python, JavaScript, native code, or an arbitrary preprocessing expression for execution on the phone.

## Local Dataset Import

The user selects a file, archive, or small directory through Android's Storage Access Framework. The application requests access only to the selected content, then validates and copies accepted data into app-private storage. A single package/archive is the preferred transport for large image collections because per-file document-provider access can be prohibitively slow.

The canonical import package is:

```text
dataset.json
records.jsonl
files/
```

`dataset.json` declares a schema version, modality, feature/input schema, label schema, record count, and referenced files. `records.jsonl` contains text or vector records and labels, or metadata referencing image files under `files/`. Kotlin owns Storage Access Framework I/O and normalizes CSV, JSONL, and image packages into this representation. Native C++ receives validated normalized batches rather than parsing arbitrary external formats.

Every successful import becomes an immutable dataset snapshot with a generated dataset ID, content hash, normalized-format version, byte size, record count, and import timestamp. A run pins one snapshot ID before qualification and keeps using it until that participation attempt ends. Re-importing or editing source content creates a new snapshot instead of mutating an active one.

Validation includes:

- schema and modality compatibility with the execution contract;
- UTF-8 and record-shape validation;
- label type/range and class-map validation;
- image type, dimensions, and decode validation;
- finite numeric values and bounded vector dimensions;
- path traversal and symbolic-link rejection;
- archive entry-count, compression-ratio, and duplicate-path limits;
- compressed and expanded size limits;
- record-count, per-record, and total-storage limits;
- sufficient free storage before copying.

Text tokenization happens on the phone using tokenizer assets declared and hash-verified by the model bundle. Tokens may be cached only in app-private storage. Image transforms and vector normalization likewise run locally. Raw records, source filenames, decoded images, and tokens are never sent to the backend, gRPC server, telemetry, or logs. Deleting an inactive snapshot removes its normalized records and token cache; an active snapshot cannot be deleted until its run is stopped or detached.

## Model and Artifact Pipeline

One recipe-driven exporter consumes the canonical recipe registry and produces the artifacts required by the execution contract. Export and promotion are CI jobs, not runtime backend work. The exporter covers:

- `TINYNET_GOLDEN`;
- `MLP`;
- `CNN`;
- `PNEUMONIA_CNN`;
- `CIFAR_RESNET18` with every declared arm;
- transformer sequence classification;
- LoRA sequence classification;
- LoRA causal-language-model training.

For each compatible recipe/arm/task, export produces:

- portable CPU loss/inference/training graphs;
- ordered trainable-parameter metadata;
- frozen-state identity and initial federated state;
- preprocessing assets and complete tokenizer packages, including vocabulary, merges or tokenizer model, normalization configuration, special-token IDs, padding/truncation policy, maximum sequence length, and package hashes;
- one or more optional accelerator-specific variants;
- deterministic parity fixtures and expected outputs;
- hashes for every file and an authenticated bundle manifest.

Initially, the manifest is delivered through the authenticated, TLS-protected backend and binds every artifact by size and cryptographic hash. A separate artifact-signing key hierarchy is not required for this online-only path. If offline or third-party artifact distribution is later allowed, its security design must add signed manifests, key rotation, and revocation before use.

Export success alone does not make an artifact eligible. Promotion requires:

1. PyTorch-to-ExecuTorch forward parity.
2. First-step gradient and parameter-delta parity.
3. Multi-step training parity, which catches stale delegated weights.
4. Byte-identical Python/C++ safetensors fixtures for the canonical wire representation, plus semantic round-trip compatibility for supported dtypes.
5. Trainable-subset and frozen-state agreement.
6. Android load and execution on at least one representative device.

Large artifacts are streamed directly to app-private files. They are not base64-expanded through the JavaScript bridge. TypeScript asks the native Android download service to fetch an authenticated artifact descriptor; the service writes to a temporary app-private file, enforces the declared size, supports range resumption only when the server validator still matches, verifies the final hash, and atomically promotes the file. Authentication credentials are obtained through the existing application session and are never placed in a query string or diagnostic log. Partial or mismatched files are quarantined and cleaned within a bounded retention period.

## Capability Discovery and Backend Qualification

Static discovery records:

- Android API level and arm64 ABI;
- total/available RAM and app memory class;
- free storage;
- CPU features and thread capacity;
- Vulkan version, GPU identity, and available memory signals;
- available vendor backends such as Qualcomm or MediaTek when built into the application;
- battery, charging, thermal, and connectivity state;
- application, native bridge, ExecuTorch, and artifact versions.

Static discovery narrows candidates but cannot approve training. Each `(device fingerprint, app/native build, runtime version, artifact hash, backend)` combination must pass an on-device qualification probe.

The probe is a bounded preflight operation, not a hidden full training round. The artifact declares the maximum probe input size, step count, peak memory, and expected duration. Android shows progress, permits cancellation, and refuses to begin when the remaining round deadline cannot accommodate the declared probe budget plus a conservative training/upload margin.

The probe verifies:

- artifact and required operators load;
- forward results agree with the portable reference within declared tolerances;
- backward results and parameter updates are finite and correct;
- at least two optimizer steps use updated rather than stale parameters;
- peak memory remains below a safety limit;
- estimated round duration fits the run deadline with margin;
- thermal and battery policy permit training.

Portable CPU is the correctness baseline. GPU/NPU training is an optional optimization and is selected only after qualification. A failed accelerated probe falls back to the next compatible backend, normally portable CPU. Presence of a GPU or NPU never implies training support.

Qualification results are cached and invalidated when any cache-key component changes. Runtime out-of-memory, numerical divergence, or operator failure quarantines that backend/artifact combination until it is requalified after an application or artifact update. A developer-only control may clear the cache or quarantine for testing; production users cannot force an unqualified backend into a federation.

## Training and Strategy Parity

The native trainer exposes a common interface over model state, batches, objectives, optimizers, and update serialization. A round request contains immutable initial model state, the ordered trainable subset, objective configuration, optimizer configuration, local-step budget, and dataset snapshot ID. A round result contains the contribution identity, final trainable state or DeComFL scalar, metrics, and a deterministic serialization descriptor.

Optimizer implementations have explicit state schemas and golden tests against the laptop implementation. The contract defines whether optimizer state is initialized at the beginning of every server round or restored from a prior local checkpoint; Android cannot choose independently. Interrupted local training may resume only from an atomic checkpoint whose contract version, artifact hash, dataset snapshot, initial-round state, and optimizer-state schema all match.

For FedProx, the trainer preserves a read-only anchor of the downloaded global trainable parameters for the entire local round and applies the declared proximal coefficient on every local optimization step. The anchor is neither updated with local weights nor carried into a later server round.

Strategy-specific behavior is explicit:

- **DeComFL:** server-supplied seeds/config, native zeroth-order local computation, scalar upload, and rebuild history.
- **FedAvg:** first-order local training followed by a trainable-state weight upload.
- **FedProx:** the FedAvg path plus the client-side proximal gradient around the round's initial global weights.
- **FedOpt:** ordinary first-order client training; optimizer adaptation remains server-side.
- **Robust:** ordinary first-order client training; robust aggregation remains server-side.
- **Central-DP runs:** Android produces the same client update as the laptop path; server-side clipping, noise, and accounting remain server responsibilities and are disclosed in the contract. Android does not silently add local clipping or noise.
- **LoRA/FFA-LoRA:** only the declared adapter subset participates in training and serialization, preserving canonical ordering.
- **Frozen and one-vs-all arms:** the execution contract selects both trainable subset and objective; frozen state does not drift locally.

The Android optimizer implementation must match the optimizer actually selected by the laptop client for the same recipe and run. SGD, Adam, AdamW, and RMSprop are implemented only when selected by the canonical run configuration. Their specifications include parameter-group ordering, momentum/beta values, epsilon placement, weight-decay semantics, bias correction, numeric dtype, gradient clipping order, step numbering, and state lifecycle.

The client refuses a strategy or optimizer it does not implement exactly. It does not silently approximate FedProx as FedAvg or replace a configured optimizer.

## Secure Aggregation and Transport Security

Android implements the existing LightSecAgg phases for DeComFL scalar updates: X25519 key publication, sealed share distribution, finite-field masking, surviving-set closure, and aggregated-share submission. Native implementation must match the Python field arithmetic, quantization, associated-data, and dropout behavior with cross-language fixtures.

This work uses a maintained Android-compatible cryptographic library for X25519 and authenticated encryption; it does not implement those primitives from scratch. The focused LightSecAgg security design must define key generation/storage, random-number sources, finite-field representation, zeroization boundaries, transcript binding, replay rejection, malformed-share handling, dropout limits, and cross-language vectors before implementation begins. LightSecAgg receives a separate security review and delivery stage from accelerator work.

The current protocol does not provide secure aggregation for model-weight updates. Android must report that limitation accurately and refuse a run that requires a security property the server cannot provide. Full Android parity does not invent a mobile-only weight-masking protocol.

TLS certificate validation, enrollment tokens, run binding, protocol-version checks, and available mTLS credentials apply to every training mode. Secrets and local records are excluded from diagnostics.

## Round Coordination and Idempotency

The TypeScript round coordinator is the single owner of participation state and permits only one active `start`, `join`, or training-loop task at a time. A synchronous in-flight guard closes the window before React state rendering, and the native layer independently rejects a second active round request instead of merely queueing it behind a mutex.

The coordinator persists through a small native app-private state-store interface that provides atomic replace, read, and delete operations. It stores, per run:

- joined run and client identity;
- last downloaded round;
- last locally completed round;
- last submitted round and upload receipt;
- selected artifact/backend qualification;
- reconnect budget and terminal state.

Each contribution has the stable identity `(run ID, client ID, server round, downloaded-state hash)`. The local result is written and fsynced before upload begins; the upload receipt is written atomically before the result becomes eligible for cleanup. Corrupt or partially written coordinator state stops participation and offers a safe reset rather than guessing whether another update should be sent.

After an accepted upload for round `N`, the client does not call a training RPC again until server status reports a round greater than `N`. Reconnect and application restart reload this state before any training decision. A server response for the same or an older round is a wait condition, not permission to retrain.

Retries distinguish safe reads from potentially repeated writes. Upload retries use round/client idempotency already enforced by the server and preserve the same locally completed result until receipt is known. `GetServerStatus.current_round` is the authoritative advancement signal. The session UI counts unique accepted rounds, not native-call completions.

## Diagnostics and Compatibility

The app maintains a bounded, app-private diagnostic journal containing:

- timestamps, run and round identifiers;
- execution stage and structured error code;
- artifact hash and selected backend;
- memory, thermal, battery, timing, and byte counters;
- retry, reconnect, and server-state transitions;
- native exception type and sanitized message.

It never records local examples, raw text, image paths, tokens, model secrets, authentication tokens, private keys, or unbounded exception payloads. The user can view or export the journal from the application.

The existing client-metrics RPC is wired to emit an opt-in, rate-limited operational summary for fleet debugging: hashed device/runtime class, run/round identifiers, stage, structured result code, selected backend, duration buckets, peak-memory bucket, thermal-state transitions, retry counts, and byte counts. It never uploads journal text or a stable cross-organization device identifier. The backend enforces organization/run scope and bounded retention. Disabling telemetry does not disable local diagnostics or federation participation.

The JavaScript and native layers exchange an ABI/build contract at startup. A mismatch blocks training and instructs the user to install a current application build. Current Metro JavaScript running against the July native APK must therefore fail before model provisioning rather than during a later round.

## Backend Process Lifecycle

The existing `ProcessHandle.onExit()` child-process watcher is extended to own run-state reconciliation as well as its current process-map eviction and port cleanup. No polling watchdog or native process integration is added. When a tracked child exits:

- a run already `COMPLETED`, `FAILED`, or `STOPPED` remains unchanged;
- an intentional stop transitions through the existing stop path;
- any other `STARTING` or `RUNNING` run becomes `FAILED`, records an end time, clears process identity/port, and broadcasts the new project status.

This applies regardless of whether the child exits inside or after the startup probe. The startup probe remains useful for synchronous error reporting but is no longer the sole correctness mechanism.

## Error Handling

User-facing failures identify the failing stage and a corrective action:

- dataset incompatible or malformed;
- insufficient storage or memory;
- no compatible artifact or operator set;
- CPU/GPU qualification failed;
- application/native version mismatch;
- model download or integrity failure;
- training numerical/runtime failure;
- upload timeout or authentication rejection;
- server run failed, stopped, completed, or timed out.

Android already samples the platform thermal status. The round coordinator turns that signal into a defined policy: nominal/fair permits work, serious pauses before the next batch and checkpoints when supported, and critical stops local execution and preserves only a resumable checkpoint. Training resumes only after the status remains below serious for a cooldown interval and the remaining round deadline is still sufficient. Battery and temporary-network conditions use similarly bounded pause/retry budgets. Data corruption, contract mismatch, security failure, and repeated deterministic execution failure are terminal for that participation attempt.

## Implementation Readiness

Stage 1 can proceed from this document because it changes existing, understood flows. Later work is intentionally decomposed. Before its implementation plan is approved, each workstream must have a focused design with concrete schemas or interfaces, ownership, migration behavior, failure handling, test fixtures, and acceptance gates:

1. **Execution contract v1:** canonical schema, generated bindings, validation, legacy compatibility window, and removal criteria. See [the focused Stage 2 design](03-android-execution-contract-v1-design.md).
2. **Artifact export and delivery:** exporter command/API, CI promotion workflow, manifest structure, tokenizer package, authenticated streaming interface, cache layout, and cleanup policy.
3. **Local dataset store:** package schema, CSV/JSONL/image normalization rules, snapshot metadata, Kotlin/native batch boundary, quotas, and deletion lifecycle.
4. **Generic native training:** model-state interface, objective interface, optimizer schemas, FedProx anchor, checkpoints, parameter ordering, and safetensors contract.
5. **Text and LoRA:** tokenizer implementation, reference vectors, sequence construction, masking, adapter naming/order, causal-LM labels, and memory controls.
6. **Capability qualification:** bounded probe API, tolerances, resource budgets, cache/quarantine schema, fallback order, deadline calculation, and thermal policy.
7. **Android LightSecAgg:** library choice, key lifecycle, field/quantization compatibility, transcript binding, dropout behavior, threat model, and security review.
8. **Diagnostics and telemetry:** journal schema, redaction rules, metrics consent, server retention, access control, sampling, and deletion.

No single schedule estimate is assigned to the umbrella design. Estimates are produced per focused design after its unknowns are resolved and its acceptance tests are enumerated. This avoids treating tokenization, stateful optimizers, accelerator correctness, and cryptographic protocol work as equivalent checklist items.

## Delivery Stages

### Stage 1: Stabilize the Existing Mixed-Device Path

- Preserve and verify the pending TinyNet desktop wire/data fixes.
- Add the single-flight round guard, durable contribution state, Android persistent diagnostics, and native/JavaScript build compatibility checks.
- Rebuild and install the current Android native layer on the vivo.
- Make the round loop wait for `current_round` advancement after one accepted upload.
- Extend the existing child-process exit callback to reconcile terminal run state.
- Repeat the four-client TinyNet run through three completed rounds.

### Stage 2: Execution Contract v1

- Approve the focused contract design and canonical schema.
- Generate bindings and validators for backend, Python, TypeScript, and C++.
- Emit legacy fields and contract v1 during the compatibility window.
- Prove legacy/v1 behavioral equivalence with laptop clients before Android depends on v1.

### Stage 3: Artifact, Dataset, and Portable CPU Foundation

- Approve the artifact/delivery, dataset-store, and qualification designs.
- Replace base64 bundle staging with authenticated, streamed, resumable file delivery.
- Add Storage Access Framework import and immutable app-private dataset snapshots.
- Implement bounded portable-CPU qualification and capability reporting.
- Prove the complete contract/artifact/data flow with TinyNet before adding recipes.

### Stage 4: Image and Vector Recipe Parity

- Approve the generic native-training design.
- Add MLP, CNN, pneumonia CNN, and ResNet-18 exporters and preprocessing.
- Add declared training arms, objectives, stateful optimizers, checkpoints, and FedProx.
- Validate DeComFL, FedAvg, FedProx, FedOpt, and Robust mixed-device runs.

### Stage 5: Text Classification

- Approve the text/tokenizer portions of the text and LoRA design.
- Add private on-device tokenization and text dataset import.
- Add transformer sequence classification with Python/Android tokenizer and training fixtures.
- Enforce resource and deadline rejection on devices that cannot safely run the model.

### Stage 6: LoRA Sequence and Causal-LM Training

- Add LoRA and FFA-LoRA adapter selection and canonical serialization.
- Add sequence-classification and causal-language-model objectives.
- Validate optimizer state, masking, label construction, parameter ordering, checkpoints, and memory limits against laptop fixtures.

### Stage 7: Accelerator Qualification

- Build and publish eligible GPU/vendor artifact variants independently of portable CPU artifacts.
- Add on-device qualification cache, ordered fallback, production quarantine, and developer reset behavior.
- Promote a backend only after forward, gradient, multi-step update, resource, deadline, and thermal checks pass on representative hardware.

### Stage 8: Android LightSecAgg

- Approve the dedicated security design and library choice.
- Implement the existing DeComFL LightSecAgg phases without custom cryptographic primitives.
- Cross-validate normal, dropout, replay, malformed-share, and recovery cases against Python fixtures and mixed clients.

### Stage 9: Operational Hardening

- Validate resumable downloads, restarts, reconnects, concurrent start attempts, background operation, low battery, thermal pause/resume, malformed imports, and long runs.
- Wire consented fleet metrics and verify retention, redaction, and organization isolation.
- Exercise mixed fleets using the vivo as the minimum Android device and Mac, g14, and Orin as heterogeneous peers.
- Produce a release build with the same native/runtime versions used by qualification.

Each stage receives its own implementation plan and test-first review. A later stage does not weaken an earlier stage's gates. Portable CPU parity is deliverable before acceleration, and first-order parity is deliverable before LightSecAgg.

## Verification Strategy

### Unit and Component Tests

- Schema conformance tests prove that generated Java, Python, TypeScript, and C++ bindings accept and reject the same execution contracts.
- TypeScript tests cover the single-flight guard, capability decisions, round idempotency, atomic retry/rejoin state, telemetry consent, and user-facing failures.
- Kotlin/Android tests cover file/archive selection, safe streaming and atomic promotion, URI permission handling, dataset snapshots, quotas, device discovery, and foreground/background lifecycle.
- C++ tests cover preprocessing, tokenization boundaries, objectives, optimizer state/lifecycle, checkpoints, FedProx anchors, LoRA parameter ordering, LightSecAgg, and serialization.
- Python tests cover recipe-driven export, manifest generation, tokenizer fixtures, deterministic parity fixtures, and artifact promotion refusal.
- Java tests cover contract generation/delivery, capability negotiation, artifact selection, telemetry retention/isolation, and child-process exit reconciliation.

### Cross-Runtime Parity Tests

For every promoted recipe/arm/task/backend combination:

1. Freeze an initial federated state and local batch.
2. Execute the laptop reference and Android implementation.
3. Compare preprocessing/tokenization output, loss, gradients where available, optimizer state, parameter deltas, and serialized update names/shapes/bytes.
4. Repeat for multiple local steps to detect stale or incorrectly ordered state.
5. Run a multi-round server trajectory with Android and laptop clients together.

Tolerance is declared per artifact and dtype. Tests fail on missing/unexpected trainable tensors, non-finite values, incorrect frozen-state changes, optimizer substitution, or a non-canonical safetensors representation. Existing byte-identical Python/C++ safetensors golden tests remain mandatory release gates.

### Live Acceptance

The full target is accepted only when:

- the vivo completes a three-round mixed-client TinyNet run without duplicate training/uploads;
- each supported non-text recipe completes at least one mixed Android/laptop run for every applicable strategy and arm;
- text classification and both LoRA task types complete representative mixed runs on a capable Android device, or are rejected on the vivo for a verified capability reason;
- DeComFL secure aggregation completes with an Android participant and a dropout case;
- CPU and any enabled accelerator produce qualified multi-step results;
- restart/reconnect resumes without repeating an accepted round;
- concurrent start attempts result in exactly one active training loop;
- a post-probe FL-server crash changes the run from `RUNNING` to `FAILED`;
- the run UI correctly distinguishes transport protection, central DP, DeComFL LightSecAgg, and unaggregated first-order updates;
- packet/log/telemetry inspection finds no raw imported records, stable cross-organization device identifier, or credentials;
- all repository test, lint, build, proto-mirror, and Android native gates pass.

## Non-Goals

- Completing the iOS ExecuTorch migration.
- Guaranteeing that the vivo or every Android phone can run every large model.
- Uploading or centrally partitioning production user data.
- Automatically reading messages, photo libraries, health stores, cameras, or sensors.
- Downloading executable preprocessing code.
- Claiming secure aggregation for weight updates before the server protocol implements it.
- Adding a new client-side differential-privacy algorithm; Android follows the existing server-side DP contract until a separately designed local-DP mode exists.
- Replacing the server's FedOpt or robust aggregation logic on the client.

## Principal Risks and Controls

- **Operator/export gaps:** detect during export and Android qualification; do not publish unusable variants.
- **Accelerator training correctness:** require multi-step parity, not timing-only benchmarks; fall back to portable CPU.
- **Large-model resource pressure:** stream artifacts, measure peak memory, enforce margins, and reject honestly.
- **Tokenizer/preprocessing drift:** version and hash assets; cross-check Android output against Python fixtures.
- **Parameter-order drift:** carry canonical names/shapes and fail on missing, unexpected, or reordered tensors.
- **Optimizer semantic drift:** specify exact equations, state lifecycle, step numbering, and reference trajectories instead of matching optimizer names alone.
- **Concurrent execution:** enforce single-flight ownership in TypeScript and independently reject overlapping native round requests.
- **Privacy regression:** use explicit file selection, app-private storage, truthful update-leakage disclosures, allowlisted transforms, metrics consent/retention, and redaction tests.
- **Dataset mutation:** pin immutable content-addressed snapshots for each participation attempt.
- **Cryptographic implementation risk:** use maintained primitives, a dedicated threat model, cross-language vectors, malformed-input tests, and separate security review.
- **Scope size:** deliver through staged plans with independent acceptance gates instead of a single cross-cutting change.
