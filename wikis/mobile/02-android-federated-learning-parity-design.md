# Android Federated-Learning Parity Design

**Date:** 2026-09-18

**Status:** Approved for implementation planning

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

## Execution Contract

The backend run manifest gains a versioned `executionContract`. It is authoritative and contains:

- contract and protocol versions;
- recipe key, base model, training arm, and task type;
- update protocol: DeComFL scalar update or first-order model update;
- loss/objective, optimizer, learning rate, local epochs/steps, gradient clipping, and FedProx coefficient;
- ordered trainable parameter names, shapes, dtypes, and frozen-state identity;
- input modality, label schema, input schema, preprocessing pipeline, and tokenizer specification;
- security requirements, including TLS, authentication, and DeComFL secure aggregation;
- available artifact variants with backend, ABI, hash, size, operator-set, and resource-envelope metadata;
- round deadline and client retry/idempotency information.

The contract replaces behavior inferred from recipe names or scattered booleans. Old manifests are rejected with an upgrade message once the new Android path becomes mandatory; they are not guessed into the new shape.

Only declarative, allowlisted transforms are accepted. A manifest cannot deliver Python, JavaScript, native code, or an arbitrary preprocessing expression for execution on the phone.

## Local Dataset Import

The user selects a file or directory through Android's Storage Access Framework. The application requests access only to the selected content, then validates and copies accepted data into app-private storage.

The canonical import package is:

```text
dataset.json
records.jsonl
files/
```

`dataset.json` declares a schema version, modality, feature/input schema, label schema, record count, and referenced files. `records.jsonl` contains text or vector records and labels, or metadata referencing image files under `files/`. CSV files and selected image directories are normalized into this internal representation during import.

Validation includes:

- schema and modality compatibility with the execution contract;
- UTF-8 and record-shape validation;
- label type/range and class-map validation;
- image type, dimensions, and decode validation;
- finite numeric values and bounded vector dimensions;
- path traversal and symbolic-link rejection;
- compressed and expanded size limits;
- record-count, per-record, and total-storage limits;
- sufficient free storage before copying.

Text tokenization happens on the phone using tokenizer assets declared and hash-verified by the model bundle. Tokens may be cached only in app-private storage. Image transforms and vector normalization likewise run locally. Raw records, source filenames, decoded images, and tokens are never sent to the backend, gRPC server, telemetry, or logs.

## Model and Artifact Pipeline

One recipe-driven exporter consumes the canonical recipe registry and produces the artifacts required by the execution contract. It covers:

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
- preprocessing and tokenizer assets;
- one or more optional accelerator-specific variants;
- deterministic parity fixtures and expected outputs;
- hashes for every file and a signed or authenticated bundle manifest.

Export success alone does not make an artifact eligible. Promotion requires:

1. PyTorch-to-ExecuTorch forward parity.
2. First-step gradient and parameter-delta parity.
3. Multi-step training parity, which catches stale delegated weights.
4. Safetensors download/upload round-trip compatibility.
5. Trainable-subset and frozen-state agreement.
6. Android load and execution on at least one representative device.

Large artifacts are streamed directly to app-private files. They are not base64-expanded through the JavaScript bridge. Downloads support resumption, size bounds, hash verification before load, and safe cleanup after failure.

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

The probe verifies:

- artifact and required operators load;
- forward results agree with the portable reference within declared tolerances;
- backward results and parameter updates are finite and correct;
- at least two optimizer steps use updated rather than stale parameters;
- peak memory remains below a safety limit;
- estimated round duration fits the run deadline with margin;
- thermal and battery policy permit training.

Portable CPU is the correctness baseline. GPU/NPU training is an optional optimization and is selected only after qualification. A failed accelerated probe falls back to the next compatible backend, normally portable CPU. Presence of a GPU or NPU never implies training support.

Qualification results are cached and invalidated when any cache-key component changes. Runtime out-of-memory, numerical divergence, or operator failure quarantines that backend/artifact combination until it is requalified after an application or artifact update.

## Training and Strategy Parity

The native trainer exposes a common interface over model state, batches, objectives, optimizers, and update serialization. Strategy-specific behavior is explicit:

- **DeComFL:** server-supplied seeds/config, native zeroth-order local computation, scalar upload, and rebuild history.
- **FedAvg:** first-order local training followed by a trainable-state weight upload.
- **FedProx:** the FedAvg path plus the client-side proximal gradient around the round's initial global weights.
- **FedOpt:** ordinary first-order client training; optimizer adaptation remains server-side.
- **Robust:** ordinary first-order client training; robust aggregation remains server-side.
- **LoRA/FFA-LoRA:** only the declared adapter subset participates in training and serialization, preserving canonical ordering.
- **Frozen and one-vs-all arms:** the execution contract selects both trainable subset and objective; frozen state does not drift locally.

The Android optimizer implementation must match the optimizer actually selected by the laptop client for the same recipe and run. SGD, Adam, AdamW, and RMSprop are implemented only when selected by the canonical run configuration, with state persisted correctly across local steps and reset/preserved across rounds according to the laptop contract.

The client refuses a strategy or optimizer it does not implement exactly. It does not silently approximate FedProx as FedAvg or replace a configured optimizer.

## Secure Aggregation and Transport Security

Android implements the existing LightSecAgg phases for DeComFL scalar updates: X25519 key publication, sealed share distribution, finite-field masking, surviving-set closure, and aggregated-share submission. Native implementation must match the Python field arithmetic, quantization, associated-data, and dropout behavior with cross-language fixtures.

The current protocol does not provide secure aggregation for model-weight updates. Android must report that limitation accurately and refuse a run that requires a security property the server cannot provide. Full Android parity does not invent a mobile-only weight-masking protocol.

TLS certificate validation, enrollment tokens, run binding, protocol-version checks, and available mTLS credentials apply to every training mode. Secrets and local records are excluded from diagnostics.

## Round Coordination and Idempotency

The client persists, per run:

- joined run and client identity;
- last downloaded round;
- last locally completed round;
- last submitted round and upload receipt;
- selected artifact/backend qualification;
- reconnect budget and terminal state.

After an accepted upload for round `N`, the client does not call a training RPC again until server status reports a round greater than `N`. Reconnect and application restart reload this state before any training decision. A server response for the same or an older round is a wait condition, not permission to retrain.

Retries distinguish safe reads from potentially repeated writes. Upload retries use round/client idempotency already enforced by the server and preserve the same locally completed result until receipt is known. The session UI counts unique accepted rounds, not native-call completions.

## Diagnostics and Compatibility

The app maintains a bounded, app-private diagnostic journal containing:

- timestamps, run and round identifiers;
- execution stage and structured error code;
- artifact hash and selected backend;
- memory, thermal, battery, timing, and byte counters;
- retry, reconnect, and server-state transitions;
- native exception type and sanitized message.

It never records local examples, raw text, image paths, tokens, model secrets, authentication tokens, or private keys. The user can view or export the journal from the application.

The JavaScript and native layers exchange an ABI/build contract at startup. A mismatch blocks training and instructs the user to install a current application build. Current Metro JavaScript running against the July native APK must therefore fail before model provisioning rather than during a later round.

## Backend Process Lifecycle

The FL child-process exit watcher owns run-state reconciliation as well as port cleanup. When a tracked child exits:

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

Thermal, battery, and temporary-network conditions pause or retry within bounded policy. Data corruption, contract mismatch, security failure, and repeated deterministic execution failure are terminal for that participation attempt.

## Delivery Stages

### Stage 1: Stabilize the Existing Mixed-Device Path

- Preserve and verify the pending TinyNet desktop wire/data fixes.
- Add Android persistent diagnostics and native/JavaScript build compatibility checks.
- Rebuild and install the current Android native layer on the vivo.
- Make the round loop wait for server round advancement after one accepted upload.
- Reconcile child-process exits into terminal run state.
- Repeat the four-client TinyNet run through three completed rounds.

### Stage 2: Shared Contract, Local Data, and CPU Qualification

- Introduce the execution contract and compatibility validation.
- Add Storage Access Framework import and app-private dataset storage.
- Replace base64 bundle staging with streamed, resumable file delivery.
- Implement portable CPU artifact qualification and capability reporting.
- Prove the new flow with TinyNet before adding recipes.

### Stage 3: Image and Vector Recipe Parity

- Add MLP, CNN, pneumonia CNN, and ResNet-18 exporters and preprocessing.
- Add declared training arms and objectives.
- Add first-order optimizer and FedProx parity.
- Validate DeComFL, FedAvg, FedProx, FedOpt, and Robust mixed-device runs.

### Stage 4: Text and LoRA Parity

- Add private on-device tokenization and text dataset import.
- Add transformer sequence classification.
- Add LoRA sequence classification and causal-language-model training.
- Enforce capability rejection for devices that cannot safely run a selected model.

### Stage 5: Acceleration and Secure Aggregation

- Build and publish eligible CPU/GPU/vendor artifacts.
- Add on-device qualification, cache, fallback, and quarantine behavior.
- Implement and cross-validate Android LightSecAgg for DeComFL.

### Stage 6: Operational Hardening

- Validate resumable downloads, restarts, reconnects, background operation, low battery, thermal throttling, malformed imports, and long runs.
- Exercise mixed fleets using the vivo as the minimum Android device and Mac, g14, and Orin as heterogeneous peers.
- Produce a release build with the same native/runtime versions used by qualification.

Each stage receives its own implementation plan and test-first review. A later stage does not weaken an earlier stage's gates.

## Verification Strategy

### Unit and Component Tests

- TypeScript tests for contract parsing, capability decisions, round idempotency, retry/rejoin state, dataset metadata, and user-facing failures.
- Kotlin/Android tests for file selection, safe copying, URI permission handling, device discovery, and foreground/background lifecycle.
- C++ tests for preprocessing, tokenization boundaries, objectives, optimizers, FedProx, LoRA parameter ordering, LightSecAgg, serialization, and durable round state.
- Python tests for recipe-driven export, manifest generation, deterministic fixtures, and cross-runtime parity.
- Java tests for manifest delivery, capability negotiation, artifact selection, and child-process exit reconciliation.

### Cross-Runtime Parity Tests

For every promoted recipe/arm/task/backend combination:

1. Freeze an initial federated state and local batch.
2. Execute the laptop reference and Android implementation.
3. Compare loss, gradients where available, parameter deltas, and serialized update names/shapes.
4. Repeat for multiple local steps to detect stale or incorrectly ordered state.
5. Run a multi-round server trajectory with Android and laptop clients together.

Tolerance is declared per artifact and dtype. Tests fail on missing/unexpected trainable tensors, non-finite values, incorrect frozen-state changes, or optimizer substitution.

### Live Acceptance

The full target is accepted only when:

- the vivo completes a three-round mixed-client TinyNet run without duplicate training/uploads;
- each supported non-text recipe completes at least one mixed Android/laptop run for every applicable strategy and arm;
- text classification and both LoRA task types complete representative mixed runs on a capable Android device, or are rejected on the vivo for a verified capability reason;
- DeComFL secure aggregation completes with an Android participant and a dropout case;
- CPU and any enabled accelerator produce qualified multi-step results;
- restart/reconnect resumes without repeating an accepted round;
- a post-probe FL-server crash changes the run from `RUNNING` to `FAILED`;
- packet/log inspection finds no raw imported records or credentials;
- all repository test, lint, build, proto-mirror, and Android native gates pass.

## Non-Goals

- Completing the iOS ExecuTorch migration.
- Guaranteeing that the vivo or every Android phone can run every large model.
- Uploading or centrally partitioning production user data.
- Automatically reading messages, photo libraries, health stores, cameras, or sensors.
- Downloading executable preprocessing code.
- Claiming secure aggregation for weight updates before the server protocol implements it.
- Replacing the server's FedOpt or robust aggregation logic on the client.

## Principal Risks and Controls

- **Operator/export gaps:** detect during export and Android qualification; do not publish unusable variants.
- **Accelerator training correctness:** require multi-step parity, not timing-only benchmarks; fall back to portable CPU.
- **Large-model resource pressure:** stream artifacts, measure peak memory, enforce margins, and reject honestly.
- **Tokenizer/preprocessing drift:** version and hash assets; cross-check Android output against Python fixtures.
- **Parameter-order drift:** carry canonical names/shapes and fail on missing, unexpected, or reordered tensors.
- **Privacy regression:** use explicit file selection, app-private storage, allowlisted transforms, and diagnostics redaction tests.
- **Scope size:** deliver through staged plans with independent acceptance gates instead of a single cross-cutting change.
