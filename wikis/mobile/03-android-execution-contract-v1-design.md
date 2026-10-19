# Execution Contract v1 — Focused Design

**Date:** 2026-09-19
**Status:** Implementation authorized 2026-09-21. The schema, generated Python/Java/TypeScript readers and shared validation rules are implemented and verified (Stage 2A); publication, client consumption and live conformance are pending. Progress: [implementation plan](06-execution-contract-v1-implementation-plan.md).

This is the Stage 2 subdesign for the [Android parity architecture](02-android-federated-learning-parity-design.md). It defines what a participant must execute, not whether a particular phone can execute it. A device still needs an artifact, local dataset, and successful qualification before joining a training round.

## Scope and authority

Version 1 is first emitted for `TINYNET_GOLDEN` vector training on the mixed-device FedAvg path. Its purpose is to establish the schema, publication lifecycle, generated readers, and laptop/Android conformance gate. It does **not** claim support for every recipe merely because an enum names that recipe. Other strategies and modalities receive contracts only after their runtime behavior and artifacts can be represented and tested without inference. There is no fallback from a rejected v1 contract to legacy behavior in an updated client.

The canonical schema will live at `proto/fedlearn/contract/v1/execution_contract.proto`. It is independent of the existing `fedlearn.v2` gRPC protocol; adding this file does not change a round RPC. Buf generates Java, Python, TypeScript, and C++ bindings. Java constructs and stores the contract, Python validates laptop behavior against it, TypeScript handles enrollment and user-facing refusal, and C++ receives only a validated, typed projection needed for native execution. No independent handwritten contract interface or recipe-name switch may become another source of truth.

The existing `/api/runs/{runId}/manifest` and enrollment response remain available during the compatibility window. `RunManifestDto` gains a `contractState` and, only when ready, an `executionContract` ProtoJSON object and opaque `contractId`. Generated ProtoJSON serializers/parsers are used at this boundary. `EnrollmentDto.manifest` remains the enclosing location; tokens and partition assignments stay outside the run-level contract.

## Normative v1 schema

The canonical schema is now `proto/fedlearn/contract/v1/execution_contract.proto`, and it supersedes the listing below. It differs in four ways: behavioral scalars whose zero is legal carry explicit presence; `LocalTraining` adds `batch_size`, `drop_last` and `batch_order`; a `ContractIssueCode` enum names refusals; and the exact rules and limits are specified with the shared conformance fixtures (`framework/tests/fixtures/execution_contract_v1/`). The listing is kept as the reviewed proposal.

The following is the proposed proto surface, with field numbers fixed when the schema is implemented. All enums reserve zero for `UNSPECIFIED`. The contract has no protobuf `map` and no arbitrary JSON/`Any` field: order-sensitive data stays in repeated fields. ProtoJSON's lowerCamelCase names are the REST names. Int64 values in ProtoJSON use the protobuf JSON string representation; clients must use generated parsing rather than JavaScript `number` coercion.

```proto
syntax = "proto3";
package fedlearn.contract.v1;

message ExecutionContract {
  uint32 contract_version = 1;             // exactly 1
  uint32 min_client_protocol_version = 2;  // minimum understood control/round protocol
  string run_id = 3;                      // UUID
  string project_id = 4;                  // UUID
  Recipe recipe = 5;
  Strategy strategy = 6;
  uint32 num_rounds = 7;
  uint32 clients_per_round = 8;
  Partitioning partitioning = 9;
  optional uint64 seed = 10;
  RoundPolicy round = 11;
  SecurityPolicy security = 12;
  oneof workload {
    ModelTraining model_training = 13;
    // Reserved for a separately specified FoT contract, not emitted in v1.
  }
}

enum Recipe {
  RECIPE_UNSPECIFIED = 0;
  RECIPE_TINYNET_GOLDEN = 1;
  RECIPE_MLP = 2;
  RECIPE_CNN = 3;
  RECIPE_PNEUMONIA_CNN = 4;
  RECIPE_CIFAR_RESNET18 = 5;
  RECIPE_TRANSFORMER = 6;
  RECIPE_LLM_LORA = 7;
}
enum Strategy {
  STRATEGY_UNSPECIFIED = 0;
  STRATEGY_DECOMFL = 1;
  STRATEGY_FEDAVG = 2;
  STRATEGY_FEDPROX = 3;
  STRATEGY_FEDOPT = 4;
  STRATEGY_ROBUST = 5;
}
enum Partitioning { PARTITIONING_UNSPECIFIED = 0; PARTITIONING_SHARDED = 1; PARTITIONING_LOCAL = 2; }
enum Arm { ARM_UNSPECIFIED = 0; ARM_FULL = 1; ARM_FROZEN_HEAD = 2; ARM_OVA_LP = 3; }
enum Task { TASK_UNSPECIFIED = 0; TASK_VECTOR_CLASSIFICATION = 1;
  TASK_IMAGE_CLASSIFICATION = 2; TASK_SEQUENCE_CLASSIFICATION = 3; TASK_CAUSAL_LM = 4; }
enum Objective { OBJECTIVE_UNSPECIFIED = 0; OBJECTIVE_CROSS_ENTROPY = 1;
  OBJECTIVE_ONE_VS_ALL = 2; OBJECTIVE_CAUSAL_LM = 3; }
enum UpdateProtocol { UPDATE_UNSPECIFIED = 0; UPDATE_DECOMFL_SCALAR = 1;
  UPDATE_TRAINABLE_STATE_F32 = 2; }
enum Transport { TRANSPORT_UNSPECIFIED = 0; TRANSPORT_TLS_REQUIRED = 1;
  TRANSPORT_PLAINTEXT_DEV = 2; }
enum ClientAuth { CLIENT_AUTH_UNSPECIFIED = 0; CLIENT_AUTH_CONNECTION_TOKEN = 1;
  CLIENT_AUTH_DISABLED_DEV = 2; }
enum SecureAggregation { SECAGG_UNSPECIFIED = 0; SECAGG_NONE = 1;
  SECAGG_LIGHTSECAGG_SCALAR = 2; }
enum ArtifactBackend { BACKEND_UNSPECIFIED = 0; BACKEND_EXECUTORCH_CPU = 1;
  BACKEND_EXECUTORCH_GPU = 2; BACKEND_EXECUTORCH_VENDOR = 3; }
enum DType { DTYPE_UNSPECIFIED = 0; DTYPE_F32 = 1; }

message RoundPolicy {
  uint64 timeout_ms = 1;                 // duration; live absolute deadline is server status
  bool one_accepted_update_per_round = 2;
  uint32 max_transient_retries = 3;
  uint64 retry_backoff_ms = 4;
}
message SecurityPolicy {
  Transport transport = 1;
  ClientAuth client_auth = 2;
  SecureAggregation secure_aggregation = 3;
  optional uint32 secure_agg_threshold = 4;
  CentralDp central_dp = 5;             // message presence; absent means no central DP
}
message CentralDp {
  double target_epsilon = 1;
  double delta = 2;
  double clip_norm = 3;
  // Server-side aggregation only; never implies local DP or secrecy from server.
}
message ModelTraining {
  string model_id = 1;                   // stable model identifier
  string model_revision = 2;             // immutable revision, not mutable project name
  Arm arm = 3;
  Task task = 4;
  Objective objective = 5;
  UpdateProtocol update_protocol = 6;
  repeated TensorSpec trainable = 7;     // canonical ordered named_parameters(requires_grad)
  string frozen_state_sha256 = 8;        // digest of canonical frozen-state artifact
  string initial_state_sha256 = 9;       // digest of initial federated-state artifact
  LocalTraining local_training = 10;
  DataRequirement data = 11;
  repeated ArtifactVariant artifacts = 12;
  optional double fedprox_mu = 13;       // required iff strategy is FEDPROX
}
message TensorSpec { string name = 1; repeated uint64 shape = 2; DType dtype = 3; }
message LocalTraining {
  uint32 local_epochs = 1;
  optional uint32 max_local_steps = 2;  // absent means every eligible batch per epoch
  optional double gradient_clip_norm = 3;
  oneof optimizer {
    Sgd sgd = 4;
    Adam adam = 5;
    AdamW adamw = 6;
    Rmsprop rmsprop = 7;
  }
  bool reset_optimizer_each_round = 8;
}
message Sgd { double learning_rate = 1; double momentum = 2; double dampening = 3;
  double weight_decay = 4; bool nesterov = 5; }
message Adam { double learning_rate = 1; double beta1 = 2; double beta2 = 3;
  double epsilon = 4; double weight_decay = 5; bool amsgrad = 6; }
message AdamW { double learning_rate = 1; double beta1 = 2; double beta2 = 3;
  double epsilon = 4; double weight_decay = 5; bool amsgrad = 6; }
message Rmsprop { double learning_rate = 1; double alpha = 2; double epsilon = 3;
  double weight_decay = 4; double momentum = 5; bool centered = 6; }
message DataRequirement {
  Task task = 1;
  repeated uint64 input_shape = 2;      // shape of one sample; no batch dimension
  DType input_dtype = 3;
  uint32 class_count = 4;               // classification tasks only
  string label_schema_id = 5;           // immutable, hash-bound schema identifier
  repeated Transform transforms = 6;   // in execution order; no executable payload
  ArtifactRef tokenizer = 7;           // message presence; absent for TinyNet
}
message Transform {
  oneof operation {
    IdentityVector identity_vector = 1;
    // Further typed operations require their own semantic specification and
    // contract-version review before any non-TinyNet recipe is published.
  }
}
message IdentityVector { uint32 width = 1; }
message ArtifactRef { string relative_path = 1; string sha256 = 2; uint64 byte_size = 3; }
message ArtifactVariant {
  string variant_id = 1;
  ArtifactBackend backend = 2;
  string abi = 3;
  repeated ArtifactRef files = 4;
  repeated string required_operators = 5; // exact runtime operator IDs, not a wildcard
  uint64 declared_peak_memory_bytes = 6;
  uint64 declared_storage_bytes = 7;
  uint64 declared_probe_ms = 8;
  uint64 declared_train_ms = 9;
}
```

This is an intentionally closed v1 transform vocabulary. New recipes must not repurpose `IdentityVector` or reinterpret an existing field. Image operations, tokenization parameters, FoT traces, and LoRA adapter semantics require their focused designs and a contract-version decision before emission. Their enum values are registry identifiers, **not** a claim that v1 can execute them. The artifact and dataset subdesigns may add typed descriptors by a reviewed v2 (or compatible optional v1 addition with no change to existing semantics).

## Required values and validation

There are no implicit defaults for behavioral fields. Java must reject a candidate before publication; Python and Android must reject it before downloading a model or opening local data. A valid protobuf parse alone is insufficient. All finite floating-point values are checked explicitly (including NaN and infinity); all hashes are lowercase 64-character SHA-256 hex; all artifact paths are relative, normalized, traversal-free, and unique within a variant.

| Area | Required v1 rule |
| --- | --- |
| Identity and version | `contract_version == 1`; minimum client protocol is positive and supported; UUIDs parse; run/project IDs match enrollment; `num_rounds`, `clients_per_round`, timeout, and retry backoff are positive; retry count is bounded by a server policy. |
| Registry | No `UNSPECIFIED` or unknown numeric enum value. Recipe/arm/task/objective/update/strategy combination must exist in the approved compatibility matrix. Unknown fields may be ignored only if they are non-behavioral; any new mandatory behavior requires a version bump. |
| TinyNet emission | `TINYNET_GOLDEN`, `FULL`, vector classification, cross-entropy, `FEDAVG`, trainable-state F32, exactly the ordered 25 trainable scalars, one typed `IdentityVector` with the exact input width, explicit class count and label schema. A staged, verified CPU trainable artifact is mandatory. |
| Parameters | Names are nonempty and unique; shapes have positive extents; their product and total count fit configured limits; wire dtype is F32. Layout, initial state, frozen state, artifact metadata, and Python recipe inventory agree. Ordered names are the canonical Python `named_parameters()` trainable order, even if a transport map arrives in another order. |
| Training | Epochs positive; optional max steps positive; learning rate positive; momentum/weight decay nonnegative; beta and alpha values strictly between 0 and 1; epsilon positive; optional clip norm positive. All optimizer values, including defaults resolved from the actual Python client, are explicit. Optimizer state reset/restore rule is explicit. FedProx coefficient is present and nonnegative only for FedProx; absent otherwise. |
| Security | Effective TLS/auth deployment settings, not desired project settings, determine enums. Central DP parameters are present exactly when server-side central DP is active; epsilon, delta, and clip norm are validated against backend policy. LightSecAgg requires DeComFL scalar updates and a valid threshold; weight updates cannot claim LightSecAgg. A dev plaintext run is never labelled TLS-protected. |
| Artifacts and data | At least one compatible variant; unique variant IDs; every referenced file has hash and nonzero size; artifact backends/ABIs/operators are explicit. Dataset shape, dtype, class count, labels, and transforms are complete. No script, URL to arbitrary code, unknown transform, or server-supplied executable expression is accepted. |

The schema is declarative, not a replacement for an optimizer or training specification. In particular, `fl-runtime/client.py` currently selects Adam or AdamW in its training path even though `fl-runtime/recipes.py` advertises several permissible optimizers. For TinyNet specifically, the recipe advertises SGD and Android's first-order path runs SGD, while the present Python training path selects Adam. A completed mixed-device round is therefore **not** optimizer parity. Stage 2 must make the actual TinyNet laptop and Android optimizer/step budgets agree (with test-first changes to the execution paths), then publish only the resolved values. The publisher reads those values from the same run configuration both clients consume; it must not copy a catalog default or encode the present mismatch as a valid contract. If that resolution cannot be established, v1 stays `UNAVAILABLE`. Likewise, a staged bundle's `trainablePtePath` alone is insufficient proof of trainability: its names, sizes, graph capability, and hashes must pass validation.

## Run lifecycle and immutability

`Run` currently snapshots strategy and some scheduling/security settings, while `Project` retains mutable model, optimizer, task, arm, and DP fields. Stage 2 adds an immutable run-intent snapshot at start, including all project-derived training/security values and effective deployment settings. The server process and contract publisher consume that snapshot. Updating the project must not change an active run's contract or its actual training configuration.

Contract publication has four explicit states:

| State | Meaning | API behavior |
| --- | --- | --- |
| `PENDING` | Run intent is captured; asynchronous artifact staging/validation is not complete. | Manifest/enrollment retain legacy fields and report `contractState=PENDING`; `executionContract` and `contractId` absent. New v1 clients wait with bounded polling; no training on an inferred contract. |
| `READY` | One complete contract and its referenced artifact manifest were validated and committed atomically. | Same immutable contract is returned on every read and in enrollment. `contractId` is SHA-256 of the stored canonical protobuf bytes, used as an opaque identity; clients do not reserialize ProtoJSON to check it. Individual artifacts are independently hash-verified on download. |
| `UNAVAILABLE` | Staging failed, timed out, or some required fact cannot be represented. | No partial contract is returned. A machine-readable reason explains refusal. Existing legacy desktop clients may proceed during the compatibility window if their path is safe; v1-dependent Android does not. |
| `LEGACY_ONLY` | Historical run predating the intent snapshot, or a run deliberately left on the old protocol during rollout. | No v1 contract is fabricated. Updated clients that require v1 refuse with a migration-specific reason. |

A failed attempt cannot change a published `READY` contract; artifact deletion or corruption after publication makes provisioning fail and surfaces an integrity error, rather than silently replacing the contract. Run terminal state remains the existing run-status API, orthogonal to contract readiness.

Absolute round deadlines, assigned round, enrollment token, per-user partition ID, and gRPC endpoint are mutable or participant-specific and therefore stay in status/enrollment/round messages, not this run-level immutable object. `RoundPolicy.timeout_ms` is a duration; it does not authorize a client to train past the live server deadline. Contribution identity is `(runId, enrolled partitionId, server round, update protocol)`. Reconnection preserves that identity, and the client records one accepted upload for it. The server remains authoritative for duplicate rejection. Retry policy applies only to transient failures and never means re-train/re-upload after acceptance.

## Migration and ownership

1. Add the canonical proto and generated bindings; CI checks generator versions, schema compatibility, and generated-file drift. The existing `fedlearn.v2` mirrors are unaffected unless a later RPC explicitly imports the new package.
2. Capture run intent and make Python's resolved training plan observable to the publisher. Validate TinyNet fixtures, effective optimizer, parameter order, frozen and initial state digests, CPU artifact, and effective security settings. Fail as `UNAVAILABLE` if any are missing.
3. Atomically publish the contract and `contractId`. `RunManifestDto` dual-emits legacy fields. A server-side equivalence check compares overlapping legacy/v1 decisions before `READY`; disagreement prevents publication and is reported.
4. Laptop clients prefer v1 when `READY`, validate it, and compare their resolved execution plan against it. Old clients continue on legacy fields in the compatibility window. Android first adds generated parsing/refusal and uses v1 only for a ready, qualified TinyNet run; later stages add recipes one by one.
5. Remove legacy interpretation only after production-style mixed-client conformance runs, a measured absence of old clients, and an explicit minimum-protocol increase. No automatic date-based removal.

The backend owns creation, validation, atomic storage, and authorization. Python owns canonical recipe/parameter/optimizer truth and parity fixtures. Android owns pre-download validation, precise refusal codes, and passing a validated projection to C++. Unknown contract version, unknown enum, unsupported strategy/optimizer/security mode, missing artifact, malformed layout, and unavailable publication receive distinct refusal codes. Logs may include run ID, contract ID, recipe, and refusal code, but never enrollment tokens, raw examples, source filenames, or model weights.

## Acceptance gates

- Schema tests cover presence, numeric bounds, unknown enums, unknown version, malformed IDs/hashes/paths, duplicate names, overflow, invalid combinations, and absent-versus-zero semantics in all generated-reader environments.
- Golden ProtoJSON and protobuf fixtures round-trip through Java, Python, TypeScript, and C++ without losing semantic fields. The stored protobuf digest remains stable on repeated server reads; clients treat it as an identity, not a recomputed integrity proof.
- A TinyNet run publishes only after staging and resolution; tests force delayed staging, failure, corrupt files, concurrent publication, project edits after start, backend restart, and artifact disappearance after `READY`.
- Legacy and v1 interpretations select the same TinyNet strategy, trainable order, objective, actual optimizer/hyperparameters and step budget, security disclosures, and update protocol. Tests demonstrate the previously mismatched Python Adam/Android SGD execution is aligned before `READY`; they fail if either path disagrees with the resolved run plan.
- Updated clients reject `PENDING`, `UNAVAILABLE`, unknown versions/enums, and mismatched security before model/data access. Old desktop clients continue to operate on a dual-emitting run.
- Mixed-device three-round TinyNet FedAvg still completes with one accepted update per participant per round. This does not certify other recipes, text, DeComFL security, or GPU training.

## Explicit later design seams

The artifact/delivery design defines promotion, authenticated streaming, resource-envelope measurement, and tokenizer package bytes. The dataset design defines `label_schema_id`, import normalization, and local snapshot binding. The generic-training design defines optimizer math/state and how native tensor ordering is enforced. The text/LoRA design adds typed transforms and adapter/task semantics. The qualification design decides whether an advertised artifact variant is actually executable on a device. The LightSecAgg design defines scalar masking; no weight-update secure aggregation is implied. FoT has a separate trace/insight protocol and requires its own workload schema; it must not masquerade as a model-weight update.
