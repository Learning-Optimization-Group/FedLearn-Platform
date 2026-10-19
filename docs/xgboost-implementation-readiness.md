# XGBoost implementation readiness

## Current status — 18 October 2026

Collaborative platform training is implemented for the pinned CPU binary presets
`xgboost_binary` (four features) and `xgboost_breast_cancer` (30 ordered features).
The packaged Mac client and Linux client completed ten real breast-cancer rounds
with exact contribution/aggregate replay, API inference and fixed pooled controls.
See the [qualification report](../research/notes/2026-10-10-breast-cancer-xgboost.md).
Trees are grown from a shared parent and appended in canonical client order;
this is parallel tree aggregation, not shared-histogram or cyclic training.

Android tree training, arbitrary schemas, multiclass and deployed TLS remain
outside that qualification. No Android XGBoost support follows from CPU desktop
success. The [current Android guide](../wikis/mobile/15-current-android-runbook.md)
records the separate mobile scope.

## Historical readiness assessment — 7 October 2026

The remainder is the original pre-integration assessment. Its then-open decisions
and implementation status describe 7 October, not the current platform.

Status on 7 October: a standalone CPU probe is implemented and verified on the local feature
branch. Collaborative platform training, production dependencies and Android
qualification remain unimplemented.

## Confirmed objective and open decisions

Add an XGBoost model that clients train collaboratively through the platform.
A centralized comparison alone does not meet this objective.

The following decisions remain open:

| Decision | Choices | Why it matters |
| --- | --- | --- |
| Dataset and task | ECG or another dataset; classification or regression | Determines feature schema, labels, objective, evaluation and partitioning |
| Client scope | Desktop/Docker; Android training as well | Determines runtime packaging, native bridge and device qualification work |
| Federation method | Parallel tree aggregation, cyclic boosting, or shared-tree histogram training | Determines coordinator behavior, update payloads and aggregation |
| Data arrangement | Horizontal partitions or vertically split features | Determines whether clients share a feature schema or need entity alignment |
| Security requirements | Authenticated transport, additional privacy protection, or a specified published method | Determines what clients may expose and whether cryptographic coordination is required |

Horizontal tabular training is a candidate starting scope, not an established
requirement. The professor's requested method or paper may change this scope.

## Work that can proceed before those decisions

1. Map integration points and identify assumptions tied to neural networks.
2. Specify dataset metadata: ordered feature names, types, missing-value rules,
   target mapping and schema fingerprint. Define train/validation/test separation
   before choosing the final dataset.
3. Draft versioned model-artifact metadata and contribution identity. Record the
   runtime/version, model format, objective, base prediction, feature schema,
   parent model digest and run identity explicitly.
4. Define acceptance tests using a small synthetic tabular fixture. Such a fixture
   checks execution and interoperability; it is not evidence of performance on
   the eventual research task.
5. Define experiment outputs and reproducibility records: configuration, partition
   assignments, artifact hashes, seeds, software versions, tree counts, metrics,
   wall time and actual bytes transferred.

## Completed CPU foundations

`framework/benchmarks/xgboost_cpu/` now provides an isolated XGBoost 3.4.1 probe
with separate pinned requirements and 16 passing real-runtime tests. It uses a
synthetic binary task with four ordered features and disjoint 256/64/128-row
training/validation/test partitions. Both native JSON and UBJSON checkpoints
preserve predictions exactly; eight saved rounds plus four additional rounds
match twelve uninterrupted rounds exactly under the pinned CPU configuration.

The report records versions, settings, split identities/hashes, class counts,
artifact hashes/bytes and metrics. Feature-order and checksum refusal are tested;
NaN missing values and incompatible numeric inputs are covered. This local
checkpoint validation does not establish an untrusted model-ingestion boundary.

The retained seed-42 run is in
`research/results/xgboost_cpu/2026-10-07-seed42/`, with interpretation in
`research/notes/2026-10-07-xgboost-cpu-probe.md`. Its synthetic accuracy is
execution evidence only. It does not establish platform or Android support.

## Integration map grounded in the current code

Paths below are relative to the repository root.

| Area | Current integration point | Required design work |
| --- | --- | --- |
| Recipe discovery | `fl-runtime/recipes.py`; backend `ModelRecipeService` and `ModelRecipeDto` | Describe XGBoost's capabilities and tree settings without advertising neural optimizers or unsupported strategies |
| Model initialization | `fl-runtime/init_model.py` | Current flow builds a model and extracts a tensor state dictionary; define initial booster/base prediction handling |
| Client training | `fl-runtime/client.py` | Add a native XGBoost library adapter for Python clients with explicit dataset schema and training settings |
| Strategy interface | `framework/src/fedlearn/server/strategy.py` | Current interface exposes tensor dictionaries; define typed tree/histogram updates without weakening existing validation |
| Strategy selection | `framework/src/fedlearn/server/strategy_factory.py`; `fl-runtime/fl_server.py` | Add the selected tree strategy and reject incompatible model/strategy pairs |
| Round coordination | `framework/src/fedlearn/server/coordinator.py` | Define participant selection, contribution identity, idempotency, missing-client behavior and method-specific progress |
| Upload/download | `framework/src/fedlearn/server/grpc_servicer.py`; `proto/fedlearn/v2/fedlearn.proto` | Current model streams use float32 safetensors; define a separately validated tree-model codec or histogram protocol |
| Execution contract | `proto/fedlearn/contract/v1/execution_contract.proto`; `fl-runtime/execution_plan.py`; Python/Java/TypeScript readers | Current workloads describe neural optimizers and fixed tensor layouts; design the versioned tree workload and fail-closed reader behavior |
| Evaluation and inference | `fl-runtime/fl_server.py`; `fl-runtime/infer.py` | Current loading/evaluation uses tensor state and neural models; add booster prediction, task metrics and registry reload checks |
| Artifact registry | `fl-runtime/fl_server.py` artifact registration and backend model loading | Define the tree artifact kind, codec and provenance; validate save, reload, inference and continued runs |
| Android | `mobile_client/bridge/`; `mobile_client/shared/`; capability and qualification gates | If required, add a separate XGBoost CPU runtime adapter and bridge; measure memory, latency and numerical agreement |

The strategy registry is only one integration point. Registering a new name or
adding a catalog row would not make these execution paths work.

## Draft contract and artifact requirements

These are requirements to review, not proposed protobuf field numbers or an
approved wire schema. The existing contract states that new mandatory behavior
requires a new contract version; resolve versioning before changing canonical
protos, regenerate readers and enforce mirror checks.

Common metadata should cover:

- Run, project, contract and participant identity, with required runtime support.
- Dataset schema fingerprint, ordered features, target encoding and missing-value
  semantics. A matching feature count alone is insufficient.
- Objective and class mapping, plus the initial base prediction. Independently
  inferred client intercepts must not silently disagree.
- Pinned XGBoost version, booster type, seed, threading policy and selected model
  format. JSON/UBJSON are candidate formats; select and test one explicitly.
- Explicit training configuration and global size/time budgets. The meaning of
  local work depends on the selected federation method.
- Parent model digest, payload digest, declared byte size and round identity for
  tree-model contributions. Histogram contributions need their own step/split
  identity and numerical validation.
- Security policy, interruption behavior, supported devices and qualification.

Method-specific decisions must remain separate:

| Method | Additional behavior to specify |
| --- | --- |
| Parallel tree aggregation | New-tree extraction, common parent state, tree/group indexing, client ordering, learning-rate scaling and ensemble growth limits |
| Cyclic boosting | Client order and selection, full-model continuation, retries, client dropout and successor handoff |
| Histogram collaboration | Shared feature bins, gradient/Hessian semantics, reduction operations, split coordination, participant synchronization and privacy protection |

Do not pass independently grown trees into FedAvg, FedOpt, FedProx, robust tensor
aggregation or DeComFL's perturbation path. A new method requires its own
compatibility declaration. Raw data remaining local does not establish privacy
protection for exposed trees or training statistics.

## Acceptance criteria to define now

These platform acceptance checks remain planned. The local CPU subset completed
above is separate from platform admission, coordination and device qualification.

| Check | Evidence required |
| --- | --- |
| Model serialization | Saving and reloading preserves predictions and continuation behavior under the pinned configuration |
| Artifact integrity | Truncated, oversized, hash-mismatched and wrong-format payloads are refused before model loading |
| Dataset compatibility | Permuted features, conflicting types, labels or missing-value semantics are refused |
| Participation identity | Wrong run, contract, parent model or round is refused; duplicate contributions cannot be applied twice |
| Compatibility | Existing clients explicitly refuse the new workload instead of interpreting it as safetensors or DeComFL scalars |
| Research separation | A fixed held-out split is excluded from client training; patient/entity grouping is used when relevant and available |
| Experiment reproducibility | Inputs, partition assignments, settings and artifact digests are retained; numerical tolerances are derived for the chosen runtime/method |

After selecting the method, add its specific oracle: tree aggregation must retain
the existing ensemble and include each valid new contribution exactly once;
cyclic training must reproduce the defined client order; histogram training must
verify reductions and split decisions against a small reference fixture.

The first platform demonstration must involve at least two real client processes
on separate local partitions, coordinated through the platform. Save the final
artifact, reload it through the registry and run inference. A single-process
experiment is an algorithm check, not proof of end-to-end integration.

Use local-only and authorized pooled-data XGBoost as comparison controls where
appropriate, with matched partitions and clearly stated work/tree budgets. Report
task-appropriate metrics, per-client results, model growth and resource costs.
Do not use a centralized comparison as the completion claim for federation.

## Work that should wait for the open decisions

- Shipping an XGBoost catalog entry, strategy selector, or production dependency.
- Editing canonical protos or publishing a new execution contract.
- Implementing tree merging, cyclic scheduling, histogram collectives or crypto.
- Selecting the final dataset, class mapping, task objective and experimental claim.
- Claiming Android support or starting its product integration without runtime
  feasibility and qualification evidence.

## Primary references

- [XGBoost model IO](https://xgboost.readthedocs.io/en/stable/tutorials/saving_model.html): model formats, model configuration and serialization boundaries.
- [XGBoost C API](https://xgboost.readthedocs.io/en/stable/tutorials/c_api_tutorial.html): native training and prediction API reference.
- [Flower XGBoost example](https://flower.ai/docs/examples/xgboost-comprehensive.html): reference behavior for parallel tree aggregation and cyclic training; using this reference does not require adopting its runtime.
- [NVFLARE federated XGBoost guide](https://nvflare.readthedocs.io/en/main/user_guide/data_scientist_guide/federated_xgboost/secure_xgboost_user_guide.html): horizontal/vertical methods and privacy considerations.
- [Upstream federated plugin](https://github.com/dmlc/xgboost/tree/master/plugin/federated): current documented Linux-only support; this does not certify an Android build of the base library.
