# Execution contract v1 fixtures and validation rules

The schema is `proto/fedlearn/contract/v1/execution_contract.proto`; the design is
`wikis/mobile/03-android-execution-contract-v1-design.md`. This directory holds the cross-language
fixtures and the exact v1 validation rules that every reader (Python, Java, TypeScript) implements.
`generate.py` produces every file here; never edit them by hand.

| File | Contents |
| --- | --- |
| `golden_tinynet_fedavg.binpb` | Canonical protobuf bytes of the TinyNet FedAvg golden contract. |
| `golden_tinynet_fedavg.json` | The same contract as ProtoJSON. |
| `conformance.json` | Valid and invalid inputs with the exact issues each must produce. |

The golden is a schema fixture, not a published contract. `generate.py` says which values are real
committed artifacts and which are fixture values.

## Reader behavior

1. **Parse.** Protobuf bytes use the generated binary parser. ProtoJSON uses the generated parser with
   unknown fields ignored; an unknown enum name therefore reads as value 0. Any parse failure yields
   exactly one issue, `ISSUE_MALFORMED` at the root path.
2. **Version.** If `contract_version` is not 1, the only issue is
   `ISSUE_UNSUPPORTED_CONTRACT_VERSION` at `contractVersion`; nothing else is evaluated.
3. **Rules.** Otherwise every rule below is evaluated and every issue is reported.

An issue is a `ContractIssueCode` and a path. Paths use ProtoJSON (lowerCamelCase) field names joined
by `.`, with `[i]` for a repeated element; an absent oneof is reported at the oneof's name; the root
path is the empty string. Readers compare issue lists as sets. A contract is accepted only when it
has no issues.

"Known" for an enum means a nonzero value that the v1 schema defines. "Finite" excludes NaN and
both infinities. "Absent" applies to `optional` fields and to messages. Unsigned bounds are compared
as unsigned integers.

## Limits and formats

| Name | Value |
| --- | --- |
| `MAX_ROUNDS`, `MAX_CLIENTS_PER_ROUND` | 10000 |
| `MAX_TIMEOUT_MS`, `MAX_DECLARED_MS` | 86400000 |
| `MAX_TRANSIENT_RETRIES` | 10 |
| `MAX_RETRY_BACKOFF_MS` | 3600000 |
| `MAX_TENSORS` | 4096 |
| `MAX_RANK` | 8 |
| `MAX_ELEMENTS` (one extent, one tensor, all trainable tensors, one sample) | 2147483647 |
| `MAX_LOCAL_EPOCHS` | 1000 |
| `MAX_LOCAL_STEPS` | 1000000 |
| `MAX_BATCH_SIZE` | 65536 |
| `MAX_CLASSES` | 1000000 |
| `MAX_TRANSFORMS`, `MAX_VARIANTS` | 16 |
| `MAX_FILES` | 64 |
| `MAX_OPERATORS` | 4096 |
| `MAX_FILE_BYTES` | 68719476736 (2^36) |
| `MAX_DECLARED_BYTES` | 1099511627776 (2^40) |

Every pattern must match the whole string (no partial or multiline match).

| Format | Pattern |
| --- | --- |
| UUID | `[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}` |
| Model ID | `[A-Za-z0-9][A-Za-z0-9._-]{0,127}` |
| Revision / label schema ID | `[A-Za-z0-9][A-Za-z0-9._:-]{0,127}` |
| Variant ID | `[A-Za-z0-9][A-Za-z0-9._-]{0,63}` |
| ABI | `[a-z0-9][a-z0-9_-]{0,31}` |
| Tensor name (at most 256 characters) | `[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)*` |
| SHA-256 | `[0-9a-f]{64}` |
| Relative path | 1 to 255 characters; 1 to 8 `/`-separated segments, each `[A-Za-z0-9_-][A-Za-z0-9._-]*` |
| Operator (at most 128 characters) | `[A-Za-z_][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?` |

A shape is **valid** when it has 1 to `MAX_RANK` extents, each in `[1, MAX_ELEMENTS]`, and their
product is at most `MAX_ELEMENTS`.

## Approved v1 matrix

Only this combination of (recipe, strategy, arm, task, objective, update protocol) is publishable:

| Recipe | Strategy | Arm | Task | Objective | Update protocol |
| --- | --- | --- | --- | --- | --- |
| `RECIPE_TINYNET_GOLDEN` | `STRATEGY_FEDAVG` | `ARM_FULL` | `TASK_VECTOR_CLASSIFICATION` | `OBJECTIVE_CROSS_ENTROPY` | `UPDATE_TRAINABLE_STATE_F32` |

Adding a row requires its runtime behavior to be implemented and tested on every participant type.

## Rules

Each rule reads *condition → code at path*.

### Contract

- `minClientProtocolVersion` is 0 or greater than the reader's protocol version →
  `UNSUPPORTED_CLIENT_PROTOCOL` at `minClientProtocolVersion`.
- `runId` is not a UUID → `INVALID_IDENTIFIER` at `runId`; otherwise, if the reader was given an
  expected run ID and it differs → `IDENTITY_MISMATCH` at `runId`. The same two rules apply to
  `projectId`.
- `recipe`, `strategy` or `partitioning` is not known → `UNKNOWN_ENUM` at that field.
- `numRounds` or `clientsPerRound` is outside `[1, 10000]` → `OUT_OF_RANGE` at that field.
- `round` absent → `MISSING_FIELD` at `round`. `security` absent → `MISSING_FIELD` at `security`.
  `modelTraining` absent → `MISSING_FIELD` at `modelTraining`.
- `modelTraining` is present, all six matrix fields are known, and their combination is not in the
  approved matrix → `UNSUPPORTED_COMBINATION` at the root.

### `round`

- `timeoutMs` outside `[1, MAX_TIMEOUT_MS]` → `OUT_OF_RANGE`.
- `oneAcceptedUpdatePerRound` is not true → `OUT_OF_RANGE`.
- `maxTransientRetries` absent → `MISSING_FIELD`; greater than `MAX_TRANSIENT_RETRIES` →
  `OUT_OF_RANGE`.
- `retryBackoffMs` outside `[1, MAX_RETRY_BACKOFF_MS]` → `OUT_OF_RANGE`.

### `security`

- `transport`, `clientAuth` or `secureAggregation` is not known → `UNKNOWN_ENUM`.
- `secureAggregation` is `SECAGG_LIGHTSECAGG_SCALAR`:
  - `secureAggThreshold` absent → `MISSING_FIELD`; less than 2 or greater than `clientsPerRound` →
    `OUT_OF_RANGE`.
  - `strategy` is known and is not `STRATEGY_DECOMFL` → `INVALID_SECURITY` at
    `security.secureAggregation`.
- `secureAggregation` is `SECAGG_NONE` and `secureAggThreshold` is present → `INVALID_SECURITY` at
  `security.secureAggThreshold`.
- `centralDp` present: `targetEpsilon` not finite and positive, `delta` not finite and strictly
  between 0 and 1, or `clipNorm` not finite and positive → `OUT_OF_RANGE` at that field.

### `modelTraining`

- `modelId` is not a model ID, or `modelRevision` is not a revision → `INVALID_IDENTIFIER`.
- `arm`, `task`, `objective` or `updateProtocol` is not known → `UNKNOWN_ENUM`.
- `frozenStateSha256` or `initialStateSha256` is not a SHA-256 → `INVALID_HASH`.
- `localTraining` absent → `MISSING_FIELD`. `data` absent → `MISSING_FIELD`.
- `strategy` is `STRATEGY_FEDPROX`: `fedproxMu` absent → `MISSING_FIELD`; not finite or negative →
  `OUT_OF_RANGE`. Otherwise, `strategy` is known and `fedproxMu` is present →
  `INVALID_STRATEGY_SETTINGS` at `modelTraining.fedproxMu`.

### `modelTraining.trainable`

- Empty, or more than `MAX_TENSORS` elements → `MALFORMED_LAYOUT` at `modelTraining.trainable`.
  When there are too many elements, the per-tensor rules are skipped.
- For each element `[i]`:
  - `name` is not a tensor name, or equals an earlier element's name → `MALFORMED_LAYOUT` at
    `[i].name`.
  - `shape` is not valid → `MALFORMED_LAYOUT` at `[i].shape`.
  - `dtype` is not known → `UNKNOWN_ENUM` at `[i].dtype`.
- The element counts of all valid shapes sum to more than `MAX_ELEMENTS` → `MALFORMED_LAYOUT` at
  `modelTraining.trainable`.

### `modelTraining.localTraining`

- `localEpochs` outside `[1, MAX_LOCAL_EPOCHS]` → `OUT_OF_RANGE`.
- `maxLocalSteps` present and outside `[1, MAX_LOCAL_STEPS]` → `OUT_OF_RANGE`.
- `gradientClipNorm` present and not finite and positive → `OUT_OF_RANGE`.
- No optimizer → `MISSING_FIELD` at `modelTraining.localTraining.optimizer`.
- `resetOptimizerEachRound` absent → `MISSING_FIELD`. `dropLast` absent → `MISSING_FIELD`.
- `batchSize` outside `[1, MAX_BATCH_SIZE]` → `OUT_OF_RANGE`.
- `batchOrder` is not known → `UNKNOWN_ENUM`.
- `sgd`: `learningRate` not finite and positive → `OUT_OF_RANGE`; `momentum`, `dampening` or
  `weightDecay` absent → `MISSING_FIELD`, or not finite and nonnegative → `OUT_OF_RANGE`;
  `nesterov` absent → `MISSING_FIELD`. When `nesterov` is true and `momentum` and `dampening` are
  present and valid, unless `momentum` is positive and `dampening` is zero → `INVALID_OPTIMIZER` at
  `sgd.nesterov`.
- `adam` and `adamw`: `learningRate` or `epsilon` not finite and positive, or `beta1` or `beta2` not
  finite and strictly between 0 and 1 → `OUT_OF_RANGE`; `weightDecay` absent → `MISSING_FIELD`, or
  not finite and nonnegative → `OUT_OF_RANGE`; `amsgrad` absent → `MISSING_FIELD`.
- `rmsprop`: `learningRate` or `epsilon` not finite and positive, or `alpha` not finite and strictly
  between 0 and 1 → `OUT_OF_RANGE`; `weightDecay` or `momentum` absent → `MISSING_FIELD`, or not
  finite and nonnegative → `OUT_OF_RANGE`; `centered` absent → `MISSING_FIELD`.

### `modelTraining.data`

- `task` is not known → `UNKNOWN_ENUM`; otherwise, `modelTraining.task` is known and differs →
  `INVALID_DATA_REQUIREMENT` at `data.task`.
- `inputShape` is not valid → `INVALID_DATA_REQUIREMENT`.
- `inputDtype` is not known → `UNKNOWN_ENUM`.
- `classCount`: when `data.task` is a classification task (vector, image or sequence), outside
  `[2, MAX_CLASSES]` → `OUT_OF_RANGE`; when it is `TASK_CAUSAL_LM`, nonzero →
  `INVALID_DATA_REQUIREMENT`.
- `labelSchemaId` is not a revision / label schema ID → `INVALID_IDENTIFIER`.
- `transforms` empty, or more than `MAX_TRANSFORMS` elements → `INVALID_DATA_REQUIREMENT` at
  `data.transforms`; when there are too many, the per-transform rules are skipped. For each `[i]`:
  - no operation → `MISSING_FIELD` at `[i].operation`.
  - `identityVector.width` outside `[1, MAX_ELEMENTS]` → `OUT_OF_RANGE`. Otherwise, when
    `data.task` is known and is not `TASK_VECTOR_CLASSIFICATION`, or is
    `TASK_VECTOR_CLASSIFICATION` with a valid `inputShape` that is not exactly `[width]` →
    `INVALID_DATA_REQUIREMENT` at `[i].identityVector.width`.
- `tokenizer`: when `data.task` is `TASK_SEQUENCE_CLASSIFICATION` or `TASK_CAUSAL_LM` and it is
  absent → `MISSING_FIELD`; when `data.task` is another known task and it is present →
  `INVALID_DATA_REQUIREMENT`. When present, the artifact reference rules apply at `data.tokenizer`.

### Artifact references

- `relativePath` is not a relative path → `INVALID_PATH`.
- `sha256` is not a SHA-256 → `INVALID_HASH`.
- `byteSize` outside `[1, MAX_FILE_BYTES]` → `OUT_OF_RANGE`.

### `modelTraining.artifacts`

- Empty → `MISSING_ARTIFACT` at `modelTraining.artifacts`. More than `MAX_VARIANTS` →
  `INVALID_ARTIFACT` there, and the per-variant rules are skipped.
- For each variant `[i]`:
  - `variantId` is not a variant ID → `INVALID_IDENTIFIER`; otherwise, equal to an earlier
    variant's ID → `INVALID_ARTIFACT` at `[i].variantId`.
  - `backend` is not known → `UNKNOWN_ENUM`. `abi` is not an ABI → `INVALID_IDENTIFIER`.
  - `files` empty → `MISSING_ARTIFACT` at `[i].files`; more than `MAX_FILES` → `INVALID_ARTIFACT`
    there, and the per-file rules are skipped. Otherwise the artifact reference rules apply at
    `[i].files[j]`, and a valid path equal, ignoring ASCII case, to an earlier valid path in the
    same variant → `INVALID_PATH` at `[i].files[j].relativePath`.
  - `requiredOperators` empty → `INVALID_ARTIFACT` at `[i].requiredOperators`; more than
    `MAX_OPERATORS` → `INVALID_ARTIFACT` there, and the per-operator rules are skipped. Otherwise an
    element that is not an operator, or equals an earlier element → `INVALID_ARTIFACT` at
    `[i].requiredOperators[k]`.
  - `declaredPeakMemoryBytes` or `declaredStorageBytes` outside `[1, MAX_DECLARED_BYTES]`, or
    `declaredProbeMs` or `declaredTrainMs` outside `[1, MAX_DECLARED_MS]` → `OUT_OF_RANGE`.
  - `declaredStorageBytes` is in range, `files` has 1 to `MAX_FILES` elements whose `byteSize`
    values are all in range, and their sum exceeds `declaredStorageBytes` → `INVALID_ARTIFACT` at
    `[i].declaredStorageBytes`.
- There are 1 to `MAX_VARIANTS` variants and none has backend `BACKEND_EXECUTORCH_CPU` →
  `MISSING_ARTIFACT` at `modelTraining.artifacts`.

Codes above omit the `ISSUE_` prefix of their `ContractIssueCode` names. Paths below
`modelTraining` are written relative to their section; `conformance.json` spells every path in
full.
