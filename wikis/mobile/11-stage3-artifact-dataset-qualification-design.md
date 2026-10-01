# Stage 3 — Artifact Delivery, On-Device Datasets and Portable-CPU Qualification: Focused Design

**Status:** Approved (2026-09-27, Anurag), with the recommendation taken on every decision: (1) fixture data only for
runs the contract marks as fixture runs, refused by release builds; (2) a new reproducible
`BATCH_ORDER_SEEDED_PERMUTATION_V1`; (3) CSV and the dataset package; (4) an exporter-recorded `probeLoss`.
Implementation follows the slices below.

**Goal:** make the phone train on data the user owns, delivered and verified the way a real model needs, and only on
hardware that has proven it can. At the end of Stage 3, a TinyNet run still passes the Stage 2 conformance checks,
but the phone's training data comes from an imported on-device dataset instead of the server. Model files arrive as
streamed, hash-verified files instead of base64 through the JavaScript bridge. The phone has qualified its CPU
backend for the run's artifact before training. New recipes are Stage 4 and are out of scope here.

## Current state (measured in the code, 2026-09-27)

| Area | Today | Why it blocks Stage 4 |
| --- | --- | --- |
| Training data | The **server** serves it. `GET /api/runs/{id}/model-bundle` returns URLs for `inputs.f32` and `targets.i64` (the committed TinyNet batch), and `modelProvisioning.ts` downloads and stages them. | Real federated learning trains on the device's own data, and none exists on the phone. Every live run so far trained on the same 8 server-supplied rows. |
| Artifact transfer | `fetchAndStage` downloads each file into JS memory, base64-encodes it (`arrayBufferToBase64`), and passes it across JSI to `stageBundleFile`, which hashes and writes it. The comment at `modelProvisioning.ts:130` already flags this as MVP-only. | A 45 MB ResNet-18 `.pte` becomes ~60 MB of base64 on the JS heap, plus copies. No resumption, no size cap before download, no streaming hash. |
| File serving | `RunController#bundleFile` returns a plain `ResponseEntity<Resource>` from a whitelist. | Range/resume and validator (ETag) behaviour are unspecified and untested. |
| Local training shape | The native trainer takes **one whole-dataset step per epoch**. `projectContract` accepts `dropLast=false` with no `maxLocalSteps`. Its comment says the dataset size is checked against `batchSize` when data is staged, but **no such check exists**: `projection.batchSize` is never read after the gate. That is harmless today only because the server always supplies exactly 8 rows at batch 8. | With imported data, a dataset larger than one batch would silently train whole-dataset steps that differ from the contract. Stage 3 must add the check (slice C0) and a real minibatch loop. |
| Capability / qualification | `DeviceState.kt` samples thermal and battery state. There is no qualification probe, and "tier" is a static label from `loadModel`. | 02 requires a bounded probe per (device, build, runtime, artifact, backend) before training. |
| Kotlin layer | `FlForegroundService`, `FlServiceModule` (start/stop), `DeviceState`, `FedLearnNative.setDataDir`. No Storage Access Framework or download code. | Both SAF import and the download service belong in Kotlin (02). |

## Scope

In scope: **(A)** streamed artifact delivery, **(B)** on-device dataset import and immutable snapshots, **(C)**
native minibatching with a specified batch order, **(D)** portable-CPU qualification and a capability report, and
**(E)** a TinyNet end-to-end proof of all four.

Not in scope: new recipes, image/text modalities beyond what TinyNet needs (vector), accelerator backends, and
tokenizers. These are Stage 4–7. The design keeps every interface modality-agnostic, so Stage 4 adds producers and
transforms, not new plumbing.

## A. Streamed artifact delivery

**Flow.** TypeScript asks Kotlin to fetch one artifact descriptor: URL path, expected size and expected SHA-256 from
the contract's `ArtifactVariant.files`, which are already hash- and size-bound. The Kotlin `ArtifactDownloader`:

1. refuses a declared size above the device's free-storage budget before opening a connection;
2. streams to `files/artifacts/tmp/<sha256>.part`, hashing incrementally and aborting as soon as the byte count passes the declared size;
3. resumes with `Range: bytes=<n>-` plus `If-Range: <validator>` only when a validator from the first response is held; otherwise it restarts from zero;
4. on completion verifies size and hash, then promotes atomically (`rename`) to `files/artifacts/<sha256>` (content-addressed, so a re-run with the same artifact re-uses it);
5. on mismatch moves the file to `quarantine/`, deleted after 24 h, and reports `ARTIFACT_HASH_MISMATCH`.

**Auth.** The session cookie must reach the download without appearing in a URL or log. The intended mechanism is
React Native's OkHttp client (`OkHttpClientProvider`, present in `react-android-0.80.0`), with the app's cookie store
attached explicitly: `JavaNetCookieJar(ForwardingCookieHandler(ctx))`, backed by the app-wide `CookieManager` where
the REST session cookie lives. **The provider's client alone carries no cookies.** React Native's networking module
attaches its jar to a client it builds for itself, so the first live A3 run got `ARTIFACT_HTTP_403` on every download,
until the jar was attached. The app ships OkHttp 4.9.2.

**Backend.** `bundleFile` must answer `Range` with `206`, and send a strong `ETag` (the file's SHA-256) and
`Content-Length`. Spring may already support byte ranges for `Resource` bodies; *slice A2 pins it with a test either
way*. Stage 3 serves only the files the contract lists, by hash, not by name.

**JS/native contract.** `stageBundleFile(base64)` is removed once all callers move. The native side takes paths only
(`loadModel(path, sha)` already verifies again).

## B. On-device dataset import and snapshots

**Import.** The user picks a file through the Storage Access Framework (`ACTION_OPEN_DOCUMENT`, read-only, no
persistable grant needed for a one-shot import). Kotlin copies it to `files/datasets/import-<uuid>/` and validates
it there, never in place. Stage 3 accepts two vector formats:

- **CSV**: header row; `label` column plus N numeric feature columns;
- **Package**: 02's canonical `dataset.json` + `records.jsonl`, needed anyway for images and text in Stage 4.

Both are normalised into the snapshot format below. Native C++ never parses a user file.

**Snapshot format** (immutable, `files/datasets/<snapshot_id>/`):

```text
snapshot.json   {schemaVersion, snapshotId, modality, inputShape, inputDtype, classNames[], labelSchemaId,
                 recordCount, inputsSha256, targetsSha256, createdAt}
inputs.f32      row-major float32, recordCount x prod(inputShape)
targets.i64     int64 class indices
```

`snapshotId` is the SHA-256 of the canonical JSON of the snapshot's content fields (everything but the id; no
timestamp), encoded exactly as Python's `json.dumps(sort_keys=True, separators=(",", ":"))`, so a Python check can
recompute it. Importing identical content therefore yields the same snapshot: import is idempotent, and still never
mutates a snapshot. `labelSchemaId` uses **the same definition as the
contract**: `"labels-sha256:" + sha256(JSON(classNames))` (`execution_plan.label_schema_id`), so compatibility is a
string comparison. The native trainer already reads exactly `inputs.f32` + `targets.i64`, so it needs no change to
consume a snapshot.

**Validation** (reject the whole import with a named reason): the 02 list, restricted to what vector data needs.
That covers UTF-8, the header, a consistent column count, finite numbers, labels present in the declared class list,
a record-count floor of 1 and a ceiling, a total-size ceiling, and free space for 2× the expanded size. Archive rules
(entry count, compression ratio, path traversal) apply once packages carry files in Stage 4.

**Binding to a run.** At join, the phone checks the chosen snapshot against the contract's `DataRequirement`: same
`input_shape`, `input_dtype`, `class_count` and `label_schema_id`. It pins that `snapshotId` for the participation
attempt. A mismatch refuses with `DATASET_INCOMPATIBLE` and names the field. Nothing about the snapshot is sent to the
server except `num_examples`, which the update already carries.

**Lifecycle.** Snapshots are listed in a new *Data* screen and can be deleted when not pinned. Deleting removes the
directory. Re-importing identical content yields the same snapshot; a snapshot is never mutated.

## C. Native minibatching and batch order

**C0, first and independent of the rest:** refuse to train when the staged dataset has more records than the
contract's `batchSize`, until C1/C2 land. This is the missing check noted above, and a one-line guard with a test.

Real snapshots exceed one batch, so the trainer must run `ceil(n / batchSize)` steps per epoch (`dropLast=false`).
Which order? The schema already answers part of this. `BatchOrder` has `SEQUENTIAL` (reproducible) and
`SHUFFLED_EACH_EPOCH`, documented as *"local to the participant and not reproducible across runtimes."* That was a
deliberate Stage 2 choice. It costs one thing: with a multi-batch shuffled epoch, a replay cannot reproduce a client's
update exactly, because the permutation is unknown. The laptop today uses
`DataLoader(shuffle=True, generator=manual_seed(partition_id))`, torch's Mersenne-Twister `randperm`.

Options (decision 2 below):

- **Keep `SHUFFLED_EACH_EPOCH` as specified.** The phone shuffles with any good RNG. Verification of multi-batch
  runs falls back to per-client statistical checks and the single-batch goldens. No proto change.
- **Add `BATCH_ORDER_SEEDED_PERMUTATION_V1`:** a specified, language-neutral permutation (Fisher–Yates over a
  SplitMix64 stream seeded with `(run seed, round, epoch)`), implemented in Python, C++ and TS and pinned by a
  shared golden. Every client's update then stays exactly replayable. This is an additive enum value (allowed by
  `buf breaking`) with corpus cases.

Single-batch TinyNet is unaffected either way: every order of one batch visits the same rows.

## D. Portable-CPU qualification and capability report

**Capability report**, sent with enrollment and shown in diagnostics: API level, ABI, total/available RAM, memory
class, free storage, CPU core count, app/native/ExecuTorch versions and the thermal/battery state `DeviceState`
already samples. It is informational: static facts never approve training (02).

**Probe.** Before the first round, run the contract's own trainable graph for **two** steps on a synthetic batch shaped
by `DataRequirement` (zeros, then a fixed pattern), from the run's initial state. Check that:

1. the artifact loads and every required operator resolves;
2. step-1 loss is finite and matches a `probeLoss` the exporter records in the artifact manifest (portable-CPU reference, declared tolerance);
3. step-2 parameters differ from step-1 (no stale weights);
4. peak RSS stays under the artifact's `declared_peak_memory_bytes` and wall time is under `declared_probe_ms` (both already in `ArtifactVariant`);
5. projected round time, `declared_train_ms × steps`, fits the round deadline with a margin.

The result is cached under `(device fingerprint, app build, ExecuTorch version, artifact sha256, backend=cpu)` and
invalidated when any part changes. A failure refuses with `QUALIFICATION_FAILED` and the failing check, and
quarantines that key until the app or artifact changes.

`probeLoss` is new manifest metadata produced by the exporter. It is not a contract field, because the contract
already binds the artifact by hash.

## E. TinyNet end-to-end proof

A TinyNet run in which the phone trains an **imported** snapshot, not server files:

- the backend stops serving `inputs.f32`/`targets.i64` for contract runs (see decision 1);
- the phone imports a CSV of TinyNet-shaped data, qualifies CPU, downloads the `.pte` files through A, and trains;
- the laptops train their own local data;
- the acceptance replay uses the recorder (`FEDLEARN_RECORD_CLIENT_UPDATES`) to compare each client's update with a
  CPU reference computed on **that client's** snapshot. The phone's snapshot is exported for the test only, from a
  debug build.

## Decisions needed before implementation

1. **Where do laptop and demo data come from once the server stops serving the TinyNet batch?** Options:
   (a) the server keeps serving it only for runs explicitly marked as fixture/demo runs, recorded in the contract,
   so a real run can never receive server data; (b) every client, including the demo, imports its data (simplest
   and most honest, but the demo needs a data file per device); (c) keep today's behaviour until Stage 4.
   *Recommendation: (a)*. The contract would gain `DataRequirement.source = LOCAL_SNAPSHOT | FIXTURE`, and
   the phone refuses `FIXTURE` in release builds.
2. **Batch order (section C).** Keep the documented non-reproducible `SHUFFLED_EACH_EPOCH`, or add a reproducible
   `SEEDED_PERMUTATION_V1`. Replicating torch's `randperm` in C++ is not proposed: it would tie the protocol to one
   library's internals. *Recommendation: add the seeded permutation.* The replay-against-contract checks were
   the most useful verification in Stages 2 and 3 (they caught the zero-start DeComFL bug and the FedProx
   vacuity), and a non-reproducible order would give them up for every real dataset.
3. **Dataset formats for Stage 3.** CSV + package (proposed), or package only.
4. **Where the probe's reference comes from.** An exporter-recorded `probeLoss` (proposed), or a probe that only
   checks finiteness and change, which is weaker but needs no exporter change.

## Slices (each test-first, each its own commit)

Progress: A1 `4eff087`, A2 `fa77ef2`, A3 `b7f4d96`, B1 `0e361b4`. Decision 1: the schema is `bf55731`
(`DataRequirement.source`, `DataSource` = `LOCAL_SNAPSHOT` | `FIXTURE`), and the reader rules and resolver come in the
next commit. TinyNet states `FIXTURE`. The phone trains a fixture run only in a development build, and refuses a
`LOCAL_SNAPSHOT` run until B2 binds a snapshot at join. A3 was checked live: a one-client TinyNet FedAvg run on
the vivo completed, with the three contract programs stored as `files/artifacts/<sha256>` at the contract's exact sizes.

Later progress:
- B2: `74b5b0f` (import), `e40296e` (training on a snapshot), `30b5c69` (choosing a dataset before Start, and the
  dataset list in Settings).
- The run intent's data source: `6396750`, plus `218fb90` (the web start dialog, and recipes that declare it).
- E: two live phone-only runs on 2026-09-28, each with a bit-exact replay of the saved model on the imported CSV.
  The first run exposed static-shape programs: a 6-example snapshot failed with ExecuTorch `NotSupported`.
  - Fixed by `8d9f933`: staged programs take 1..8 examples.
  - `fe10d4e`: that error is no longer retried.
  - `86fa1f8`: a contract whose `batch_size` exceeds the programs' `maxBatch` is not published.
- C1: `84528a2` adds `BATCH_ORDER_SEEDED_PERMUTATION_V1`, in Python and C++ with a shared golden. Decision 2 took
  the recommendation.
- C2: `d6dae1d`. The native first-order round trains seeded minibatches, and first-order own-data plans state the
  order. `70fda73` evaluates a multi-batch dataset in chunks: the first live minibatch run found whole-dataset
  evaluation failing after every round. Live, a 20-example snapshot replays bit-exactly.
- D2: the probe core is `ba51358`, wired end to end in `1edf80e` (stager, bundle DTO, bridge ABI 3, a per-device
  cache, and `QUALIFICATION_FAILED`). Decision 4 took the exporter-recorded reference: `probeLoss` became two step
  losses. It was checked live: "Model qualified on this device (probe 7 ms)".
- Found along the way: `1ac1e89` (server status reports each round's real deadline) and `6fbbe50` (the privacy
  label is true for own-data and DeComFL runs).
- D1: `4559249`. A device sends a bounded capability report with enrollment (V31, `run_enrollments.capability_report`),
  informational only, and the shared diagnostics end with it. Checked live, it exposed that thermal state was never
  sampled before training ("NOMINAL" was a default): fixed in `afbc4fb`, where the holder defaults to UNKNOWN and the
  app samples on resume.

With D1 every slice of Stage 3 has landed and been checked on a device.

- **A1** Kotlin `ArtifactDownloader` with size cap, streaming hash, atomic promote and quarantine; auth via the shared client. Unit tests use a local HTTP server (resume, mismatch, oversize).
- **A2** Backend: range, strong ETag and hash-addressed file serving, with tests.
- **A3** Move `modelProvisioning.ts` to A; remove base64 staging; jest tests plus a live TinyNet run.
- **B1** Snapshot format and validators in Kotlin, with fixture tests for every rejection reason.
- **B2** SAF import UI, *Data* screen, pin-at-join and `DATASET_INCOMPATIBLE`.
- **C0** Refuse a staged dataset larger than one batch until C2 lands (the missing check); one test.
- **C1** (if decision 2 picks it) Proto enum + Python/C++/TS permutation + shared golden + corpus cases.
- **C2** Native minibatch loop, with endpoint parity against the Python trainer on a 3-batch snapshot.
- **D1** Capability report; **D2** qualification probe and cache.
- **E**  Live TinyNet run on an imported snapshot, with per-client replay.

## Verification strategy

- **Unit:** every validator and download failure mode has a test that fails without the check.
- **Cross-language:** the batch permutation and `labelSchemaId` are pinned by goldens that the Python, C++ and TS tests all read.
- **Parity:** C2's multi-batch endpoint gets its own tolerance, derived from the movement it must detect. That is the lesson of 06/09: never inherit a tolerance.
- **Live:** slice E, plus the four-platform run repeated with imported data.

## Risks

- **SAF on vivo/Funtouch** may add vendor quirks; B2 is tested on the vivo first.
- **Batch-order change** touches the laptop client's training. It is gated by the contract, so Stage 2 contracts keep today's order.
- **Serving no data** breaks today's demo path until decision 1 is implemented.
