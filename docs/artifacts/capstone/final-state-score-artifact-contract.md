# Qualified final-state and score artifacts

20 October 2026. Research evidence contract for the
[frozen NIH study](nih-phase2-study-v1-protocol.md). The study itself has not run.

## Capture and failure ownership

`engine.run(..., on_final_parameters=callback)` optionally supplies owned, detached
CPU copies of its last returned server parameters. Generic engine completion can
be partial; the callback alone does not establish qualified execution.

`run_core(..., on_final_parameters=callback)` buffers that capture internally and
invokes the caller only after successful preflight, complete-round checks and
postflight qualification. A stopped/refused/incomplete run does not invoke it.
An ordinary callback exception becomes `CoreQualificationError` with the complete
qualified simulation record in `.record`, preserving evidence for its runner.
Its existing marker means simulation execution only, not successful export.
Without a callback, the default training/record path remains unchanged.

## Artifact API

[Source](../../../framework/benchmarks/nih_final_artifacts.py):
`write_final_artifacts(cfg, record, params, output, *, feature_report_sha256)` and
`verify_final_artifacts(cfg, record, output, *, feature_report_sha256)`.

Both reconcile the resolved recipe and qualification marker, exact cache identity,
source/cohort/backbone/manifest hashes, split/sample counts and model size.
The expected feature digest is supplied separately by the study caller.
Parameters must match declared keys/shapes, CPU float32, finite values and the
record's final-state digest. The full serialized checkpoint also has a SHA-256.

| File | Bound content |
| --- | --- |
| `model.safetensors` | Actual final head parameters; no pickle or optimizer state |
| `validation.npz`, `test.npz` | Float32 logits, float64 margins, int64 labels, Unicode image/patient IDs in exact cache order |
| `manifest.json` | Recipe/record/source/cache/model/split identities, software/thread/prediction recipe, file hashes/sizes and explicit non-private research marker |

Predict with CPU eval/no-grad and a whole-split forward matching the engine's
matrix shape. Convert each logit to float64 **before** subtracting; subtracting
in float32 and casting afterward can collapse distinct ranks. Store logits too,
so later analysis can independently reconcile loss, calibration and argmax ties.

Output must be fresh and outside the cache. Payloads are written first and the
atomic manifest last. An interrupted write leaves inspectable incomplete files,
without a completed export; refuse reuse rather than overwrite evidence.
Verification reloads the checkpoint and recomputes every score/label/identity,
rejecting inconsistent payloads even if their hashes were rewritten. It requires
the recorded software/thread recipe and exact re-forward equality, not cross-
hardware floating-point equivalence. Atomic completion is not a power-loss
storage-durability guarantee.

The caller still owns prospective protocol enforcement, source/runtime proofs,
exact study-cell enumeration, independent score-based AUROC and inference,
threshold selection/transfer, patient/mask analyses, budgets and failed-case
reporting. This artifact API does not complete those tasks.

Real test scoring occurs only through the verified prospective study workflow;
unit checks use explicitly identified synthetic NIH feature controls. Parameters,
scores, labels/identities and seeded diagnostics remain private research evidence,
with `research_record_dp_safe:false`. Neither a completed manifest nor the
per-run model accountant makes these files a protected deployment export.
