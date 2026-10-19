# Explicit NIH evaluation and patient-preserving simulation plan

Status: both tasks implemented and exercised; final whole-slice review pending.
Full suite:1,787passes,14deselected,81.21%coverage. All11pilot cells/102rounds and
the separate three-replay score diagnostic independently verified. The pilot
exposed poor calibration and a real probability-saturation AUROC defect; these
are study-readiness blockers, not reasons to discard or edit v1 results. See the
[report](../../artifacts/capstone/nih-validation-pilot-v1-report.md). The preceding
frozen-feature slice passed complete qualification and review.

**Goal:** Run the existing simulation engine on the verified NIH cache while
preserving patient ownership, explicit validation/test selection and exact cache
identity. Establish a bounded validation-only learning pilot before the protocol.

**Spec:** [NIH workload design](../specs/2026-10-18-capstone-nih-workload-design.md).

**Architecture:** Add a separate NIH feature workload that reuses existing head,
client and evaluation behavior but loads the fixed verified cache and partitions
whole patient groups. Select it explicitly in the engine. Extend configuration
only for the NIH dataset; retain the exact legacy resolved configuration, hashes
and numerical reproduction behavior. Keep public test performance unseen.

## Constraints and interfaces

- `workload.dataset = "nih"` selects the new workload. Its `features_path` is a
  cache directory; `features_report_sha256` is a required lowercase 64-digit trusted
  digest, and `evaluation_split` is explicitly `validation` or `test`. Absence is
  an error; no implicit test selection. NIH `balance` must explicitly be false.
- NIH requires `federation.partition.type = "patient_iid"`; refuse image-level
  IID/Dirichlet rather than silently pretending they preserve patients. The partition
  uses the existing `patient_iid_partition` and the resolved partition seed.
- Do not add new default keys to resolved Kermany configurations. Accept NIH-only
  configuration fields through a dataset-specific merge template; reject those
  keys on legacy datasets. Keep historical config hashes stable.
- `NihFeatureHeadWorkload` reuses head construction/training interface and exposes
  the chosen evaluation tensor through the existing internal evaluation interface.
  Its provenance explicitly names the evaluation split; metadata must never imply
  validation is official test. Record source dataset, cache/cohort/model/manifest
  identities, patient/image counts, partition algorithm, assignment digest and
  actual per-client sample/patient/class counts.
- Copy read-only mapped arrays into owned CPU tensors before training. No writable
  alias of the cache. No source image access or feature recomputation in the engine.
- Store no raw patient/image lists in run JSON; an assignment digest plus aggregate
  client counts suffices. Tests may inspect private workload identities directly.
- Keep triggered/backdoor features absent and refuse backdoor configs before any
  training. Do not add trigger synthesis in this slice.
- Evaluation uses selected natural prevalence. No balancing/subsetting, changed
  labels, validation redraw, or threshold tuning on test. Report-derived labels
  and simulated-client DP limits stay explicit.
- Existing diagnostic run records are research outputs; do not claim the entire
  record is privatized. The output-release privacy inventory remains a protocol gate.

## Task 1: Configuration and workload integration

1. Write failing tests with a tiny real NIH feature cache. Require explicit split,
   digest, false balancing and patient IID; refuse missing/invalid values and
   incompatible backdoor config before training. Confirm legacy resolved dict and
   config hash are byte-equivalent to the current baseline for representative
   configurations.
2. Implement NIH-specific configuration validation and new workload selection.
   Reuse `load_nih_cache(..., expected_report_sha256=...)`; reject mismatches before
   building clients. NIH-only `allow_random_features` defaults to false and must
   be a strict boolean. Require explicit true to use a labelled random-weight
   control; record both this setting and actual cache pretrained status. Real
   pilot configurations keep it false. Tiny tests use true with actual random
   weights, without network access or mislabelling their provenance.
3. Test whole-patient client ownership, nonempty exact training union, reproducible
   assignment, seed changes, too many clients and preserved feature/label alignment.
   Hash ordered client image assignments using a versioned canonical encoding.
   Confirm every client's class/sample/patient counts independently.
4. Run a tiny no-DP/no-attack engine case on validation; prove its evaluation labels
   and counts are validation rather than test. Assert selected cache and assignment
   identities appear in saved metadata. A separate synthetic test may select test;
   no real official-test performance is inspected.
5. Run workload/config/engine/NIH tests, unchanged reproduction anchors and the full
   framework suite with coverage and unhandled thread exceptions promoted to errors.
6. Commit `feat(capstone): integrate patient-preserving NIH simulation workload`.

## Task 2: Bounded real validation pilot and cost check

1. Before running, save a pilot specification and exact cell list. Start with a
   centralized linear-head no-DP optimization sanity check and a matched small
   federation, using only training and validation. Fix seeds, rounds, local steps,
   optimizer, learning-rate candidates, resource cap and failure/stop rules before
   observing results. This is an exploratory pilot, not the confirmatory study.
2. Report natural-prevalence majority baseline, loss, AUROC, balanced accuracy,
   confusion counts and calibration at the existing fixed threshold. Record
   parameter count, patient/client counts, hashes, elapsed time and memory scope.
   Examine learning curves and optimization failures; do not hide weak learning.
3. If learning is weak, diagnose optimization or representation using a separately
   recorded bounded extension. Do not launch an unrestricted tuning sweep or use
   test results to decide which extension looks favorable.
4. Independently reload and reconcile the retained validation results. Run one
   1,000-client, two-round case as an execution/cost check; state that its tiny
   round budget does not establish useful learning or population-scale validity.
5. Update the current capstone plan with measured conclusions, limitations and a
   practical budget for a separately versioned prospective phase-two protocol.
   Commit the pilot code/specification and aggregate findings; keep private cache,
   manifests and raw records ignored and backed up off this laptop.

## Review focus

Check explicit evaluation identity, digest binding before training, no patient
crossing clients, exact row coverage, no mutation of read-only cache, no implicit
balancing, backdoor refusal, legacy config-hash/numerical stability and honest
pilot versus confirmatory claims. Judge run-record privacy scope separately from
the model mechanism; no full-record privacy guarantee is introduced here.
