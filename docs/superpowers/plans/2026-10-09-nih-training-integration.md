# NIH training integration plan

Goal: execute the NIH training design inline, retaining the existing worktree
and local branch. Spec: ../specs/2026-10-09-nih-training-integration.md.

## Tasks

- [x] Add `fl-runtime/nih_runtime.py` with `load_manifest(source, prepared, name)`
  verifying preparation hashes/metadata/partitions; lazy `NihDataset`; explicit
  deterministic sampling/provenance; validation-only server loader and binary
  metrics. Add runtime tests with synthetic patient/image fixtures. Observe
  failures, implement and verify; no source images changed.
- [x] Add explicit NIH recipe metadata/model/data dispatch to `recipes.py`,
  canonical client/server and inference. Extend seeded snapshot loader to map
  datasets; add manifest path dispatch only for NIH client manifests with an
  explicit source root. Test recipe semantics, no fallback/leakage, seeded order,
  partial batches and registry evaluation provenance.
- [x] Add canonical recipe enum 9, regenerate mirrors and pinned Python/TS stubs,
  and approve FedAvg/Robust/FULL NIH matrices in Python/Java/TS. Extend NIH
  first-order execution plan, LOCAL_SNAPSHOT-only enforcement, artifact export
  shape/budget and backend stager recipe dispatch. Test exact class-schema
  binding, old-recipe isolation and unsupported strategies/data sources.
- [x] Add `scripts/make_nih_image_package.py` over verified client manifests,
  explicit count/balance/seed and complete provenance; test train versus
  validation/test intent and bit-identical package preparation. Add user guide.
- [x] Run full changed-area suites and protocol regeneration/mirror checks.
  Independent correctness review; reproduce and fix substantive findings.
- [x] Align g14 within codebase/FedLearn_Platform, update validation backend and
  phone JS, import a new NIH package without clearing datasets, then qualify
  two-round phone/g14 training with retained parents and separate patient
  partitions. Verify registry, metrics, original-runtime/aggregate replay and
  old dataset preservation. Record results and any live limitations.
- [x] Final checks, status/guide update and local commits; preserve branch and
  user checkout. No merge or publication.

## Review focus

- A maliciously altered report cannot legitimize changed labels or split leakage.
- Client manifest selection cannot train on held-out or another client's rows.
- Serving raw PNGs lazily must preserve the phone's exact prepared tensor bits.
- Validation sampling must record its changed counts and never consume official
  test images or silently report whole-set results.
- Old NORMAL/PNEUMONIA and version-2 XGBoost behavior must remain isolated.

## Rulings and work log

Ruling: separate NIH recipe rather than mutating the existing pneumonia label
contract. Cost if wrong: more catalog/schema entries; old projects remain valid.
Ruling: small explicit patient-separated samples qualify execution; the complete
NIH partitions remain available for a subsequent research protocol. Cost if
wrong: weak model quality; report metrics and limitations plainly.

Verified source checkpoints: 72bd908 integrated NIH; 0b279a7 corrected the
export batch bound and case-alias guards. The default export originally used
FIXTURE and fell back to maxBatch8, causing live publication refusal against
batch16; a regression reproduced this and the corrected live bundle published.
Lowercase/mixed-case NIH aliases now refuse before creation, and strategy/start
guards cannot bypass the supported configuration through FoT.

Independent review also found and verified fixes for a generic accuracy target
in NIH benchmarks and a feasible patient-unique sampling request rejected by
positive-first assignment. No remaining confirmed source finding after review.

Final source suites: runtime569 passed (30 slow deselected); framework1345
passed (13 deselected), coverage78.52%; backend1159 tests and required coverage;
mobile647 tests, lint/typecheck; frontend137 tests, lint/build with explicit
validation origin. Python/TS contract regeneration is a no-op and all five
proto mirrors match. g14 NIH focused32 tests pass.

Live run adf7860c-9566-4777-9fc3-fdacc851ca83 completed two rounds with
32 balanced patient-unique images per participant. All four accepted updates
replay bitwise on their original runtimes, changed-label controls differ, and
the aggregate reconstructs the registered artifact bitwise. Independent reload
reproduces all binary validation metrics/provenance. All 21 original dataset
files are preserved; three NIH files added. The installed APK hash matches the
retained native build. Original-phone replay temporary files were removed.

NIH Android catalog support is cleared after actual program/device qualification.
Final accuracy 1.17%, balanced accuracy 50%, AUROC 0.4384: model quality is poor.
This is engineering qualification, not full NIH training. Detailed receipts
and limits are recorded in research/notes/on-device/2026-10-09-nih-pneumonia-mixed.md.
