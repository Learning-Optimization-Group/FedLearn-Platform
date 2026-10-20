# NIH phase-2 reproduction guide

20 October 2026. The fixed study completed once: 360 qualified cells/7,200 rounds.
The standalone checker replayed every model and both score splits; the original
source reader reproduced the entire saved analysis exactly. Twelve tables and six
figures were exported/reconciled and visually checked. Study review and its validation correction are complete. The final assessed
report, presentation and working dashboard remain open.

The [frozen protocol](nih-phase2-study-v1-protocol.md) defines the science. The
[runner contract](nih-phase2-runner-contract.md) describes execution and failure
handling. The [TODO](capstone-todo.md) records current delivery status. This guide
separates verifying retained evidence from starting a new reproduction; neither
replaces failed original cells or adds favorable seeds to the original analysis.

## Evidence and runtime

| Artifact | Required identity or role |
| --- | --- |
| Frozen protocol | First committed at `aa7ae53`; SHA-256 `4245a8446d3d88a93cd921d24f7a4f565273a7285663e446e43f1fcb123012a0` |
| Original execution source | `3baea7d3f1c3d518d689c6f12fee4a85ee03a00c`; all loaded source proofs remain in raw evidence |
| Qualified feature report | SHA-256 `6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd` |
| Curated cohort report | SHA-256 `287dca8359300f08cea79098d367793a73e43205bcfe3213e9d22115e3bb7267` |
| Frozen sensitivity mask | SHA-256 `352b10c16342e4160d06a1e536ea82d201ea739eec17ed67fc9801141e60470d` |
| Classification runtime | Python 3.12.7, PyTorch 2.12.0, NumPy 2.1.2; CPU, four threads |
| Feature preparation | Frozen pretrained ResNet-18/ImageNet weights; original extraction used MPS, batch 64, seed 1234, four audit/image workers; torchvision 0.27.0 and Pillow 12.3.0 |
| Raw study directory | All 360 configs/index entries, protocol/mask, logs, source proofs, records, qualified final checkpoints and aligned validation/test scores |
| Independent audit | Recomputed model/score/accounting/assignment/cost/inference evidence; separate script and raw snapshot hashes |
| Curated output | Aggregate tables/figures/report, with hashes and raw-evidence reconciliation; no patient-row dumps |

The source and data artifacts are different things. The original private Git bundle
preserves the branch; it does not contain ignored caches/models/score arrays. Those
are kept in separately verified private archives. The code is on a local capstone
branch, not assumed published to a remote repository. Obtain the required artifacts
through an authorized handoff; this guide does not publish raw patient-row data, checkpoints or score arrays.

The strict reforward check targets the reference environment. A copied cache can be
read at a new location, but another hardware/kernel stack may produce different
floating-point logits. A failed exact match must be reported or investigated, not
silently fixed by editing records, changing tolerances or replacing scores.

The fast primary-summary regressions exercise complete and partial evidence, exact
ten-seed coverage, declared inference scope/method, missing identities and malformed
statistics. Run them before the full model audit:

```bash
python docs/artifacts/capstone/repro/test_primary_validation.py
```

The checker reconstructs the full expected primary object from qualified scores;
extra/omitted rows, duplicate seeds, empty intervals or false metadata cannot pass.

## 1. Verify retained evidence without retraining

Wait until capture is terminal. Keep the original raw study and cache immutable.
Choose a new review directory outside both. From a source checkout containing the
original Git objects and the standalone reproduction scripts:

```sh
export CAPSTONE_PYTHON=/path/to/python3.12
export SOURCE_ROOT=/path/to/FedLearn-Platform
export STUDY_DIR=/path/to/retained-study
export FROZEN_CACHE=/path/to/qualified-features
export REVIEW_DIR=/path/to/new-review-directory
mkdir "$REVIEW_DIR"

"$CAPSTONE_PYTHON" "$SOURCE_ROOT/docs/artifacts/capstone/repro/check_verification_cli.py" \
  --source-root "$SOURCE_ROOT"

"$CAPSTONE_PYTHON" "$SOURCE_ROOT/docs/artifacts/capstone/repro/verify_phase2_study.py" \
  --source-root "$SOURCE_ROOT" --study "$STUDY_DIR" --cache "$FROZEN_CACHE" \
  --output "$REVIEW_DIR/independent-audit.json"

"$CAPSTONE_PYTHON" "$SOURCE_ROOT/docs/artifacts/capstone/repro/export_phase2_tables.py" \
  --study "$STUDY_DIR" --audit "$REVIEW_DIR/independent-audit.json" \
  --output "$REVIEW_DIR/tables"

"$CAPSTONE_PYTHON" "$SOURCE_ROOT/docs/artifacts/capstone/repro/plot_phase2_study.py" \
  --study "$STUDY_DIR" --audit "$REVIEW_DIR/independent-audit.json" \
  --output "$REVIEW_DIR/figures"
```

The checker reads the cache at the supplied location while preserving the original
recorded cache-path string in each configuration identity. It checks the actual
cache report/payload hashes; a path override is not permission to substitute data.
It imports no project runner/model/codec/metric/accountant. It parses checkpoints,
reforwards both splits, recomputes score arithmetic and resolves source bytes from
Git at the recorded commit. The commands explicitly refuse disabled assertions (-O or PYTHONOPTIMIZE). The
quick mode check covers six prohibited-mode invocations and three normal help
controls. All output paths must be fresh; the checker does not edit the
raw study or refill missing outcomes.

The exporter creates 12 tables: every planned outcome; all seven evaluation modes
with confusion counts; primary seed effects; all 64 recipe groups; thresholds and
patient/content sensitivity; observed/modeled costs; reported update diagnostics;
and matched head/clipping/attack/cohort comparisons. Missing cells stay visible with
blank metrics and controlled reason codes. Detailed raw errors stay private because
they may contain host paths or commands. Inverse-patient confusion counts are
fractional diagnostic weights, not patient-level predictions.

The figure script creates six figures. Bars describe observed seed ranges except
for the single prespecified primary t interval, which exists only with all ten
complete seed effects. It records every plotted value and file hash. Rendered exports
and public report tables must be visually/numerically checked for each reproduction.
The original curated package passed those checks, including exact binding of all 370
plotted records; it does not establish completion of the capstone deliverables.

The source-bound production reanalysis CLI is an additional check in the original
matched source/runtime/path environment:

```sh
PYTHONPATH=framework/src:framework "$CAPSTONE_PYTHON" \
  -m benchmarks.nih_phase2_analysis "$STUDY_DIR" \
  --output "$REVIEW_DIR/source-bound-analysis.json"
```

Run that command from a clean checkout of the original execution source. Its strict
loaded-source binding and recorded I/O paths are intentional. It is not the portable
cache-override checker above; do not alter original records to make it accept a new
filesystem layout.

## 2. Reproduce execution as a separate run

This is optional retraining, not the default way to check reported numbers. It takes
up to three hours of child time plus parent control. Do not run it alongside the
original study or another measured workload. Use a separate clean checkout of the
original source, the exact pinned cache and the frozen mask, and a new output path:

```sh
PYTHONPATH=framework/src:framework OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$CAPSTONE_PYTHON" -m benchmarks.nih_phase2_study \
  --cache "$FROZEN_CACHE" \
  --expected-report-sha256 6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd \
  --test-mask /path/to/frozen-mask.json \
  --output /path/to/new-reproduction-run
```

Keep this reproduction's index/failures independent. Its timings and possibly its
floating-point results belong to its recorded environment. Never promote a repeated
cell to replace an original failure, or pool repeated seeds as new independent
replications. The fixed source refuses existing output and altered pins/recipes.

## 3. Rebuild and qualify data as a separate workflow

Raw NIH images/metadata/official lists are required here. Do not rerun preparation
just to inspect retained results. The implemented builder sequence is:

```sh
export NIH_DATA_ROOT=/path/to/NIH
export PREP_DIR=/path/to/new-original-preparation
export AUDIT_DIR=/path/to/new-image-audit
export COHORT_DIR=/path/to/new-curated-cohort
export FEATURES_DIR=/path/to/new-features

PYTHONPATH=framework/src:framework "$CAPSTONE_PYTHON" -m benchmarks.nih_data \
  --data-root "$NIH_DATA_ROOT" --output "$PREP_DIR" \
  --clients 10 --validation-fraction 0.2 --seed 42
PYTHONPATH=framework/src:framework "$CAPSTONE_PYTHON" -m benchmarks.nih_image_audit \
  --data-root "$NIH_DATA_ROOT" --preparation "$PREP_DIR" \
  --output "$AUDIT_DIR" --workers 4
PYTHONPATH=framework/src:framework "$CAPSTONE_PYTHON" -m benchmarks.nih_cohort \
  --data-root "$NIH_DATA_ROOT" --original-preparation "$PREP_DIR" \
  --image-audit "$AUDIT_DIR" --output "$COHORT_DIR" --audit-workers 4
PYTHONPATH=framework/src:framework OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$CAPSTONE_PYTHON" -m benchmarks.prepare_nih_features \
  --data-root "$NIH_DATA_ROOT" --original-preparation "$PREP_DIR" \
  --image-audit "$AUDIT_DIR" --curated-cohort "$COHORT_DIR" \
  --output "$FEATURES_DIR" --device mps --batch-size 64 \
  --backbone-seed 1234 --audit-workers 4 --image-workers 4
```

Replace related placeholders consistently and use only available devices. The original
preparation's full-content audit fails; the conservative curated cohort is the input
to features. Do not skip that distinction or train the failed original preparation.
No random-weight control may be presented as the qualified pretrained workload.

A rebuilt report/cache on a different runtime/device need not have the original
byte identity. Qualify it independently. The frozen original runner intentionally
rejects a different pin; a new qualified replication needs its own declared data/
source identity and must not be relabelled as the original confirmatory evidence.
Feature-extraction timing is separate from the classifier's measured costs.

## Interpretation and delivery checks

- Retain all 360 statuses and the exact fixed cell order. Missing primary components
  suppress the planned ten-seed interval; secondary comparisons remain descriptive.
- Do not pick an intermediate round, best seed or threshold using official-test results.
- Report all absolute primary components beside the interaction, including negative
  findings. A positive interaction is not necessarily better absolute private utility.
- Distinguish image-level prediction, inverse-patient weighting and a patient model;
  report-derived finding labels are not certified diagnoses.
- Label observed wall/process/memory scopes versus modeled payload/FLOPs. Disclose
  fixed order, power changes and background load; do not claim physical-client scaling.
- Per-run noise targets, matched random streams and diagnostic exports do not establish
  a jointly protected matrix, patient-level DP or a privacy-safe research transcript.
- Review figures, reconcile tables/report/deck against the same audit and preserve a
  fresh source/evidence backup before final handoff. Publication/messages remain
  separate from preparing these local artifacts.
