# Fixed NIH study runner and analysis

The [prospective protocol](nih-phase2-study-v1-protocol.md) is the scientific
specification. The runner implements its 360 fixed cells; this document describes
execution and verification, not completed empirical results.

## Execution

`framework/benchmarks/nih_phase2_study.py` requires the qualified pretrained
feature-cache report pin and the frozen nine-image test sensitivity mask. It
refuses an existing output directory, arbitrary recipes and a modified protocol.
Before its first child starts it persists every configuration, configuration hash
and all 360 pending outcomes. Each child runs in a fresh isolated Python process.
The parent and standalone reanalysis explicitly establish four CPU threads,
matching the saved final-prediction recipe.

A child receives at most 180 seconds; combined child execution is limited to
10,800 seconds. Parent control and verification time are recorded separately.
Peak RSS is an engine-completion process high-water observation, checked against
2 GiB; it is not an enforced process-tree or late-artifact memory cap. Ordinary
failed/timed-out cells remain in the study and later fixed cells continue.
Source, cache, mask or generated protocol/configuration integrity failures stop
later launches. Malformed JSON/schema evidence is an integrity failure. The
parent compares generated protocol/configuration evidence against its retained
originals before launch, after return and during failure handling.

Start-source proof is written before engine entry and survives an interrupted
child. Returned records survive later artifact or analysis failure. Successful
children retain the actual final model, aligned validation/test logits, float64
margins, labels and identities under the [artifact contract](final-state-score-artifact-contract.md).
The parent checks loaded source fingerprints, complete cohort/round counts,
record/runtime identities, timing, memory, byte/FLOP totals and saved predictions.
It recomputes final score metrics rather than accepting the index's summary.

## Analysis

`framework/benchmarks/nih_phase2_analysis.py` implements tie-aware binary AUROC,
confusion metrics, logistic loss and calibration directly in NumPy. It does not
import the engine's metric helpers. Scores use subtraction after conversion to
float64; default binary prediction is positive only for a strictly positive
margin. Operating thresholds come exclusively from each final model's validation
positives: the highest threshold reaching at least 80% recall, including ties.
That threshold transfers unchanged to test. Inverse images-per-patient weighting
and the predeclared ambiguous-content mask are additional diagnostics.

Standalone saved-study analysis reopens configurations, raw records, checkpoints
and prediction arrays. It verifies the complete artifact registry and file hashes,
reforwards the saved model and rebuilds each qualified summary. Editing an index
metric or dropping an artifact hash cannot silently alter the analysis.

The primary effect uses all eight declared components in each of ten matched
seeds. Its t interval is produced only when all ten pairs are complete; missing
components suppress the interval and remain visible. All other comparisons are
descriptive. Positive interaction alone does not establish better absolute
utility, so all component AUROCs are retained.

## Reproduction and limits

From a clean committed source checkout with the private qualified cache/mask:

```sh
PYTHONPATH=framework/src:framework python -m benchmarks.nih_phase2_study \
  --cache <qualified-cache> --expected-report-sha256 <frozen-report-sha256> \
  --test-mask <frozen-mask.json> --output <new-study-directory>
PYTHONPATH=framework/src:framework python -m benchmarks.nih_phase2_analysis \
  <study-directory> --output <new-analysis.json>
```

The runtime must match the recorded library versions. Never reuse an output
folder or rerun failed seeds to replace their outcomes. Independent reconciliation
of real study evidence remains a separate gate; synthetic regressions do not
establish useful NIH performance or completion of the study.

Models, IDs, seeds and diagnostic logs are research artifacts, not DP-safe exports.
Privacy accounting is conditional per-run simulated honest-client replace-one DP,
not patient-level, whole-matrix or clinical protection. Trusted local sequential
execution is assumed; no malicious-host or power-loss durability guarantee is made.

## Verification checkpoint

All 68 focused runner/analysis checks pass, including eight regressions that first
reproduced the parent-thread and malformed-evidence failures. One independent
review preceded those fixes. The full framework suite passes 2,076 tests, with
14 deselected and81.66% coverage; unhandled thread exceptions are treated as errors.
The suite emitted1,974 dependency/runtime warnings. No real NIH official-test
classifier scores or completed main-study results are claimed at this checkpoint.
