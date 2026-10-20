# NIH full-schedule mechanism qualification v1

19 October 2026. Prospective exploratory validation protocol. This document must
be committed, followed by the verified runner, before any declared real-data cell.
It is not the confirmatory protocol and does not authorize official-test scoring.

## Purpose and fixed recipes

Exercise the complete source-bound central-DP/attack path on the real NIH cache,
including matched no-DP controls, full accepted cohorts and full-round accounting.
The earlier no-attack/no-noise baseline and short head-cost checks do not establish
that this combined path works. No outcome selects a winner or triggers extra cells.

Exactly twelve linear-head cells, in this fixed nesting order:

1. Condition: clean, sign flip, label flip.
2. Within condition: mean, coordinate-wise trimmed mean (trim ratio 0.2).
3. Within rule: no DP, central DP with epsilon 4 and delta 0.00001.

All cells: frozen 512-feature ResNet-18 ImageNet representation, natural prevalence,
whole-patient IID allocation, all 80 clients accepted in every one of 20 rounds,
7 local epochs, SGD LR 0.003, batch 256, no proximal/server optimizer, four CPU
threads, validation `margin_v2`. Base seed 20261022; resolved seed streams are
partition/init/client 20261022, attack 20262022 and noise 20263022. Keep these
streams matched across all twelve cells; trajectories diverge after treatments.
No subsampling/privacy amplification credit is taken.

Clipping C = 0.05 for all cells, including no-DP controls. This is a declared
exploratory choice informed by earlier clean-upload mean norms 0.019–0.058 and
inactive C = 0.4. Mean norms do not determine a quantile or optimal bound.
This qualification does not tune C or certify useful utility. The eventual
confirmatory protocol must identify this validation exposure and fix its own
clipping factors before test performance is examined.

Sign flip uses 16 fixed attacker slots (fraction 0.2), multiplying each attacker's
own update by -4 before common clipping. Label flip uses 16 slots with the same
attack-assignment seed, reversing binary labels on each attacker's own data.
Neither attack reads another client's current raw update. Clean cells have no
attacker slots. IPM/ALIE, local DP, non-IID partitions and fault/dropout/deadline
behaviors are excluded; historical entry points remain available unchanged.

Pinned feature report SHA-256:
`6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd`.
Random features are refused. Cache payload hashes and pretrained identity must
validate before the first child. Official-test membership and cache integrity
were already inspected; official-test classifier performance remains unseen.

## Resource and failure contract

- Per-child timeout 180 seconds; combined child execution at most 1,800 seconds.
  Parent control/check time is reported separately and does not consume this cap.
- Observed child process RSS, sampled at engine completion, must be at most
  2 GiB. This is a post-run gate, not an OS-enforced or whole-process-tree limit.
- Exactly twelve pending index entries, resolved configurations and their hashes
  must exist before the first child. Refuse existing output/cache-contained paths.
- Sequential isolated children, no concurrent regression suite. Persist atomic
  before/after loaded-source proofs; the before proof survives interruption.
  Bind source tree, this protocol, resolved recipe and cache identity.
- Every child calls `run_core` preflight before training and postflight afterward.
  Parent independently requalifies its raw record and marker, source/config/model/
  thread/CPU identity, 80 accepted slots, 20 contiguous rounds and noise/accounting.
- Retain failed/nonfinite returned records, traceback logs, source proofs and
  execution manifests with artifact hashes. No retries or overwritten outputs.
  Ordinary cell failure continues the fixed list; source-integrity failure stops
  later launches. Budget exhaustion leaves explicit unrun entries.
- No utility threshold, early stopping, best-round selection or adaptive extension.
  Poor finite performance is a result; a missing/inconsistent/partial record fails
  execution qualification. All twelve must qualify for the combined-path gate.

## Checks, analysis and limits

Report all twelve final-round validation AUROCs, loss, balanced accuracy, finding
recall, specificity, accuracy and ECE. Include complete curves and clipping/noise
diagnostics in retained research records. Reconcile case identities, complete
participation, assignment matching, sensitivity/noise/epsilon, payload/FLOPs and
child/core/loop timing using a checker independent of runner qualification logic.
No p-values, seed uncertainty intervals, clinical conclusion or confirmatory
interaction claim are supported by this single-seed validation qualification.

The privacy unit is one protected honest simulated client under replace-one
adjacency, conditional on fixed public recipe, cohort, accepted set/cardinality
and permitted attacker information. The server is trusted with raw updates;
per-round Gaussian mechanisms compose across the fixed 20 rounds. Patient-disjoint
data do not establish patient-level DP. Seeded research diagnostics remain a
non-private audit record; this is not an end-to-end DP-safe release workflow.
No production privacy guarantee or one universally safe epsilon is inferred.

The next study must freeze exact hypotheses, matched cells/new seeds, final-round
endpoint, uncertainty/multiplicity and failure rules before official-test scoring.
Preserve this exploratory protocol and all outcomes without retrospective edits.
