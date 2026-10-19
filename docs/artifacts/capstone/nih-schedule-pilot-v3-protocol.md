# NIH schedule pilot v3 — prospective exploratory budget

19 October 2026. Written after the reconciled [v2 results](nih-optimization-pilot-v2-report.md),
before any v3 training. Exploratory only; this does not amend v2, constitute the
confirmatory protocol or permit official-test classifier evaluation.

## Question and fixed inputs

Does a longer optimization schedule give a stable eighty-client baseline at LR
0.003? V2's ten-client model at that LR passed the one-seed gate, while its
80-client model had lower ranking and only 3–5 local minibatch steps per client
per round, versus 26–30 at ten clients. That observed confound is not proof of
causation. No new learning-rate, feature-transform or representation search is
allowed in this probe.

Use the identical qualified cache report SHA-256
`6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd`,
all 68,650 training and 17,763 validation rows, natural prevalence, eighty whole-
patient IID clients and all-client participation. Retain the frozen ResNet-18
features without scaling/normalization, linear two-class head (1,026 parameters),
SGD without momentum/weight decay, LR **0.003**, batch **256**, CPU/four threads,
no server optimizer, no DP, attacks, dropouts, balancing, fairness or threshold
fitting. Prediction is argmax with class zero on a tie. Primary AUROC is
`margin_v2`, with `auc_probability` separately retained.

## Exact maximum budget: six cells / at most 720 rounds

| Order | ID | Local epochs | Rounds | Seed | Clip / rule |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `local7_round20` | 7 | 20 | 20261019 | none / mean |
| 2 | `local1_round140` | 1 | 140 | 20261019 | none / mean |
| 3–4 | `selected_repeat_20261020`, `selected_repeat_20261021` | selected schedule | selected schedule | named | none / mean |
| 5 | `selected_clip_mean` | selected schedule | selected schedule | 20261019 | 0.4 / mean |
| 6 | `selected_clip_trimmed` | selected schedule | selected schedule | 20261019 | 0.4 / trimmed mean, trim 0.2 |

Both initial schedules expose every image to 140 nominal local epochs, with the
same number of client minibatch calls over the entire study. They change local
trajectory length, aggregation frequency and round-derived shuffle streams. They
are not numerically equivalent or a clean single-factor causal experiment. Seven
local epochs give 21–35 client steps per round, close to v2's ten-client range.
Use the existing ten-client v2 result as a historical reference, not an additional
new run or a matched total-image-exposure comparator.

Qualification: both initial cells must complete with exact config/cache/source/
score identities, finite in-range metrics, all declared rounds and no refusal or
early stop before any selection. Do not select among survivors. Finite poor
performance is a result, not an execution failure.

Eligibility uses the same final-round-only gates as v2: margin AUROC >= **0.68**,
balanced accuracy >= **0.60**, loss <= **0.65**, ECE <= **0.10**. These are pragmatic
research gates fixed after earlier exploratory results, not clinical/significance
thresholds. If `local7_round20` is eligible select it; otherwise select
`local1_round140` only if eligible. Prefer fewer aggregation releases, rather than
choose by tiny loss differences across different privacy-composition schedules.
This preference does not claim a better eventual DP utility result. If neither
is eligible, mark the four follow-ons unrun and stop; do not extend the sweep.

For the selected schedule run two named seed repeats and both unchanged clipping/
rule diagnostics. No retuning. Report eligibility of each separately. The core
optimization gate requires all three selected-schedule unclipped seeds eligible.
A failed clipping/rule gate prevents calling that setting an established clean
baseline; report it rather than replace the result. No official-test evaluation,
privacy/attack matrix or larger-head experiment is part of this budget.

## Execution, reconciliation and interpretation

Commit this protocol and the exact executable recipes before the first child.
Enforce the exact cache pin and clean tracked/source tree. Bind parent helpers,
isolated child source origins and bytes to this checkout, record source/protocol/
config/cache/assignment hashes, and verify source fingerprints before/after each
child. Keep an executed-source proof for every child. This guards local source
mix-ups, not a hostile runtime or dependency-attestation threat model.

Each child has a 300-second cap; the total probe has an 1,800-second computation
budget. Recompute remaining time immediately before launching each child; control
and final verification overhead are reported separately from bounded child time.
Retain partial logs, errors/timeouts and explicitly unrun cases with no silent
retry. Run sequentially after regression checks finish; ordinary background load
is uncontrolled. Use a fresh output directory outside the immutable cache.

Before viewing outputs, prepare an independent checker that imports no production
pilot/config/partition/metric helper. Reload configs, records, source proofs and
hashes, rebuild patient assignments/counts from retained CSVs, reconcile confusion
fractions, independently apply initial eligibility/selection and the three-seed
gate. Report all cases, final metrics, complete curves, clipping/shrink rates,
process RSS and child/engine wall time. Plot curves against nominal image passes
as well as stating aggregation rounds; 20 and 140 rounds are not interchangeable
for communication or privacy accounting. Seed ranges describe simulation variation,
not clinical-population uncertainty. No test-derived threshold fitting or causal
privacy/cohort/clinical conclusion is permitted.

Keep raw records/private row identities ignored and backed up off-device; publish
only aggregate research evidence. Current research seeds/diagnostics are not a
protected deployment export or patient-level DP guarantee. If the baseline gate
passes, measure the larger-head budget before freezing a separate bounded
confirmatory privacy/robustness protocol. If it fails, report the specific failure
and redesign explicitly instead of silently adding favorable conditions.
