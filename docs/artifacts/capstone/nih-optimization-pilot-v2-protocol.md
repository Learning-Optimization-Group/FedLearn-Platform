# NIH optimization pilot v2 — prospective exploratory budget

19 October 2026. Fixed after the eleven-cell v1 pilot and its three-replay score
diagnostic, before any v2 training. This is a new bounded exploratory study, not
confirmatory evidence or an amendment to v1. The continuous capstone-work mandate
authorizes this necessary baseline gate. Official-test classifier performance
remains unseen; metadata/prevalence/content audits were already inspected.

## Question and alternatives

Can conservative raw-feature SGD produce useful ranking and stable confidence
at ten and eighty simulated clients? V1 selected high LR by AUROC alone and had
poor cross-entropy/ECE. Its low-rate pooled model had useful ranking and confidence.
Do not infer that normalization is necessary, or that a score correction repairs
optimization.

First test smaller learning rates and one local epoch, using the existing engine
without a feature-transform change. A row-local L2 transform is a possible later
separate diagnostic if this fails. Train-fitted shared standardization is not
used: it would introduce a new cross-client data dependency requiring explicit
privacy assumptions. Optimizer/backbone/feature tuning is outside this budget.

## Fixed computation

- Same qualified pretrained ResNet-18 cache report SHA-256
  `6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd`.
- All 68,650 training and 17,763 validation rows, natural prevalence, whole-patient
  IID allocation. No redraw, rebalancing, standardization, normalization or
  threshold tuning; prediction is argmax, class zero on an exact tie.
- Linear two-class head, 1,026 trainable parameters. CPU, four torch threads,
  SGD without momentum/weight decay, batch 256, **one local epoch, twenty rounds**,
  no server optimizer, all clients each round, no DP/attacks/dropout/fairness.
- Primary ranking is explicitly `margin_v2`; retain `auc_probability` diagnostic.
  Loss, balanced accuracy, sensitivity/specificity, confusion reconstruction and
  ten-bin ECE accompany every case. Only final-round metrics enter eligibility
  or selection; retain all twenty-round curves.
- Base seed **20261019**, engine seed derivation unchanged. Repeat seeds below
  change partition, initialization, shuffling together; no individual-source
  variance attribution.

## Exact maximum budget: ten cells / two hundred rounds

| Order | IDs | Clients | LR | Seed | Clip / rule |
| --- | --- | ---: | --- | ---: | --- |
| 1–3 | `clients10_lr_0.001`, `clients10_lr_0.003`, `clients10_lr_0.01` | 10 | named | 20261019 | none / mean |
| 4–6 | `clients80_lr_0.001`, `clients80_lr_0.003`, `clients80_lr_0.01` | 80 | named | 20261019 | none / mean |
| 7–8 | `clients80_repeat_20261020`, `clients80_repeat_20261021` | 80 | selected below | named | none / mean |
| 9 | `clients80_clip_mean` | 80 | selected below | 20261019 | 0.4 / mean |
| 10 | `clients80_clip_trimmed` | 80 | selected below | 20261019 | 0.4 / trimmed mean, trim 0.2 |

Eighty clients is a working core-study cohort between the small federation and
scale check, not a measured deployment population. The two cohort groups are
optimization diagnostics, not a causal client-count experiment: per-client
sample counts, minibatch steps and uniform-client weighting change. Relative to
v1, rounds and local epochs also change; no single-factor causal claim follows.

All six initial cells must complete and qualify before selection. Qualification
means correct immutable inputs, score/code/config identity, exact twenty rounds,
no refusal/early stop and finite in-range metrics. Finite poor results qualify
as recorded results and remain visible. No selection among surviving cells after
failure, no replacement seed, no intermediate-round selection or added LR.

Among the three eighty-client initial cases, a case is **eligible** only if final
margin AUROC >= **0.68**, balanced accuracy >= **0.60**, loss <= **0.65** and ECE <=
**0.10**. These are pragmatic research-baseline gates chosen after v1, not clinical
acceptance criteria or significance thresholds. Among eligible cases select the
lowest final loss; exact ties select higher AUROC, then lower LR. Ten-client
cases do not enter selection. If none is eligible, record the failure of the
baseline gate and mark all four follow-ons unrun; a further diagnostic requires
a separately written prospective budget.

Run both repeat seeds and both clipping/rule diagnostics only for the selected
LR. Do not retune clipped/trimmed cases. Report eligibility for every repeated
and clipped case separately. Passing the core optimization gate requires all
three selected-LR unclipped seeds to be eligible. Clipped mean/trimmed results
do not block reporting this narrow optimization result, but any failed clipped
gate blocks treating that aggregation setting as an established clean baseline
for the privacy/robustness matrix without a separately bounded correction.

## Execution and evidence gates

Cap each child at 300 seconds and the entire study at 1,800 seconds. Retain logs,
partial outputs, failures and explicitly unrun cells; no silent retries. Run
children sequentially without a concurrent regression suite during cost capture.
Report end-to-end child time separately from engine time and peak process RSS;
ordinary desktop background load is uncontrolled. These are cached-head CPU
simulation measurements, not physical-client or network benchmarks.

Before the first child, commit this protocol and exact runner/config recipes.
Record protocol/config/cache digests, source commit and the executed source-byte
fingerprints; refuse a dirty tracked tree at launch and verify it and source
fingerprints before/after every child. Include the framework source, benchmark
runner and protocol rather than relying on HEAD alone. A detected mutation
invalidates qualification; retain the affected case and mark follow-ons unrun.
Ignored outputs do not count as source edits. This is local execution integrity,
not protection against a hostile filesystem actor racing the checks.

Independently reload all records, hashes, config values, patient/client counts,
assignment digests and selection/eligibility, using retained manifests and a
separate reconciliation calculation. Report all attempted cases and complete
curves, plus the three-seed descriptive mean/range if applicable. No clinical
population CI or hypothesis test comes from seed repeats. Raw cache, row IDs,
logits and records stay private/ignored with an off-device backup; public
aggregate evidence is a research artifact, not a DP-protected deployment release.

## Stop and next decision

The matrix remains on hold until a stable core baseline and any required clean
clipping/rule baseline are established. Then measure the larger-head budget
before freezing the separate confirmatory protocol. Do not inspect official-test
performance or launch privacy/attack conditions within this pilot. A failed gate
is an informative result to report, not permission for an unlimited tuning sweep.
