# NIH validation pilot — prospective exploratory budget v1

19 October 2026. Written before the first real NIH classifier run. This is a
bounded exploratory pilot, **not** the confirmatory phase-two pre-registration.
Only training and validation may be used. Official-test classifier performance
remains unseen. The prior metadata/content audits, test prevalence and duplicate
membership were already inspected and are disclosed separately.

## Fixed inputs and computation

- Cache report SHA-256: `6c9e6969c6fd762156ef2246323fbdbc5e9cb272c9910a8b1c9b691ab33783fd`.
- Verified ImageNet ResNet-18 features; all 68,650 training and 17,763 validation
  images at natural prevalence. No balancing, feature scaling, subsetting or
  feature/backbone tuning. Labels are No Finding/any listed finding.
- CPU training, four torch CPU threads, a two-class linear head (1,026 parameters),
  SGD without momentum/weight decay, batch size 256, three local epochs and ten
  federation rounds. Server optimizer none; all clients participate each round.
- Whole-patient IID assignment, fixed seeds derived by the engine from base seed
  **20261019**. No DP, no attacks, no dropouts/stragglers or per-client fairness.
- Initial cells have uniform mean and no clipping. Learning rates are exactly
  **0.001, 0.01 and 0.1**. Fixed prediction rule is argmax, class 0 on an exact tie.

The pooled optimization check is one simulated client containing all training
patients. The small federation has ten clients. Both expose each image to the
same nominal number of local epochs, but averaging frequency, shard endpoints
and the uniform-client versus pooled-image objective differ. This is a learning
sanity comparison, not a claim of numerically equivalent optimization.

## Exact cell budget and order

| Order | Cell IDs | Clients | LR | Seed | Rounds | Clip/rule |
| --- | --- | ---: | --- | ---: | ---: | --- |
| 1–3 | `pooled_lr_0.001`, `pooled_lr_0.01`, `pooled_lr_0.1` | 1 | named LR | 20261019 | 10 | none / mean |
| 4–6 | `federated_lr_0.001`, `federated_lr_0.01`, `federated_lr_0.1` | 10 | named LR | 20261019 | 10 | none / mean |
| 7–8 | `federated_repeat_20261020`, `federated_repeat_20261021` | 10 | selected below | named seed | 10 | none / mean |
| 9 | `federated_clip_mean` | 10 | selected below | 20261019 | 10 | 0.4 / mean |
| 10 | `federated_clip_trimmed` | 10 | selected below | 20261019 | 10 | 0.4 / trimmed mean, trim 0.2 |
| 11 | `scale_1000_two_rounds` | 1000 | selected below | 20261019 | 2 | none / mean |

Choose the LR with the highest **final-round validation AUROC** among the three
completed, finite ten-client LR cells. Exact ties select the lower LR. Pooled
cells do not enter selection. Do not select an intermediate round, add an LR,
retune for the clipped/trimmed cells or replace an unfavorable seed. Follow-on
configs are resolved and saved before each run; their only adaptive value is
this predeclared selected LR.

The first six cells must all complete successfully before selection/follow-ons.
After a failure, retain all attempted initial records within the budget, report
the failure, and mark follow-ons unrun rather than selecting among survivors.
Each subprocess has a 300-second cap; the whole pilot has an 1,800-second cap.
Kill a timed-out child, retain its log/partial output and record timeout. Do not
silently retry. Nonfinite metrics, early stopping, missing rounds, refusal or a
mismatched cache/evaluation identity fail qualification. Finite poor performance
is a result, not a reason to discard a cell.

## Outputs, interpretation and stop gate

Retain exact configs, code/config/cache/assignment identities, raw round records,
subprocess logs, failures/unrun cells and a reconciled summary. Report final loss,
AUROC, natural-prevalence accuracy/majority baseline, balanced accuracy, sensitivity,
specificity, confusion counts, ECE, learning curves, parameter/client/patient counts,
wall time and measured-memory scope. Confusion counts may be reconstructed from
known validation class counts and the recorded sensitivity/specificity, with
integer/fraction/accuracy consistency checks; this must not be described as an
independent prediction-level recomputation.

Report all three seeds for the chosen unclipped federation, with descriptive
range/mean only. No population confidence interval, hypothesis test, attack
resistance or privacy/utility claim comes from this pilot. The clipped cases are
paired exploratory diagnostics of clean clipping/aggregation cost. The 1,000-client
two-round run is an execution/cost check, not evidence of convergence or clinical
population-scale validity. Keep simulated network/FLOP costs separate from measured
wall time and RSS; GPU memory is irrelevant to this CPU head-training run.

If the best ten-client final validation AUROC is below **0.60**, call this a weak
baseline for the intended study and require a separately recorded, bounded
optimization/representation diagnostic before confirmatory runs. This is a
pragmatic study-readiness gate, not a clinical threshold or significance test.
Complete the already declared finite follow-ons without opening a new sweep.

After independent record reconciliation, use the measured costs and utility to
draft a separately versioned prospective confirmatory protocol. Do not inspect
official-test performance until that protocol is frozen. Research seed/diagnostic
records remain research outputs; the [privacy-output boundary](privacy-output-boundary-audit.md)
does not turn them into protected deployment exports.
