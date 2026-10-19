# NIH score/optimization diagnostic — bounded extension v1

19 October 2026. Written after the eleven-cell exploratory pilot and before any
real validation-logit capture. This extension is exploratory and does not change,
replace or reselect any v1 pilot result.

**Trigger:** v1 selected LR 0.1 by final probability-score AUROC, but validation loss
was 6.127 and ECE 0.349. Pooled LR 0.001 had AUROC 0.720 and loss 0.613. The code
uses float32 softmax probabilities as AUROC scores. An already-run synthetic probe
with margins 20,21,22,23 and labels 0,0,1,1 produced probabilities all exactly 1,
probability AUROC 0.5 and margin AUROC 1.0. This proves a numerical failure mode;
its impact on the real pilot remains unknown before these replays.

Replay **exactly three** existing v1 configs: `pooled_lr_0.001`, `pooled_lr_0.1`,
and `federated_lr_0.1`. Retain all original hyperparameters, seeds, cache, grouping,
evaluation split and code behavior. Use CPU/four threads. Cap each replay at
300 seconds and the extension at 900 seconds; no retries or replacement cells.
No test evaluation, normalization, parameter tuning or new training condition.

Observe validation logits through a wrapper that returns the original metric
result unchanged. Check source-file bytes against v1 code `27e10bb`, final model
digest equality and equality of every recorded utility metric for all ten rounds.
If equality fails, retain the failure and do not attribute differences to scoring.
Write captured final validation logits/labels privately for independent scoring.

On identical captured logits, compare original probability AUROC with AUROC of
the double-precision decision margin `logit_1 - logit_0`. Independently calculate
tie-aware rank AUROC; report probability zero/one counts, number of unique scores,
margin range, loss, ECE, predictions and confusion counts. Distinguish numerical
ranking loss from genuine poor confidence/calibration; a better scoring method
does not fix an overconfident classifier.

Decision values are valid binary AUROC inputs according to the
[scikit-learn metric documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
That reference supports the score convention, not the magnitude of any numerical
error in this project; the synthetic and real-logit checks establish the latter.

Report all three replays, even unfavorable results. Before confirmatory runs,
any scoring fix must be explicitly versioned and regression-tested while retaining
historical behavior/results. Any conditioning or optimization change requires a
separately bounded prospective pilot; do not silently edit v1 or its readiness flag.
