# Additional published-claim corrections

25 October 2026. Report-integration review of published notes only. No raw holdout data,
holdout predictions or saved holdout models were read or evaluated. Numeric scientific
tables and original artifact hashes in both affected notes were checked before and after
editing and remain unchanged.

## A10 internal holdout note

The published FedAvg-minus-convex point is +0.0011 with a conditional patient interval
[−0.0021, +0.0044]. This supports close observed points, not an equivalence claim without
a prospective margin and suitable test. A higher holdout point than selection does not
rule out selection optimism. The original confidence intervals condition on the realized
heads rather than including training-seed variability.

The operating points were chosen before the original holdout opening. Their observed
sensitivities remained near the target while specificity fell for the tested private
recipes. This does not establish that DP generally leaves sensitivity unchanged, and the
clinical burden of a high negative-flag rate does not by itself establish triage usefulness.
Recipe and noise differences remain combined in those operational contrasts.

The [A10 note](effusion-holdout-v1-result-2026-10-22.md) now states those boundaries. Its
original one-time evaluation and frozen thresholds remain the historical evidence.

## A17 zero initialization and release count

Zero initialization adds no data-dependent privacy expenditure. Its observed utility
improvements apply to the matching tested cells; they do not establish a universal
utility improvement. The T5-minus-T20 patient intervals crossing zero at epsilon4/8
do not establish equivalence or noninferiority. The epsilon1 conditional interval is
positive, while training-seed uncertainty remains a separate question.

The [A17 note](effusion-zero-head-r0-v1-result-2026-10-22.md) now describes fewer releases
as a candidate with measured conditional effects and one-quarter of the modeled
round-dependent communication. No new A17 seed interval was computed here. The earlier
173-contrast reanalysis retains its declared A11/A12/A19/A22/A23 scope.

## Honest-client boundary in coalition-owned attacks

The owned-data mechanism preparation already limits its attacked-run sensitivity argument to replacement of one honest client, with coalition identities/data fixed and conditional on released history. A coalition member can influence several malicious uploads, so its replacement is not covered by a single-upload sensitivity bound. The final comparison retains its exact frozen methods and analysis; this is an explicit interpretation boundary, not a method amendment. Clean primary cells have no malicious coalition. The integrated report and final result note must state this restriction. The qualified PGD uses a uniform-mean surrogate and is not a worst-case attack tailored to each reducer.
