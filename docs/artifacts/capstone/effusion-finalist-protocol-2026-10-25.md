# Prospective fresh-seed Effusion finalist comparison

25 October 2026. Freeze this protocol, source, analysis, input/parent hashes and all resolved
cells in a local checkpoint before any finalist fit. This is a prespecified training-seed
comparison conditional on the repeatedly used development-selection cohort and the selected
policies. It is not untouched-cohort confirmation, external validation or a whole-study DP release.
The existing holdout remains closed.

## Decision after the diagnostic phase

The 126-cell pilot completed without replacement. Owned-data PGD qualified as damaging,
whereas the original and coordinated sign-flip controls did not clear the declared nonprivate
positive-control threshold. All finalists therefore use the same qualified owned-data PGD
policy under attack, with the same knowledge and per-round budget.

The private clean projection screen selected width 128: paired gains were 0.0360543 and
0.0452283 over the full-width mean. Center tau/C=0.25 is retained to check a mechanism trade-off:
it substantially improved nonprivate attacked utility with a small clean loss, but harmed the
private absolute points. It is not being called the best private centered radius. Tau/C=1
collapsed private clean utility; tau/C=4 behaved near the mean in this pilot. The scalar
output-perturbed comparator is not advanced because its tested private points were weak,
without claiming other private convex methods fail. The bucket study remains a separate
nonprivate heterogeneity/contamination diagnostic; private bucketing is unqualified.

These are exploratory candidate-selection decisions. Final results will not retune widths,
radii, optimizer settings, attacks or the seed budget.

## Frozen matrix and recipe

There are 192 main cells: four policies × two noise settings × two attack settings × twelve
fresh seeds. Four additional native-mean anchor cells, paired with the corresponding main
ByzFL mean cells, verify backend agreement before any candidate/attacked fitting: total 196.

| Policy | Width | Reducer | Public parameters |
| --- | ---: | --- | --- |
| mean512 | 512 | ByzFL clipped mean | C=0.05 |
| projected128 | 128 | ByzFL clipped mean | same C; fixed public projection seed 20261025 |
| trim16 | 512 | ByzFL coordinate trimmed mean | k=16 of 80; same outer C |
| center_quarter | 512 | one-step released-history centered clipping | tau=0.0125, same outer C |

Noise settings are zero-noise matched controls and central Gaussian mechanisms calibrated to
GDP epsilon 4, delta 1e-5 over 20 releases. Each uses its own justified sensitivity; matching
epsilon does not imply identical noise standard deviation. The private center is the previous
released delta, not a raw unnoised state.

All cells use the pinned R0 train/development-selection bundle, 80 full-participation whole-patient
IID clients, zero head initialization, 20 rounds, seven local epochs, batch 256 and LR=0.03.
This controlled recipe isolates composite mechanism choices; the separate nonprivate recipe
study is not an optimality certificate for every method. Attacks are clean or owned PGD with
16 persistent malicious clients. PGD uses 20 steps, three restarts and step ratio 0.25,
with 63 loss and 60 gradient evaluations per round, coalition-owned examples/updates and
previously released global/history only. No selection labels or current raw honest updates enter it.

Fresh seed identifiers are 13001 through 13012 inclusive. They identify seeded partition,
local-training, noise and attack streams, not dates. The fixed projection basis and zero
initialization are shared design choices. The four native anchors use the first two seeds
and both noise settings; their corresponding mean cells belong to the main matrix and are not rerun.

## Primary endpoint and inference target

The primary endpoint is the mean paired selection AUROC gain of projected128 over mean512
in the private clean condition across the twelve fresh training seeds. The technical practical
margin is +0.01 AUROC; it is not a clinical-benefit threshold.

Use the prespecified two-sided 95% Student t interval on the twelve paired seed effects.
Report every effect, sample SD, standard error and degrees of freedom. This approximate
interval describes training-seed variation conditional on this fixed evaluation cohort and
the selected policies. Normality and independence are assumptions, not empirically certified
by twelve seeds. Do not reinterpret it as a patient/population interval.

If its lower endpoint exceeds +0.01, the result supports a practically positive seed-mean
gain for this frozen recipe/cohort. A lower endpoint above zero but not +0.01 supports only
the corresponding positive-effect statement. An interval crossing zero or the practical
margin remains unresolved for that claim. No equivalence/noninferiority claim follows.
An interval wholly below zero is evidence of a negative conditional seed-mean effect;
report that harm rather than labelling it merely unresolved.

The pilot paired SD was about 0.006487; a nominal twelve-seed t half-width using that unstable
two-seed estimate is 0.004122. If the actual SD is 0.02 or 0.04, nominal half-widths are about
0.01271 or 0.02541. Twelve seeds are a bounded precision choice, not a guarantee. Do not add
seeds after viewing the final interval to manufacture a desired conclusion.

Separately compute a 2,000-replicate patient-cluster percentile bootstrap for that same primary
contrast, with public bootstrap seed 20261025 and all twelve realized head pairs held fixed.
Sample whole patients with replacement, carrying all their images and sharing multiplicities
across all heads and both policies. The point remains image-level AUROC, not inverse-patient
weighting. Count skipped single-class draws and cap attempts at 20,000. This interval is
conditional on the selected fitted heads and does not include training-seed variation or
adjust for selecting the policy on the same cohort. The two intervals are separate; neither
is a joint patient-and-training or external-population interval.

## Secondary outcomes and components

Report per-seed and mean AUROC, average precision, Brier score, and sensitivity/specificity
at fixed probability 0.5 for every main arm. The operating point is descriptive, not a
new clinical threshold or a replacement for A10's frozen calibration.

Prespecified secondary paired AUROC contrasts cover each alternative minus mean in each
noise/attack setting; each policy's clean-minus-attacked damage and zero-noise-minus-private
cost; and the change under privacy in attack-adjusted aggregation advantage. That interaction
is `(method_attacked-mean_attacked)-(method_clean-mean_clean)` under private noise, minus
the same expression without noise. Absolute clean/attacked components must accompany it.
Secondary intervals are descriptive and unadjusted for multiplicity; no confirmatory
family-wide discovery or universal ranking is claimed.

Retain center/inner versus outer clipping, actual noise, attack budgets and represented
pre-server norms as research diagnostics. Mandatory server clipping controls reducer inputs;
the final intended PGD norm is not separately logged. Diagnostics, raw fits, validation and
model selection are not covered by the released-model calibration claim.

## Integrity, failures and analysis

Bind the complete screen/bucket summaries, source, public manifests, saved raw records and
phase verification before final preflight. Include the design and inference helpers in the
prospective source snapshot. Refuse existing outputs, changed hashes/runtime/configurations,
incomplete cohorts or nonfinite cells. Journal completed rounds and retain failed returned
records and available final heads. Baseline/anchor failure or unexplained native/ByzFL head difference
above 1e-6 stops subsequent fitting. Saved-head and engine AUROC must agree within 1e-10.
Failed and unrun cells stay in the declared accounting; no replacement or silent tolerance change.

Final analysis must reconcile all 196 planned cells, every noise/sensitivity calibration,
all head/record hashes and the twelve exact seed identities in each main contrast. A partial
primary pair set is failed execution, not an analysis on a conveniently smaller sample.
Keep patient identities and predictions local; export only aggregates and nonidentifying
hashes. A later method/data change needs a separate amendment and cannot reuse the holdout
to validate the choices made here.
