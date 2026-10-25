# Prospective Effusion ByzFL implementation qualification

24 October 2026. This protocol prepares an implementation reconciliation, not a new aggregation-effect study. No patient-data model fit is authorized merely by generating a preflight plan. The existing holdout remains closed.

## Fixed design

Four cells: native mean and ByzFL mean, each without noise and with central GDP epsilon4/delta1e-5. Use R0's pinned train/selection bundle, zero-initialized linear head, seed20261101, 80 whole-patient clients with full participation, 20 rounds, seven local epochs, batch256, LR0.03 and clip0.05. The historical references are A17's matching clean/private cells on that seed. There are no attackers, model selection, tuning, dropout or held-out evaluation.

The runner derives the exact GDP noise multiplier and records the different RDP-equivalent budget supplied to the existing engine. It verifies the actual multiplier and exact GDP epsilon rather than relabeling the engine's RDP field.

## Preflight and checkpoint

Preflight verifies the canonical train/selection-only bundle, the pinned A17 summary, reference parameter shapes/noise, zero initialization and nonempty patient-whole partitions. It freezes all four fully resolved configurations, input/reference hashes, source bytes, dependency/runtime versions and settings. Full configurations remain in the private plan because they contain runtime paths; their hashes bind them to the public manifest. It does not train or score a model.

The private plan contains local runtime paths and stays outside version control. Its public manifest contains nonidentifying hashes, settings, source paths and aggregate reference endpoints. A local checkpoint must contain that public manifest, this protocol and the exact repository source bytes before execution. Explicit authorization is required for the local commit under REVIEW-HANDOFF.md section8; no push or merge is implied.

The current preflight hashes of the A17 raw reference records become part of this new checkpoint. Do not claim that the original summary already cryptographically bound those raw head files. The pinned original AUROC scalars and shape/noise checks supply additional reference constraints.

## Acceptance and failure handling

Require four complete finite runs with all80 clients aggregated in all20 rounds. Require native/ByzFL final parameters and native/prior-A17 parameters to differ by at most1e-6, AUROC by at most1e-10, and GDP noise within the existing calibration tolerance. Investigate any mismatch; do not silently relax tolerance or replace an unsuccessful cell.

Save all raw records, final heads, receipt hashes, source snapshots and failures in a fresh local output directory. Journal each completed round before proceeding, retain returned records when validation fails, and preserve the completed-round journal if execution raises. Refuse existing output paths and any changed source, dependency/runtime, bundle, reference record or prospective plan. A qualification result does not establish method superiority, clinical performance, external generalization or privacy of research diagnostics.

Execution checks the archived source inventory and bytes as well as current sources, revalidates the bundle's synthetic/real classification before accepting a checkpoint bypass, and preserves captured final heads even when later validation fails. A filesystem failure may abort publication of later records or the summary; already published round journals remain evidence, and an aborted output is not a completed qualification.

The implementation has an explicit synthetic-fixture mode for tests with generated bundles and bounded settings. The canonical medical bundle cannot be run in that mode. Actual execution requires the committed checkpoint and public manifest.

## One-step mechanism for subsequent pilots

The mean reconciliation above does not exercise history-centered clipping. Its separate mechanism checks use a fixed accepted cohort of size n, uniform averaging, outer Euclidean projection P_C, a single centered Euclidean projection P_tau, and a center m fixed by prior released history:

`q(u) = m + (1/n) * sum_i P_tau(P_C(u_i) - m)`.

Under replacement of one upload, nonexpansiveness of P_tau bounds the changed summand by 2C. Its output ball also bounds the change by 2tau. Thus the conditional L2 sensitivity is at most `2*min(C,tau)/n`. With no outer clip the bound is `2*tau/n`. C and tau must be fixed or functions of public/private-released history. Adaptive composition accounts for successive releases conditional on the same previously released history; the neighboring histories need not be identical in an unconditional comparison.

This is a one-step aggregate-release argument. It does not cover repeated recentering on raw updates, bucketing, changing the accepted cohort, data-dependent filtering, shared private preprocessing, or raw research diagnostics. Center and inner-clipping diagnostics are research outputs and must not be presented as covered by the training epsilon. Synthetic neighboring-vector tests illustrate the bound; they do not replace this argument or establish whole-program privacy.
