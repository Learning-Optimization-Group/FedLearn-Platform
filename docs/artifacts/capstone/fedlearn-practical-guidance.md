# Practical decisions for FedLearn

20 October 2026. The fixed NIH study completed all 360 cells and passed independent
model/score/accounting reconciliation. The [study report](nih-phase2-study-v1-report.md)
and retained tables support the conditional recommendations below. The additional
source-bound reader reproduced the full analysis exactly. Study review and its
validation correction are verified. The worker and persistent API now pass their
execution gates; the dashboard UI/browser evidence and final assessed deliverables
remain open.

## Decisions supported by completed work

| Decision | Evidence and consequence | Concrete project action | Current boundary |
| --- | --- | --- | --- |
| Qualify data content as well as patient membership | The original NIH preparation had patient-separated groups but exact RGB overlap across splits. The curated development cohort quarantines111 images from nine patients and keeps official test intact. | Put patient ownership, exact-content audit, label definition and pinned manifests before image-training readiness. | Implemented for the capstone NIH workload. Do not assume every mobile/production image recipe now uses this audit; near-duplicates and clinical label truth remain unqualified. |
| Score ranking from unsaturated decision values | The v1 diagnostic found a substantial probability-versus-margin AUROC discrepancy; three exact logit replays and synthetic rank fixtures isolate float32 probability saturation. | Use versioned float64 logit differences for binary AUROC and retain raw logits. Preserve old metric versions/records instead of silently rewriting history. | Implemented for NIH simulation and final-score artifacts. Audit other platform scorers before extending the claim to them. |
| Separate ranking, confidence and operating-point quality | The validation pilots showed that high AUROC alone can coexist with poor calibration/threshold behavior; finding prevalence differs between development and test. | Display AUROC, loss/ECE, balanced accuracy and actual confusion counts. Select an operating threshold using validation and transfer it unchanged. | The80%recall threshold is a prespecified research diagnostic, not a clinical requirement. Prevalence difference alone does not explain all distribution shift. |
| Count successful execution at the accepted aggregation boundary | Rejected admissions previously inflated completion counts and could leave repeated unresolved rounds. Participation schema2separates submission, admission, acceptance and aggregation. | Gate successful runs on actual accepted cohorts, complete rounds and qualified final state; retain failed/partial evidence. | Framework corrections/core gates are implemented. Do not assume every physical-device UI/endpoint has been independently requalified by these simulator tests. |
| Compare privacy settings using each rule's actual sensitivity | At80 clients, mean uses denominator80and20%trimming uses 48. Its coordinate noise SD is5/3 larger at the same multiplier. | Present epsilon/delta, adjacency, accepted cohort, clipping and rule-specific noise together. Match each private model with its appropriate no-DP control. | Source/accounting checks support this mechanism arithmetic. It does not alone predict utility or choose a robust rule. |
| Enforce permitted information through the whole attack pipeline | The source-backed IPM counterexample changes multiple attacker uploads when one raw honest update changes; a downstream single-upload bound does not cover that dependence. | Keep raw-honest-update oracle stress tests separate from protected-release claims. For the qualified core use own-data/own-update attackers and fixed accepted cardinality. | This demonstrates a raw-update-domain IPM/mean counterexample, not a measured NIH patient attack or a proof that every rule/ALIE fails the same way. |
| Separate experiment seeds from protected-release randomness | The experiment deliberately shares standard draws across matched cells. The [first-round algebraic example](matched-noise-release-boundary.md) shows that shared noise at different amplitudes can cancel jointly, even if its seed is hidden. | Use matched random numbers for declared public-data experiments. For a protected deployment, design secret per-release randomness, conditional/joint accounting and an explicit export boundary. Do not simply add marginal budgets for correlated releases. | Algebraic first-round/upload-domain clarification; no demonstrated recovery from final-only NIH checkpoints. A source-bound scalar production check is complete; no final-only NIH reconstruction is demonstrated. |
| Treat diagnostic export as research data | Seeds, identities and pre-noise diagnostics sit outside a model-only privacy accountant. | Label research exports honestly. A deployment needs a reviewed allowlist and protected randomness/accounting for everything released. | Current records are not DP-safe deployment exports; no patient-level, whole-matrix or clinical guarantee follows. |
| Make provenance/failure recovery inspectable | Source proofs, strict config/cache/model/score bindings and exact reforward checks expose stale or edited evidence. The completed study retains all 360 outcomes and a hash-verified terminal backup. | Retain recipe identity, source/runtime, actual final state, metric version and complete/partial/error status. Stop/drain real owned processes before freeing resources. | Final artifact/core/runner and persistent API checks pass. Actual dashboard ownership/cancellation/restart/browser acceptance remains open. |

Supporting reports: [data qualification](nih-frozen-features-2026-10-19.md),
[scoring/pilot](nih-validation-pilot-v1-report.md),
[schedule probe](nih-schedule-pilot-v3-report.md),
[attack-information audit](attack-knowledge-privacy-boundary-audit.md),
[participation contract](participation-accounting-contract.md),
[release-boundary audit](privacy-output-boundary-audit.md),
[final artifact contract](final-state-score-artifact-contract.md),
[runner contract](nih-phase2-runner-contract.md).

## Decisions supported by the fixed study

The primary interval is conditional on fixed cohorts and ten simulation seeds.
Head/cohort/clipping results use three matched seeds and are descriptive. None is
a production privacy-budget choice, clinical recommendation or universal rule ranking.

| Decision | Measured evidence | Concrete platform action | Applicability |
| --- | --- | --- | --- |
| Choose aggregation from absolute utility and its control | At C0.05 / epsilon 4/sign-flip, mean AUROC 0.627389 versus trimming 0.618250. The primary interaction is nevertheless +0.002266, with paired 95% CI [+0.000704,+0.003828]. | Show clean/attacked absolute endpoints and each matched no-DP control beside the interaction and rule-specific sensitivity/noise. | Ten-seed linear-head primary setting; this does not establish a universal winner. |
| Qualify clipping jointly with attacks and DP | At C0.4/sign-flip, no-DP mean is 0.495854 versus trimming 0.658057; epsilon 4 values are 0.520096 and 0.546027. | Validate clipping with the intended threat/privacy recipe. Preserve the frozen study; new tuning belongs on validation in a separate experiment. | Three seeds; clipping changes useful signal, permitted attack magnitude and noise together. |
| Demand a learning benefit before increasing the head | Clean epsilon 4 mean: linear 0.644213 versus MLP512 0.643233; modeled upload 6.8 MB versus 1,688.0688 MB across 80 clients/20 rounds. | Put matched learning controls and payload estimates into head/model admission and comparison views. | About 248.245 times the payload; no equivalence test or claim that a differently tuned larger head cannot improve. |
| Qualify learning as cohort size changes | Clean epsilon 4 mean: 10 clients 0.548356, 80 clients 0.644213, 1,000 clients 0.630644. The thousand-client private/no-DP gap is smaller, but its private point mean is lower than at 80. | Report accepted cohort and local training recipe together; requalify local steps before scaling client count. | Synthetic patient groups and three seeds; not physical-device scaling or a pure 1/n noise experiment. |
| Treat threshold transfer as a measured operating point | For clean epsilon 4 mean, validation 80% recall transfers to test recall 0.915729 and specificity 0.241375; default test recall is 0.619790. | Display the validation target, transferred threshold and actual held-out confusion counts/recall/specificity. | Ten-model mean; no demonstrated clinical threshold or guarantee of 80% recall on new data. |
| Keep the evaluation unit visible | For that same group, inverse-patient weighting raises AUROC by 0.077657 on average, from 0.646336 to 0.723993. | Retain the original image endpoint and show patient-weight sensitivity alongside it. | A change of weighting, not a patient classifier; do not replace the prespecified endpoint after seeing the higher value. |
| Preserve the content-audit boundary | Removing the nine prespecified ambiguous images changes any model's AUROC by at most 0.000165502. | Keep the official primary cohort and report the fixed secondary mask, rather than remove inconvenient test cases. | Small measured sensitivity does not certify all labels, near-duplicates or clinical validity. |
| Expose actual execution and resource scopes | All 360 cells/7,200 rounds qualified; child 10446.316 s plus parent 2834.088 s. Maximum engine-completion high-water RSS 1573.469 MiB. | Show lifecycle, stage timings, accepted counts, completion/failure and measured-versus-modeled fields explicitly. | No process-tree/late-artifact memory cap or isolated hardware runtime comparison; power/background/fixed-order limitations apply. |

All comparisons refer to the pinned frozen-feature NIH workload. The mobile
pneumonia task and production image-training recipes require their own qualification.
The Simulation Lab should expose these comparisons and limitations. Its persistent
API has passed the execution gate; the React workflow and actual browser acceptance
remain open. The [follow-up readiness audit](privacy-security-readiness-2026-10-20.md)
separates central DP, generic local DP and DeComFL secure aggregation, with the
remaining qualification/reporting gates. It supplies no new disease-model result.
