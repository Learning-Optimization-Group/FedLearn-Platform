# Final report: evidence map and completion gates

20 October 2026. Working outline and claim register, not a finished final report.
The [current capstone plan](../../../wikis/capstone/README.md) governs scope.
The title still needs advisor confirmation. No confirmatory NIH study has run;
results below remain exploratory and must be labelled that way in every artifact.

## Research question and intended contribution

Under a declared simulated-client privacy/threat model, when do clipped mean and
coordinate-wise trimmed mean retain useful utility with central DP and malicious
updates? How do local training schedules, cohort size and trainable head size
change the observed trade-off and measured cost?

The contribution must be reproducible experimental evidence and practical platform
decisions. Implementing a GUI, running many configurations or reporting epsilon
alone is insufficient. Negative findings and confounds are useful when measured
and scoped correctly. Do not claim a new DP theorem, clinical model, protected
end-to-end research log or 1,000 physical-client deployment.

## Claim register

| ID | Claim allowed by current evidence | Authoritative evidence | Limit / remaining check |
| --- | --- | --- | --- |
| D1 | Official metadata/list membership is patient-disjoint, but exact RGB content overlaps those groups | [Cohort preparation and qualification](nih-frozen-features-2026-10-19.md), retained full image audit and source hashes | Patient separation alone is insufficient; no near-duplicate or clinical-truth certificate |
| D2 | Conservative development quarantine removes 111 images from nine non-test patients; official test remains unchanged | Same cohort/feature qualification and private curated manifests | Known ambiguous groups remain within test; define any sensitivity analysis before test performance |
| M1 | Frozen pretrained ResNet-18 features are source/model/transform/manifest-bound and match direct reference forwards | [Feature qualification](nih-frozen-features-2026-10-19.md), cache report SHA6c9e…83fd | Frozen representation, not end-to-end radiology training; preparation timing overlapped regressions |
| M2 | NIH simulation preserves whole-patient clients, natural prevalence and explicit validation/test identity | [Engine implementation plan and verification](../../superpowers/plans/2026-10-19-capstone-nih-engine-integration.md), config/workload/engine tests and independent allocation checks | Patient IID only; clients are synthetic groups, not verified institutions |
| M3 | A separate core wrapper enforces permitted-information recipes before engine access and complete fixed-cohort execution afterward | [Core execution qualification](core-execution-qualification.md), real coordinator/strategy regressions and malformed-record tests | Execution premises only; persistent source-bound runner, protocol freeze, feature/learning readiness and protected exports remain separate |
| M4 | Qualified final-state capture and reloadable model/score artifacts preserve cache/model/record/row identity, including float64 subtraction before margins | [Artifact contract](final-state-score-artifact-contract.md), synthetic linear/MLP roundtrips, 24 reproduced identity regressions and payload-rehash corruption checks | Verified infrastructure on synthetic controls; no actual official-test score, independent real-study reconciliation, private export or cross-hardware floating-point guarantee |
| R1 | Float32 probability saturation can materially understate binary AUROC | [V1 pilot and three source-locked logit replays](nih-validation-pilot-v1-report.md); pooled high-LR gap0.153445 | Historical magnitude outside three replays unknown; preserve old records, use versioned margins prospectively |
| R2 | Highest AUROC can select a poor confidence/threshold operating point | [V1](nih-validation-pilot-v1-report.md) and [v2](nih-optimization-pilot-v2-report.md), all cases/curves | ECE is a ten-bin aggregate diagnostic, not clinical or subgroup calibration |
| R3 | Ten-client LR0.003 is a one-seed stable candidate under the v2 research gates; eighty-client v2 gate failed | [V2 report](nih-optimization-pilot-v2-report.md), independent records/config/source/assignment reconciliation | No selected eighty-client LR; no v2 repeats/clipping follow-ons; no clinical readiness |
| R4 | Client-count comparisons change local minibatch trajectories even with the same nominal image passes | [V2 step-budget audit and report](nih-optimization-pilot-v2-report.md):26–30 versus3–5 steps/client/round | Not proof this alone causes utility loss; the separately frozen v3 probe supplies later evidence |
| R5 | Eighty-client E7/R20 schedule passes across three declared seeds; matched-exposure E1/R140 gives similar validation utility with seven times the modeled payload | [V3 report](nih-schedule-pilot-v3-report.md), six independently reconciled cells/240rounds, source proofs and cost/clipping audit | Exploratory, no clinical/DP/attack claim; shuffle streams and local trajectories differ; modeled payload is not physical traffic |
| R6 | A raw-honest-update IPM oracle can invalidate reuse of a single-final-upload sensitivity bound through the upstream pipeline | [Attack-knowledge audit](attack-knowledge-privacy-boundary-audit.md), production-code scalar counterexample and fixed-attacker control | No demonstrated NIH data-neighbor breach, ALIE/trimmed-mean bound violation or replacement-noise theorem; enforce accepted-set and permitted-information conditions before the core study |
| R7 | Rejected admissions previously caused false completion counts and repeated unresolved rounds; version 2 records separate submission, admission, validation and aggregation | [Participation contract and reproduced regressions](participation-accounting-contract.md), actual coordinator/strategy tests | Synthetic execution evidence, not an observed NIH baseline failure; fixed-cohort enforcement and privacy scope still require separate checks |
| R8 | In one matched NIH validation seed at C0.05, trimmed mean improves attacked AUROC without DP but is lower than mean with central epsilon4; its coordinate noise SD is 5/3 larger | [Twelve-cell mechanism qualification](nih-mechanism-qualification-v1-report.md), all240rounds and independent Gaussian/accounting reconciliation | Exploratory single seed; no general rule ranking, causal noise-only attribution, clinical or confirmatory interaction claim; final score-based independent AUROC still required |
| C1 | A1,000-client/two-round tiny-head NIH simulation executed in7.063s end-to-end | [V1 cost check](nih-validation-pilot-v1-report.md) with actual counts, CPU/RSS scope | Execution only; not convergence, clinical validity, physical-device latency or large-head performance |
| C2 | All four head sizes pass the declared eighty-client/two-round cost gate; MLP512 has 248.245 times the linear modeled payload and a 1,149.66 MiB observed memory peak | [Prospective head-cost report](nih-head-cost-probe-v1-report.md), four cells/eight rounds and independent source/allocation/FLOP/payload audit | One fixed-order timing sample per head; engine-completion RSS sample; not convergence, clinical/physical-device performance or a general runtime scaling law |
| P1 | Current diagnostic JSON is a research record, not a DP-safe deployment export | [Output-boundary audit](privacy-output-boundary-audit.md), seeded-noise reconstruction proof | Model accountant alone does not protect disclosed randomness/pre-noise diagnostics or patient identity |
| H1 | Fixed512-feature heads contain1,026/4,122/65,922/263,682 trainable parameters | Current plan and model construction tests | Capacity/optimization change with width; measure cost and match each head's no-DP baseline |
| E1 | Earlier945-run Milestone2 grids are exploratory, not NIH confirmatory findings | [Retrospective Milestone2 report](capstone-milestone-2-report.md), immutable raw records/summary | Keep dataset, metric version and tuning history separate from phase2 |

## Conclusions still requiring evidence

- The selected core NIH schedule now passes all three declared seed gates, and
  clean clipped mean/trimmed mean pass in the base seed. Twelve exploratory
  full-schedule privacy/attack validation cells also qualify, exposing a trade-off;
  confirmatory contrasts remain unrun and these checks do not substitute for them.
- Larger-head learning and privacy contrasts under matched no-DP baselines.
  The completed short cost qualification cannot establish convergence or usefulness.
- Execution of the [frozen protocol](nih-phase2-study-v1-protocol.md): one primary
  final-round contrast, ten matched new seeds and bounded head/cohort/clipping
  families (360 cells total). Exact settings, release assumptions and inference/
  failure/resource rules are fixed; final model/score artifact API is verified on
  synthetic controls. The source-bound study runner and independent score analysis
  are implemented and verified (68 focused checks;2,076 full-suite passes,
  14 deselected,81.66% coverage). A committed source checkpoint must precede
  execution. See the [runner contract](nih-phase2-runner-contract.md).
- Paired privacy/robustness results against matched no-DP clean/attacked baselines.
  Do not attribute every raw head-size/cohort difference solely to dimension/noise.
- Official-test classifier evaluation only after the confirmatory protocol is
  frozen. Preserve declared prevalence and patient-aware sensitivity checks;
  retain bound final model/scores to independently recompute AUROC and transfer
  a validation-selected threshold rather than infer these from aggregate JSON.
- A practical recommendation table with measured setting, utility/cost effect,
  uncertainty, applicable assumptions and a concrete FedLearn action.

## Eight-page working structure

This is a layout budget to refine against the advisor's actual submission rules,
not an assertion that formatting is approved.

| Section | Approximate pages | Material and gate |
| --- | ---: | --- |
| Abstract, problem and contributions |0.75| State actual question, result and limits after confirmatory analysis; no invented result placeholders |
| Related work and declared threat/privacy model |1.00| Read primary sources fully where a theorem is invoked; distinguish robust-optimization assumptions from empirical behavior |
| System, dataset and reproducibility |1.25| Patient/content audit, fixed representation, client ownership, production-strategy simulator, versioned scoring and private/public artifact boundary |
| Prospective study design |1.25| Explicit tuning history, exact frozen core/sensitivity cells, matched controls, failure policy and uncertainty units |
| Results |2.00| Primary paired contrast first, then bounded secondary privacy/attack/head-size/cost findings; retain negative results |
| Practical decisions, limitations and conclusion |1.00| Recommendations linked to claim IDs; simulated-client DP, report-derived labels, shift, private-release limits and unmeasured deployment scope |
| References / formatting allowance |0.75| Verified citations and final template checks |

Detailed configs, manifests, raw research records, complete curves and diagnostics
belong in the reproducibility evidence package/appendix; the report must still
contain enough method detail to assess the primary finding without hidden logs.

## Presentation and demonstrable workflow

The final deck should reuse the problem/method framing of Milestone2 but replace
result claims with the verified final evidence. Keep old milestone decks intact.
Show one comprehensible privacy/robustness comparison, its matched baseline and
one practical conclusion before secondary grids. Label exploratory pilots clearly.

A demonstration must load a qualified workload, configure a bounded simulation,
show actual client ownership/counts and metric version, run a small case and
inspect outcomes/failure handling. The Simulation Lab still needs a design pass
before implementation. A replayed artifact is a useful fallback, but must be
identified as a replay; no network/phone availability is required for the cached-
feature simulator. Neither slides nor a smooth demo can substitute for study
validity, the written report or reproducible underlying evidence.
