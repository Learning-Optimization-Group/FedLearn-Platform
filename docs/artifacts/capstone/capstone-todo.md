# Capstone work checklist

Updated 20 October 2026. Scope: the capstone and the FedLearn foundations it uses.
This is the current completion checklist; older reports/decks remain historical
records. Checked items have completed evidence; an unchecked item may be underway.
The final capstone is **not complete**. The fixed 360-cell study has finished and
independent and original-source analyses agree. Study review and its validation
correction are complete; assessed deliverables and the working dashboard remain open.

## Live checkpoint

| Item | Current evidence |
| --- | --- |
| Working branch | `capstone`, isolated worktree |
| Tested source | `3baea7d` — fixed NIH study runner and score analysis |
| Full framework verification | 2,076 passed; 14 deselected; 81.66% coverage; unhandled thread exceptions treated as errors |
| Focused runner/analysis verification | 68 passed; both Important review findings reproduced before correction |
| Backup | Exact source commit/Git bundle and 518 private evidence files hash-verified on Orin |
| Main study | **360/360 qualified**, 7,200 rounds; 0 failed, timed out or unrun |
| Run status | Original process exited normally; all four fixed families completed once |
| Study verification | 360 models/both splits replayed, 148 source files and 16 allocations verified; primary D=+0.002266150, paired 95% CI [+0.000704398,+0.003827901], conditional on fixed cohorts |
| Results | [Study report](nih-phase2-study-v1-report.md), 12 tables/3,202 rows and six rendered figures; additional reader and corrected-checker audit passed; study review completed |

The terminal original study now has a separate second-device backup: all 3,964
files hash-verified, including final models/scores/proofs/configs/logs. Earlier source,
cache, historical and first 80-cell checkpoints remain preserved. Audit/report/figure
and final source-checkpoint backup is still pending. Charging resumed during capture
after battery operation; observed timing interpretation discloses that transition
and the in-flight backup as background load.

The study index and verification receipts are authoritative. The overview and
claim register now reflect the completed execution and reconciled outcomes; older
implementation entries are explicitly historical. Original execution source stays
pinned at 3baea7d. Delivery documents will be committed with the reviewed checkpoint.

## Completed — preservation, scope and existing research

- [x] Create a dedicated `capstone` branch/worktree and preserve the original
  simulation research history.
- [x] Integrate the merged FedLearn baseline while retaining mobile/XGBoost work.
- [x] Preserve the earlier 945 exploratory Milestone 2 run records, notes,
  configurations and presentation assets in a private recovery backup.
- [x] Write the previously missing [Milestone 2 report](capstone-milestone-2-report.md).
- [x] Consolidate the [capstone plan](../../../wikis/capstone/README.md), schedule,
  scope boundaries and [final-report claim/evidence map](final-report-evidence-map.md).
- [x] Preserve source plus ignored research evidence on a second device; verify
  file hashes and the exact Git branch head, rather than assuming copying succeeded.
- [x] Draft an [advisor title/scope message](advisor-title-confirmation-draft.md).
  Drafting is complete; advisor approval and sending remain separate open actions.

## Completed — NIH data and learning foundation

- [x] Prepare the **No Finding = 0 / any listed finding = 1** task; keep it separate
  from the mobile pneumonia-only task.
- [x] Verify the metadata, official lists and all 112,120 image decodes.
- [x] Audit patient/image membership and exact RGB duplicates; detect that patient
  separation alone left content leakage in the original preparation.
- [x] Independently confirm the reported duplicate/contradictory-label groups.
- [x] Quarantine 111 development images from nine non-test patients; preserve the
  complete official test set and retain the failed original audit.
- [x] Qualify curated splits: **68,650 training / 17,763 validation / 25,596 test**
  images, with separate patients across these groups.
- [x] Extract and verify frozen pretrained ResNet-18 features, with pinned model,
  transformation, source, manifest and payload identities.
- [x] Add the NIH workload and whole-patient simulated-client allocation; preserve
  natural prevalence and require explicit validation/test selection.
- [x] Preserve historical workload behavior and verify selected earlier reproduction
  anchors after integrating the platform baseline.

Evidence: [NIH audit](nih-data-audit-2026-10-19.md),
[qualified features/cohort](nih-frozen-features-2026-10-19.md).

## Completed — exploratory learning and methodological corrections

- [x] Run and report the first no-DP validation pilot and a separate 1,000-client,
  two-round execution check. The latter demonstrates simulation execution only.
- [x] Identify float32 probability-saturation error in AUROC; introduce versioned
  float64 logit margins and preserve old records unchanged.
- [x] Run the bounded optimization pilot; identify that cohort size also changes
  local minibatch trajectories, so nominal image exposure is insufficient alone.
- [x] Qualify the eighty-client E7/R20 schedule across three fixed seeds.
- [x] Compare matched image exposure with E1/R140: similar exploratory validation
  utility and seven times the modeled payload; retain the trajectory caveat.
- [x] Audit attack knowledge and diagnostic release boundaries. Restrict the main
  study to attacks using their own data/updates; distinguish model DP accounting
  from protection of research logs, seeds and patient identities.
- [x] Correct participation accounting; distinguish submitted, admitted, accepted
  and aggregated clients and prevent false completed-round claims.
- [x] Add a core qualification wrapper enforcing permitted recipes, complete fixed
  cohorts, round counts and explicit evaluation identity.
- [x] Qualify all four trainable head sizes in the bounded cost probe: 1,026,
  4,122, 65,922 and 263,682 parameters. This is a cost/execution check, not convergence.
- [x] Run and independently reconcile the 12-cell exploratory privacy/attack
  qualification: 240 rounds, clean/sign-flip/label-flip, mean/trimmed mean,
  no DP/central epsilon 4.
- [x] Publish exploratory findings with caveats: trimming helps the attacked no-DP
  result in that seed, but gives lower absolute DP utility; matched epsilon also
  gives trimming 5/3 the coordinate noise SD in the eighty-client comparison.

Evidence: [v1](nih-validation-pilot-v1-report.md),
[v2](nih-optimization-pilot-v2-report.md), [v3](nih-schedule-pilot-v3-report.md),
[head costs](nih-head-cost-probe-v1-report.md),
[mechanism qualification](nih-mechanism-qualification-v1-report.md),
[core checks](core-execution-qualification.md),
[privacy boundary](privacy-output-boundary-audit.md).

## Completed — confirmatory study preparation

- [x] Prospectively freeze the [360-cell protocol](nih-phase2-study-v1-protocol.md)
  before official-test classifier scoring: 240 primary, 72 head, 24 cohort and
  24 clipping cells, with ten new primary seeds.
- [x] Freeze the primary contrast, final-round endpoint, paired interval method,
  missing-cell rules, resource budgets and per-run privacy assumptions.
- [x] Freeze the nine-image ambiguous-content sensitivity mask while preserving
  primary official-test membership.
- [x] Capture the actual qualified final server state using owned CPU parameters.
- [x] Save and reload final models and aligned validation/test logits, float64
  margins, labels and identities; validate record/cache/model/source bindings.
- [x] Implement the fixed isolated-process runner: all configurations persisted
  before launch, retained failures, source proofs, timing/cost gates and integrity stop.
- [x] Implement separate NumPy score analysis: tied-score AUROC, threshold transfer,
  inverse-patient weighting, mask sensitivity and complete-pair primary inference.
- [x] Make saved-study analysis reopen raw artifacts and recompute summaries;
  reject edited scores, stale identities and missing artifact hashes.
- [x] Reproduce and fix both launch blockers from final review: uncontrolled parent
  CPU threads and malformed protocol/configuration failures continuing the study.
- [x] Pass focused/full regression gates, commit source and verify the private backup.
- [x] Launch the fixed study once on 20 October; persist all 360 initial entries,
  keep the Mac awake and avoid concurrent tests/source edits during timing capture.

Evidence: [artifact contract](final-state-score-artifact-contract.md),
[runner contract](nih-phase2-runner-contract.md),
[implementation plan](../../superpowers/plans/2026-10-20-capstone-confirmatory-study.md).

## Completed — dashboard demonstration design

- [x] Choose integration into the existing FedLearn dashboard and write a concrete
  [Simulation Lab design](../../superpowers/specs/2026-10-20-simulation-lab-design.md).
  Its fresh builder check found no material unanswered questions; the
  [implementation plan](../../superpowers/plans/2026-10-20-simulation-lab.md) specifies
  the Python worker, persistent Spring ownership/lifecycle, React workflow and
  actual browser evidence. Implementation/testing follows the audited study report.

## Study checks and remaining reporting

- [x] **Finish the single 360-cell execution:** all four families qualified once.
  Retain all outcomes and unrun cells; never replace failed cells to improve coverage.
- [x] **Independently reconcile the real evidence after capture:** committed source,
  configs, allocations, attackers, counts, checkpoint decoding, both split forwards,
  AUROC, privacy accounting, payload/FLOPs, timings and memory scope.
  The independent checker passed all 360 models and 7,200 rounds. The additional
  source-bound reader also reproduced the complete saved analysis. Review found one
  Important primary-summary validation gap; complete/partial/malformed regressions
  and a fresh all-model audit pass after its correction.
- [x] **Recompute the primary ten-seed effect and interval independently.** If any
  required component is missing, suppress the planned interval and report missingness.
- [x] Verify validation-selected threshold transfer, patient weighting and the
  frozen content-mask sensitivity results for every qualifying model.
- [x] Reconcile all 64 descriptive groups and matched head/cohort/clipping controls.
- [x] Write the main study report and create/render/check scientific figures and
  tables against the independently checked evidence.
- [x] Produce practical FedLearn recommendations with measured effect, cost,
  uncertainty, applicable assumptions and concrete implementation implications.
- [ ] Update the current plan/evidence map and this checklist from actual outcomes;
  back up the new raw study, audit, figures and source checkpoint.

## Planned — complete the capstone deliverables

- [ ] Obtain advisor confirmation of the exact working title/scope. Message draft
  exists; no approval or message delivery is assumed.
- [ ] Finish the [v2 working report](capstone-final-report-v2-working.md), using the
  evidence map and current eight-page structure. Methods, abstract, audited results
  and conclusions are drafted and scientifically reviewed. The [working PDF](capstone-final-report-v2-working.pdf)
  has eight visually checked pages; final submission layout and delivery checks remain open.
  Verify actual advisor formatting/submission requirements.
- [ ] Integrate methods, tuning history, primary/secondary results, practical
  conclusions and limitations into one coherent argument.
- [ ] Verify primary literature/theorem citations used in final privacy/robustness claims.
- [ ] Create the final presentation from verified results; preserve the presented
  Milestone 2 deck and label older exploratory material accurately.
- [ ] Implement the designed Simulation Lab workflow: configuration,
  start/stop, progress/results and record export for the cached-feature simulator.
- [ ] Verify the demonstration through the actual UI and real execution; clearly
  label a replay fallback as a replay.
- [ ] Prepare reproducibility instructions and a curated shareable evidence package;
  keep private models/identities/research diagnostics out of public artifacts.
- [ ] Perform final report/figures/deck/demo consistency and reproduction checks.
- [ ] Prepare the final submission and delivery rehearsal; obtain separate approval
  for any publication, push, merge or advisor message when required.

## Schedule and scope decisions

| Window | Focus |
| --- | --- |
| Now–2 November | Complete/audit/report the frozen experiment matrix; write alongside it |
| 3–16 November | Minimum Simulation Lab and usable demonstration |
| 17–30 November | Final report, presentation and rehearsal |
| 1–5 December | Correction/reproduction buffer; stated deadline 5 December |

Keep the primary study, honest reporting and reproducibility ahead of GUI extras.
Local-DP expansion is optional and not part of the frozen matrix. LoRA/full image
network training and physical thousand-device deployment remain outside this phase.
Mobile release/signing/production TLS remains parked while capstone work has priority.
There is no defensible single completion percentage: experiment preparation is
complete and the scientific result is measured and reconciled. The working
dashboard, final assessed report, presentation and rehearsal remain open.
