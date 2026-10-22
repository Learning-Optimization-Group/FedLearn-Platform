# Capstone work checklist

Updated 20 October 2026. Scope: the capstone and the FedLearn foundations it uses.
This is the current completion checklist; older reports/decks remain historical
records. Checked items have completed evidence; an unchecked item may be underway.
The final capstone is **not complete**. The fixed 360-cell study has finished and
independent and original-source analyses agree. Study review and its validation
correction are complete; assessed deliverables and the working dashboard remain open.

## Live checkpoint

**Priority update, 20 October:** follow the approved
[next-phase plan](capstone-next-phase-plan.md): stronger centralized and federated
learning baselines first, then clean privacy/security comparisons, attacks and
heatmaps. A good integrated Simulation Lab remains a required final deliverable.
The completed study stays intact as the earlier reference. The older schedule
below is superseded where this order differs.

**Follow-up target:** one specific condition, as requested. Condition selection is
pending; do not silently reuse the earlier any-finding labels or the mobile
pneumonia task. Preserve the completed study separately.

### Next-phase work, in priority order

- [x] Audit all fourteen candidate conditions in the existing development cohort
  without fitting or test scoring. [Feasibility audit](condition-feasibility-audit-2026-10-20.md):
  independent counts agree; positive-patient scarcity and misleading majority
  accuracy identified. No condition or new model has been selected by this audit.
- [x] Prepare a source-bound medical feature candidate without NIH fitting:
  [CheXpert DenseNet-121 check](medical-backbone-candidate-2026-10-21.md). Exact
  checkpoint/library pinned; two synthetic CPU inputs yield finite 1,024-value
  features with exact fresh-process replay. Nine reproduction tests pass.
  The 512-wide legacy cache is unchanged; no new disease accuracy is available.
- [x] Check real development inputs for that candidate without fitting:
  [eight-image probe](medical-input-probe-2026-10-22.md). Pinned cache/manifests,
  whole-set patient separation, label-free hash selection, raw-byte and decoded-pixel
  audit bindings and an explicit grayscale contract pass; 8 × 1,024 frozen features
  are finite and replay byte-identically. No test manifest, labels, fitting or scores.
- [ ] Select the condition and define the prediction purpose, labels, baseline acceptance measures and a genuinely
  untouched final evaluation; do not tune to the already examined NIH test results.
  **22 Oct:** user selected `Effusion` and a locked internal holdout. Label-blind,
  patient-whole 50/50 validation carve locked (2,800/2,800 patients; receipt
  `083aa7ce…c794`). [Protocol v1](effusion-baseline-protocol-v1.md) drafted and
  frozen on the user's approval (22 Oct).
- [x] Run the frozen centralized gate on development data:
  [gate v1 result](effusion-gate-v1-result-2026-10-22.md). Versioned 1,024-wide R1
  cache built (77,725 images, none refused) and replay-verified; 20/20 fits converged.
  Selection AUROC R0 0.822, R1 0.829; ΔAUROC +0.007 [−0.004, +0.018], so **the gate
  failed on criterion (a)**; both exceed the 0.80 floor. Federated baseline not started.
  **Decision pending:** amend to proceed on R0 or R1, or add bounded fine-tuning.
- [ ] Audit and improve a bounded centralized learning baseline on development data.
  The [initial audit](learning-baseline-audit-2026-10-20.md) is complete: development
  payload identities/statistics checked, no centralized NIH comparison found, and
  representation/optimization/federation effects remain unresolved. No new fitting
  or test scoring has been performed; the prospective diagnostic is next.
- [ ] Qualify a matched federated baseline without DP or attacks.
- [x] Audit privacy/security readiness against source and fresh controls:
  [readiness report](privacy-security-readiness-2026-10-20.md), 218 checks passed.
  Local DP exists in the generic engine but is excluded from the qualified NIH
  path; secure aggregation is on DeComFL's scalar channel. No new NIH runs.
- [ ] After the learning gate, qualify local DP with separate client/server noise
  fields and explicit clipping/release accounting. Preserve historical record semantics.
- [ ] Choose and qualify a matched secure/plain training path before attributing
  performance differences to secure aggregation; mark incompatible combinations unavailable.
- [ ] Freeze a supported follow-up comparison of central DP, local DP and secure
  aggregation, including clean controls and explicit privacy/trust/compatibility scope.
- [ ] Run the bounded attack study across declared seeds, retaining all outcomes.
- [ ] Build absolute-performance and attack-drop heatmaps with uncertainty and costs.
- [ ] Complete and browser-verify the integrated simulator, including persistence,
  ownership, live progress, stop/reopen, comparison, heatmaps and research export.

The Spring API checkpoint passes 32 focused checks and the full backend gate:
1,207 tests, no skips, 84.09% line coverage. An earlier admin-login HTTP EOF did not
recur in isolation or the instrumented full run; its cause is unresolved, not fixed.
React and actual browser acceptance remain open. Preserve the tested checkpoint
while prioritizing learning quality.
The API is committed locally at `53d60fc`; no public push or merge is implied.

| Item | Current evidence |
| --- | --- |
| Working branch | `capstone`, isolated worktree |
| Tested source | `3baea7d` — fixed NIH study runner and score analysis |
| Full framework verification | Study source: 2,076 passed/81.66%; later worker: 2,183 passed/79.82%; 14 deselected; thread exceptions treated as errors |
| Focused runner/analysis verification | 68 passed; both Important review findings reproduced before correction |
| Backup | Delivery commit `4e80033`, exact Git bundle and 4,729 source/evidence files hash-verified on Orin; original raw-study backup retained |
| Main study | **360/360 qualified**, 7,200 rounds; 0 failed, timed out or unrun |
| Run status | Original process exited normally; all four fixed families completed once |
| Study verification | 360 models/both splits replayed, 148 source files and 16 allocations verified; primary D=+0.002266150, paired 95% CI [+0.000704398,+0.003827901], conditional on fixed cohorts |
| Results | [Study report](nih-phase2-study-v1-report.md), 12 tables/3,202 rows and six rendered figures; additional reader and corrected-checker audit passed; study review completed |

The terminal original study now has a separate second-device backup: all 3,964
files hash-verified, including final models/scores/proofs/configs/logs. Earlier source,
cache, historical and first 80-cell checkpoints remain preserved. The reviewed
audit/report/figures and exact delivery source checkpoint now also have a verified
4,729-file private backup. Charging resumed during capture
after battery operation; observed timing interpretation discloses that transition
and the in-flight backup as background load.

The study index and verification receipts are authoritative. The overview and
claim register now reflect the completed execution and reconciled outcomes; older
implementation entries are explicitly historical. Original execution source stays
pinned at 3baea7d. Reviewed delivery documents are committed at 4e80033.

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

Implementation has begun: the bounded worker, choice resolver and canonical
finite/tagged record writer pass **107 focused tests**. Two lifecycle review findings
were reproduced and corrected; an installed-wheel control completed two real
synthetic-cache rounds. The full framework gate passed **2,183 tests**, 14 deselected, with 79.82% coverage.
The backend/API checkpoint also passes its full gate. The React implementation
passes its component/service/route checks and frontend release gates. Actual
authenticated browser-to-worker, keyboard and responsive acceptance now also pass;
see the verification receipt and demonstration guide below.

- [x] Implement and qualify the [bounded Simulation Lab worker](simulation-lab-worker-contract.md),
  including actual subprocess lifecycle and isolated installed-wheel verification.
  One minor oversized-integer CLI error-formatting issue remains documented.
- [x] Implement and qualify the [owned persistent Spring API](simulation-lab-api-contract.md),
  including real cookie/PostgreSQL authorization, process supervision, stop/restart,
  safe persisted research exports and the opt-in numerical worker controls.
  All 1,207 backend tests passed with 84.09% line coverage. Synthetic execution
  controls do not establish NIH model quality. One busy-response link minor and
  the unreproduced admin-login transport failure remain documented.


- [x] Implement the dashboard route, recipe review/start, actual-round polling,
  stop/reopen/export, matched-workload comparisons and historical-study page.
  The 21 October checkpoint passes **175 frontend tests**, lint and production
  build, with **57.59% line coverage**. Both tagged-nonfinite chart crashes were
  reproduced before correction. The report link now names the actual HTML file;
  all ten public manifest entries were fetched and hash-checked in the browser.
  These checks do not establish authenticated UI-to-worker execution or learning
  quality. The original Markdown download retains repository-relative references,
  disclosed on its download page; a self-contained Markdown package is deferred.
- [x] Finish authenticated keyboard/responsive verification and the real
  browser-to-API-to-NIH-worker demonstration, including ownership and stop/export.
  The [21 October receipt](simulation-lab-browser-verification-2026-10-21.json)
  binds two completed three-round jobs and one stopped two-of-twenty-round job.
  The disabled-workload phase returned 503 and retained all three jobs. A real
  screenshot exposed clipped mobile controls despite an initially green width
  check; responsive navigation and stronger checks now pass. **179 frontend tests**,
  lint/build and 57.74% line coverage pass.
- [x] Write the [live demonstration and explicitly labelled replay guide](simulation-lab-demo-guide.md).
- [ ] Verify off-device preservation of the new dashboard/browser checkpoint;
  local evidence is retained and the historical study already has verified backups.

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
- [x] Update the current plan/evidence map and this checklist from actual outcomes;
  back up the new raw study, audit, figures and source checkpoint.

## Planned — complete the capstone deliverables

- [ ] Obtain advisor confirmation of the exact working title/scope. Message draft
  exists; no approval or message delivery is assumed.
- [ ] Finish the [v2 working report](capstone-final-report-v2-working.md), using the
  evidence map and current eight-page structure. Methods, abstract, audited results
  and conclusions are drafted and scientifically reviewed. The [working PDF](capstone-final-report-v2-working.pdf)
  now includes the 21 October dashboard checkpoint in eight visually checked pages.
  The [rendering instructions](repro/RENDERING.md) and source-bound layout receipt
  record fonts/dependencies and two byte-identical fresh renders. Methods/results
  text and historical study identities are unchanged. Advisor-approved final
  layout/submission and delivery checks remain open.
  Verify actual advisor formatting/submission requirements.
- [ ] Integrate methods, tuning history, primary/secondary results, practical
  conclusions and limitations into one coherent argument.
- [ ] Verify primary literature/theorem citations used in final privacy/robustness claims.
- [ ] Create the final presentation from verified results; preserve the presented
  Milestone 2 deck and label older exploratory material accurately.
- [x] Implement the designed Simulation Lab workflow: configuration,
  start/stop, progress/results and record export for the cached-feature simulator.
- [x] Verify the demonstration through the actual UI and real execution; clearly
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
Local DP remains outside the completed frozen matrix, but is now a required
follow-up investigation after the learning-baseline gate. Candidate model/fine-tuning
scope will be bounded by the new baseline audit; physical thousand-device deployment
remains outside scope.
Mobile release/signing/production TLS remains parked while capstone work has priority.
There is no defensible single completion percentage: experiment preparation is
complete and the scientific result is measured and reconciled. The working
dashboard, final assessed report, presentation and rehearsal remain open.
