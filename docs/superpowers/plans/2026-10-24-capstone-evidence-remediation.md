# Capstone evidence remediation and finalist study

## Objective

Complete the accepted sequence: correct existing claims and inference, qualify the real Effusion/private mechanism pipeline, diagnose aggregation and attack choices, freeze and execute a compact finalist study, then integrate the evidence into the assessed working report. Critically assess each phase before implementation and report the outcome of every phase. Completion of an earlier implementation checkpoint does not establish completion of this objective.

## Invariants

- Work in the existing local `capstone` worktree, preserving the in-progress ByzFL integration.
- Never reopen or evaluate the existing raw holdout/test cohorts or their saved predictions. Published aggregate results may be cited unchanged.
- Keep private evidence local. Share only aggregate results, source code and nonidentifying provenance.
- Preserve original experiment artifacts and numeric results; corrections and reanalysis are explicitly retrospective.
- Use new output directories, hash-pinned inputs/source and retained failures. No replacement runs that hide failures.
- No commit, push, merge, rebase, stash or deletion without the required explicit authorization. Prospective experiment amendments must satisfy the existing pre-run checkpoint rule before new patient-data training begins.
- Scientific inference targets must distinguish fixed-head patient uncertainty, training-seed variation and external population validity.
- No clinical-readiness claim, universal method ranking, or whole-program DP claim follows from these experiments.

## Phase 1 — Correct and reanalyse existing evidence

Critical questions: which claims are unsupported rather than merely imprecise; can saved scalar metrics reproduce the published contrast points without reading any holdout; which conclusions survive training-seed uncertainty?

Deliverables:
- Correct A14/A19/A21/A22/A23, A11/A12, guidance, checklist, handoff and literature status; preserve original result tables/artifact hashes.
- Implement a tested selection-only seed reanalysis from hash-verified aggregate summaries for A11/A12/A19/A22/A23 and their declared A8/A14/A17 controls.
- Reconcile every original contrast point, retain the original conditional patient intervals, add explicitly post-hoc paired seed t intervals and identify unresolved differences. Do not reinterpret these as joint patient-and-seed intervals or equivalence tests.
- Publish a correction addendum and phase receipt; independently review the changed claims and reanalysis.

Acceptance: no unqualified oracle-DP/collusion/equivalence/impossibility claims in the current synthesis, complete contrast coverage with reproduced points, no input artifact mutation or holdout access.

## Phase 2 — Qualify methods and the real-workload bridge

Critical questions: is GDP epsilon correctly distinguished from the engine's RDP-equivalent calibration; is the zero-initialized feature workload identical; does the one-step composed clipping sensitivity use a justified bound; can inner clipping and historical noise be diagnosed?

Deliverables:
- Document/test the one-step bound 2*min(C,tau)/n, with fixed prior released center and fixed accepted cohort; keep unsupported mechanisms refused.
- Add separate inner-clipping/center diagnostics and tests preserving preview, failure and RNG isolation.
- Build a hash-pinned train/selection-only runner that records actual noise calibration, zero initialization, source identity, final parameters and retained failures.
- Prepare a concrete prospective qualification amendment. Obtain the required local checkpoint authorization before patient-data training if it has not already been explicitly supplied.
- Reconcile native and ByzFL mean on the intended Effusion zero-init/GDP pipeline and an admissible prior reference. Stop on unexplained mismatches.

Acceptance: real-workload parity within declared tolerance, exact GDP/noise checks pass, DP output/export boundaries remain explicit.

## Phase 3 — Diagnostic pilots and attack qualification

Critical questions: does a historical private center help or merely preserve noise; are bucket choices compatible with contamination; do methods receive fair recipe/tuning budgets; is the attack genuinely damaging within its permitted knowledge?

Deliverables:
- Small declared tau/C screen with matched noise-free/private controls and inner-clipping diagnostics.
- Nonprivate bucket/median screen with an appropriate bounded recipe check, low/high attacker counts, IID/skew and recorded contamination.
- Qualify at least one clip-aware permitted-information adversary with fixed objective/knowledge/budget, constraint checks and a damaging positive control. Keep oracle stress arms outside formal client-DP claims.
- Include a small fixed/public dimension-reduction utility screen and evaluate whether a suitable private convex-head comparator can be implemented with a matched privacy unit; explicitly justify any deferral rather than claiming it was tested.
- Retain every candidate/outcome and use pilot variation to propose finalist precision. No later tuning on finalist results.

Acceptance: pilot findings support or reject the proposed candidates with interpretable diagnostics; arbitrary radius/bucket choices and weak attacks cannot determine the final ranking.

## Phase 4 — Freeze and execute the finalist study

Critical questions: does the matrix answer one useful question; are practical effects and inference targets specified before execution; is the planned precision sufficient for the claim?

Deliverables:
- Freeze a compact matrix with matched controls, fair tuning provenance, fresh seeds, explicit attack/heterogeneity settings, practical margins and failure rules.
- Complete the required prospective source/protocol checkpoint before launch.
- Run the declared selection-only study, retain all outcomes, independently reconcile configurations/noise/final metrics and compute the declared seed analysis.
- Report conditional and exploratory boundaries; do not reopen the existing holdout to confirm later discoveries.

Acceptance: complete accounting of planned/failed/unrun cells, independently reproduced results and no unsupported equivalence/generalization claims.

## Phase 5 — Report integration and completion audit

Critical questions: do the abstract, methods, tables and conclusions tell the same story; is the contribution useful without a state-of-the-art claim; is historical confirmatory evidence distinct from later exploratory work?

Deliverables:
- Integrate historical NIH, A10's published internal evaluation, corrected Effusion follow-ups and the new method comparisons into a coherent working report.
- Generate figures/tables from machine-readable evidence, update guidance/evidence map and provide reproducibility instructions.
- Render and verify the working report and reading copies; audit the current claims against primary citations and actual artifacts.
- Run appropriate source/test gates, preserve outputs and perform a requirement-by-requirement completion audit.

Acceptance: all requested phases have inspectable evidence and no remaining required work. Advisor-specific approval, publication, push/merge and submission are separate actions, not assumed.

## Phase ledger

Initial state: active goal created from the accepted sequence; previous turn produced a critical next-step assessment and synthetic diagnostic evidence. Existing ByzFL checkpoint is uncommitted and preserved. No new patient-data experiment is running.

- Phase 1: complete. Eight original summary hashes verified, 173 original contrasts reproduced, 30 synthetic tests passed, all173 intervals independently checked. Seven result-note scientific tables and artifact hashes preserved. See phase-1-verification-2026-10-24.json and selection-seed-reanalysis-2026-10-24.html.
- Phase 2: in progress. The128 exact source files underlying synthetic-v3 were recovered and hash-verified before changing the uncommitted adapter. The one-step composed sensitivity and separate diagnostics are implemented. Framework gate: 2,348 passed,14 deselected,79.95% coverage. Adapter/qualification/synthetic-runner checks:33/12/3 passed. Three subsequent review findings were reproduced, fixed and inspected. Real preflight v2 froze130 source files and four resolved configurations with zero model fits; public manifest and phase-2-preflight-verification-2026-10-24.json are available. Preflight v1 is retained but superseded after runner changes. Awaiting explicit local checkpoint authorization before the four real reconciliation cells. Phase2 is not complete until those cells meet the declared acceptance criteria.
- Phase 3: actual-workload pilots pending. On25October, separate preparation added a public nested projection and a constrained owned-data loss-ascent helper without modifying the130 frozen Phase2 source files. Generated-data gate:24 passed; qualification-runner regression gate:12 passed. The synthetic clipped-mean control was damaged but tied with full-norm sign flip, so it does not establish a stronger attack. A reviewed feasibility note derives a scalar client-private convex-head comparator with a finite-iteration sensitivity allowance; no such solver or medical pilot is implemented yet. See phase-3-mechanism-preparation-2026-10-25.md and pilot-mechanism-preparation-verification-2026-10-25.json. Engine integration and patient-data gates remain open.
- Phase 4: pending.
- Phase 5: pending.

### Phase 2 completion — 25 October

All four real qualification cells completed80 client-round journals under checkpoint016677f. Native/ByzFL and native/prior-A17 final heads and AUROC matched exactly: clean0.8075528388932567 and GDP-epsilon4 0.7035003272499432. A separate NumPy/source audit checked hashes/cohorts/heads; the Gaussian normal-CDF tradeoff independently gave delta9.999999984911015e-6 at epsilon4. Phase2 is complete; see effusion-byzfl-qualification-result-2026-10-25.md and phase-2-qualification-verification-2026-10-25.json. Its earlier in-progress ledger line is historical.

Phase3 implementation now prepares a malicious-owned-data-only attack integration and a scalar client-private ERM comparator independently, followed by a source-frozen pilot runner. No Phase3 patient-data cells have started.

### Phase 3 screen launch — 25 October

Owned attacks, scalar ERM and aggregate analysis passed their scoped reviews. Two runner findings were reproduced and fixed: anchor failures now stop subsequent fitting, and bucket recipe selection binds the complete declared real screen to its committed manifest, frozen plan and hashed decision records. The ERM numerical path uses non-BLAS contractions after a post-Torch warning regression exposed spurious BLAS flags; 113 combined checks passed with RuntimeWarnings treated as errors.

The default sandbox framework run ended12 failures/9 errors from denied local sockets (2350 passed,7 skipped,14 deselected). An isolated denied gRPC test passed with localhost permission. The permitted full rerun passed2378 tests,14 deselected,2079 warnings,80.15% coverage in562.12s; both logs are retained locally.

Ruling: include attacked controls at every centered radius, increasing the prospective screen from68 to80 cells before any fitting. Clean-only radius selection would not test robust utility. Its cost is12 additional declared development runs; no original outcome is replaced.

Preflight v2 freezes140 files, all80 resolved configurations and the reviewed analysis source. It supersedes the unexecuted v1 after analysis-source inclusion and EOF normalization; both preflights remain preserved. Local checkpoint8ea4b7d contains the exact v2 source/protocol/public manifest. The80-cell screen was launched in a fresh output directory against private-plan SHA777b1edf934290afa29b4fba5e6467e89683231e1579f285dcec1aea75acf9bb. The46-cell conditional nonprivate bucket stage remains pending. No source used by the live screen may be changed until it finishes.

Deferred minor: scalar ERM raw records do not separately repeat the workload assignment hash; the exact cohort remains replayable from the pinned bundle, source, cell seed and patient-whole partition. This is an audit convenience limitation, not a missing privacy-unit definition.

### Phase 3 screen audit and bucket launch — 25 October

All80 screen cells completed (68 iterative,12 scalar ERM), with1360 complete80-client round journals and four exact native/ByzFL anchor comparisons. All140 pinned source hashes remained unchanged. Separate sensitivity/noise/normal-CDF checks verified all38 private cells; all owned PGD rounds used63 losses/60 gradients. A direct saved-head scorer independently recomputed rank-sum AUROC, tied-threshold AP and fixed-threshold metrics for all80 cells; maximum metric difference was2.23e-16.

The stronger owned-data attack qualified: nonprivate mean AUROC loss0.2872 vs0.0050 for originalsignflip and0.0075 forfullnormownmeansignflip. Projection128 gave private point0.7702 vsmean512 0.7295, with cleanpointcost0.0147. Historical-noise-centered clipping at tau=C harmed cleanprivateutility (0.5604); smaller tau helped nonprivate attackedutility but notprivateabsoluteutility. ScalaroutputperturbedERM remainedweak. These aretwo-seed exploratoryfindings, notfinalistconfirmation. See effusion-diagnostic-screen-result-2026-10-25.md and phase-3-screen-verification-2026-10-25.json.

The declared medianrecipe rule selected LR0.1/C1 (cleanpoint0.81983 vs0.80629). Bucketpreflight hash c2e457060835a6f7f01899d48f6b3f01e168eba8ad826937695518449979804b freezes46 newnonprivatecells and bindsthescreen/rawrecords. Checkpoint5940d13 commits its prospective manifest and public aggregate screen results; the46 cells were launched in a fresh directory. Source remains frozen until completion. PublicCSVcopies were normalized toLF; parsedfields were verified unchanged and private generated outputs remain preserved.

Report-integration audit additionally corrected A10/A17 equivalence/optimism/no-worse wording using published notes only. All original scientific table lines and evidence hashes were checked unchanged. No raw holdout inputs/predictions/models were read or evaluated.

### Phase 3 complete; Phase 4 preparation — 25 October

All46 bucket cells completed and passed separate source/record/cohort/budget/zero-noise and saved-head metric audits (920 rounds; maximum metricdifference1.12e-16). Combined Phase3 accounting is126newcells plus2reusedreferences, with2280 iterative rounds. All140 pinned sources matched at the audit before beginning final-stage source changes. See phase-3-verification-2026-10-25.json and effusion-bucket-pilot-result-2026-10-25.md.

The bucket pilot found a clean-skew gain withlargerbuckets but worsehigh-attackerpoints; itdoesnotjustify a universalbucketdefault or privatebucketing. The privateclean projection128 paired gains are0.0360543 and0.0452283 (exact seed identities), SD0.006487; a prior prose transcription using min/max pairing was corrected without changing sourceJSON/CSV or the mean effect.

Ruling: the final matrix contains12freshseeds13001–13012, fourprivacy-compatiblepolicies(mean512,projected128,trim16,center_quarter), clean/ownedPGD and noise-free/GDP4 controls,192mainfits plus4nativeanchors. This bounded budget estimates the primary privatecleanprojectiongain; the two-seed variance projection is unstable and doesnotguarantee precision. If wrong, intervalsremainunresolved; no adaptive extra seeds or retuning are permitted.

Ruling: retain center_quarter to examine its nonprivate robustness/privateutility reversal, not because itwonprivateutility. Scalaroutputperturbation and privateunqualifiedbucketpolicies are not advanced; their pilotresults remainreported ratherthan hidden. This costs a negative/comparator arm and limits conclusions to the testedpolicies.

The prospective finalist protocol separately specifies a pairedseedtinterval and a fixed-headpatientclusterbootstrap for one primarycontrast; nojointinterval, equivalence or untouched-cohort confirmation is claimed. Inferencehelpers56tests and design3tests passed scoped mathematical review. Final driver/analysis integration and its sourcecheckpoint remain pending; nofinalistpatientfits have begun.

## Checkpoint authorization blocker — 25 October

Resolved: the user explicitly replied "continue" to the concrete checkpoint-and-four-runs approval request. The14 reviewed qualification source/test/protocol/manifest files were committed locally as016677f after48 focused tests passed again. The four-cell qualification was launched against the unchanged frozen preflight v2 in a fresh25October output directory. No push or merge was performed. The paragraphs below retain the preceding blocker history.

The required local checkpoint authorization remains unanswered across the request turn and two subsequent goal continuations. The first continuation completed independent generated-data preparation; the second rechecked the authoritative branch and frozen package. HEAD remains e9e21639d5bc7d20ceb3d2ac190a67cd363d3d61, the prospective manifest is absent from that commit, and all130 frozen source hashes still match. No patient-data fit was launched.

The next required operation is creating the exact local source/protocol/manifest checkpoint, followed by the four qualification cells. REVIEW-HANDOFF.md section8 prohibits the commit without explicit authorization. Further actual-workload pilots and final study choices depend on the qualification results; the independent preparation is not a substitute for that gate. Work is blocked pending explicit permission for this local checkpoint. On authorization, verify the frozen package, commit the intended source/tests/protocol/public manifest locally, execute the four cells, and continue the remaining phases. Nothing is marked complete by this blocker.

### Phase 4 launch — 25 October

The final preflight verified all196 declared configurations, the complete126-cell parents and150 source files with zero model fits. Two integration findings were reproduced and fixed before freezing: exact bootstrap settings are enforced before artifact access, and completed raw records must match their resolved configurations, seeds, full workload/partition/projection identity and behavior assignment. The31 analyzer checks and93 other design/inference/runner/pilot checks passed; scoped review confirmed both closures and used the actual workload constructor on generated inputs.

Local checkpoint1a9f0ef freezes source/protocol/public manifest before fitting. The196 cells were launched against private-plan SHA c8b261d059e02891ebecc3e4658217cce4d8e40aa3c1b43cda4afd09d123ddc9 in a fresh output directory. No methods, seed budget or pinned analysis source may change during this run. Report preparation is independent. Phase4 remains incomplete until every outcome and declared analysis is audited.

### Phase 4 complete — 25 October

All196 cells completed3,920 full80-client rounds; all150 frozen/archive/live source files matched. Four native/ByzFL anchors had exact parameter agreement. The saved-head analyzer reconciled five metrics and independent weighted AUROC for all196 heads, then computed the frozen2,000-draw patient bootstrap with zero invalid draws. A separate audit checked98 private cells and1,920 PGD rounds and reproduced all31 seed contrasts (maximumdifference5.56e-17). See phase-4-verification-2026-10-25.json.

Primary projected128-minus-mean512 private-clean AUROC is+0.019175643, seedt95[+0.011011254,+0.027340033], narrowly clearing the declared+0.01 margin. The separate fixed-head patient interval is[+0.013135936,+0.025893490]. This is less than half the pilot point, with one negative seed; no post-result retuning or extra seeds were added. Projection led attacked point means too, but clean no-noise utility cost was0.0149231. Trimming helped private attackedutility while harming privatecleanutility; center_quarter harmedprivateutility. Secondary intervals are descriptive/unadjusted.

Report review made the pre-existing honest-client restriction explicit: attacked-run calibration does not cover a coalition member whose data affects multiple uploads. The mean-surrogate PGD objective is not a worst-case reducer-specific adversary. No holdout or official test was reopened. Phase5 report integration and visual verification are now underway.

### Phase 5 complete — 25 October

The integrated v3 working report combines historical360-cell evidence, corrected A10/public follow-ups and all326 new remediation cells. Four figures are reproducibly generated from aggregate JSON, with exact plotting data/hashes. The11-page PDF and standalone embedded-figure HTML passed numeric-token checks and visual inspection; prior layouts are preserved. Separate report review reconciled every final table cell and all31 contrasts and verified the limited primary/practical/pilot-shrinkage claims. Reproduction and completion guides, current guidance and phase receipts are available. Nine original result-note measurement tables and hashes were preserved.

All five accepted research-remediation phases are complete. The subsequent user-provided review is being critically assessed; no new experiment or submission rewrite is implicitly authorized by asking whether that review is correct. Advisor formatting/approval, presentation/rehearsal, external evaluation, production crypto, push/merge and actual submission remain outside this completed scope.
