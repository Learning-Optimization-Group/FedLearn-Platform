# NIH confirmatory study implementation plan

**Execution:** inline under the continuing capstone mandate. Use the existing
isolated `capstone` worktree. One fresh read-only review of each independently
rejectable code slice; verified source commits precede real-data study runs.

**Goal:** Execute and analyze the prospectively frozen NIH matched study with
reloadable final predictions, honest uncertainty and practical conclusions.

**Architecture:** Extend the engine/core with optional final-parameter capture,
preserving default numerical behavior. A dedicated research-artifact module binds
final state and aligned scores to the qualified record/cache. A fixed source-bound
runner enumerates the protocol's360cases; analysis consumes verified artifacts,
never reruns or selects models from performance.

**Tech stack:** Python3.12, existing PyTorch/NumPy/simulation/codec/pytest stack;
no additional training infrastructure or dependency installation.

**Spec:** [Frozen study protocol](../../artifacts/capstone/nih-phase2-study-v1-protocol.md).

## Global constraints

- Exactly240primary+72head+24cohort+24clipping cells, in protocol order.
- All settings/seeds/data/mask hashes, final-round endpoints and inference rules
  come from the protocol. No adaptive pilots, retries, replacement seeds or tuning.
- Qualified pretrained NIH cache, natural prevalence, whole-patient allocation;
  no source/config/cache mutation during capture or concurrent regression suite.
- Per-child180s,totalchild10800s,engine-completionRSS2GiB observation gate; complete
  source/config/artifact provenance, initial360entries and failure retention.
- Per-run simulated honest-client replace-one DP is conditional; no clinical,
  patient-DP, full-matrix/private-log or physical-device claim.
- Generic/historical default numerical behavior and old records remain unchanged.
- Private model/prediction/raw identities stay ignored; public evidence is curated
  aggregate reports/figures and reproducible source, not real-data dumps.

## Review focus

1. Final capture must use actual final server state, with no callback mutation of
   engine state and no capture promoted as qualified after a failed postflight.
2. Reloaded checkpoint and scores must match model/config/cache/row order, including
   float64 margin subtraction, binary ties and pretrained/control identity.
3. All360cells and reused controls must match exactly; fail before test scoring
   for arbitrary recipes, altered mask or unsupported raw-honest-update oracles.
4. Threshold selection uses validation only; patient weights/mask and exact tie-aware
   AUROC must be checked independently, not inferred from aggregate JSON.
5. Missing primary components prevent the planned ten-seed interval; diagnostics,
   memory/timing scopes and source failures must not silently select survivors.

### Task 1: Qualified final-state and prediction artifacts

**Files:** modify `framework/src/fedlearn/simulation/engine.py` and
`framework/src/fedlearn/simulation/core_qualification.py`; create
`framework/benchmarks/nih_final_artifacts.py`; tests in
`framework/tests/test_nih_final_artifacts.py`.

**Interfaces:** optional `on_final_parameters` callback for `engine.run` and
`run_core`, receiving owned CPU tensor copies. The core invokes the caller only
AFTER successful postflight, using internally captured final engine state.
`write_final_artifacts(cfg, record, params, output, *, feature_report_sha256)`
returns a JSON manifest; `verify_final_artifacts(...)` validates it and returns
aligned arrays/model metadata for analysis. Output must be fresh/outside cache.
Manifest binds config hash, final model digest/full file hash, cache/mask-independent
split identity, payload sizes/hashes, parameter names/shapes, logits and float64
margins/labels/IDs for validation/test. Research output is explicitly non-private.

- [x] Write failing tests for actual final-state capture/digest, default identical
  run, mutation isolation, postflight refusal/no callback and callback errors.
  Use small synthetic NIH controls; no real official-test predictions here.
- [x] Implement optional capture with unchanged default engine/core APIs and JSON.
  Preserve returned failed records if the later caller's artifact writing fails.
- [x] Write failing artifact roundtrip tests using real small NIH cache/model:
  linear and MLP, source-record/config/state mismatches, existing/inside-cache
  output refusal, score/label/identity/checkpoint tampering, partial output, nonfinite
  state/scores. No pickle/object arrays or silent dtype/coercion changes.
- [x] Implement atomic completion manifest last. Predict on CPU under eval/no-grad,
  record execution/library/batch/thread settings. Check saved model roundtrip and
  final digest; strict model key/shape matching. Derive float64 margins before
  subtraction, preserve raw float32 logits for loss/ECE/tie reconciliation.
- [x] Focused tests, one independent review; substantive fixes RED→GREEN, no re-review.
  Full framework suite/coverage/thread errors gate; commit verified source.

### Task 2: Fixed study runner and independent analysis

**Files:** create `framework/benchmarks/nih_phase2_study.py`,
`framework/benchmarks/nih_phase2_analysis.py` and corresponding tests.

**Consumes:** Task1 qualified core callback/artifact API, frozen protocol/cache/mask.
**Produces:** all360fixed config hashes and records, study index/source proofs,
analysis JSON with per-seed effects and correctly scoped summaries/failures.

- [x] Write failing cell-enumeration/order/count/control/seed tests with literal
  expected cases and selected exact recipes. Test child refusals before training
  for protocol mismatch, altered mask/pin and test-selection inconsistencies.
- [x] Implement bounded source-bound sequential children using `run_core`, actual
  declared cohort10/80/1000 and explicit evaluation split. Persist raw records,
  final-state/artifact errors and before/after source proofs. Requalify in parent;
  bind final artifacts to exact record and source. Initial360index/configs/hashes
  before launch, no overwrites/retries, ordinary failure continuation and integrity
  stop/global-budget behavior, correct child/core/loop/artifact timing scopes.
- [x] Write independent metric tests with hand-checked tied-score/weighted AUC and
  confusion fixtures, default positive iffmargin>0, validation recall threshold
  ceil(.8*Npositive) with>=ties, unchanged threshold transfer, inverse-patient weights
  and exact bound mask alignment. Include single-class/empty/nonfinite failures.
- [x] Implement analysis directly from saved scores, not production metric helpers.
  Report all endpoint components and complete pairs. Ten-seed primary D mean,
  SD/range/t interval df9 only when all80required components qualify; all other
  summaries descriptive. Missing component/duplicate/wrong-split/hash/seed/stale
  checkpoint cannot produce a formal primary interval. No unadjusted winner test.
- [x] Synthetic full CLI/failure/tamper/interrupt integration checks; one independent
  code review, fixes with RED→GREEN; complete framework suite and commit before data.

### Task 3: Execute, reconcile and report the fixed study

**Files:** private new `phase2-study-v1/` outputs and independent verification scripts;
public study report/figures/current status/evidence map and reproducibility instructions.

**Consumes:** committed tested Task2 source plus original frozen protocol and pins.
**Produces:** retained all360outcomes, independently verified scores/calibration/
matching/costs, primary analysis or explicit incomplete status, practical guidance.

- [x] Execute once with no concurrent regressions. Poll the same live session/job;
  observation timeout is not a failed run or permission to restart. Retain all statuses.
- [x] Independently verify committed source/protocol/config/cache/mask/checkpoint/
  score hashes and loaded imports, exact counts/assignments/attackers, model
  predictions and final AUROC, Gaussian/prefix accounting, payload/FLOPs and budgets.
- [x] Recompute primary effects/interval with separate arithmetic; verify threshold
  transfer and both sensitivity analyses. Publish every planned failure and all
  components; no convergence/population/clinical claim from simulator execution.
- [x] Create report and scientific figures with exact evidence-linked practical
  recommendations, assumptions and limits. Render/inspect exports and reconcile
  public numbers against independently checked records before committing.
- [x] Update capstone status/claim register and preserve execution ledger/checks/logs;
  verify fresh private Orin Gitbundle plus per-file hashes before workspace cleanup.

The full capstone goal remains active afterward: integrate verified findings into
the final written report, presentation and a designed Simulation Lab/demo workflow.
Advisor title sign-off/submission and public branch publication remain separate;
do not send messages, push, merge or claim overall completion from this slice.
