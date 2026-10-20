# NIH mechanism qualification implementation plan

**Execution:** inline under the continuing capstone mandate; existing isolated
capstone worktree. One final read-only review of the runner slice before real data.

**Goal:** Qualify full-schedule real-data central-DP and own-data/update attacks
under matched controls before freezing the confirmatory study.

**Architecture:** A fixed twelve-cell persistent runner calls the existing core
preflight/postflight wrapper in isolated source-bound children. Reuse tested atomic
record/proof helpers; preserve earlier pilots and generic engine behavior.

**Tech stack:** Python 3.12, PyTorch, pytest, existing simulation engine.

**Spec:** [Prospective validation protocol](../../artifacts/capstone/nih-mechanism-qualification-v1-protocol.md).

## Global constraints

- Exactly 12 cells: clean/sign_flip/label_flip × mean/trim0.2 × none/central eps4,
  delta1e-5; linear1026,80/80clients,E7/R20,LR.003,batch256,C.05,CPU4,seed20261022.
- Validation margin_v2 only; pinned pretrained cache, natural prevalence, patient IID.
- Per-child180s,totalchild1800s,RSS2GiB engine-completion postgate; parent time separate.
- Source/config/protocol/cache binding, initial12entries/hashes, retained failed
  records/logs/proofs, no overwrite/retry/adaptive cells, stop on source mismatch.
- No utility gate or model selection, test scoring or full-record DP guarantee.
- Commit protocol and tested/reviewed source before real runs. No concurrent tests.

## Review focus

1. A child must reject arbitrary modified recipes and official-test selection before training.
2. Matched conditions need the same partition/init/client/attack/noise streams, with actual attacker sets checked.
3. Parent must requalify DP fields and full participation rather than trusting a marker.
4. Failed/timeout/budget/source-corrupted cells remain visible with usable evidence.
5. Full-call/loop/child time and memory sampling scopes must remain distinct.

### Task 1: Fixed mechanism runner

**Files:** create `framework/benchmarks/nih_mechanism_qualification.py` and
`framework/tests/test_nih_mechanism_qualification.py`.

**Interfaces:** `configs(cache,pin) -> list[dict]`; `run_child(root,config_path,
output,protocol_path) -> int`; `summarize_record(record,expected_config,commit)`;
`run_qualification(cache,pin,output) -> dict`; CLI cache/pin/output, exit2 on incomplete.

- [ ] Write tests for all12exact recipes, matching seeds/attack assignment,
  refusal before core, DP/marker/participation/source tampering, initial hash/index
  persistence, ordinary failure/timeout/budget/source-stop evidence. Run RED.
- [ ] Implement fixed builder/isolated core child/parent orchestration; use existing
  source/import verifier and atomic/failed-record helpers without changing history.
- [ ] Run focused tests; expect all pass, including actual small-core DP invocation
  and interruption proof survival. One fresh final review; substantive fixes RED→GREEN.
- [ ] Full framework suite with thread exceptions as errors; expect green/coverage
  gate. Commit verified source before real execution.

### Task 2: Real validation mechanism evidence

**Files:** ignored private mechanism-qualification-v1 output/checker/audit/logs;
public report and current capstone status/evidence-map.
**Consumes:** committed Task1 CLI and frozen protocol, qualified real cache.
**Produces:** twelve retained cell outcomes and qualified or failed combined-path gate.

- [ ] Run once with no concurrent tests. Retain every failure/unrun outcome.
- [ ] Independently reconcile source/config/cache identities, 240rounds if complete,
  common assignments, noise/accounting, metrics/cost totals and timing/resource caps.
- [ ] Report all results and exact limitations; no test/convergence/clinical claim
  from a single seed. Commit results/status, then verify private second-device backup.

The confirmatory protocol and study remain subsequent work; this is an exploratory
combined-path gate, not permission to relabel validation outcomes confirmatory.
