# NIH head-cost probe implementation plan

**Execution:** inline, test-driven development, one final independent review before
real measurements. Existing isolated `capstone` workspace, based on `aa8ae48`.

**Goal:** Execute and independently reconcile the four prospectively declared
head-cost cells so the final study has an evidence-based resource budget.

**Architecture:** A new bounded runner reuses source/import verification and atomic
checkpoint helpers without changing historical pilots. Isolated children call the
new core execution wrapper and persist successful or failed records. Parent-side
checks bind recipes, proofs, model size and resource readiness.

**Tech stack:** Python 3.12, subprocesses, existing simulation engine, pytest.

**Specification:** [Committed prospective protocol](../../artifacts/capstone/nih-head-cost-probe-protocol.md).

## Constraints

- Exactly linear/MLP8/MLP128/MLP512, counts1026/4122/65922/263682;80/80 clients,
  E7/R2,LR0.003,batch256,C0.4,mean,noDP/noattack,CPU4,base seed20261019.
- Fixed qualified cache pin, validation only, margin_v2, natural prevalence.
- Per-child300s, combined child600s; parent control time separate. RSS above2GiB
  fails resource qualification. No utility ranking or convergence claim.
- All four cells exist in the initial index and remain visible on failure/timeout/
  budget exhaustion. Source-integrity failure stops further execution.
- Refuse existing outputs. Save atomic before/after source proofs and raw records.
  Failed nonfinite records use an explicit tagged JSON encoding rather than being
  discarded. Do not overwrite historical evidence or change generic engine math.
- Commit runner before real measurements. No concurrent suite during measurement.

## Review focus

1. Start proof survives an actual interrupted child, without a completion claim.
2. Parent source/import/proof mismatch prevents subsequent launches.
3. A partial cohort, bad core marker, wrong recipe/model size or excessive RSS
   cannot appear qualified; failed records remain inspectable.
4. Parent verification overhead cannot consume the declared child budget; timeout
   and budget paths retain all four cases and log/artifact hashes.
5. Poor but finite short-run utility is retained rather than selecting survivors.

### Task 1: Source-bound runner and execution checks

**Files:** create `framework/benchmarks/nih_head_cost_probe.py`,
`framework/tests/test_nih_head_cost_probe.py`.

**Interfaces:** `configs(cache, pin) -> list[dict]`; `run_child(root, config_path,
output, protocol_path) -> int`; `child_command(root, config, output, protocol)`;
`summarize_record(record, expected_config, source_commit) -> dict`;
`run_probe(cache, pin, output) -> dict`; CLI accepts cache/pin/output arguments,
exits2 if any cell is not qualified.

- [ ] Write recipe/model-count, preflight, child failure/interrupt, record validation
  and four-case orchestration tests; run RED before implementation.
- [ ] Implement the fixed runner with initial protocol/config/index checkpointing,
  independent core requalification and required before/after import proofs.
- [ ] Run focused tests; expected all pass, including poor-utility retention,
  tagged failed records, wrong-proof stop and child/control timing separation.
- [ ] Review this bounded slice once; fix substantive findings with RED→GREEN.
- [ ] Run the complete framework suite after final changes; expected green and
  coverage above73%, unhandled thread exceptions treated as errors. Commit.

### Task 2: Real cost evidence and practical budget

**Files:** ignored private `head-cost-probe-v1/` run/config/log/proof records;
new report and cost figure under `docs/artifacts/capstone/`; update current plan
and final-report claim register.

**Consumes:** committed Task1 CLI and the already qualified cache.
**Produces:** all four declared cell outcomes, independently reconciled source,
configuration, counters, model sizes, resource/cost totals and bounded conclusions.

- [ ] Execute once, isolated from regression runs. Expected four retained statuses;
  success or failure is evidence, not permission for adaptive new cells.
- [ ] Independently check config/source/protocol/artifact hashes, all round counts,
  model sizes, allocation identity, FLOPs, timing sums and resource gate.
- [ ] Report exact measured scope and failures, render/inspect any cost figure;
  distinguish two-round qualification from convergence and planning extrapolation.
- [ ] Commit evidence-backed report/status and verify a private Orin checkpoint.

No confirmatory/test-classifier evaluation is part of this plan. Final study
protocol, matched experiments, report, presentation and demo remain subsequent work.
