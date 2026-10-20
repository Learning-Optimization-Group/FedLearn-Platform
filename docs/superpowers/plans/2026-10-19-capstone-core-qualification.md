# NIH core execution qualification implementation plan

**Execution:** inline, using test-driven development and one final independent review.

**Goal:** Reject unsupported attack knowledge before training and refuse incomplete
or inconsistent fixed-cohort records before accepting a core study cell.

**Architecture:** A separate simulation module wraps the existing engine. It
resolves and checks the recipe before dispatch, then validates the complete result.
The generic engine, historical runners and attack algorithms remain unchanged.

**Tech stack:** Python 3.12, existing simulation config/engine and pytest.

**Specification:** [Attack-information boundary](../../artifacts/capstone/attack-knowledge-privacy-boundary-audit.md)
and [participation schema](../../artifacts/capstone/participation-accounting-contract.md).

## Constraints and decisions

- Default fixed cohort is 80/80; an explicit positive integer cohort argument
  supports bounded controls and tests. No sampling credit is claimed.
- The expected report hash and evaluation split come from the caller's declared
  protocol, independently of the candidate recipe. Default split is validation;
  test requires explicit caller selection after protocol freeze. This helper
  does not implement or certify that future freeze or source-bound subprocess runner.
- Require NIH, patient IID, preserved prevalence, margin_v2, pinned cache identity,
  no deadline, central/no DP only, clipped mean/trimmed mean, and at least one
  honest client after actual deterministic assignment. Permit only clean,
  own-update sign_flip_scale and own-data label_flip recipes. Reject other
  behaviors even at zero fraction, matrices and nonfinite numeric settings.
- Cache loading remains responsible for validating actual report/payload hashes.
  The independently chosen pin must identify the qualified pretrained cache for
  the real study. Synthetic random-feature controls exercise the same execution
  contract in tests; this helper does not certify representation quality.
- Qualification scope is simulation execution only. Secret noise, a protected
  export, clinical validity and a complete privacy proof are separate concerns.
- No historical data, algorithm, config schema, model math or anchors change.

## Review focus

1. Unsupported recipes must fail before any engine/model/data access.
2. A rejected admission or strategy validation must not pass the full-cohort gate.
3. Missing/old/boolean/fractional counters and incomplete rounds must fail closed.
4. Mismatched config, workload identity, noise or sensitivity must not qualify.
5. Failed qualification must retain the complete returned record for diagnosis.

### Task 1: Preflight, postflight and actual execution wrapper

**Files:** create `framework/src/fedlearn/simulation/core_qualification.py` and
`framework/tests/test_simulation_core_qualification.py`; update capstone status/docs.

**Interfaces:** `prepare_core_config(cfg, *, feature_report_sha256, clients=80,
evaluation_split="validation") -> dict`; `qualify_core_record(record, cfg, *,
feature_report_sha256, clients=80, evaluation_split="validation") -> dict`;
`run_core(cfg, *, feature_report_sha256, clients=80, evaluation_split="validation",
on_round=None, should_stop=None) -> dict`. `CoreQualificationError(ValueError)`
carries `.record` for postflight refusal, otherwise `None`.

- [ ] Write tests for the five review conditions, using the actual NIH workload,
  coordinator and strategy for success and invalid-admission routes. Expected
  counts and sensitivity are literal hand-derived values for three-client fixtures.
- [ ] Run focused tests before implementation. Expected: failures for missing
  qualification functions, then behavioral failures until implementation is complete.
- [ ] Implement preflight and wrapper; compare metadata/config hash, expected
  workload identity/assignment, complete sequential rounds, strict schema2 counts,
  no refusal/forced/stop, finite utility and expected sensitivity/noise. Only attach
  `meta.core_qualification` on success; reject a preexisting success marker.
- [ ] Run focused tests. Expected: all pass, including record retention and the
  pre-training rejection sentinel.
- [ ] Run the complete framework suite with unhandled thread exceptions as errors.
  Expected: all selected tests pass and coverage exceeds repository minimum.
- [ ] Obtain one read-only review of this bounded slice, fix substantive findings
  with RED→GREEN regressions, document scope and verify the final diff.
- [ ] Commit verified code/docs and create a hash-verified private Orin checkpoint.

No real NIH privacy/attack run is authorized by this implementation plan alone.
Larger-head qualification and the frozen prospective study remain subsequent work.
