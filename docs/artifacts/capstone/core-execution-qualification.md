# NIH core execution qualification

19 October 2026. Implements the execution gates identified by the
[attack-information audit](attack-knowledge-privacy-boundary-audit.md) and
[participation contract](participation-accounting-contract.md).

## Implemented boundary

`fedlearn.simulation.core_qualification.run_core` resolves and checks a candidate
recipe **before calling the engine**, runs the existing engine unchanged, then
qualifies the returned record. The expected cache-report digest, cohort size and
evaluation split are policy arguments supplied separately from the candidate
configuration. They must come from the future runner's frozen protocol, not be
copied blindly from an arbitrary candidate.

Preflight requires NIH, whole-patient IID allocation, natural prevalence,
`margin_v2`, a matching cache pin and evaluation split, full declared participation
(default 80/80), no deadline, clipping, mean or trimmed mean, and central DP or a
matched no-DP control. Only clean, own-update sign flip/scale and own-data label
flip behaviors are permitted. IPM/ALIE and faulty-client behaviors are refused,
including zero-fraction declarations. Actual rounded attacker assignment must
leave at least one honest client. Matrices, nonfinite settings and numeric strings
(including `"Infinity"`) are refused; numeric knobs require actual JSON numbers.

Postflight checks resolved configuration/hash, workload identity, attack
assignment, privacy-mode/clipping metadata, contiguous complete rounds and schema 2
participation. Every round must have the full sampled/submitted/admitted/accepted/
aggregated count, zero dropout/stragglers, no force-resolution and no refusal.
Stopped or missing rounds fail. Sensitivity is checked against `2C/n` or
`2C/(n-2 floor(trim_ratio*n))`, and noise standard deviation against the recorded
multiplier times that sensitivity. The multiplier is independently recomputed
from configured epsilon, delta and total releases using the existing accountant,
and epsilon is recomputed for every release prefix. This checks consistency with
the implemented accountant, not an independent theorem verification. Utility must
be finite and in range.

Only successful results gain `meta.core_qualification`, version 1, with scope
`simulation_execution_only`. A record already carrying that marker is refused by
the standalone postflight validator rather than blindly trusted. The input recipe
is copied during resolution so changes to the caller's configuration in a round
callback do not alter the declared execution recipe.

## Failure and provenance handling

`CoreQualificationError.record` holds the complete returned record on postflight
failure, without adding a success marker. The future persistent runner must save
that record and record a failed cell. Preflight errors carry `record=None` because
training did not run. Engine exceptions propagate; its runner must retain the
traceback, logs and any existing artifacts. The wrapper alone is not a persistent
experiment runner and does not silently replace a failed cell with a new seed.

The NIH cache loader still checks actual report and payload hashes. The expected
pin must identify the independently qualified pretrained cache for the real study.
Synthetic random-feature caches exercise the complete workload/coordinator/strategy
route in tests; successful synthetic records explicitly say `feature_control:true`.
Execution qualification is not a certificate of useful features or learning.

## Privacy and study limits

The marker explicitly says `research_record_dp_safe:false`. Published seeds and
pre-noise diagnostic fields remain research artifacts. These checks enforce
specific execution premises; they do not prove a complete privacy theorem,
protect diagnostics, generate secret deployment randomness, certify clinical
utility or establish patient-level DP. The protected-input/release assumptions
still need to be stated in the study protocol.

Historical engine entry points, oracle algorithms, generic config defaults and
retained experiment records are unchanged. The old runners do not retroactively
gain these checks. The new helper must be called by the future study runner;
merely adding the module cannot constrain another entry point.

Default evaluation is validation. Test evaluation requires an explicit expected
split argument matching the candidate. The helper does not certify protocol
freeze: the actual study runner must bind the frozen protocol and source before
any official-test classifier evaluation. No official-test performance was
evaluated while implementing this slice.

## Verification scope

The initial delegating implementation failed all 42 initial cases. Review found
two substantive omissions: calibration/accounting was only checked for internal
consistency, and string-encoded infinity bypassed JSON finite-value serialization.
Both were reproduced with failing tests before being fixed. The checks include
before-engine rejection sentinels, actual
three-client mean/trimmed-mean execution, own-data/own-update attacks, rejected
admissions, strategy-invalid updates, stopped runs and malformed record fields.
The final focused module passes 52 tests. A fresh full suite after both fixes
passes 1,889 tests, 14 deselected, 81.60% coverage in 188.22 seconds, with unhandled
thread exceptions treated as errors. Existing dependency/runtime warnings remain.
The two review findings were verified with RED→GREEN tests and that fresh suite;
no second review was used. These synthetic checks are not new NIH performance
experiments.
