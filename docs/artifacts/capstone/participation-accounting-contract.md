# Participation accounting before phase two

19 October 2026. Execution-correctness gate; no new NIH performance experiment.

## Reproduced defect

The simulator previously ignored the coordinator's `False` acknowledgment for a
nonpositive example count. It added that submission to `RoundRecord.reported`
anyway. A nominally full cohort could then skip incomplete-round resolution even
though the coordinator was still waiting for an update. Regression fixtures
reproduced repeated round 1 records and no aggregation; an all-rejected cohort
produced three records instead of stopping after one under the existing policy.

The engine also wrote `participants.aggregated = len(submissions)` before ingress
and strategy validation. A three-client Krum fixture reported three aggregated
inputs despite the reducer refusing the round. These are observed synthetic
regressions, not evidence that the retained NIH baseline suffered these failures.

## Version 2 record contract

New engine records declare `meta.participation_schema_version = 2`.

| Round participant field | Meaning |
| --- | --- |
| `sampled` | Selected client slots before availability handling |
| `dropped` | Selected clients removed by simulated dropout |
| `straggled` | Arrived uploads excluded by the modeled deadline |
| `submitted` | Candidate uploads passed to coordinator admission after availability handling |
| `admitted` | Submissions acknowledged `True` by the coordinator; `len(RoundRecord.reported)` |
| `accepted` | Inputs passing strategy validation; `null` when the strategy never ran |
| `aggregated` | Accepted input count on successful model production; zero on refusal |

`aggregated` is an input count, not a count of nonzero weights or coordinates
retained by a robust reducer. Krum may select one of several accepted inputs;
coordinate-wise trimming can retain different contributors at each coordinate.
The strategy's existing internal `last_round_stats["aggregated"]` remains the
accepted-input counter, including when the reducer later refuses. The engine
translates it into the two distinct public fields above.

Both direct and collect-then-submit simulator paths count an update as reported
only after a successful acknowledgment. Incomplete cohorts use the coordinator's
existing resolution policy: aggregate when its minimum is met, otherwise stop.
No quorum, aggregation rule, noise calibration or attack algorithm is changed.

## Regression evidence

- One of three direct clients returns zero examples: only clients 1 and 2 are
  reported, both requested rounds advance, and each force-aggregates their updates.
- One of ten engine clients returns zero examples: submitted 10, admitted 9,
  accepted 9, aggregated 9. Both rounds advance and mean sensitivity is `2C/9`.
- One client adds an extra parameter key: submitted/admitted 10, accepted and
  aggregated 9. This exercises the strategy's stricter validation after ingress.
- All ten return zero examples: admitted 0, accepted `null`, aggregated 0; one
  refused round and an early stop, with no fabricated utility.
- Three valid Krum inputs with `f=1`: accepted 3, aggregated 0, refusal recorded.
- Ordinary late-client handling still records seven submitted, admitted, accepted
  and aggregated inputs from ten selected clients with three stragglers.

The RED run contained five failing cases before the fix. The two focused test
modules pass 48 tests after it. The full framework suite passes 1,837 tests,
14 deselected, with 81.31% coverage in 189.19 seconds. Unhandled thread exceptions
were treated as errors. Existing dependency deprecation/export warnings remain.

## Compatibility and scientific limits

Historical records are preserved. Absence of the version marker means the old
`aggregated` field must not be interpreted as proof of strategy acceptance or
model production. Config resolution and hashes, training math and historical
Kermany numerical anchors are unchanged by this metadata contract. Rejected
submission trajectories intentionally change because they previously failed to
resolve correctly.

Correct counts are necessary for enforcing the fixed accepted cohort in a
prospective study; they do not establish a privacy guarantee by themselves. The
[attack-knowledge audit](attack-knowledge-privacy-boundary-audit.md) and
[research-output boundary](privacy-output-boundary-audit.md) remain separate gates.
The future confirmatory runner must reject missing or old participation metadata,
partial or invalid cohorts and reducer refusals when claiming the fixed-cohort
mechanism assumptions. That runner enforcement is not implemented by this fix.
