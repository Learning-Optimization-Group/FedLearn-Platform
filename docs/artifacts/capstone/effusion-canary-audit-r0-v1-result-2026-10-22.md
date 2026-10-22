# Effusion R0 one-run canary audit of central DP (A16): result

22 October 2026. Executes amendment A16, which was committed before the run. Everything is on the
selection half; the holdout is untouched.

**Outcome: the audit failed its own pre-registered power check, so it tests nothing.** This is a
negative result about the audit design, not evidence about the DP mechanism.

**Integrity:**
- All 40 runs were finite with no refusals.
- Each run had exactly 8 canaries, and each DP run's z matched its exact-GDP calibration.
- Bootstrap: 2,000 resamples, seed 20261035.

## Results

Scores are pooled over 10 seeds: 80 inserted canaries and 10,000 unseen directions per level.

| Privacy | μ̂ [95%] | ε̂ [95%] | Theory μ, inserted vs not (√T/2z) | Mean AUROC with canaries |
| --- | --- | --- | ---: | ---: |
| clean (positive control) | 0.150 [−0.065, 0.362] | 0.53 [0, 1.39] | ∞ | 0.767 |
| ε 8 | 0.158 [−0.069, 0.387] | 0.56 [0, 1.50] | 0.833 | 0.745 |
| ε 4 | 0.150 [−0.078, 0.388] | 0.53 [0, 1.50] | 0.462 | 0.710 |
| ε 1 | 0.101 [−0.117, 0.326] | 0.35 [0, 1.24] | 0.134 | 0.589 |

With **no DP noise at all**, canaries are indistinguishable from unseen directions: the interval
includes 0. The DP rows match the clean row, so the estimate measures the audit's own background,
not the mechanism. "No leak beyond claim" holds at every level only trivially, and must not be read
as confirmation.

## Why the audit had no power

There are two causes, both measured.

1. **Background spread from the parameter vector itself.**
   - A random unit direction's projection onto the final head has a spread of 0.030 per run.
   - That is ‖θ‖/√d ≈ 0.96/√1026. It comes mostly from the random initialisation and the bias,
     not from DP noise.
2. **Honest training erases most of the canary.**
   - Each canary adds C/n · u to every round's mean, so the expected final shift is
     T·C/n = 0.0125.
   - The observed shift was 0.0046 clean and 0.0059 at ε 4, only 37–47% of that. A canary
     direction is noise to the task, and later honest local training partly undoes it.

Against a background of 0.030, a 0.005 shift gives μ ≈ 0.15 whatever the DP setting.

## What follows

- The final-model threat model is the weakest observer, so a weak result is expected there. At
  n = 80 and d = 1,026 it is too weak to test a central-DP claim.
- The remedy follows the same paper: the **all-iterates** threat model, scoring each round's
  released update θ_t − θ_{t−1} against u.
  - Round t's canary contribution C/n is present in that release untouched by later training.
  - The background is the per-round DP noise plus a small honest projection.
  - μ̂ should then approach √T/(2z) under DP and be very large in the clean control.
  - This was pre-registered as amendment A18. It was then withdrawn before any run, because the
    averaged score telescopes back to the final model (see the follow-up below).

Private evidence: `summary.json`
`9c6692759411903cb583433a024d85e14d78baa253753e060874c9e3cbb4c42c`.

## Follow-up: A18 withdrawn, mechanism checked instead

A18 was withdrawn before any run. The averaged all-iterates score telescopes to ⟨θ_T − θ_0, u⟩/T.
That removes the init spread but not the erasure, and the projected clean μ (about 0.2, or about 0.6
for weights-only canaries) could not test even the ε 8 claim. See the protocol.

What can be verified instead is that the mechanism adds the noise it claims. Three links establish
it together:
1. `test_noise_std_is_multiplier_times_rule_sensitivity`
   (`framework/tests/test_dp_robust_aggregate.py`) measures the noise `dp_robust_aggregate`
   actually draws and checks it equals z × the rule's replace-one sensitivity.
2. `test_bit_exact_with_dp_robust_aggregate_over_three_rounds`
   (`framework/tests/test_composed_strategy.py`) shows the simulator's strategy is bit-identical to
   that function.
3. `repro/check_effusion_dp_noise_records.py` re-derives z·2C/(n − 2k) for every recorded
   central-DP round from its own config and cohort:
   - 404 runs and 7,235 rounds across A8, A9, A13, A14, A15, A16 and A17;
   - 0 mismatches, worst relative error 2.2 × 10⁻¹⁶;
   - private output `dp-noise-record-check.json`
     `1eee2af54cf6254ece05605fa7c46c12436f7b5d6117dd13cee588a3d4276e9e`.

This is a correctness check of the implementation against its own accounting. It is not an
empirical privacy estimate. An informative one-run audit at this scale would need many more
canaries and clients than the 80-client Effusion setup has; that is left open.
