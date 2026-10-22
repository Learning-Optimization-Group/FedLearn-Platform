# Effusion R0 number of central-DP releases (A15): result

22 October 2026. Executes amendment A15, which was committed before the run. Everything is on the
selection half; the holdout is untouched.

**Integrity:**
- All 75 runs were finite with no refusals.
- Each DP run's z matched its exact-GDP calibration.
- The T = 20 cells reproduced A13 (DP) and A8 (clean) margins exactly.
- Intervals are paired patient bootstraps (2,000 resamples, seed 20261034).

**Setup:** R0, A8 recipe (LR 0.03, clip 0.05, 7 local epochs), 80 clients, five seeds. z is
calibrated per T with exact GDP. At claimed ε 4 it is 1.08 / 1.53 / 2.42 / 3.42 / 4.84 for
T = 1 / 2 / 5 / 10 / 20.

## Seed-mean AUROC

| T | Clean | ε 8 | ε 4 | DP cost at ε 4 (clean − DP) |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.501 | 0.501 | 0.501 | −0.0005 [−0.0011, +0.0002] |
| 2 | 0.523 | 0.524 | 0.525 | −0.0019 [−0.0033, −0.0004] |
| 5 | 0.595 | 0.592 | 0.588 | +0.0062 [+0.0033, +0.0092] |
| 10 | 0.698 | 0.691 | 0.678 | +0.0208 [+0.0161, +0.0255] |
| 20 | 0.774 | 0.750 | 0.712 | +0.0614 [+0.0545, +0.0678] |

Each T < 20 is below T = 20 at every privacy level; every interval excludes zero. At ε 4:
- T = 10 is −0.034 [−0.043, −0.026] relative to T = 20;
- T = 1 is −0.211 [−0.235, −0.187].

## Findings

1. **Negative result for this recipe: fewer releases lose.**
   - Fewer rounds do shrink the DP cost, from 0.061 at T = 20 to zero within noise at T ≤ 2.
   - But the clean model collapses faster, to 0.501 at T = 1, so T = 20 is best at every
     privacy level.
2. **Why the clean model collapses:** the clip-bounded travel from a random initialisation.
   - With a binding clip of 0.05, the global head can move at most 0.05 per round.
   - The randomly initialised head has weight norm 0.81. At T = 1 the final weight norm is still
     0.81, so the weight *direction*, which alone sets AUROC, is essentially random.
   - Even over T = 1 → 20, the weights move only 0.24, while the bias moves 0.65. Effusion is
     rare, so much of the clipped budget goes to the bias.
3. **What this suggests:**
   - At this recipe, the dominant limit on DP utility is not noise alone. The random head
     initialisation also acts as a fixed perturbation that a binding clip cannot overcome.
   - A zero-initialised head removes it at no privacy cost. That is amendment A17, pre-registered
     before this note was written.

**Limits:**
- The recipe was tuned at T = 20 and held fixed. A few-round recipe (larger clip and LR,
  zero init) could change the ranking; this result does not show that fewer rounds can never win.
- One workload; IID; selection half.

Private evidence: `summary.json`
`8260acba9de0a68b6d2caddb92e4eaf97008c8fa24175700217abe3943c3c05e`.
