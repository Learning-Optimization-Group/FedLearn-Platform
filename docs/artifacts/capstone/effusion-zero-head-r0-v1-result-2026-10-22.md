# Effusion R0 zero-initialised head under central DP (A17): result

22 October 2026. Executes amendment A17, which was committed before the run. Everything is on the
selection half; the holdout is untouched.

**Integrity:**
- All 40 runs were finite with no refusals.
- Each DP run's z matched its exact-GDP calibration.
- Intervals are paired patient bootstraps (2,000 resamples, seed 20261036).
  They condition on the five realized heads and do not quantify training-seed variability.
  Claim wording below was corrected on 25 October; numeric tables and artifact hashes are unchanged.

**Setup:**
- Identical to A13/A15: R0, A8 recipe (LR 0.03, clip 0.05), 80 clients, same seeds, exact-GDP z.
- The only change: the initial global head (weights and bias) is zero instead of PyTorch's
  default random init. This is data-independent, so it costs no privacy.

## Pre-registered results

| T | Privacy | AUROC, zero init | Zero − random init (same seed) | DP cost vs zero-init clean |
| ---: | --- | ---: | --- | --- |
| 20 | clean | 0.8074 | +0.0339 [+0.0291, +0.0387] | — |
| 20 | ε 8 | 0.7779 | +0.0280 [+0.0237, +0.0323] | 0.0295 [0.0252, 0.0339] |
| 20 | ε 4 | 0.7313 | +0.0192 [+0.0155, +0.0226] | 0.0760 [0.0681, 0.0840] |
| 20 | ε 1 | 0.5867 | +0.0026 [+0.0006, +0.0043] | 0.2207 [0.2055, 0.2351] |
| 5 | clean | 0.7966 | +0.2021 [+0.1824, +0.2190] | — |
| 5 | ε 8 | 0.7733 | +0.1812 [+0.1615, +0.1980] | 0.0234 [0.0198, 0.0270] |
| 5 | ε 4 | 0.7377 | +0.1494 [+0.1304, +0.1654] | 0.0589 [0.0523, 0.0659] |
| 5 | ε 1 | 0.5990 | no random-init counterpart | 0.1976 [0.1829, 0.2123] |

## Exploratory contrasts (not pre-registered; computed after the run, seed 20261037)

| Contrast | ε 8 | ε 4 | ε 1 |
| --- | --- | --- | --- |
| Zero init: T 5 − T 20 | −0.0046 [−0.0101, +0.0010] | +0.0064 [−0.0014, +0.0142] | +0.0124 [+0.0008, +0.0244] |
| Zero init T 5 − A13 (random init, T 20, exact z) | +0.0233 [+0.0159, +0.0304] | +0.0256 [+0.0167, +0.0343] | +0.0149 [+0.0030, +0.0268] |
| Zero init T 5 − A8 (random init, T 20, RDP z) | +0.0299 [+0.0222, +0.0371] | +0.0430 [+0.0335, +0.0524] | +0.0333 [+0.0212, +0.0456] |

Without DP, T 5 − T 20 is −0.0107 [−0.0142, −0.0075].

## Findings

1. **Zero initialization improved the observed points in every tested matching cell.**
   It is data-independent and adds no privacy expenditure. The utility finding is
   conditional on these heads, recipes and cohort; it is not a universal improvement guarantee.
   - At T = 20 it adds 0.019–0.034 AUROC except at ε 1.
   - It also improves the clean DP-recipe model from 0.774 to 0.807.
2. **It removes A15's collapse.** With random init, T = 5 was 0.59 clean. With zero init it is
   0.80, because the learned direction is no longer swamped by the random starting weights.
3. **Fewer releases now pay off under strong privacy (exploratory).**
   - With zero init, the T = 5 minus T = 20 point differences are −0.0046 at ε 8,
     +0.0064 at ε 4 and +0.0124 at ε 1. The first two conditional patient intervals
     include zero; that does not establish equivalence or noninferiority. The ε 1
     interval is positive conditional on these fitted heads.
   - Five releases need much less noise per release (exact-GDP z 2.42 vs 4.84 at ε 4), which
     offsets having fewer steps.
4. **Cumulative effect of this session's DP work at ε 4 (selection half):**

   | Step | AUROC |
   | --- | ---: |
   | A8 DP recipe | 0.695 |
   | + exact accounting (A13) | 0.712 |
   | + zero-initialised head (A17, T 20) | 0.731 |
   | + zero init with T 5 (exploratory) | 0.738 |

   That is +0.043 [+0.034, +0.052] for the same claimed privacy. The ε 4 DP cost against the best
   clean federated model (A7, 0.819) falls from 0.124 to 0.081.

## Platform implications

- **For DP runs that train a fresh linear head on frozen features, initialise the head at zero.**
  It is data-independent, so it has no privacy cost. This applies to the frozen-head training arms.
- With exact accounting and zero init, **fewer rounds are a useful candidate to test**.
  The observed ε 4–8 differences remain unresolved by the conditional patient intervals;
  the ε 1 point and interval favor T = 5. Five rounds use one-quarter of the modeled
  round-dependent communication of twenty rounds, with the same payload and participation.

**Limits:**
- Selection half, five seeds, one workload, IID.
- The recipe (LR, clip) was tuned in A8 for random init at T = 20; zero init may shift the optimum.
- The T 5 vs T 20 and "best vs A8/A13" comparisons were chosen after seeing the pre-registered
  cells, so they are exploratory. Independent future data under a frozen plan would be needed to confirm a recommended
  configuration; the existing A10 holdout must not be reopened.

Private evidence:
- `summary.json` `54d584fcc0dd92f7ba86d4edb530f433e02c08cb965f7e80208a068f8f3fc225`.
- Exploratory contrasts: `zero-head-r0-v1-posthoc-contrasts.json`
  `37cf4a855d97d1e384fed5424e67700958000a1fb3188ecb8623183c8f6e16b6`.
