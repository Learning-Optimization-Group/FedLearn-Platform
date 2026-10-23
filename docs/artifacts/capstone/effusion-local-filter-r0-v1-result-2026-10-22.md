# Effusion R0 noise-aware norm filtering under local DP (A20): result

22 October 2026. Executes amendment A20, which was committed before the run. Everything is on the
selection half; the holdout is untouched.

**Integrity:**
- All 105 runs were finite with no refusals; every attacked run had 16 of 80 attackers.
- Filter-off clean and non-compliant cells are A12's, compared seed by seed from the pinned A12
  output.
- Intervals are paired patient bootstraps (2,000 resamples, seed 20261040).

**Setup:**
- R0, A8 recipe (LR 0.03, clip C 0.05), 20 rounds, local DP, honest noise N(0, (2zC)²), rule `mean`.
- Filter: `server.norm_filter = {"k": 4}`. The server drops uploads whose delta norm is outside
  the band an honest privatised update occupies. That is post-processing, so it costs no privacy.

## How the filter behaved

| | Dropped |
| --- | --- |
| Honest uploads | **9 of 120,000** (7.5 × 10⁻⁵; designed for about 10⁻⁴) |
| Non-compliant label flip | 4,798 of 4,800 |
| Non-compliant sign flip | 4,710 of 4,800 (ε 8: 1,600/1,600; ε 32: 1,592/1,600; ε 128: 1,518/1,600) |
| Noise-matching label flip | 2 of 4,800 |
| Noise-matching sign flip | 2 of 4,800 |

## Seed-mean AUROC

| ε | Clean | Non-compliant sign flip, filter on (off, A12) | Non-compliant label flip, filter on (off, A12) | Noise-matching sign flip | Noise-matching label flip |
| ---: | ---: | --- | --- | ---: | ---: |
| 8 | 0.558 | 0.557 (0.305) | 0.557 (0.452) | 0.543 | 0.539 |
| 32 | 0.653 | 0.630 (0.300) | 0.642 (0.385) | 0.605 | 0.603 |
| 128 | 0.742 | 0.669 (0.310) | 0.735 (0.307) | 0.684 | 0.688 |

**Filter on − off, same cell and seed:**
- Non-compliant sign flip: +0.252 [+0.225, +0.280] at ε 8, +0.331 at ε 32, +0.359 at ε 128.
- Non-compliant label flip: +0.105 [+0.094, +0.116] at ε 8, +0.256 at ε 32, +0.427 at ε 128.
- Clean and noise-matching cells: +0.0001 or less at every ε.

**Attack drop with the filter on (clean − attacked):**

| | ε 8 | ε 32 | ε 128 |
| --- | --- | --- | --- |
| Non-compliant sign flip | +0.0005 [−0.006, +0.007] | +0.023 | +0.073 [+0.065, +0.081] |
| Non-compliant label flip | +0.0005 | +0.011 | +0.008 |
| Noise-matching sign flip | +0.015 [+0.014, +0.017] | +0.048 | +0.059 |
| Noise-matching label flip | +0.018 | +0.050 | +0.054 |

## Findings

1. **The filter fixes A12's failure at no cost.**
   - Attackers who skip the noise are dropped almost always.
   - AUROC recovers by up to +0.43, back to within 0.023 of clean, except one case (finding 3).
   - Clean utility and honest participation are unaffected: 9 false drops in 120,000.
2. **It forces attackers to follow the noise protocol, which caps their damage.**
   - Noise-matching attackers pass the filter by construction, so the filter does nothing against
     them.
   - But following the protocol limits each attacker to a clipped, noised, honest-sized vote.
     Their damage (0.015–0.059) is a fraction of the unfiltered non-compliant damage
     (0.25–0.43).
3. **A norm-only filter can be evaded by norm-matching without noise.**
   - At ε 128 the band is narrow and sits at norm about 1.2. As training converges, the
     non-compliant sign-flippers' updates shrink toward it, and 82 of 1,600 land inside.
   - That leaves a 0.073 drop, larger than the noise-matching attack does.
   - A deliberate attacker would scale a noise-free push to the band's norm every round.
   - Xiang et al. pair the norm test with a distribution test (KS against N(0, s²) per
     coordinate), which rejects such a vector. That is the next step if local DP ships.

## Platform implication

If a local-DP option ever ships, it should include this norm filter and a coordinate-distribution
test. With both, the residual threat is protocol-following attackers, whose damage the clip and
noise already bound. Local DP's utility cost (A11) remains the main reason not to default to it.

**Limits:**
- One workload, IID, rule `mean`, 20% attackers.
- The norm-matching evasion is observed incidentally, not run as a designed attack.
- The KS test is not implemented.

Private evidence: `summary.json`
`6c11ef11f6ce60396a12a71ffde0e4e744f0d75e754ae95aec85c87445188119`.
