# Effusion R0 norm-matching attackers and a KS filter under local DP (A21): result

22 October 2026. Executes amendment A21, which was committed before the run. Everything is on the
selection half; the holdout is untouched.

**Claim correction, 24 October 2026:** the tested filters fail to provide sufficient protection
against the tested norm-matching attacks. This is not an impossibility result for all local-DP
filters. No additional honest drops or clean utility loss was observed in these finite runs;
absence of harm for other runs is unestablished. Numeric results and artifact identities are
retained. See the [review](review-notes-2026-10-23.md).

**Integrity:**
- All 75 runs were finite with no refusals; every attacked run had 16 of 80 attackers.
- Norm-only clean cells are A20's, compared seed by seed from its pinned output.
- Intervals are paired patient bootstraps (2,000 resamples, seed 20261041), conditional on the
  five realized trained heads per cell. They exclude variation over training seeds, partitions,
  initialization and local-DP noise draws; they are not procedure-level intervals.

**Setup:**
- As A20: R0, A8 recipe, local DP, rule `mean`.
- Attackers send the A9 sign flip ×−4 or label-flip delta with no noise, rescaled every round to
  the expected honest privatised norm √(C² + s²d), with s = 2zC.
- Filters:
  - **norm only**: A20, k = 4;
  - **norm + KS**: an in-band upload is also dropped when the KS distance of its coordinates from
    N(0, s²) exceeds the asymptotic critical value at α = 10⁻⁴.

## Results

| ε | Filter | Clean AUROC | Sign flip, matched: AUROC (drop) | Attackers dropped | Label flip, matched: AUROC (drop) | Attackers dropped |
| ---: | --- | ---: | --- | ---: | --- | ---: |
| 8 | norm only | 0.558 (A20) | 0.296 (0.262 [0.235, 0.288]) | 0 / 1,600 | 0.462 (0.096) | 0 / 1,600 |
| 8 | norm + KS | 0.558 | 0.349 (0.208) | 91 / 1,600 | 0.406 (0.152) | 100 / 1,600 |
| 32 | norm only | 0.653 (A20) | 0.300 (0.353) | 0 / 1,600 | 0.397 (0.256) | 0 / 1,600 |
| 32 | norm + KS | 0.653 | 0.339 (0.314) | 328 / 1,600 | 0.433 (0.220) | 1,161 / 1,600 |
| 128 | norm only | 0.742 (A20) | 0.313 (0.429) | 0 / 1,600 | 0.207 (0.536) | 0 / 1,600 |
| 128 | norm + KS | 0.742 | 0.420 (0.323) | 1,013 / 1,600 | 0.669 (0.073) | 1,516 / 1,600 |

Honest uploads dropped: 1 of 8,000 per ε under norm + KS, exactly as under norm only. The KS
test added no false drops, and clean AUROC is identical (difference 0.0000).

**Norm + KS minus norm only, same cell and seed:**

| | ε 8 | ε 32 | ε 128 |
| --- | --- | --- | --- |
| Sign flip | +0.054 [+0.045, +0.062] | +0.039 [+0.032, +0.046] | +0.107 [+0.094, +0.120] |
| Label flip | **−0.056 [−0.068, −0.043]** | +0.037 [+0.017, +0.055] | +0.462 [+0.429, +0.492] |

## Findings

1. **The tested norm filter fails against these norm-matching attacks.**
   - Attackers who skip the noise but rescale to the expected honest norm pass it in every
     recorded upload.
   - They do far more damage than A12's crude attackers did unfiltered: up to 0.54 AUROC, and
     below chance under sign flip.
   - A20's noise-matching attackers had smaller drops (0.015–0.059). Matching an upload's norm
     alone does not verify that it contains the protocol's independent noise; this comparison
     does not prove that adding noise neutralises every attacker.
2. **A coordinate-distribution test is a partial defence.**
   - It adds no observed honest drops and no clean AUROC loss in these runs. Finite runs do
     not establish a zero false-drop probability or absence of utility cost generally.
   - It catches more attackers as ε grows: 6% at ε 8, up to 95% at ε 128, where it recovers
     +0.46 AUROC under label flip.
   - It leaves large residual damage, 0.07–0.32. Dense, roughly Gaussian rescaled deltas are a
     possible explanation for partial detection; that mechanism was not isolated here.
3. **One adverse result is unexplained:** at ε 8 under label flip, adding KS lowered AUROC by
   0.056, although it removed 100 attacker uploads and no honest ones. No mechanism has been
   checked, so it is reported as observed.
4. **Platform conclusion:**
   - The tested norm-only and norm-plus-KS filters leave substantial damage under these
     informed attacks. Other filters, joint-coordinate tests and robust protocols remain open;
     the measurements do not establish their impossibility.
   - Verifiable clipping/noise protocols or distributed noise with secure aggregation are
     candidate directions. They require separate privacy, robustness and implementation
     qualification; this experiment establishes neither their necessity nor their sufficiency.
   - **Do not present these local-DP filters as a qualified poisoning defence.** Central DP
     permits server-side clipping and remains a candidate under a trusted-server model, but
     its robustness is limited to tested configurations. A14's oracle attacked ε values are
     nominal aggregate-input calibrations, not established client-data guarantees.

**Limits:**
- One workload, IID, rule `mean`, 20% attackers.
- The KS test is marginal: it ignores the exact negative correlation between the two class
  rows' coordinates, which a joint test could exploit.
- The attacker does not adapt to KS.

Related uncertainty update: the [24 October selection-only seed reanalysis](selection-seed-reanalysis-2026-10-24.md) covers A11/A12/A19/A22/A23. A21 was not recalculated in that addendum; its intervals here remain conditional on realized heads, not procedure-level intervals.

Private evidence: `summary.json`
`209332623d4eb2d1d6de870f74c634f1b77cf59bbbe4780d173037cbdcb0c7e7`.
