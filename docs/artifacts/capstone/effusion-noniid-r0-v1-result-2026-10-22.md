# Effusion R0 non-IID label-skewed clients (A19): result

22 October 2026. Executes amendment A19, which was committed before the run. Everything is on the
selection half; the holdout is untouched.

**Claim correction, 24 October 2026:** the private-AUROC decline under strong skew is 0.0384;
the additional privacy cost after subtracting the clean decline is about 0.0326. Recommendations
below are restricted to the tested label/quantity-skew simulation. Numeric tables and artifact
identities are retained. See the [review](review-notes-2026-10-23.md).

**Integrity:**
- All 60 runs were finite with no refusals and no attack notes.
- Every attacked run had exactly 16 of 80 attackers.
- Each cell differs from its A14 IID counterpart only in the partition: same seeds, attacker
  slots and recipe.
- Intervals are paired patient bootstraps (2,000 resamples, seed 20261039), conditional on the
  five realized trained heads per cell. They exclude variation over training seeds, partitions,
  initialization and DP noise draws; they are not procedure-level intervals.

**Partition:** `patient_dirichlet_partition` keeps patients whole, gives every client at least one
patient, and splits each class's remaining patients in Dirichlet(α) proportions. Heterogeneity
across the five seeds; positive share is shown for seed 20261101:

| α | Clients without a positive image | Image-count CV | Positive share, min / median / max |
| ---: | ---: | ---: | --- |
| IID (A14) | not recorded | patient counts equal (±1) | not recorded |
| 1.0 | 0–2 of 80 | 0.64–0.74 | 0 / 0.078 / 0.30 |
| 0.1 | 29–41 of 80 | 1.69–2.02 | 0 / 0.094 / 0.70 |

The Dirichlet split also creates **quantity skew**: at α 0.1 client sizes vary by a factor of
several.

## Results (seed-mean AUROC; non-IID − IID is the same cell against A14)

| Recipe | Rule | Condition | IID (A14) | α 1.0 | α 0.1 | α 1.0 − IID | α 0.1 − IID |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| No DP | mean | clean | 0.819 | 0.818 | 0.813 | −0.0004 [−0.0014, +0.0008] | −0.0058 [−0.0083, −0.0036] |
| No DP | trimmed k 16 | clean | 0.818 | 0.819 | 0.810 | +0.0003 [−0.0015, +0.0021] | −0.0084 [−0.0151, −0.0022] |
| No DP | median | clean | 0.817 | 0.818 | 0.798 | +0.0010 [−0.0015, +0.0034] | −0.0183 [−0.0278, −0.0090] |
| No DP | mean | Fang, 16 | 0.664 | 0.634 | 0.508 | −0.0297 [−0.0368, −0.0221] | −0.1555 [−0.1726, −0.1385] |
| No DP | median | Fang, 16 | 0.753 | 0.750 | 0.637 | −0.0031 [−0.0091, +0.0030] | −0.1158 [−0.1330, −0.0992] |
| DP ε 4 | mean | clean | 0.695 | 0.691 | 0.656 | −0.0036 [−0.0054, −0.0017] | −0.0384 [−0.0425, −0.0341] |

The IID column is A14's seed mean for the same cells (Fang rows: 0.819 − 0.155 and 0.817 − 0.063).

**Median minus mean, clean:** α 1.0 −0.0007 [−0.0023, +0.0009]; α 0.1 −0.0146 [−0.0231, −0.0063].

**Under Fang's attack at 16 of 80:**

| | IID (A14) | α 1.0 | α 0.1 |
| --- | --- | --- | --- |
| Median − mean | +0.089 | +0.116 [+0.097, +0.135] | +0.129 [+0.094, +0.164] |
| Mean drop | 0.155 | 0.184 | 0.305 |
| Median drop | 0.063 | 0.068 | 0.161 |

## Findings

1. **Moderate skew has small clean point differences in this simulation.** At α 1.0, FedAvg,
   the trimmed mean and the median differ from their clean IID point means by about 0.001 or
   less. Private mean AUROC declines by 0.0036; this is not a test of equivalence. Under Fang,
   the nonprivate mean declines by 0.0297, so this result does not make moderate skew benign
   across all tested conditions.
2. **Strong skew costs little without DP, more under DP.**
   - At α 0.1 the clean mean loses 0.0058 and the median loses 0.0183 (about 0.015 below
     the mean). These are observed penalties for these configurations.
   - Central DP at ε 4 loses 0.0384 AUROC, about 6.6 times the nonprivate mean's decline.
     The increase in the clean-minus-private gap is approximately 0.0384 − 0.0058 = 0.0326.
     The retrospective seed analysis gives +0.03253 [0.01162, 0.05343] from the saved unrounded run values; the subtraction above uses rounded table values.
   - **Hypothesis, not tested:** with uniform averaging and a binding clip, every client
     contributes a full-norm-C vote however little data it has. Under skew those votes disagree
     more and partly cancel, which shrinks the signal against fixed noise.
3. **The tested skew amplifies Fang's damage, and the median retains higher attacked AUROC.**
   - Fang's attack drives the mean to chance at α 0.1 (drop 0.305).
   - The median keeps a large advantage (+0.129), but its own drop rises from 0.063 to 0.161.
   - Widening honest disagreement is a possible explanation, not an isolated causal result.
4. **Platform guidance:**
   - These results qualify only the tested patient-whole label/quantity-skew construction;
     they do not establish generalization to moderately heterogeneous hospitals or sites.
   - Strong simulated skew lowers private AUROC and increases the clean-minus-private gap in
     this recipe; requalify both privacy cost and attack damage for an intended federation.
   - For the tested nonprivate Fang conditions, the median has higher AUROC than the mean,
     with a larger margin under skew. This is not a universal recommendation for all attacks.

**Limits:**
- Label skew plus the quantity skew it induces, not feature or site shift. A real hospital
  split (view, age, scanner) is a separate test.
- One workload, random-init head, five seeds, selection half.
- The clean DP recipe uses the older RDP calibration; these results do not qualify the later
  zero-init/GDP defaults. Fang cells are nonprivate raw-update-oracle stress tests.
- The DP mechanism hypothesis in finding 2 is untested.

Retrospective uncertainty update: the [24 October selection-only seed reanalysis](selection-seed-reanalysis-2026-10-24.md) reproduces the original points and reports separate paired-seed intervals. The original patient intervals remain conditional and unchanged.

Private evidence: `summary.json`
`6c1e4010dff279db5a974d0fea9e219dd0850d2fb0113cf98dd6cb8f9dd8215e`.
