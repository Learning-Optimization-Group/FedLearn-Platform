# Effusion R0 non-IID label-skewed clients (A19): result

22 October 2026. Executes amendment A19, which was committed before the run. Everything is on the
selection half; the holdout is untouched.

**Integrity:**
- All 60 runs were finite with no refusals and no attack notes.
- Every attacked run had exactly 16 of 80 attackers.
- Each cell differs from its A14 IID counterpart only in the partition: same seeds, attacker
  slots and recipe.
- Intervals are paired patient bootstraps (2,000 resamples, seed 20261039).

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

1. **Moderate skew is benign.** At α 1.0, FedAvg, the trimmed mean and the median all match
   their IID results within ±0.001, and DP costs only 0.004 more.
2. **Strong skew costs little without DP, more under DP.**
   - At α 0.1 the clean mean loses 0.006. The median loses 0.018, which reproduces the known
     heterogeneity penalty of median-type rules (0.015 below the mean).
   - Central DP at ε 4 loses 0.038, ten times what the non-private mean loses.
   - **Hypothesis, not tested:** with uniform averaging and a binding clip, every client
     contributes a full-norm-C vote however little data it has. Under skew those votes disagree
     more and partly cancel, which shrinks the signal against fixed noise.
3. **Skew amplifies attacks, and the median is still the better rule.**
   - Fang's attack drives the mean to chance at α 0.1 (drop 0.305).
   - The median keeps a large advantage (+0.129), but its own drop rises from 0.063 to 0.161.
   - Honest disagreement widens the range attackers can hide in.
4. **Platform guidance:**
   - IID-based recommendations hold for moderately heterogeneous sites.
   - For strongly skewed federations, expect a larger DP cost and stronger attacks, and budget
     for both.
   - The A14 recommendation (median when no DP and attackers are plausible) holds; its margin
     over the mean grows with skew.

**Limits:**
- Label skew plus the quantity skew it induces, not feature or site shift. A real hospital
  split (view, age, scanner) is a separate test.
- One workload, random-init head, five seeds, selection half.
- The DP mechanism hypothesis in finding 2 is untested.

Private evidence: `summary.json`
`6c1e4010dff279db5a974d0fea9e219dd0850d2fb0113cf98dd6cb8f9dd8215e`.
