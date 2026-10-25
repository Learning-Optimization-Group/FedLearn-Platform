# Effusion one-time holdout evaluation (§8 / A10): result

22 October 2026. This is the single pre-registered look at the locked internal holdout.
The model list was frozen in A10 (`85ca42a`) before any holdout feature or label was read.
Every model's selection-half margin was recomputed from its saved head and matched the
stored margin exactly (maximum difference 0.0). Platt recalibration and the
sensitivity-0.90 operating points were then written to a hashed file
(`23da497351b9b47c3b63c700e929a8766055b87a2b127da9130b58e33a351e23`) before holdout labels
were loaded. The holdout has not been scored again, and any later scoring is post-hoc.

**Holdout:**
- 8,688 images from 2,800 patients, 884 Effusion-positive (prevalence 0.102).
- Internal: patient-disjoint and Effusion-untouched, but drawn from the same NIH
  validation split used historically for any-finding selection. It is not an external
  cohort.

Uncertainty: 2,000 patient-cluster bootstrap resamples (seed 20261029); none skipped.
Federated rows are five-seed means, with the mean taken inside each replicate.
These intervals condition on the realized fitted heads; they do not include training-seed
variation or establish equivalence. Claim wording below was corrected on 25 October without
changing the published numeric tables, artifact hashes, thresholds or evaluating the holdout again.

| Model | AUROC | AP | Sensitivity | Specificity | PPV | NPV | Brier | Cal. slope | Patient AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| C0 constant | 0.500 | 0.102 | 1.000 | 0.000 | 0.102 | — | 0.091 | — | — |
| Convex R0 (centralised) | 0.835 [0.818, 0.852] | 0.377 | 0.912 | 0.559 | 0.190 | 0.982 | 0.076 | 1.07 [0.99, 1.16] | 0.915 |
| FedAvg, no DP (A7 recipe) | **0.836 [0.819, 0.853]** | 0.381 | 0.923 | 0.545 | 0.187 | 0.984 | 0.075 | 1.10 [1.01, 1.19] | 0.913 |
| Central DP ε 8 (A8 recipe) | 0.763 [0.746, 0.780] | 0.264 | 0.911 | 0.386 | 0.144 | 0.975 | 0.083 | 1.08 [1.00, 1.18] | 0.871 |
| Central DP ε 4 (A8 recipe) | 0.715 [0.700, 0.730] | 0.221 | 0.909 | 0.296 | 0.128 | 0.966 | 0.086 | 1.12 [1.04, 1.22] | 0.843 |
| Central DP ε 1 (A8 recipe) | 0.580 [0.570, 0.592] | 0.137 | 0.908 | 0.141 | 0.107 | 0.927 | 0.091 | 1.19 [1.02, 1.36] | 0.758 |

| Paired contrast (AUROC) | Point [95% CI] |
| --- | --- |
| FedAvg − convex | +0.0011 [−0.0021, +0.0044] |
| DP ε 8 − FedAvg no DP | −0.0729 [−0.0816, −0.0633] |
| DP ε 4 − FedAvg no DP | −0.1213 [−0.1315, −0.1104] |
| DP ε 1 − FedAvg no DP | −0.2557 [−0.2718, −0.2381] |

The DP contrasts compare against the clean-optimal A7 recipe, so they include the
roughly 0.045 cost of the DP recipe itself as well as the noise. This is the observed
cost of switching to these tested recipes, not an isolated or universal cost of DP.

## Conclusions

1. **The clean federated and centralised point estimates are close on this internal cohort.**
   The A7 recipe gives a FedAvg-minus-convex difference of +0.0011, with a conditional
   patient interval [−0.0021, +0.0044]. No prospective equivalence margin or equivalence
   test is supplied. Selection-half points (0.819–0.822) were below the holdout points;
   this does not rule out selection optimism or establish external generalization.
2. **The preselected operating points retained high observed sensitivity.** Sensitivities
   ranged from 0.908 to 0.923 against a selection target of 0.90. Post-Platt clean slopes
   were about 1.07–1.10 and ECE ≤ 0.01. Those measurements do not establish perfect
   calibration or transfer beyond this internal cohort.
3. **The tested private recipes had substantially lower specificity at those operating points.**
   At sensitivities near 0.91,
   specificity falls from 0.55 (no DP) to 0.39 (ε 8), 0.30 (ε 4) and 0.14 (ε 1). At ε 1
   the model flags about 86% of negatives. This is a substantial benchmark burden;
   clinical triage usefulness was not evaluated. It is not a general finding that DP
   preserves sensitivity while affecting only specificity.
4. **Clinical usefulness is not established.** Even the best model has PPV 0.19 at
   prevalence 0.10: about one flagged image in five is Effusion-positive. Labels are
   report-derived; this is a research triage signal on an internal holdout, not a
   qualified clinical tool.

Private evidence: `holdout-evaluation.json`
`dd634be473b9b71875d7fb65890fda3dc7a707f6174094220ce658f6cd981599`; frozen calibration
as above; holdout margins in the same private directory.
