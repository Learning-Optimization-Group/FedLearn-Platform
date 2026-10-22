# Effusion R0 federated recipe study (A7): result

22 October 2026. This is the exploratory optimiser check pre-registered as amendment A7,
run on R0 (the primary representation since A6). Everything is on the selection half;
the holdout is untouched.

## Grid

The grid is 12 configurations, selected on seeds 20261101 and 20261102. All runs are
retained, and all were eligible. The table shows mean federated selection AUROC, with
the last-round clip rate in brackets.

| Client LR \ clip | 0.05 | 0.25 | 1.0 |
| --- | --- | --- | --- |
| 0.003 | 0.7556 (0.62) | 0.7675 (0.00) | 0.7675 (0.00) |
| 0.01 | 0.7684 (1.00) | 0.8061 (0.01) | 0.8061 (0.00) |
| 0.03 | 0.7764 (1.00) | 0.8163 (0.04) | 0.8167 (0.00) |
| 0.1 | 0.7707 (1.00) | 0.8177 (1.00) | **0.8198 (0.01)** ✓ |

**Chosen recipe:** LR 0.1 with clip 1.0. The §6 recipe was LR 0.003 with clip 0.05.

## Five-seed confirmation of the chosen recipe

Values are seed means, compared with the A4 R0 cells at the §6 recipe. Intervals are
paired patient bootstraps (2,000 resamples, seed 20261026). All 25 runs completed with
finite outputs and no refusals.

| Arm | AUROC |
| --- | ---: |
| Federated, 80 clients | 0.8187 |
| Central-SGD control | 0.8060 |
| Central DP ε 8 / 4 / 1 | 0.6775 / 0.6318 / 0.5573 |
| Convex R0 head (gate) | 0.8220 |

| Contrast | Point [95% CI] |
| --- | --- |
| Recipe gain, federated new − federated §6 | **+0.0642 [+0.0562, +0.0726]** |
| Federated − convex | −0.0033 [−0.0067, −0.0001] |
| Federated − central-SGD | +0.0127 [+0.0078, +0.0177] |
| DP utility, new − §6 recipe, at ε 8 / 4 / 1 | −0.0493 [−0.0583, −0.0400] / −0.0510 [−0.0594, −0.0427] / −0.0165 [−0.0187, −0.0142] |
| DP cost at the new recipe, ε 8 / 4 / 1 | 0.1412 / 0.1869 / 0.2614 |

## What it means for the platform

1. **The default recipe, not federation, was the bottleneck.** With LR 0.1 and a
   non-binding clip, 80 whole-patient clients reach within 0.003 AUROC of the
   centralised convex probe. The §6 "federation gap" was almost entirely recipe.
2. **Clip and LR interact.** At clip 0.05 every LR is clip-bound (rate 1.00) and stuck
   near 0.77. Raising the LR only helps once the clip stops binding.
3. **A clean-tuned clip is the wrong DP setting.** Noise is proportional to the clip.
   Moving from 0.05 to 1.0 made DP utility worse by about 0.05 AUROC at ε 8 and ε 4,
   even though clean utility improved. DP runs need their own clip and LR choice;
   amendment A8 studies that.
4. **The central-SGD control is not monotone in the recipe.** At LR 0.1, one client
   making 269 steps per epoch lands slightly below the federated average (0.806 vs
   0.819), so "federated beats central" is a property of this step budget, not a
   general claim.

**Limits.** Seeds 1–2 chose the recipe, so the five-seed means are optimistic.
Seeds 3–5 reuse the same selection data. Only the §8 holdout is unbiased. Everything
is client-level, IID whole-patient allocation, and simulated.

Private evidence: `summary.json`
`7d724e38f60db45d048690e225fb6c11d91fc47f5728079d0bb79d55dabeb954`.
