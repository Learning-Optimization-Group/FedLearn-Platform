# Effusion R0 DP-specific recipe study (A8): result

22 October 2026. This is the exploratory DP recipe check pre-registered as amendment A8,
run on R0. Everything is on the selection half; the holdout is untouched.

DP here is central, client-level, q = 1, δ = 10⁻⁵. Accounted ε equals the target in every
confirmation cell. Choosing the recipe from DP runs consumes privacy that the stated ε does
not cover; this is disclosed.

## Grid under ε = 4

Mean AUROC over seeds 20261101–20261102. Every cell clipped in 100% of last-round updates.

| Client LR \ clip | 0.01 | 0.025 | 0.05 | 0.1 |
| --- | --- | --- | --- | --- |
| 0.03 | 0.6358 | 0.6876 | **0.6961** ✓ | 0.6848 |
| 0.1 | 0.6172 | 0.6639 | 0.6845 | 0.6830 |
| 0.3 | 0.6094 | 0.6476 | 0.6676 | 0.6719 |

**Chosen recipe:** LR 0.03 with clip 0.05.

**Seed noise:** under DP, the two selection seeds differ by up to about 0.07 AUROC within a
cell, while the top cells are within about 0.012 of each other. The robust conclusion is
that a **small binding clip (0.025–0.1) with a moderate LR** is best. Very small clips (0.01)
or large LRs (0.3) are worse. The exact optimum is not resolved.

## Five-seed confirmation of the chosen recipe

Paired patient bootstraps (2,000 resamples, seed 20261027). All 20 runs completed with
finite outputs and no refusals.

| | Clean | ε 8 | ε 4 | ε 1 |
| --- | ---: | ---: | ---: | ---: |
| AUROC, DP recipe (A8) | 0.7735 | 0.7434 | 0.6947 | 0.5657 |
| DP cost (clean − DP) | — | 0.0301 [0.0255, 0.0344] | 0.0788 [0.0706, 0.0862] | 0.2078 [0.1922, 0.2219] |
| Gain vs §6 recipe (A4) | — | +0.0166 [+0.0143, +0.0188] | +0.0119 [+0.0096, +0.0142] | −0.0081 [−0.0106, −0.0057] |
| Gain vs clean-tuned recipe (A7) | — | +0.0659 [+0.0558, +0.0750] | +0.0629 [+0.0539, +0.0710] | +0.0084 [+0.0060, +0.0109] |

Running the DP recipe without DP costs −0.0452 [−0.0537, −0.0371] relative to the
clean-optimal A7 recipe (0.7735 vs 0.8187).

## Defaults this supports, for this workload class

These defaults are for a linear head over frozen 512-wide features with 80 clients and
20 rounds.

| Run type | Client LR | Update clip | Selection AUROC (5 seeds) |
| --- | ---: | ---: | --- |
| No DP | 0.1 | 1.0 (non-binding) | 0.819 |
| Central DP (ε 4–8) | 0.03 | 0.05 (binding) | 0.695–0.743 |

- **Never reuse a clean-tuned clip for DP.** It loses about 0.06 AUROC at ε 4–8.
- **Use the planner first.** At ε 4 this workload's noise-norm-to-clip lower bound is
  z√d/N ≈ 2.3 (z 5.80, d 1,026, N 80). The planner reproduces that number before a run.
- **Client-level ε = 1 is near chance** for this setup whatever the recipe.

**Limits:**
- Two-seed selection on the same selection data, with tuning privacy unaccounted.
- One dataset, one representation, IID whole-patient clients, simulated.
- These numbers are evidence for a default, not a universal rule. Only the §8 holdout
  gives an unbiased figure.

Private evidence: `summary.json`
`bd9868ae3300411e44371348aebf97856c4a7b178996ed1d7d32a42d210da42d`.
