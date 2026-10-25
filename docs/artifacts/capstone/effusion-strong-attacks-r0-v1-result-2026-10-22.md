# Effusion R0 attacks built for robust rules, attacker fractions and small-k trimming (A14): result

22 October 2026. Executes amendment A14, which was committed before the run. Everything is on the
selection half; the holdout is untouched.

**Claim correction, 24 October 2026:** the attacked noisy cells are oracle stress tests with
nominal aggregate-input ε 4 calibration, not an established client-data ε 4 guarantee. The
attacks read current raw honest updates, so replacing one honest input can change several
attacker uploads and invalidate the one-upload sensitivity premise. Numeric results and
artifact identities below are retained. See the [review](review-notes-2026-10-23.md).

**Integrity:**
- All 350 runs were finite with no refusals.
- Every attacked run had exactly the declared 4, 8 or 16 attackers.
- No run fell back to attacker-only statistics (no attack notes).
- The 25 clean cells shared with A9 reproduced A9's margins exactly.
- Intervals are paired patient bootstraps (2,000 resamples, seed 20261033), conditional on the
  five realized trained heads per cell. They exclude variation over training seeds, partitions,
  initialization and DP noise draws; they are not procedure-level intervals.

**Setup:**
- As A9: no-DP recipe LR 0.1 with clip 1.0; DP-oriented recipe LR 0.03 with clip 0.05 and
  central Gaussian noise nominally calibrated to ε 4 (RDP z, replace-one, aggregate-input
  boundary). The historical table label `DP ε 4` denotes this calibration for oracle attacked
  cells, not a proven client-data guarantee through the oracle.
- R0, 80 clients, 20 rounds, five seeds, random-init head.
- Attacks, all with full knowledge of the honest updates and none adapted to the server clip:
  - **min-max**: Shejwalkar and Houmansadr, std perturbation;
  - **Fang**: trimmed-mean attack, b = 2;
  - **IPM** with scale 0.1.

## Attack drop (clean − attacked, same rule and seed; seed means)

Each cell shows the drop at 4 / 8 / 16 attackers of 80.

| Recipe | Rule | Clean AUROC | min-max | Fang | IPM 0.1 |
| --- | --- | ---: | --- | --- | --- |
| No DP | mean | 0.819 | 0.000 / 0.000 / 0.001 | 0.006 / 0.030 / **0.155** | 0.000 / 0.000 / 0.001 |
| No DP | trimmed k = 16 | 0.818 | 0.000 / 0.000 / 0.001 | 0.003 / 0.013 / 0.106 | 0.000 / 0.001 / 0.002 |
| No DP | median | 0.817 | 0.000 / 0.000 / 0.000 | 0.004 / 0.012 / **0.063** | 0.001 / 0.003 / 0.017 |
| DP ε 4 | mean | 0.695 | 0.001 / 0.003 / 0.005 | 0.014 / 0.031 / 0.072 | 0.010 / 0.022 / **0.053** |
| DP ε 4 | trimmed k = 1 | 0.692 | 0.001 / 0.003 / 0.005 | 0.014 / 0.031 / 0.072 | 0.010 / 0.022 / 0.054 |
| DP ε 4 | trimmed k = 4 | 0.684 | 0.001 / 0.003 / 0.005 | 0.013 / 0.029 / 0.070 | 0.009 / 0.022 / 0.055 |
| DP ε 4 | trimmed k = 16 | 0.638 | 0.000 / 0.001 / 0.003 | 0.007 / 0.015 / 0.041 | 0.006 / 0.014 / 0.036 |

Selected intervals:
- No DP, 16 attackers, Fang: mean drop 0.155 [0.139, 0.171]; median 0.063 [0.056, 0.071].
- DP, 16 attackers, IPM 0.1: mean drop 0.053 [0.049, 0.057].

## Robust rule minus mean (same condition and seed)

- **No DP:**
  - Under Fang at 16 attackers, the median is +0.089 [+0.079, +0.101] and the trimmed mean
    +0.048 [+0.043, +0.055].
  - At 8 attackers, both are about +0.016; at 4, about zero.
  - The median costs −0.002 clean and −0.018 [−0.023, −0.013] under IPM at 16 attackers.
- **DP ε 4:** every trimmed variant is **below** the mean in **every** condition, and every
  interval excludes 0.
  - k = 1: −0.0025 clean, at most −0.0033 attacked.
  - k = 4: about −0.01 everywhere.
  - k = 16: −0.056 clean, and still −0.026 [−0.029, −0.022] under Fang at 16 attackers, the
    attack designed against it.

## Findings

1. **Without DP, Fang causes the largest observed damage, and the median performs best
   against it at the higher tested attacker fractions.**
   - At 20% attackers Fang removes 0.155 AUROC from the mean, 0.106 from the trimmed mean and
     0.063 from the median.
   - At the lowest tested fraction (5%) every rule loses at most 0.006.
   - Min-max and IPM 0.1 do not change AUROC without DP. They move the model (min-max's
     deviation ratio is about 5), but not along the direction that ranks patients.
2. **With the DP-oriented recipe, the mean has the highest observed AUROC among the tested rules.**
   - The clip bounds each final upload to C. Trimming uses the larger one-upload sensitivity
     calibration (2C/(n − 2k)), so these cells receive more noise. That calibration does not
     establish client-data privacy through the oracle attacks.
   - Small k keeps the clean cost small (−0.0025 at k = 1) but shows no material reduction in
     attack drops at the reported precision.
   - Large k buys some robustness but costs more clean AUROC than it saves under any attack
     tested.
   - This supports A9's observed rule ordering within the tested configurations; it does not
     rank untested private robust methods or validate the later zero-init/GDP recipe.
3. **The DP-oriented recipe is more vulnerable to the tested IPM attack.**
   - IPM with scale 0.1 costs about 0.001 without DP and 0.053 with the DP-oriented recipe at
     20% attackers. Learning rate, clip and noise all differ between these recipes.
   - Clipping limits uploads above C to norm C; an opposing clipped upload can cancel part of
     the honest aggregate while the configured noise remains. Clipping does not amplify an
     update whose norm is below C.
   - A matched attacked zero-noise control at the same LR and clip was not run here, so the
     contribution of DP noise to the difference is unresolved.
4. **Damage rises with attacker fraction in these cells.** At 5% the worst observed drop is
   0.014 (DP-oriented recipe, Fang). The three tested fractions do not establish a general
   scaling law or rank attacker count against attack strength.

## Platform guidance

| Setting | Recommendation |
| --- | --- |
| No DP, Fang-like coordinated attackers at the tested 10–20% fractions | Median is the strongest tested option against Fang; compare its clean cost and its poorer IPM result before choosing. |
| No DP, tested attacks at 5% | Mean or trimmed mean have small observed damage; this does not cover arbitrary informed attacks. |
| DP-oriented recipe and tested oracle stresses | Prefer mean over the tested trimming settings for this workload. Treat attacked ε as nominal calibration and vet participation; broader robustness and client-data privacy require separate qualification. |

**Limits:**
- IID, one workload, simulated, random-init head (A17 shows zero init changes absolute levels).
- One nominal ε under the DP-oriented recipe; no established client-data guarantee for oracle
  attacked cells. Raw-honest-update dependence is separate from adaptation to clipping.
- The LR/clip change prevents attributing cross-recipe attack differences to DP noise alone.
- Attacks not adaptive to the clip; an adversary who scales uploads to exactly C may do better.
- min-max with std perturbation only.

Private evidence: `summary.json`
`57ff024b4ef9dc461606f9dd003c92e84495c4672c8dac31e7aa9e4f3de9d22f`.
