# Effusion R0 exact Gaussian-DP noise calibration (A13): result

22 October 2026. Executes amendment A13, which was committed before the run. Everything is on the
selection half; the holdout is untouched.

**Integrity:**
- All 15 runs were finite with no refusals.
- Each run recorded a noise multiplier within 10⁻⁶ (relative) of the exact-GDP z. The largest gap
  was 5 × 10⁻¹⁰.
- Each run's exact ε is at or below its claim, by at most 2 × 10⁻¹⁰.
- Intervals are paired patient bootstraps (2,000 resamples, seed 20261032).

**Setup:**
- Identical to the A8 confirmation DP cells: R0, LR 0.03, clip 0.05, 80 clients, 20 rounds,
  replace-one central DP, δ 10⁻⁵, the same five seeds.
- The only change is z, which is now the smallest multiplier whose exact GDP ε meets the claim.

| Claimed ε | z (RDP, A8) | z (exact GDP, A13) | Less noise |
| ---: | ---: | ---: | ---: |
| 8 | 3.087 | 2.684 | 13% |
| 4 | 5.796 | 4.835 | 17% |
| 1 | 21.92 | 16.68 | 24% |

## Seed-mean AUROC

| Claimed ε | A8 (RDP-calibrated) | A13 (exact) | Gain, A13 − A8 | DP cost vs A8 clean (0.7735) |
| ---: | ---: | ---: | --- | --- |
| 8 | 0.7434 | 0.7499 | +0.0065 [+0.0059, +0.0071] | 0.0236 [0.0196, 0.0273] |
| 4 | 0.6947 | 0.7121 | +0.0174 [+0.0161, +0.0186] | 0.0614 [0.0546, 0.0679] |
| 1 | 0.5657 | 0.5841 | +0.0184 [+0.0167, +0.0200] | 0.1894 [0.1744, 0.2029] |

## Findings

1. **Exact accounting is free utility.** The privacy claim is identical; the old one was simply
   loose. Every interval excludes zero. At ε 4 the DP cost falls from 0.079 (A8) to 0.061, which
   recovers about a fifth of it.
2. The gain is largest where noise dominates (ε 1–4) and smallest at ε 8, where the noise is
   already small relative to the signal.
3. **Platform implication:** the platform's live DP runs always have q = 1, which is exactly the
   case GDP accounting covers. `--dp-accountant gdp` (shipped in `edad4aa`) should become the
   recommended setting for full-participation runs. RDP remains necessary for subsampled runs.

**Limits:**
- Selection half, five seeds, one workload and one recipe. The recipe was tuned under RDP noise in
  A8, so a GDP-tuned recipe could gain slightly more.
- These runs are replace-one, as everywhere in the capstone. The relative noise saving is the same
  under add/remove.

Private evidence: `summary.json`
`366fb6b0d4e37e40418911f4fd43f7ace0c9b7b2033c38bb6ae5e2cff867116b`.
