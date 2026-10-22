# Effusion R0 local DP (A11): result

22 October 2026. Executes amendment A11, which was committed before the run. Everything is
on the selection half; the holdout was opened once in A10 and is not reused here.

**Setup:**
- R0 with the A8 DP recipe (LR 0.03, clip 0.05), 80 whole-patient IID clients, 20 rounds,
  mean rule, seeds 20261101–20261105.
- Each honest client clips to C = 0.05 and adds N(0, (2zC)²) per coordinate. The server adds
  nothing; its records now report `client_noise_std` and `server_noise_std` explicitly
  (engine fix `6a1b4ca`).
- The privacy unit is one simulated client with replace-one adjacency. There is no trusted
  aggregator, so equal ε is **not** an equivalent claim to central DP.

**Integrity:** all 25 runs were finite with no refusals, and accounted ε equals the target
in every cell. Intervals are paired patient bootstraps (2,000 resamples, seed 20261030).

| ε | Noise multiplier z | Client noise SD per coordinate | Local DP AUROC (seed mean) | Seed range | Local − central (A8), same ε | Cost vs A8 clean arm (0.7735) |
| --- | ---: | ---: | --- | --- | --- | --- |
| 1 | 21.92 | 2.192 | 0.5252 [0.5162, 0.5347] | 0.389–0.648 | −0.0405 [−0.0550, −0.0261] | 0.2483 |
| 4 | 5.80 | 0.580 | 0.5399 [0.5304, 0.5499] | 0.432–0.653 | −0.1548 [−0.1735, −0.1358] | 0.2336 |
| 8 | 3.09 | 0.309 | 0.5574 [0.5472, 0.5676] | 0.473–0.662 | −0.1860 [−0.2042, −0.1672] | 0.2161 |
| 32 | 0.99 | 0.099 | 0.6529 [0.6386, 0.6660] | 0.612–0.711 | not run (A8 has no ε 32) | 0.1206 |
| 128 | 0.38 | 0.038 | 0.7423 [0.7242, 0.7598] | 0.730–0.759 | not run | 0.0312 |

## Findings

1. **At useful privacy levels (ε ≤ 8), client-level local DP is near chance on this
   workload.** Seed-to-seed spread is wide, and some runs fall below 0.5.
2. **Removing the trusted server costs about a 16× larger ε here.** Local DP at ε 128
   (0.742) roughly matches central DP at ε 8 (0.743, A8). This is the expected √n penalty:
   each of the 80 clients adds noise at the full 2zC scale, against one noise draw at
   2zC/n for central DP.
3. **Local DP is not a drop-in alternative** for small-cohort, frozen-feature federations
   like this one. Options are to keep central DP with a trusted aggregator (or secure
   aggregation plus distributed noise, not implemented here), or to accept much weaker
   privacy.

**Platform implication:** production has no local-DP path, and this evidence does not argue
for adding one as a default. If one is added, it should ship with the planner's
noise-to-signal warning and this utility evidence.

**Limits:** one recipe (the A8 DP recipe, not re-tuned for local DP); client-level privacy;
80 simulated IID clients. A larger cohort would reduce the gap, and that was not measured.

Private evidence: `summary.json`
`6175fcbe5ee94c18896704417d2655e355ef5d5d9f147e38d17d5e9343b86fb5`.
