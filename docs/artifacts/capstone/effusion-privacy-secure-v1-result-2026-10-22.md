# Effusion central DP (with R0 width arm) and secure aggregation v1: results

22 October 2026. Executes protocol amendments A4 (central-DP sweep) and A5
(secure-aggregation scope). Both were committed before their runs. Everything is
on the selection half; the holdout and official test were not touched.

The setup is the §6 recipe:
- 80 whole-patient IID clients, 20 rounds, 7 epochs, LR 0.003, batch 256,
  clip 0.05, mean rule;
- seeds 20261101–20261105;
- the production engine.

R1 is primary by amendment A2 and is not claimed better than R0 at the gate.

## A4: central DP, and does head width change its cost?

**Mechanism:**
- Gaussian noise is added to the clipped mean using replace-one sensitivity
  2C/n (C = clip, n = clients).
- The multiplier is calibrated for 20 rounds at q = 1 and δ = 10⁻⁵, with no
  amplification credit.
- Accounted ε equals the target in every cell.
- **The privacy unit is one simulated client (about 280 patients), not a patient.**

**Integrity:** all 40 runs completed with finite outputs and no refusals, and the five
clean R1 cells reproduced the §6 federated margins exactly.

| ε | Noise multiplier | R1 AUROC (range) | R0 AUROC (range) | R1 / R0 noise-to-signal, last round |
| --- | ---: | --- | --- | --- |
| clean | 0 | 0.7968 (0.7940–0.7993) | 0.7545 (0.7478–0.7604) | — |
| 8 | 3.09 | 0.7617 (0.7540–0.7714) | 0.7268 (0.7140–0.7390) | 6.48 / 4.45 |
| 4 | 5.80 | 0.7086 (0.6951–0.7259) | 0.6828 (0.6619–0.7000) | 10.82 / 7.60 |
| 1 | 21.92 | 0.5650 (0.5221–0.6140) | 0.5738 (0.5357–0.6141) | 31.66 / 22.78 |

DP cost is clean − DP, same arm and seed, seed-averaged. Intervals are patient-cluster
bootstraps (2,000 resamples, seed 20261024), seed-averaged inside each replicate.

| ε | R1 cost [95% CI] | R0 cost [95% CI] | Width contrast cost(R1) − cost(R0) [95% CI] |
| --- | --- | --- | --- |
| 8 | 0.0351 [0.0306, 0.0397] | 0.0277 [0.0230, 0.0320] | **+0.0074 [+0.0014, +0.0138]** |
| 4 | 0.0882 [0.0800, 0.0962] | 0.0717 [0.0640, 0.0792] | **+0.0165 [+0.0061, +0.0271]** |
| 1 | 0.2317 [0.2095, 0.2518] | 0.1807 [0.1659, 0.1944] | **+0.0511 [+0.0307, +0.0709]** |

**Findings:**
- **The wider head pays more for the same ε, at every ε.** Each width interval excludes
  zero. At each ε the noise SD per coordinate is identical across arms (0.0039, 0.0072
  and 0.0274 for ε 8, 4 and 1), so the noise norm grows with √d. R1's noise-to-signal
  ratio is 1.39–1.46× R0's, close to √2.
- **That is a cost of width, not a verdict on usefulness.** The wider R1 head still
  scores higher in absolute terms at ε = 8 and ε = 4, because it starts from a higher
  clean federated AUROC (0.797 vs 0.755). At ε = 1 both arms are near chance and R0
  is marginally higher.
- **The arms were not tied under this federated recipe.** They tied as centralised
  convex probes (0.822 vs 0.829), but the fixed federated recipe reaches 0.755 on R0
  versus 0.797 on R1, with last-round clip rates of 0.52 vs 0.71. The width contrast
  compares each arm's DP cost against its own clean run, so this does not bias it. It
  does mean "clean utility" depends on the recipe.
- **Client-level ε of 1 makes the model nearly useless here; ε = 8 costs about 3.5 AUROC
  points.** None of these ε values is patient-level, and none was tuned.

## A5: secure aggregation, equivalence and overhead

**Method:** the existing LightSecAgg client and session run in-process, in place of only
the engine's mean reduction:
- X25519 key agreement; Shamir shares of an `os.urandom` mask; sealed relay through the
  real servicer handlers; 10⁶ fixed-point masking.
- 80 clients, threshold 41, no dropouts.
- One protocol run per round: 20 per seed and 100 in total. The engine's two noise-free
  previews per round reused each result.

| Measure | Result |
| --- | --- |
| Aggregate error vs float mean, max over 100 rounds | 1.49e-7, within fixed-point bound 5.07e-7 in every round |
| Final selection margins vs plain, max absolute difference over seeds | 1.9e-4 (accumulated over 20 rounds; not bit-identical) |
| Per-round AUROC vs plain, max absolute difference | 8.7e-7 |
| Seed-mean AUROC, secure − plain | −1.0e-7 [−7.7e-7, +6.2e-7] (2,000 resamples, seed 20261025) |
| Upload per client per round | 1,315,641 bytes (seed mean) vs 8,348 plain (**158×**); seed 20261101 breakdown: sealed shares 1,300,415, masked vector 10,018, summed share 5,128, key 82 |
| Download per client per round | 1,952,494 bytes |
| Protocol wall time per round, 80 simulated clients on one CPU | 7.24 s: share and seal 5.39, collect and open 1.68, mask and submit 0.10, summed shares 0.06, recover 0.001 |

**Findings:**
- **Utility is equivalent within fixed-point error.** Secure aggregation neither helps nor
  hurts accuracy.
- **The cost is communication and client compute.** This implementation sends each peer
  a sealed full-length Shamir share of the mask, so upload grows with
  cohort size × model width. Over a 20-round run that is about 2.1 GB uploaded, versus
  13.4 MB plain.
- **The timings exclude the network.** No latency, retries, interceptors or physical
  devices are included.
- **Trust change:** clients self-clip before masking. The server can no longer verify
  clipping, so a malicious client could exceed the clip that robustness and central DP
  rely on.

## Combinations marked unavailable (not run)

| Combination | Status | Reason |
| --- | --- | --- |
| Secure aggregation + trimmed mean / median | Unavailable | Needs per-coordinate order of individual updates; the server learns only the sum. The harness refuses it. |
| Secure aggregation + Krum / Multi-Krum / Bulyan | Unavailable | Needs pairwise distances between individual updates |
| Secure aggregation + weighted mean | Unavailable as built | The masked-sum path is a uniform sum |
| Secure aggregation + central DP | Mechanically possible, not run | The server would add noise to the recovered sum; trust and clipping caveats as above |
| Local DP | Not run | Needs its own qualification amendment (client and server noise reporting) |

Private evidence: DP `summary.json`
`5e692a6aa5ac9c161b305cfefa5e37e919bc0c575ab4c728b9c77f4518b1c34a`; secure
`summary.json` `687d9870747928d2a5a6fc7687f14de72c3d42009963280fce217f19de7f367c`.
Per-run records, protocol logs and margins are in the same private directories.
