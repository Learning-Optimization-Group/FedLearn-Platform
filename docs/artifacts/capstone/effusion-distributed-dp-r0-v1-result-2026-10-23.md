# Effusion distributed DP (Skellam over coded LightSecAgg, A23): result

23 October 2026. Executes amendment A23, which was committed (`beda39e`) before the run. Everything
is on the selection half; the holdout is untouched.

**Integrity:**
- 90 runs completed: 55 distributed, 15 central and 20 attacked. None was refused or non-finite,
  and there were no integrity failures.
- Every one of the 1,300 decoded rounds in the 65 distributed runs (20 per run) was exactly equal to the direct modular sum of
  the client messages.
- Every run's accounted ε was within its claim. Every central run's z equalled the RDP calibration.
- The five b = 12, ε 1 cells are recorded as "not run": 12 bits cannot hold that aggregate at any
  scale, as pre-declared.
- Pre-registered intervals are paired patient bootstraps (2,000 resamples, seed 20261044).

**Setup:**
- A17's T = 20 zero-head configuration: R0, A8 recipe (LR 0.03, clip 0.05), 80 clients, mean,
  five seeds.
- In the distributed cells the server receives only the decoded secure sum.
- Coded LightSecAgg uses T = 40 colluders and U = 64 survivors.

## Utility (AUROC, five-seed mean)

| Claimed ε | Central exact GDP (A17) | Central RDP | Distributed b 20 | b 16 | b 14 | b 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.7779 | 0.7698 | 0.7767 | 0.7723 | 0.7709 | 0.7700 |
| 4 | 0.7313 | 0.7105 | 0.6987 | 0.7148 | 0.7213 | 0.6896 |
| 1 | 0.5867 | 0.5673 | 0.5732 | 0.5694 | 0.5051 | not run (infeasible) |

**Effective noise multiplier** (aggregate noise SD on the sum ÷ 2C):

| ε | Central RDP z | b 20 | b 16 | b 14 | b 12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 | 3.087 | 3.100 | 3.102 | 3.113 | 3.261 |
| 4 | 5.796 | 5.800 | 5.807 | 5.868 | 6.924 |
| 1 | 21.920 | 21.925 | 22.150 | 25.547 | — |

Fewer bits mean a smaller scale, so rounding inflates the sensitivity and more noise is needed.
At 16–20 bits the mechanism is within 0.5% of central RDP at ε 8 and ε 4, and within 1.1% at ε 1. Part of that comes from Skellam's bound holding only at integer orders.
At 12–14 bits the extra noise is 17–19% where the field binds: b 12 at ε 4, and b 14 at ε 1.

## Pre-registered contrasts

| Contrast | ε 8 | ε 4 | ε 1 |
| --- | --- | --- | --- |
| Central RDP − exact GDP (accounting gap, zero head) | −0.0081 [−0.0089, −0.0073] | −0.0208 [−0.0222, −0.0193] | −0.0194 [−0.0211, −0.0178] |
| Distributed b 16 − central RDP | +0.0025 [−0.0039, +0.0083] | +0.0043 [−0.0048, +0.0127] | +0.0021 [−0.0157, +0.0175] |
| Distributed b 20 − central RDP | +0.0069 [+0.0006, +0.0127] | −0.0118 [−0.0214, −0.0023] | +0.0059 [−0.0081, +0.0191] |
| Distributed b 14 − central RDP | +0.0011 [−0.0050, +0.0067] | +0.0108 [+0.0016, +0.0187] | −0.0622 [−0.0792, −0.0451] |
| Distributed b 12 − central RDP | +0.0002 [−0.0053, +0.0057] | −0.0209 [−0.0316, −0.0109] | — |
| Distributed b 16 − exact GDP (A17) | −0.0056 [−0.0116, −0.0000] | −0.0165 [−0.0252, −0.0085] | −0.0173 [−0.0356, −0.0013] |

**Reading these intervals:**
- The patient bootstrap resamples patients, not the DP noise.
- Central RDP and exact GDP share their noise draws up to scale, so their gap is well resolved.
- A distributed cell and a central cell draw independent noise. With per-seed AUROC SD of
  0.006–0.008 at ε 8, 0.02–0.04 at ε 4 and about 0.05 at ε 1, their bootstrap intervals are too
  narrow.
- Seed-level paired t intervals, computed after the run (exploratory, `distributed-dp-r0-v1-seed-intervals.json`):
  - Every distributed − central RDP interval includes 0.
  - Widths are about ±0.012 at ε 8, ±0.05 at ε 4 and ±0.08–0.12 at ε 1.
- The opposite-signed b 20 bootstrap results (+0.007 at ε 8, −0.012 at ε 4) are therefore read as
  noise-draw variation, not as an effect of the mechanism.

## Clip-skipping attackers (ε 4)

The attack is sign flip ×10. Under distributed DP at b = 16 the attackers skip the clip; under
central DP the server clips every upload.

| Attackers of 80 | Central RDP AUROC | Distributed b 16 AUROC | Damage, central | Damage, distributed | Distributed − central damage |
| ---: | ---: | ---: | --- | --- | --- |
| 4 | 0.7000 | **0.2540** | 0.0106 [0.0097, 0.0116] | **0.4608** [0.4220, 0.5000] | +0.4502 [+0.4121, +0.4886] |
| 16 | 0.6582 | 0.4819 | 0.0524 [0.0489, 0.0561] | 0.2329 [0.2122, 0.2526] | +0.1805 [+0.1617, +0.1983] |

**Field wraps:**
- Clean distributed runs had 0–5 wrapped coordinates in 40,960 coordinate-rounds. That matches
  the 4σ tail implied by k = 4, so it is expected and costs far less than the DP noise.
- The attacked runs had about 4,400 (4 attackers) and 10,800 (16 attackers) wrapped coordinates.

**Why more attackers did less damage:**
- Four unclipped attackers push the model coherently below chance in every seed (AUROC
  0.246–0.274).
- Sixteen overflow the field so often that the wraps scramble their direction, and the result is near chance with
  a large spread (0.385–0.560).
- Neither is safe.

## Communication (analytic, bit-packed, per client per round)

Plain float32 upload of the 1,026-parameter head: 4,104 bytes.

| Bits | Coded LightSecAgg | Full-length Shamir shares (`lightsecagg.py` scheme) | Coded ÷ plain |
| ---: | ---: | ---: | ---: |
| 12 | 13,392 | 248,832 | 3.3× |
| 16 | 17,856 | 331,776 | 4.4× |
| 20 | 22,320 | 414,720 | 5.4× |

**What the coded total is made of (b 16):**
- 79 offline shares of 86 elements each: 13,588 bytes.
- The masked vector of 2,048 elements: 4,096 bytes. Padding to a power of two doubles it.
- One recovery share: 172 bytes.

Coding cuts the traffic 18.6×. A5's measured 158× also includes protobuf and sealing overhead,
about 4× over the analytic element count, so these figures are not wire measurements.

## Findings

1. **The trusted server can be removed without losing utility beyond the accounting.**
   - At 16 bits, distributed Skellam is indistinguishable from central DP with the same (RDP)
     accountant at every ε. The effective noise is within about 1% of it.
   - At this cohort size, 16 bits is the practical minimum; 12–14 bits add 17–19% noise where the
     field binds (b 12 at ε 4, b 14 at ε 1).
2. **What it does cost is the exact accounting.**
   - Skellam has no exact-GDP analysis, so against A17's exact-GDP central DP it loses 0.006 /
     0.017 / 0.017 at ε 8 / 4 / 1.
   - This is the same gap as central RDP vs GDP, and it replicates A13 with a zero head.
3. **Without a server that can see updates, one unverified clip undoes the defence.**
   - Four clip-skipping clients out of 80 (5%) drove AUROC to 0.25, below chance, under distributed
     DP. Central DP with server-side clipping lost 0.011 to the same attack.
   - Secure aggregation removes the server's ability to clip, filter or inspect, and the clip is
     what bounded the attack.
   - Distributed DP is safe against poisoning only if each client's clip is verified, for example by
     a zero-knowledge norm proof or a two-server check. That is not built here.
4. **Coded shares make secure aggregation affordable.**
   - At 16 bits a client sends about 4.4× a plain float32 upload instead of 81× (full-length
     Shamir shares, same accounting). A5 measured 158× on the wire.

**Predictions (A23) vs outcome:**
- **"b 20 within ±0.005 of central RDP":** not met on the bootstrap intervals (+0.007 at ε 8,
  −0.012 at ε 4). It is met in effective noise (≤ 0.5%), and every seed-level interval includes 0.
  Noise-draw variation is the reason.
- **"Fewer bits cost first at ε 1":** met. b 14 at ε 1 lost 0.062 on the bootstrap. b 12 also lost
  0.021 at ε 4, where its field binds.
- **"Distributed loses about A13's gain against A17":** met (0.017 at ε 4).
- **"Clip-skipping attackers do far more damage under distributed DP":** met (0.46 vs 0.011).

## Platform implications

- **Distributed DP is a credible option for "no trusted server" deployments that can verify
  clients:**
  - use 16-bit fields and coded LightSecAgg;
  - accept RDP accounting (about 0.02 AUROC at ε 4 compared with exact central DP).
- **Do not offer it to federations with untrusted clients until clipping is verified.**
  Server-side clipping under central DP is a much stronger poisoning defence than anything
  available once updates are hidden.
- The production DeComFL secure-aggregation path still uses full-length Shamir shares. Coding
  them would cut its offline traffic by about U − T = 24×.

**Limits:**
- Simulator only, with no production wiring. The secure sum runs in-process.
- Bytes are analytic. Key agreement and authenticated channels are assumed.
- No dropouts. Honest-but-curious server. One attack family, non-adaptive.
- IID, one workload, five seeds; the recipe was tuned for central DP.
- The seed-level intervals were added after the run.

Private evidence:
- `distributed-dp-r0-v1/summary.json` `cba952640c2dd8b9f71ecf77315765ab117574daaf5d7efc92bb680f43443465`.
- `distributed-dp-r0-v1-seed-intervals.json` `f80e370f9664f9c4f4c95e298b8edd44974d9ba9ee6d143120ac213e6cf60cd7`.
- Commits:
  - framework `4d4ff3d`;
  - harness `31d8234`.
