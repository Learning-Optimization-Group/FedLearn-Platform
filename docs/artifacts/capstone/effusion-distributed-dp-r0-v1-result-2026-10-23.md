# Effusion distributed DP (Skellam over coded LightSecAgg, A23): result

23 October 2026. Executes amendment A23, which was committed (`beda39e`) before the run. Everything
is on the selection half; the holdout is untouched.

**Correction notice — 24 October 2026:** the recorded runs, numeric tables and evidence hashes below
are unchanged. The original conclusions overstated both the composed privacy guarantee and utility
equivalence. The μ/n noise allocation is calibrated for an ideal sum in which every client's noise
is generated and remains unknown to the server: no server/client collusion, noise disclosure or
noise withholding. The T = 40 coded-share threshold does not extend that DP guarantee to 40 colluders.
The reproducible masks and experiment seeds also make confidentiality a simulated interface, rather
than an operational cryptographic property. See the [review](review-notes-2026-10-23.md) and its
[synthetic collusion proof](review-2026-10-23/privacy_collusion_proof.py). The interpretation below
supersedes the original equivalence and deployment claims; the planned
[selection seed reanalysis](selection-seed-reanalysis-2026-10-24.md) addresses procedure-level
uncertainty without reopening the holdout.

**Integrity:**
- 90 runs completed: 55 distributed, 15 central and 20 attacked. None was refused or non-finite,
  and there were no integrity failures.
- Every one of the 1,300 decoded rounds in the 65 distributed runs (20 per run) was exactly equal to the direct modular sum of
  the client messages.
- Every run's accounted ε was within its claim under its stated calibration; for distributed runs,
  that calibration requires the full unknown-noise contribution described above. This is an
  accounting consistency check, not validation against collusion or noise withholding. Every central
  run's z equalled the RDP calibration.
- The five b = 12, ε 1 cells are recorded as "not run": 12 bits cannot hold that aggregate at any
  scale, as pre-declared.
- Pre-registered intervals are paired patient bootstraps (2,000 resamples, seed 20261044).

**Setup:**
- A17's T = 20 zero-head configuration: R0, A8 recipe (LR 0.03, clip 0.05), 80 clients, mean,
  five seeds.
- In the distributed cells the modeled server interface receives only the decoded secure sum. The
  in-process implementation has diagnostic access to individual messages, and its masks are
  deterministic functions of client IDs and experiment seeds. It does not establish operational
  confidentiality.
- Coded LightSecAgg uses the algebraic coded-share parameters T = 40 colluders and U = 64 survivors.
  The threshold describes the sharing construction under private-randomness/channel assumptions;
  neither implemented mask secrecy nor a DP guarantee against those colluders follows from it.

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
At 16–20 bits the calibrated aggregate noise SD is within 0.5% of central RDP at ε 8 and ε 4,
and within 1.1% at ε 1. This is a noise-scale comparison, not evidence of equivalent AUROC.
Part of the noise-scale gap comes from Skellam's bound holding only at integer orders.
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
- The patient bootstrap resamples patients, not training seeds or DP noise. Its intervals are
  conditional on the realized fitted heads.
- Central RDP and exact GDP share their noise draws up to scale. This pairing helps compare the
  realized heads, but it does not make the patient bootstrap a procedure-level uncertainty interval.
- A distributed cell and a central cell draw independent noise. With per-seed AUROC SD of
  0.006–0.008 at ε 8, 0.02–0.04 at ε 4 and about 0.05 at ε 1, their bootstrap intervals are too
  narrow for claims about the training procedures.
- Seed-level paired t intervals, computed after the run (exploratory, `distributed-dp-r0-v1-seed-intervals.json`):
  - Every distributed − central RDP interval includes 0.
  - Widths are about ±0.012 at ε 8, ±0.05 at ε 4 and ±0.08–0.12 at ε 1.
- These broad seed intervals leave meaningful utility differences unresolved. Including zero is
  not evidence of equivalence. Noise-draw variation is a plausible contributor to the opposite-signed
  b 20 results (+0.007 at ε 8, −0.012 at ε 4), but these five seeds do not identify its contribution
  separately from a mechanism effect. The original ±0.005 AUROC prediction remains unestablished.

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

Coding cuts this analytic traffic estimate 18.6×. A5's measured 158× also includes protobuf and sealing overhead,
about 4× over the analytic element count, so these figures are not wire measurements.
They do not establish deployment affordability, latency or operational cost.

## Findings

1. **The tested distributed calibration has noise close to central RDP; utility equivalence is unresolved.**
   - At 16 bits, the calibrated effective noise is within about 1% of central RDP. Five seeds do not
     resolve a utility difference or establish equivalence at any ε.
   - For this configuration, 16 bits is a candidate worth further evaluation. The tested 12–14-bit
     cases add 17–19% noise where the field binds (b 12 at ε 4, b 14 at ε 1). This does not establish
     a universal minimum bit width.
2. **An accounting-related noise difference is established; its utility contribution is not isolated.**
   - This implementation uses a Skellam RDP bound rather than an exact-GDP analysis. Against A17's
     exact-GDP central DP, the observed distributed b 16 AUROC differences are −0.006 / −0.017 /
     −0.017 at ε 8 / 4 / 1.
   - Those arms draw independent noise. Their observed differences cannot be attributed entirely
     to the accountant. The paired central RDP versus GDP comparison more directly examines the
     calibration change with a zero head; its patient intervals still condition on the fitted heads.
3. **Unverified clipping permits the measured attack; verifying it does not establish robustness.**
   - Four clip-skipping clients out of 80 (5%) drove AUROC to 0.25, below chance, under distributed
     DP. Central DP with server-side clipping lost 0.011 to the same attack.
   - Secure aggregation removes the server's ability to clip, filter or inspect, and the clip is
     what bounded the attack.
   - Enforcing each client's norm bound, for example by a norm proof or a two-server check, is
     necessary to justify the clipping premise. It is not sufficient protection against malicious
     directions inside the bound. Such validation is not built here.
   - Privacy additionally requires verified noise generation and a declared minimum number of honest
     contributors whose noise remains unknown. Even protocol-following colluders can reveal their
     own noise; a norm proof cannot replace that missing variance.
4. **Coded shares reduce the analytic traffic estimate.**
   - At 16 bits the estimate is about 4.4× a plain float32 upload instead of 81× (full-length
     Shamir shares, same accounting). A5 measured 158× on the wire. The analytic reduction is not
     a wire-cost measurement or an affordability result.

**Predictions (A23) vs outcome:**
- **"b 20 within ±0.005 of central RDP":** unestablished. The observed differences exceed the
  margin (+0.007 at ε 8, −0.012 at ε 4); the patient intervals condition on the fitted heads, and
  the broad seed intervals do not establish equivalence. Close effective noise (≤ 0.5%) is a
  different outcome and does not satisfy an AUROC prediction.
- **"Fewer bits cost first at ε 1":** the noise calibration supports this direction; b 14 at ε 1
  lost 0.062 on the conditional patient bootstrap. b 12 also lost 0.021 at ε 4, where its field
  binds. The procedure-level utility differences remain uncertain.
- **"Distributed loses about A13's gain against A17":** the observed difference is 0.017 at ε 4.
  This does not isolate the accounting contribution because the mechanisms draw independent noise.
- **"Clip-skipping attackers do far more damage under distributed DP":** met (0.46 vs 0.011).

## Platform implications

- **Treat this as a simulation candidate requiring privacy and protocol qualification.** For the
  tested configuration, 16-bit fields and coded sharing give close calibrated noise and lower
  analytic traffic. They do not establish equivalent utility, confidentiality or deployment readiness.
- A deployment needs private client randomness, authenticated confidential share transport, norm
  validation, and a noise protocol calibrated to a declared minimum unknown honest contribution.
  With μ/n per-client noise, the current DP calibration assumes no server/client collusion, noise
  disclosure or withholding. T = 40 coded sharing does not make that calibration collusion-safe.
- Norm validation constrains this clip-skipping attack but leaves in-bound poisoning possible.
  Further threat-specific robustness evaluation is required; clipping verification alone is not an
  approval criterion. The observed distributed-versus-GDP AUROC gap is not a fixed deployment cost.
- The production DeComFL secure-aggregation path still uses full-length Shamir shares. Coding
  them would cut its offline traffic by about U − T = 24×.

**Limits:**
- Simulator only, with no production wiring. The secure sum runs in-process.
- Bytes are analytic; operational affordability is unmeasured. Key agreement and authenticated
  confidential channels are assumed. Deterministic public masks implement a reproducible algebra
  experiment, not confidential client uploads.
- No dropouts. Honest-but-curious server; the distributed DP calibration additionally excludes
  colluding clients, noise disclosure and noise withholding. The reported poisoned runs are utility
  stress tests, not evidence that the composed privacy guarantee covers arbitrary malicious clients.
- One attack family, non-adaptive. Verified norms would still permit in-bound poisoning.
- IID, one workload, five seeds; the recipe was tuned for central DP.
- The seed-level intervals were added after the run.

Private evidence:
- `distributed-dp-r0-v1/summary.json` `cba952640c2dd8b9f71ecf77315765ab117574daaf5d7efc92bb680f43443465`.
- `distributed-dp-r0-v1-seed-intervals.json` `f80e370f9664f9c4f4c95e298b8edd44974d9ba9ee6d143120ac213e6cf60cd7`.
- Commits:
  - framework `4d4ff3d`;
  - harness `31d8234`.
