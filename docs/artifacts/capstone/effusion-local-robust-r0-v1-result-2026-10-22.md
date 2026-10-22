# Effusion R0 robust aggregation under local DP (A12): result

22 October 2026. Executes amendment A12, which was committed before the run. Everything is
on the selection half.

**Integrity:**
- All 135 runs were finite with no refusals.
- Every attacked run had exactly 16 of 80 attacker slots.
- The ε 8 mean clean cells reproduced the A11 margins exactly on all five seeds.
- Intervals are paired patient bootstraps (2,000 resamples, seed 20261031).

**Setup:**
- R0 with the A8 recipe (LR 0.03, clip C = 0.05), 20 rounds.
- Honest clients clip to C and add N(0, (2zC)²). Attackers do not follow the protocol and add no noise.
- The server applies only the engine's local-mode post-processing clip, C + 2zC(√d + 4). It is
  set high enough not to shrink honest privatised updates.

## Seed-mean AUROC

| ε | Rule | Clean | Sign flip ×−4 | Label flip |
| --- | --- | ---: | ---: | ---: |
| 8 | mean | 0.557 | 0.305 | 0.452 |
| 8 | trimmed mean | 0.561 | 0.298 | 0.413 |
| 8 | median | 0.552 | 0.299 | 0.218 |
| 32 | mean | 0.653 | 0.300 | 0.385 |
| 32 | trimmed mean | 0.651 | 0.298 | 0.319 |
| 32 | median | 0.633 | 0.299 | 0.242 |
| 128 | mean | 0.742 | 0.310 | 0.307 |
| 128 | trimmed mean | 0.740 | 0.310 | 0.276 |
| 128 | median | 0.729 | 0.311 | 0.258 |

**Robust rule minus mean** (same ε, condition and seed):
- ε 8, label flip: trimmed mean −0.0391 [−0.0445, −0.0337], median −0.2348 [−0.2500, −0.2189].
- ε 128, label flip: trimmed mean −0.0314, median −0.0493.
- Sign flip: between −0.007 and +0.0003 at every ε, for both rules.
- Clean: trimmed mean within ±0.003 of mean; median −0.006 to −0.020.

**Attack drops for the mean** (clean − attacked):

| ε | Sign flip | Label flip |
| --- | --- | --- |
| 8 | 0.253 [0.226, 0.277] | 0.105 [0.094, 0.116] |
| 32 | 0.353 | 0.268 |
| 128 | 0.432 | 0.435 |

## Findings

1. **Under local DP, 20% non-compliant attackers break every rule tested.** Sign flip drives
   AUROC to about 0.30, below chance: the model learns the inverted direction. This happens at
   every ε and under every rule. Label flip does similar damage, growing with ε (0.45 → 0.31 for
   the mean).
2. **Robust aggregation does not help under local DP.** It is often worse: the median under
   label flip loses up to 0.23 AUROC relative to the mean.
3. **The mechanism:**
   - Local DP makes honest updates very noisy, so the server's norm bound has to be generous.
     That effectively removes the norm bounding that kept attacks small without DP and under
     central DP (A9).
   - Coordinate-wise rules cannot separate a consistent attacker direction from noise-dominated
     honest updates. That is the noise-dominance failure described by Guerraoui et al. (PODC'21)
     and Allouah et al. (ICML'23).
   - Damage grows with ε here because honest updates become *less* noisy while attackers keep
     the same large, consistent pushes.
4. **What to try next:** the literature's remedy is noise-aware filtering (Xiang et al.,
   SIGMOD'23). Honest privatised update norms concentrate around √(C² + (2zC)²d), so the server
   can reject uploads that don't look like "clip plus noise". Rejection is post-processing, so it
   costs no privacy. This is shortlist item 8 in the
   [literature check](literature-review-robust-private-fl-2026-10-22.md).

## Platform implication

Local DP is not just costly in utility (A11). Without norm validation it is also **fragile
against poisoning**: noise that hides honest clients also hides attackers. If a local-DP option
ever ships, it needs server-side noise-aware norm validation, or a verifiable client norm proof.
It should not be paired with coordinate-wise robust rules as a defence.

**Limits:**
- One attacker fraction (20%), non-adaptive attacks, attackers that skip local noise.
- The engine's generous local-mode server clip is one reasonable design. A tighter, noise-aware
  bound is the remedy and has not been tested here.

Private evidence: `summary.json`
`5de3c331fb35693a057320fa882c28ed75c568d44c3e08fe2701ff0683b1bc17`.
