# Effusion matched clean federated baseline v1: result

22 October 2026. Executes protocol §6 with amendments A1 (seeds, engine, control),
A2 (proceed on R1 after the failed gate) and A3 (reading rule). A3 was committed
before any result was opened. All results are on the selection half; the holdout and
official test were not touched.

The R1 representation is used by decision (A2). It was statistically tied with R0 at
the gate and is not claimed to be better.

## Setup

- **Engine:** the production engine (`ComposedStrategy` + `FLCoordinator` via
  `SimulatedFederation`), with the Effusion/R1 workload passed in.
- **Recipe:** a 2,050-parameter linear head; 20 rounds; 7 local epochs; SGD LR 0.003;
  batch 256; update clip 0.05; mean rule.
- **Seeds:** 20261101–20261105.
- **Arms:**
  - Federated: 80 whole-patient IID clients. Every client holds positives.
  - Central-SGD control: one client holding all of train, otherwise identical.
- **Integrity:** all 10 runs completed with no refusals. The engine source hash is
  recorded in the summary.
- **A first attempt aborted at import** because the harness pointed at a nonexistent
  framework path. It produced no results; the harness was fixed and now has a
  fresh-process guard.

## Results

AUROC uses float64 logit margins. Intervals are paired patient-cluster bootstraps
(2,000 resamples, seed 20261023).

| Seed | Federated | Central-SGD | Federated − central-SGD [95% CI] | Last-round clip rate (fed / central) |
| --- | ---: | ---: | --- | --- |
| 20261101 | 0.7940 | 0.8035 | −0.0095 [−0.0169, −0.0027] | 0.78 / 1.00 |
| 20261102 | 0.7993 | 0.8053 | −0.0060 [−0.0178, +0.0048] | 0.75 / 1.00 |
| 20261103 | 0.7966 | 0.8042 | −0.0076 [−0.0154, +0.0002] | 0.66 / 1.00 |
| 20261104 | 0.7992 | 0.8012 | −0.0020 [−0.0109, +0.0069] | 0.74 / 1.00 |
| 20261105 | 0.7948 | 0.7876 | +0.0072 [−0.0046, +0.0188] | 0.61 / 1.00 |
| **Seed mean** | **0.7968** | **0.8004** | −0.0036 | 0.71 / 1.00 |

The convex R1 head from the gate scores 0.8289. Federated − convex ranges from −0.030
to −0.035 across seeds, and central-SGD − convex from −0.024 to −0.041. Every one of
those intervals excludes zero.

## Reading under A3

- **Primary contrast.** Federation's cost relative to the matched central-SGD control
  is small: −0.004 AUROC on the seed mean. Only one of five seed intervals excludes
  zero, and one seed favours the federated arm.
- **The recipe is limiting**, on both A3 tests:
  - the central-SGD arm clipped in every round of every seed (rate 1.00 ≥ 0.95);
  - its seed mean is 0.029 below the convex head (> 0.02).

  The gaps to the convex head are therefore **recipe-limited**. They are not the cost of
  federation. The central-SGD curve was still rising at round 20 (0.550 → 0.804 on
  seed 1), while the federated curve was flattening (0.662 → 0.794).
- **The comparison is at a fixed 20-round, clip-0.05 budget, not at convergence.**
  Under this clip, one client taking 269 steps per epoch moves no further per round
  than the average of 80 clipped client updates.
- **Communication.** Each federated run uploaded 13.4 MB over 20 rounds (80 × 8,348
  bytes per round). The central-SGD run uploaded 0.17 MB.

Not shown: patient-level (as opposed to client-level) effects, non-IID allocation,
physical devices, calibration and holdout performance. No LR, clip, epoch or round
setting was changed after these results (A3).

Private evidence: `summary.json` `12c0788a0b39d3026598cc6c57dc0b192ec7e929290f5716ed599af7a05b8c0b`,
plus per-run records and selection margins in the same private directory.
