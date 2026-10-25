# Effusion stronger frozen encoders (A22): result

23 October 2026. Executes amendment A22, which was committed before any A22 cache was built.
Everything is on the selection half; the holdout is untouched.

**Claim correction, 24 October 2026:** the gate is a development screen after selecting fits
on the same patients, and the federated intervals are conditional on realized heads. Intervals
containing zero do not establish encoder equivalence or unchanged privacy cost. Numeric
tables, gate outcomes and artifact identities are retained. See the
[review](review-notes-2026-10-23.md).

**Integrity:**
- Both new caches cover 68,650 train and 9,075 selection images, with none refused.
  - R2 (DINOv2 ViT-B/14, MPS) built in 847 s.
  - R3 (XRV `densenet121-res224-mimic_ch`, CPU, 4 threads) built in 4,521 s.
- Each cache was verified by exact recomputation of 32 train and 32 selection rows, chosen by hash.
  - Every sampled row was bit-identical.
  - The parameter-state digest was unchanged after extraction.
  - Payload hashes matched the reports.
- **Gate:**
  - All 40 fits (4 representations × 10 grid points) converged and were finite.
  - The bootstrap used 2,000 patient-cluster resamples (seed 20261042), with no single-class draws skipped.
- **DP follow-up:**
  - All 20 runs were finite, with no refusals.
  - Every DP run's z reproduced its claimed ε exactly under GDP accounting at δ = 10⁻⁵.

## Pre-registered gate (centralised)

Pass rule: the AUROC − R0 interval is above 0 and the AUROC is at least 0.80.
The historical pass/fail outcomes below apply this rule mechanically. Each representation's
best of ten fits was chosen using these same selection patients; the bootstrap holds that
choice fixed and does not adjust for fit selection or the three candidate comparisons. The
intervals are conditional on selected heads, not independent confirmation of encoder gains.

| Representation | Width | AUROC [95%] | − R0 [95%] | Selected fit | Passed |
| --- | ---: | --- | --- | --- | --- |
| R0 ResNet-18 (historical) | 512 | 0.8220 [0.8030, 0.8386] | — | C 0.01, balanced | — |
| R1 XRV CheXpert DenseNet | 1,024 | 0.8289 [0.8102, 0.8461] | +0.0069 [−0.0052, +0.0187] | C 0.01, unweighted | no |
| R2 DINOv2 ViT-B/14 CLS | 768 | 0.8352 [0.8171, 0.8518] | +0.0132 [+0.0007, +0.0253] | C 0.01, balanced | **yes** |
| R3 XRV MIMIC DenseNet | 1,024 | 0.8314 [0.8125, 0.8479] | +0.0094 [−0.0025, +0.0208] | C 0.01, unweighted | no |

C 0.01 is an interior grid point for every representation (the grid runs 0.001–10).

## Pre-registered DP follow-up (R2 only, federated)

- **Setup:** A17's configuration, with only the features changed.
  - A8 recipe (LR 0.03, clip 0.05), 80 clients, zero-initialised head, exact-GDP z, T = 20.
  - Five seeds, paired seed by seed with A17's R0 `zero_T20` cells.
- **Intervals:** paired patient bootstrap, 2,000 resamples, seed 20261043, conditional on the
  five realized trained heads per cell. These intervals exclude training-seed, partition and
  DP-noise variability and do not establish differences between training procedures.

| Privacy | R2 AUROC | R2 − R0 (A17) | R2 DP cost | R2 DP cost − R0 DP cost |
| --- | ---: | --- | --- | --- |
| clean | 0.8158 [0.7954, 0.8346] | +0.0085 [−0.0022, +0.0195] | — | — |
| ε 8 | 0.7875 [0.7669, 0.8070] | +0.0096 [−0.0011, +0.0212] | 0.0283 [0.0241, 0.0329] | −0.0011 [−0.0056, +0.0038] |
| ε 4 | 0.7431 [0.7226, 0.7636] | +0.0117 [−0.0007, +0.0241] | 0.0728 [0.0644, 0.0819] | −0.0033 [−0.0119, +0.0060] |
| ε 1 | 0.6006 [0.5815, 0.6217] | +0.0140 [−0.0042, +0.0312] | 0.2152 [0.1975, 0.2331] | −0.0055 [−0.0234, +0.0135] |

**Mean final-round noise-to-signal, R0 → R2:**
- ε 8: 3.27 → 4.11
- ε 4: 5.68 → 7.09
- ε 1: 16.72 → 20.42

The ratio is about 1.25, close to the √(768/512) ≈ 1.22 expected from the wider head.

## Findings

1. **DINOv2 has about +0.01 selection AUROC and is the only candidate to clear the declared gate.**
   - DINOv2, a general-purpose self-supervised model, passes by a 0.0007 margin.
   - Neither chest-X-ray model passes. The CheXpert DenseNet (R1) gains +0.007 and the MIMIC DenseNet (R3) gains +0.009.
   - The screen does not identify a general representation ceiling; the small pass margin is
     subject to selection optimism and unresolved pretraining overlap.
2. **Differences in DP cost are unresolved.**
   - R2's DP-cost point estimates are within 0.006 of R0's at every ε, and every conditional
     patient interval includes 0. This does not establish equal cost or absence of benefit.
   - The wider head raises the measured noise-to-signal ratio by about 1.25, close to the √d
     scaling expectation. Width, features and optimization differ together, so these runs do
     not show that extra noise cancels a representation gain.
   - This screen does not isolate or refute a general benefit from better public representations.
3. **Under DP, R2 has positive point differences of about +0.01, with uncertainty spanning zero.**
   - At ε 4 the point estimate is 0.743, against A17's 0.731. The interval just includes 0.
   - Fewer rounds (T = 5) were not run with R2; that was not pre-registered.
4. **Observed federated-vs-centralised point gaps are 0.019 for R2 (0.835 → 0.816) and 0.015
   for R0 (0.822 → 0.807).** No equivalence margin or procedure-level gap comparison was tested.

## Platform implications

- **Retain R0 as a conservative default pending stronger evidence.** These five-seed selection
  comparisons do not establish that its utility per privacy budget equals R2's, or that a
  stronger encoder cannot improve it.
- **DINOv2 is a candidate for further evaluation.** Its observed gain is about +0.01, but
  selection and procedure-level uncertainty prevent treating that gain as confirmed. It is
  costlier to compute: a ViT-B/14 forward pass per image (847 s for 77,725 images on MPS).
- Data-independent or public-data-fitted dimension reduction is one untested way to control
  head dimension. Recipe tuning is another open factor. Neither is guaranteed to improve
  utility, and dimension reduction is not established as the only route to improvement.

**Limits:**
- Selection half only, one workload, IID, five seeds.
- The recipe was tuned on R0. A wider head may want a different LR or clip.
- R2 passed by a margin of 0.0007. This fragile development gate is not confirmation; later
  confirmation requires genuinely new evaluation data under a frozen plan. The existing
  holdout remains closed.
- **Grid selection on the selection half:** the gate picks the best of 10 grid points there, so
  the gate AUROCs are subject to optimism. Applying the same procedure to each representation
  does not establish that selection bias cancels in their differences. Repeated selection and
  three nominal candidate intervals are not a familywise confirmatory test.
- **R2 leakage caveat:** web-crawled chest X-rays in LVD-142M cannot be ruled out. NIH
  ChestX-ray14 was not a listed source.

At epsilon 8, the retrospective R2-minus-R0 seed interval is [0.00196, 0.01725] while its patient interval includes zero. These different targets do not establish a population-wide improvement.

Retrospective uncertainty update: the [24 October selection-only seed reanalysis](selection-seed-reanalysis-2026-10-24.md) reproduces the original points and reports separate paired-seed intervals. The original patient intervals remain conditional and unchanged.

Private evidence:
- **Gate:** `encoder-gate-v1/summary.json` `f7bf0f4966a709f13b20fb3096fb8b7b07cdc505d29f329c0684b2bb4846d706`.
- **DP follow-up:** `encoder-dp-r2-v1/summary.json` `4196c213fb89e51face7a9ef1053ec9fc3fe2d74fe980827498377f887f61820`.
- **Cache reports:**
  - R2: `dcc9d2f4d49609acb58d943094cac62c906cd252cc28a1b5b25cac3cc675f5b8`
  - R3: `bd31220bd506aa66e5dde165270bbe63478f23413153182395d425000b25891c`
- **Verification receipts:** `features-dinov2-base-mps-v1-verify.json` and `features-xrv-mimic-cpu-v1-verify.json`.
