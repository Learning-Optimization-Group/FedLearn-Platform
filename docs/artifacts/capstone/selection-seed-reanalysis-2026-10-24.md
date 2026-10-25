# Selection-only paired-seed reanalysis

24 October 2026. Retrospective correction of uncertainty interpretation; no new model fits.

Eight pinned aggregate summaries were hash-verified. All 173 original contrast points in A11, A12, A19, A22 and A23 were reproduced from saved per-run AUROC scalars to tolerance 1e-10. No patient manifests, per-image scores or raw holdout/test inputs were read. Original artifacts and conditional patient intervals remain unchanged.

## What these intervals measure

The original patient-cluster intervals condition on five realized heads. The new paired seed-mean t intervals describe variation across those five development training replicates on the fixed selection cohort. These are separate uncertainty targets: the new intervals can be wider or narrower, and neither is a joint patient-and-training interval.

The t intervals are approximate, retrospective and unadjusted for multiple comparisons. Normality is unverified. Five seeds, including seeds used during recipe development, do not establish independent confirmation, equivalence, population generalization or privacy of the research exports.

## Complete coverage

| Study | Original contrasts reproduced | Effect intervals excluding zero: patient | Effect intervals excluding zero: seed |
| --- | ---: | ---: | ---: |
| A11 | 13 | 8 | 7 |
| A12 | 63 | 29 | 26 |
| A19 | 34 | 15 | 12 |
| A22 | 14 | 3 | 5 |
| A23 | 49 | 24 | 9 |

Interval classifications are descriptive, not multiplicity-adjusted discovery counts. Absolute-AUROC contrasts are excluded from the zero-comparison columns.

## Selected decision-relevant contrasts

| Study / contrast | Point | Original conditional patient 95% interval | New paired-seed 95% interval |
| --- | ---: | --- | --- |
| A11: Local DP epsilon 1 | +0.5252 | [+0.5162, +0.5347] | [+0.4024, +0.6481] |
| A11: Local minus central, epsilon 8 | -0.1860 | [-0.2042, -0.1672] | [-0.2839, -0.0881] |
| A12: Local mean sign-flip damage, epsilon 8 | +0.2528 | [+0.2258, +0.2767] | [+0.1093, +0.3964] |
| A12: Local mean label-flip damage, epsilon 8 | +0.1050 | [+0.0935, +0.1155] | [+0.0583, +0.1517] |
| A19: Strong-skew private AUROC change | -0.0384 | [-0.0425, -0.0341] | [-0.0615, -0.0152] |
| A22: DINOv2 minus R0, epsilon 8 | +0.0096 | [-0.0011, +0.0212] | [+0.0020, +0.0172] |
| A22: DINOv2 minus R0, epsilon 4 | +0.0117 | [-0.0007, +0.0241] | [-0.0074, +0.0309] |
| A22: DINOv2 minus R0 privacy cost, epsilon 4 | -0.0033 | [-0.0119, +0.0060] | [-0.0225, +0.0160] |
| A23: Distributed b16 minus central RDP, epsilon 4 | +0.0043 | [-0.0048, +0.0127] | [-0.0450, +0.0535] |
| A23: Distributed b16 minus central RDP, epsilon 1 | +0.0021 | [-0.0157, +0.0175] | [-0.1148, +0.1190] |
| A23: Extra distributed damage, four attackers | +0.4502 | [+0.4121, +0.4886] | [+0.4007, +0.4998] |

## Interpretation

- A11's epsilon-1 local-DP interval is much wider over seeds than over patients. Its observed point remains weak, while the local-minus-central epsilon-8 gap remains negative under the seed view. This is not a conversion between their different trust guarantees.
- A12's large attack damage survives the seed analysis in the highlighted mean cells. This supports a failure of the tested local protocol/rules, not a general impossibility for all local-DP defences.
- A19's strong-skew private utility decline remains negative. The additional privacy-cost contrast below separates that decline from the clean nonprivate decline.
- A22's epsilon-8 encoder gain has a positive seed interval while its original patient interval includes zero. At epsilon 4 the gain remains unresolved; all DP-cost differences remain unresolved. This illustrates the different inference targets and does not justify either equivalence or a confirmed population-wide encoder upgrade.
- A23's b16-minus-central intervals permit material gains and losses. Close calibrated noise and a seed interval crossing zero do not establish utility equivalence. The large additional damage from four clip-skipping clients remains clear across seeds.

## Additional exploratory A19 privacy-cost contrasts

These four-term contrasts were added retrospectively: (non-IID clean minus private) minus (IID clean minus private). They are not original preregistered endpoints.

| Skew | Additional privacy cost | Paired-seed 95% interval |
| --- | ---: | --- |
| a0p1 | +0.03253 | [+0.0116, +0.0534] |
| a1 | +0.00321 | [-0.0034, +0.0099] |

The strong-skew value is 0.03253 from saved run scalars; subtraction of the rounded published table entries gives approximately 0.0326. It is not the full 0.03836 private-AUROC decline.

## Claim boundaries

A14 raw-honest oracle attacked cells retain only nominal aggregate-input noise calibration; the reused one-upload argument does not establish their client-data epsilon. A23's mu/N noise allocation does not establish protection against its coded-share collusion threshold. Those mechanism issues are independent of this statistical reanalysis. Public/shared preprocessing, deterministic research randomness, selection/tuning and diagnostic export limitations remain.

## Reproduction and provenance

[All derived contrasts](selection-seed-reanalysis-2026-10-24.json) and [CSV table](selection-seed-reanalysis-2026-10-24.csv) contain only aggregate metrics and nonidentifying source provenance. SHA-256 of the JSON: `c56a6f087dc7bd3be2a5c66627bf78c0d65836b29d7a673dc06bfc04ab971468`.

Run `repro/reanalyze_selection_seeds.py` with the declared selection evidence root and a fresh output directory, then render this addendum with `repro/render_selection_seed_addendum.py`. The analysis script accepts only its eight pinned summary inputs and reconciles complete seed/contrast coverage before writing results.

The analysis and contrast-specification tests pass 30 synthetic checks. No full framework gate was repeated for this analysis-only phase; the prior ByzFL gate remains historical evidence.
