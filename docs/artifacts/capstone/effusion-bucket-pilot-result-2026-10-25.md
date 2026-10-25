# Nonprivate median/bucket diagnostic: result

25 October 2026. All 46 new cells completed under prospective checkpoint `5940d13`,
with two clean IID unbucketed references reused from the pinned screen. The declared
recipe rule selected LR=0.1/C=1. This is a two-seed development diagnostic; none of
these arms claims differential privacy.

| Partition / condition | Median, bucket 1 | Bucket 2 | Bucket 4 |
| --- | ---: | ---: | ---: |
| IID, clean | 0.8198 | 0.8198 | 0.8200 |
| IID, owned PGD 4/80 | 0.8119 | 0.8101 | 0.8053 |
| IID, owned PGD 16/80 | 0.6719 | 0.5819 | 0.5123 |
| IID, oracle Fang 16/80 | 0.7668 | 0.6852 | 0.6566 |
| Alpha 0.1, clean | 0.7947 | 0.8089 | 0.8122 |
| Alpha 0.1, owned PGD 4/80 | 0.7827 | 0.7964 | 0.7971 |
| Alpha 0.1, owned PGD 16/80 | 0.6527 | 0.6189 | 0.5987 |
| Alpha 0.1, oracle Fang 16/80 | 0.6313 | 0.4342 | 0.4115 |

Bucketing improved the clean strong-skew point while worsening several high-attacker
points. At 16/80 attackers, buckets of four have an expected contaminated fraction
of 0.5983 under uniform grouping. The actual round contamination was recorded and
checked against the bucket counts. Contamination is a diagnosis, not a proof that
the reducer must fail or an isolated causal explanation of the utility changes.

Seed variation matters: the skewed unbucketed PGD-16 AUROCs were about 0.552 and 0.754;
bucket-four points were about 0.461 and 0.736. Those wide ranges preclude treating
the two-seed ordering as a stable universal ranking. The Fang condition is a distinct
raw-honest oracle stress arm, kept outside the permitted-information/private comparisons.

All 920 new rounds had all 80 clients aggregated. Record/source hashes, owned attack
budgets, bucket contamination arithmetic and zero-noise settings passed a separate audit.
Direct saved-head scoring reproduced every reported metric to within 1.12e-16, sharing
Torch's linear kernel while independently recalculating rank AUROC and tied-threshold AP.
The holdout was not read or scored.

See [combined pilot aggregates](effusion-diagnostic-pilots-2026-10-25.json),
[arm CSV](effusion-diagnostic-pilots-2026-10-25-arms.csv),
[contrast CSV](effusion-diagnostic-pilots-2026-10-25-contrasts.csv),
[diagnostics](effusion-diagnostic-pilots-2026-10-25-diagnostics.json) and
[phase verification](phase-3-verification-2026-10-25.json).
Private bucket summary SHA-256: `249eabd87615839c289ded26bfefee75cd9bbd8bdaefd92ee0df832f63a48996`.

This completes the declared diagnostic phase. It supports keeping bucket choice and
attacker fraction explicit, and carrying the tested privacy-compatible policies into
a separately frozen fresh-seed comparison. It does not justify adding unsupported
private bucketing or a state-of-the-art claim.
