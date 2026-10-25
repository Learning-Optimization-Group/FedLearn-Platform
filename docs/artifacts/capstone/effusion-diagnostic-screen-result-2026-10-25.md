# Effusion diagnostic screen: result

25 October 2026. All80 prospectively frozen cells completed under local checkpoint `8ea4b7d`.
This is an exploratory two-seed screen on the existing development-selection cohort, not a new
clinical evaluation or a universal method ranking. The holdout remained closed.

## Main measured components

AUROC values are two-seed means. All iterative rows use the matched zero-initialized
LR0.03/C0.05 recipe,80 full-participation clients and20 rounds. Private rows target exact
GDP epsilon4/delta1e-5; each method uses its declared sensitivity. The attack has16 malicious clients.

| Method | No-noise clean | No-noise PGD | Private clean | Private PGD |
| --- | ---: | ---: | ---: | ---: |
| Clipped mean,512 features | 0.8073 | 0.5201 | 0.7295 | 0.5963 |
| Trimmed mean,k16 | 0.8070 | 0.5864 | 0.6586 | 0.6136 |
| Centered,tau/C0.25 | 0.8051 | 0.7028 | 0.6714 | 0.5423 |
| Centered,tau/C1 | 0.8073 | 0.6033 | 0.5604 | 0.5302 |
| Centered,tau/C4 | 0.8073 | 0.5201 | 0.7289 | 0.5967 |

The private results do not support assuming a released historical center is beneficial.
At tau=C, low clean utility must remain visible alongside the smaller clean-to-attack drop;
that drop alone would create a misleading robustness recommendation. Tau/C0.25 helps
nonprivate attacked utility, yet both private absolute points are below the private mean's.
Tau/C4 behaves close to the mean in this screen. These are conditional observations,
not a reproduction or refutation of the full client-momentum history method.

## Attack qualification

The nonprivate mean loses0.0050 AUROC under individual sign flip x-4,0.0075 under the
coordinated full-norm own-mean sign flip, and0.2872 under owned-data PGD. PGD's two losses
are0.28635 and0.28805, clearing the predeclared0.02 positive-control threshold. The simpler
attacks fail that qualification. The private mean's PGD loss is0.1332 on average, with
substantial variation between the two noise/partition seeds. This qualifies a damaging
permitted-information attack for the tested workload; it does not establish worst-case robustness.

Every owned PGD round used the declared63 loss and60 gradient evaluations. The largest
represented pre-server norm excess was about1.06e-8; the server's mandatory outer clip
enforces the reducer input radius. Final intended PGD norms are not separately logged.

## Utility comparators

| Public projection width | No-noise AUROC | Private AUROC |
| --- | ---: | ---: |
| 512,identity | 0.8073 | 0.7295 |
| 256 | 0.7975 | 0.7560 |
| 128 | 0.7926 | 0.7702 |
| 64 | 0.7693 | 0.7465 |

Width128 is the strongest private point in this bounded screen. Its paired private gains
over width512 are about0.0361 and0.0452, while its clean mean falls by0.0147. These paired
values follow the exact seed identities in the source JSON/CSV; the aggregate point is unchanged.
The clean-to-private
gap therefore shrinks for two separate reasons: better private utility and worse clean utility.
Both components must be reported. Width selection used this same development cohort and
needs a fresh-seed comparison; it does not provide untouched-cohort confirmation.

The scalar output-perturbed ERM comparator's private means are0.5345/0.5434/0.5483 at
lambda0.01/0.1/1, with wide two-seed ranges spanning below and above chance. Its clean
means are0.7843/0.7691/0.7557. This conservative bounded-sensitivity construction did not
improve utility here. It does not rule out other private convex algorithms, clipping bounds,
representations or optimizers, and it changes scalar geometry and trusted-coordinator access.

## Conditional bucket stage

The declared recipe-selection rule chooses LR0.1/C1: clean unbucketed median mean0.81983,
versus0.80629 at LR0.03/C0.05. The next46 nonprivate cells use that one shared recipe across
bucket sizes and include low/high owned PGD counts, IID/skew and a distinct oracle Fang stress arm.
The two clean IID unbucketed references are reused from this pinned screen.

## Verification and interpretation

The audit checked all80 record hashes, all1360 iterative round journals, full cohorts,
all140 unchanged source files, actual noise/sensitivity and all owned attack budgets.
All four native/ByzFL anchor comparisons matched exactly. A separate direct saved-head
scorer reproduced all five metrics for all80 cells; AUROC and AP differences were at most
2.23e-16. That scorer shares Torch's linear kernel for two-logit heads, while rank-sum AUROC
and tied-threshold AP were recalculated separately. Scalar heads were checked directly.

Independent Gaussian trade-off calculations verified every private cell at epsilon4;
the largest implied delta was9.999999999854343e-6. This remains a per-mechanism prepared-input
accounting check, not privacy of raw diagnostics, private preprocessing, model selection or
the complete multi-run study.

See [all aggregate arms and contrasts](effusion-diagnostic-screen-2026-10-25.json),
[arm CSV](effusion-diagnostic-screen-2026-10-25-arms.csv),
[contrast CSV](effusion-diagnostic-screen-2026-10-25-contrasts.csv) and
[research diagnostics](effusion-diagnostic-screen-2026-10-25-diagnostics.json).
Private summary SHA-256: `034084f9c1b96c8bd680c1136d37c213bc19adb74ba0d1bcdadff0830001214c`.

The screen is complete; the diagnostic phase remains open until the declared bucket stage
is evaluated and the finalist question, seed budget and inference plan are frozen.
