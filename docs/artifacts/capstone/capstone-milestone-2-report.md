# Milestone 2 report: reproducible privacy and robustness simulation

Status: retrospective account of the 6 October 2026 exploratory experiments. Written 19 October 2026. This is not a pre-registration or a final NIH result. The project title still awaits advisor confirmation.

## Purpose and contribution

Milestone 2 moved the earlier matched-study harness into a configurable simulation engine. The engine runs through the platform's FL coordinator with composed clipping, aggregation and privacy stages, reproducible behavior schedules, and structured per-round records. Its purpose is to make controlled comparisons reproducible and eventually useful for selecting FedLearn settings.

The contribution at this milestone is an executable experimental workflow and bounded evidence about interactions between privacy, clipping, aggregation and attacks. It does not establish a generally best algorithm or clinical performance.

## Workload and experiment design

The workload was balanced Kermany chest X-ray classification using a frozen ImageNet ResNet-18 representation and a 1,026-parameter linear head. The prepared workload contained 2,682 training images and 468 test images. The main grids used 80 IID clients, eight rounds, five local epochs, learning rate 0.5, common server clipping norm 0.4, and trimming ratio 0.2. Image-row allocation was used; these clients are simulation shards, not verified hospitals, and this workload does not establish patient-separated data ownership.

Three exploratory grids completed:

| Grid | Configurations and repetitions | Completed records |
| --- | --- | ---: |
| Attack suite | Six conditions, seven aggregation rules, ten seeds | 420 |
| Local versus central DP | Two modes, six epsilon values, two rules, clean/IPM, ten seeds | 480 |
| Scale and cost | Three client counts, five rules, three seeds, three rounds | 45 |

All 945 records have status `ok`. Attack-suite conditions were clean, sign flip, IPM, ALIE, label flip and free rider, with 20% bad clients when attacked. Default strengths were sign-flip scale −10, IPM scale 4 and ALIE z = 20. That ALIE value is a configured stress setting, not a validated paper-calibrated choice for this cohort.

Accuracy means final-round test accuracy. Historical confidence intervals use Student-t intervals across seeds. There was no prospective multiplicity plan. These tables describe exploratory patterns; they do not support a collection of confirmatory significance claims.

## Findings

### Robust aggregation had a clean-data cost in this clipped regime

Final accuracy, percent, averaged over ten seeds:

| Aggregation | Clean | IPM, 20% |
| --- | ---: | ---: |
| Mean | 79.7 | 79.3 |
| Trimmed mean | 79.2 | 80.8 |
| Median | 76.9 | 80.9 |
| Krum | 74.7 | 77.2 |
| Multi-Krum | 78.1 | 77.7 |
| Bulyan | 76.7 | 78.4 |
| Centered clipping | 79.7 | 79.3 |

No tested attack condition reduced a rule's mean accuracy by more than 1.4 percentage points under the common clipping rule. Honest upload norms were reported around 8–10, far above the 0.4 clip. Clipping therefore strongly changed both honest and malicious updates. These results cannot establish that an attack is intrinsically weak or that robust aggregation is unnecessary under other clipping or heterogeneity regimes.

The median improved from 76.9% clean to 80.9% under IPM. This is an unexplained anomaly. A smaller effective update at this learning rate is a hypothesis, not a demonstrated mechanism; do not recommend adding attackers to improve learning.

### Local and central privacy paths behaved differently

Mean aggregation accuracy, percent, ten seeds per cell:

| Epsilon | Central, clean | Local, clean | Central, IPM | Local, IPM |
| --- | ---: | ---: | ---: | ---: |
| 1 | 71.9 | 53.9 | 64.1 | 50.0 |
| 2 | 76.6 | 56.9 | 71.5 | 50.0 |
| 4 | 77.2 | 63.3 | 75.6 | 50.0 |
| 8 | 78.7 | 71.4 | 77.8 | 47.1 |
| 16 | 78.8 | 76.3 | 77.8 | 47.0 |
| 32 | 79.1 | 78.4 | 78.8 | 48.9 |

Local IPM runs remained near chance throughout this epsilon range, including for trimmed mean. In this implementation honest local uploads contain noise while worst-case attackers need not add it. The local-mode server bound is enlarged to accommodate noisy honest uploads. Consequently the local/central contrast changes the noisy-upload geometry and clipping policy as well as where privacy noise is applied. Its magnitude depends on those choices.

A utility comparison alone does not choose the privacy architecture: local and central mechanisms have different trust assumptions. A production recommendation must state who is trusted, the protected unit, adjacency, release mechanism and accounting assumptions. Neither these experiments nor patient-disjoint NIH preparation provides a patient-level privacy claim.

### Cost and scale require separate interpretation

The linear head's serialized upload was 4,250 bytes per client per round. The historical training-compute estimate was 0.69 MFLOP per client. These figures exclude full image feature extraction and real device/network overhead.

At 1,000 clients, median aggregation times were approximately 0.58 ms for mean, 2.7 ms for centered clipping, 15 ms for median, 29 ms for trimmed mean and 44 ms for Krum. The grids ran concurrently on one machine; absolute times include contention. Trimmed-mean timing includes an exclusion-rate diagnostic. These are exploratory implementation costs, not deployment latency guarantees.

Krum reached 50% accuracy on all three 1,000-client scale repetitions. Each client held only about three images, and Krum selects one update. Increasing client count therefore changed per-client data as well as aggregation scale. This is not evidence that every 1,000-client Krum deployment fails.

## Reproducibility and evidence

The original engine commit is `8a9b89dc62b7ccb8e57f4816416411c8af1f43d9`. The historical engine-to-harness check reproduced all 160 stored cells of one matched configuration family on per-round accuracy, initial upload digest and final model digest under torch 2.9.1. It did not cover every engine setting or establish cross-platform equality.

On 19 October, after integration with the mobile/XGBoost main baseline, a bounded eight-cell check of repetition zero reproduced those fields under torch 2.12.0. That is a new limited check, not a rerun of the complete historical verification.

Evidence inventory:

- `research/notes/capstone/2026-10-06-m2-exploratory-engine-experiments.md`: historical methods, findings, anomalies and caveats.
- `research/results/capstone/2026-10-06-m2-summary.json`: tabulated source values.
- `research/results/capstone/2026-10-06-m2_attack_suite/`, `2026-10-06-m2_local_vs_central_dp/`, `2026-10-06-m2_scale_cost/`: per-run records and indexes.
- `research/benchmarks/configs/m2_*.json`: exact experiment settings.
- `research/notes/capstone/2026-10-06-engine-reproduces-capstone-records.md`: historical reproduction scope.
- `docs/artifacts/capstone/milestone2/Capstone-Milestone-2.html` in the retained presentation assets: the 7 October presentation. The older PowerPoint misses those edits.

The research tree is ignored by git; a private off-laptop evidence backup exists. A repository checkout alone is insufficient to reproduce these tables without the retained evidence and feature cache. Summary SHA-256: `3108b58f8db687aaae8ff1633ac22931d055ce472051ea7d60a4b0b52b6826c7`.

## Practical decisions to test in phase 2

1. Compare each robust rule against the same clipped mean under both clean and attacked conditions. Report its clean-data cost alongside protection; do not rank it only by attacked accuracy.
2. Keep clipping, privacy-noise calibration and attacker behavior explicit. Inspect honest-update norms and clipping rates so a weak effective learning rate is not mislabeled as privacy or robustness failure.
3. Compare head sizes on the same frozen representation. Changing backbone and trainable dimension together cannot isolate a head-size effect.
4. Report cohort size with actual per-client patient/image counts, communication and server cost. Distinguish a scale execution check from useful learning.
5. Retain NIH's official-test membership and prevalence, tune only on validation, and report threshold behavior as well as discrimination. The 19 October metadata audit found 41.53% findings in training, 42.05% in validation and 61.47% in official test; this was known before the phase-2 protocol and must be disclosed.

These are evidence-informed study requirements, not validated deployment defaults. The next gates are complete NIH image integrity, frozen features, patient-preserving engine integration, a learning pilot and a prospectively frozen bounded protocol. The current schedule and completion criteria live in [the capstone plan](../../../wikis/capstone/README.md).
