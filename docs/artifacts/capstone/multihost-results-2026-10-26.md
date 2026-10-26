# Three-host CPU execution: verified results and limits

26 October 2026. **The three machines train one model correctly, but this workload does not show an overall speedup.** All 52 prospective cells completed without failed or replacement fitting: 320 rounds and 13,180 client fits/acknowledged uploads. This establishes CPU transport and multiplexed-client execution across Mac, G14 and Orin. It is weak evidence for a performance improvement: every five-round placement pair took longer end to end on three hosts.

## Protocol and correctness

Mac coordinated and evaluated selection while all three machines trained. The 48 five-round systems cells crossed 3/6/24/80 logical clients, mean512/projected128, three new paired training seeds and two placements. Four twenty-round 80-client anchors replayed historical seed 13001. Full cohorts, zero heads, SGD 0.03, seven local epochs, batch 256, clipping C=0.05 and uniform reduction order were frozen. Two fits per host permits two total on Mac-only and six total on three hosts; at three clients, only three fits can actually be active.

The prospective source/input/runtime checkpoint is `8ac59c4`; public-manifest SHA-256 is `a02da5d262fc59d054f67bdff6c5ad7ca71dd42f721d141af7cb1acebd9dc935`. Three physical contributors were established using Mac-local execution plus two distinct SSH-launched Linux machine identities. Host labels alone were not treated as hardware attestation.

All 26 placement pairs met the prospective maximum absolute parameter tolerance 1e-5. Observed maximum was 2.9802322387695312e-8. The Mac-only anchors exactly reproduced their stored simulator head values; the largest three-host/historical difference was 3.725290298461914e-9. Independent rescoring reconciled AUROC and loss for all 52 saved heads. Maximum placement AUROC difference was 2.494283103393613e-7. These are numerical execution checks, not statistical equivalence margins or bytewise-equality claims.

The separate audit reconciled raw host JSONL with summaries, exactly one fit and acknowledged upload per client round, the seven-epoch partial-batch budget, declared sample counts, every evaluated aggregate and independent final-head digests. It regenerated the entire public analysis to verify metrics, unique comparisons, timing ratios and acceptance flags. All 68,650 training rows remained represented in every medical cell. No holdout/test data or saved holdout models were used.

## Timing

Ratios are Mac-only time divided by three-host time: above 1 favors three hosts. Values below are medians of three paired training seeds. The printed paired median ratios are not necessarily the ratios of the two separately printed time medians. Systems cells have five rounds; anchors are excluded from this table.

| Method | Logical clients | Mac-only median s | Three-host median s | Median whole-cell ratio | Median steady-round ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean512 | 3 | 13.51 | 32.25 | 0.419 | 1.134 |
| mean512 | 6 | 13.34 | 33.48 | 0.392 | 0.945 |
| mean512 | 24 | 13.34 | 35.84 | 0.370 | 0.864 |
| mean512 | 80 | 13.37 | 35.81 | 0.367 | 0.921 |
| projected128 | 3 | 13.40 | 26.28 | 0.510 | 1.131 |
| projected128 | 6 | 13.36 | 27.50 | 0.478 | 1.011 |
| projected128 | 24 | 13.30 | 28.12 | 0.468 | 0.873 |
| projected128 | 80 | 13.10 | 28.81 | 0.455 | 0.927 |

Whole-cell time includes process startup, per-cell data staging and shutdown. Across the 24 five-round pairs, three-host time was 1.87–2.93 times Mac-only time. This is a protocol-specific result; real sites with stationary data would not necessarily repeat this staging cost.

Steady-round time uses adjacent server monotonic completion times after the first round. Every three-client pair improved steady-round time; every 24/80-client pair worsened it. At six clients, projected128 was mixed and mean512 was slower in all three pairs. The experiment did not isolate whether polling, serialization, transport, host speed or background work caused these differences. Three seeds support descriptive timing, not broad hardware rankings.

![Paired CPU timing ratios. Each marker is the median of three training-seed pairs; whiskers are observed minimum/maximum, not confidence intervals. Faint points show each pair. The dashed line is equal time. Two fits per host add resources in the three-host placement.](figures/multihost-2026-10-26/cpu-placement-speedup.png)

## Memory and communication

Peak client-process RSS across all cells was 626.7 MiB on Mac, 413.6 MiB on G14 and 474.6 MiB on Orin. These are per-process maxima, not total machine memory or simultaneous summed usage; the Mac coordinator is separate. Existing background workloads remained active. Start/end load, available memory and process CPU time are retained in the aggregate export.

For 80-client five-round cells, median training-RPC protobuf traffic was 7061.7/8780.2 KiB for mean512 (Mac-only/three-host), and 2275.4/2676.6 KiB for projected128. Counts include repeated model downloads and actual message metadata. They exclude registration, heartbeats, gRPC framing, TLS and SSH; they do not measure network-interface throughput. Serialization time was not separately isolated.

## What to do next

1. Profile coordinator polling, repeated downloads, worker startup and data staging before spending on larger CPU scaling runs. A separate persistent-worker/cached-data experiment can distinguish startup cost from training throughput. Do not change this frozen benchmark after seeing its results.
2. If the research question is accelerator or deep-model scaling, specify one bounded adapter/backbone workload and qualify actual CUDA/MPS execution separately. These small frozen-feature heads do not establish GPU performance.
3. Treat private distributed training as a separate protocol amendment. Full participation, release count, randomness visibility, clipping and the privacy unit must be qualified together. Moving 80 logical clients onto three machines does not protect against correlated host compromise; changing the privacy cohort to three changes mean sensitivity.

Three physical machines are not 80 independent devices, institutions or failure domains. This adds real execution evidence to the capstone while preserving the original scientific result's simulation/selection boundary. It supplies no DP, attack-resilience, clinical or hospital-deployment guarantee.

## Artifacts and reproduction

- [Prospective protocol](multihost-protocol-2026-10-26.md), [manifest](multihost-public-manifest-2026-10-26.json) and [qualification](multihost-qualification-2026-10-26.json).
- [Aggregate measurements](multihost-results-2026-10-26.json) and [independent verification](multihost-verification-2026-10-26.json).
- [Pre-benchmark review](multihost-review-2026-10-26.md) and [reproduction instructions](multihost-reproduction.md).

Reviewed implementation gates passed 2,403 framework tests (14 deselected, 80.23% coverage), 802 runtime tests (30 deselected), 17 preparation/supervisor tests and the final two real-transport tests. Seven postprocessing tamper/audit regressions passed. Source review found and corrected admission, public-seal, cleanup, runtime, final-head, queue-reporting and export-reconciliation defects. No remaining material source-review blocker was reported; test counts do not certify security or external validity.
