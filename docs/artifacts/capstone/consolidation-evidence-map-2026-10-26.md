# Defense claim and evidence map

26 October 2026. This is the current defense map; the older final-report evidence map is historical. Report v5 is the frozen research manuscript. All intervals below have distinct conditional targets.

| ID | Defensible claim | Exact evidence | Boundary to state |
| --- | --- | --- | --- |
| C1 | Projection improves private-clean AUROC by +0.019176 among the tested policies on the fixed selection cohort. | [Finalists](effusion-finalists-2026-10-25.json), primary paired-seed interval [+0.011011,+0.027340]; fixed-head patient interval [+0.013136,+0.025893]. | Twelve training seeds do not create new patients. Neither interval is joint training/population uncertainty; the cohort was repeatedly used for development. |
| C2 | Projection trades away 0.014923 clean no-noise AUROC. | Finalist clean mean512 0.807298 versus projected128 0.792375; [report§4](capstone-final-report-v5-working.md). | No universally best model or causal attribution to noise dimension alone. |
| C3 | Aggregator ranking changes with privacy and attack conditions. | Finalist private-clean/owned-PGD values: mean512 0.740361/0.551304; projected128 0.759536/0.683331; trim16 0.678365/0.605114; center_quarter 0.666497/0.508300. | The permitted-information PGD is a uniform-mean surrogate, not a reducer-specific worst-case adversary. |
| C4 | The engineering path preserves configuration, ownership, completion and failure semantics. | [Worker contract](simulation-lab-worker-contract.md), [API contract](simulation-lab-api-contract.md), [bounded browser receipt](simulation-lab-effusion-browser-verification-2026-10-22.json). | A source-bound research record is not a protected end-to-end DP transcript. The bounded UI is not a multi-host scheduler. |
| C5 | Canonical transport reproduces training on three physical hosts with multiplexed logical clients. | [CPU verification](multihost-verification-2026-10-26.json): 52 cells, 320 rounds, 13,180 client-rounds; max placement parameter difference 2.9802322387695312e-8; historical reference max 3.725290298461914e-9. | Half the cells are Mac-only controls. Three hosts are not 80 independent devices, hospitals or failure domains. CPU only, no DP/attacks in this extension. |
| C6 | Every five-round three-host pair takes longer end to end under the declared protocol. | [CPU measurements](multihost-results-2026-10-26.json): 24 pairs, 1.866–2.931 times Mac-only time. | Includes startup/staging/shutdown. Every 3-client steady-round pair improves, every 24/80-client pair worsens. No isolated causal bottleneck or universal hardware ranking. |
| C7 | The bounded CPU recipe reproduces in clean environments and a tracked-files-only source export. | [Clean verification](clean-reproduction-verification-2026-10-26.json): 3 fresh environments, 164 profile checks; generated three-host 4 cells/20 rounds/60 fits; source-export local 2 cells/10 rounds/30 fits. | Minimal CPU profile, not every platform dependency or independent external-party replication. |
| C8 | The research uses calibrated client-replacement update mechanisms under declared fixed-input assumptions. | [Report§3/§6](capstone-final-report-v5-working.md), [phase4 audit](phase-4-verification-2026-10-25.json). | Trusted coordinator; public research randomness/diagnostics and the cumulative workflow are not qualified as a private release. No production or malicious-host protection claim. |
| C9 | Independent-cohort predictive validation is not available in this submission scope. | [Freeze decision](research-freeze-2026-10-26.md): user-confirmed access failure; no evaluation attempted. | Treat this as an unresolved validity limitation, not an unfavorable external score or proof that the method fails externally. |

## Statements to avoid

- “Best aggregation method” or “better than NVIDIA”: the comparison is among four scoped policies, with no matched FLARE benchmark.
- “Clinically validated,” “hospital federation,” “patient-private pipeline” or “production secure”: none is established.
- “Three hosts accelerate training”: whole-cell timing is worse in every five-round pair.
- “Bitwise cross-platform equality”: measured float32 parameter differences satisfy an engineering tolerance.
- “The tests prove correctness/security”: checks support specified execution and artifact contracts.
- “Every criticism is fixed”: some were corrected; others remain explicit limitations.

## Evidence hierarchy

Use report v5 for the narrative; use the bound public JSON and verification receipts for exact values. Older any-finding, A1–A23 and pilot results provide background only unless clearly labelled. Do not silently combine their budgets, uncertainty targets, privacy claims or endpoints with the finalist or CPU matrices.
