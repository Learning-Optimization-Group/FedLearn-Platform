# Effusion ByzFL workload qualification

25 October 2026. Phase 2 complete under local checkpoint `016677f6ab6b4e4292a4afc3f681ba8f40fbabe4` and the [prospective protocol](effusion-byzfl-qualification-protocol-2026-10-24.md).

All four runs completed20 rounds with all80 clients aggregated in every round. The workload used the pinned train/development-selection feature bundle, zero initialization and the declared fixed recipe. The existing holdout was not accessed.

| Condition | Native mean AUROC | ByzFL mean AUROC | Maximum parameter difference | Native/prior A17 parameter difference |
| --- | ---: | ---: | ---: | ---: |
| No noise | 0.8075528388932567 | 0.8075528388932567 | 0 | 0 |
| GDP epsilon4, delta1e-5 | 0.7035003272499432 | 0.7035003272499432 | 0 | 0 |

All corresponding AUROC differences were also exactly zero. These are single-seed implementation checks, not seed-averaged method comparisons.

The actual private noise multiplier was4.835102780756167, giving coordinate standard deviation0.006043878475945209 for sensitivity2*0.05/80. A separate normal-CDF calculation gave delta9.999999984911015e-6 at epsilon4 over20 Gaussian releases. The engine's RDP-equivalent input remains distinct from that GDP claim.

The separate audit checked all saved record hashes,80 completed round journals, complete cohorts, finite final heads and every native/external/prior head coordinate. See the [verification receipt](phase-2-qualification-verification-2026-10-25.json). SHA-256 of the private aggregate summary is `e0f784c41e8954fb133cc742261a1e0e8a41d393534d8814211526cd8398205e`.

This removes an implementation reconciliation blocker. It does not establish the utility of centered clipping, attack strength, comparative robustness, population validity or privacy of preprocessing and research diagnostics. Phase 3 must qualify those candidates on the permitted development workload before a finalist study is frozen.
