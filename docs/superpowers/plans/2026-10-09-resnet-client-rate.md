# ResNet client-rate stabilization

Base: 956b31a. Continue on the isolated feature branch with local commits only.

Goal: diagnose the exploding adaptive ResNet updates, select a bounded client-rate correction, and verify it on desktop and vivo.

## Design and limits

Compare .01, .001 (the existing base recipe rate), and .0001 on the same initial checkpoint, two 64-example partitions, two rounds and identical recorded seeds. Use production Python client fitting and aggregation; FedOpt server Adam remains .01 and FedProx mu remains .1. Offline counterfactual contracts and resolver constants are explicitly overridden only within the diagnostic process. No official test data informs selection. Diagnostic training accuracy cannot establish generalization.

If .001 removes the explosion, make only CIFAR_RESNET18 adaptive client rates recipe-aware in the Torch-free shared settings module. Server round settings and execution plans must agree. Keep other recipes at their existing values until independently diagnosed. Stored old contracts remain unchanged; incompatible client/server settings must continue failing closed. NIH adaptive support remains unavailable.

## Work and verification

- [x] Reproduce the .01 failure and complete all six bounded desktop controls; retain checked provenance.
- [x] RED/GREEN tests for ResNet server/plan agreement, legacy recipe settings and contract mismatch refusal. Implement the narrow shared resolver and run runtime regression checks.
- [x] Commit and synchronize verified code to g14; use the existing installed native runtime and 64-image snapshot for a fresh two-round mixed qualification. Preserve phone datasets.
- [x] Check accepted updates, aggregate reconstruction, independent reload/evaluation and original-runtime replay. Retain evidence and limitations in the guide and public receipt.
- [x] One fresh whole-slice review, corrections, local commits and clean-state checks. No push or merge.

## Review focus

- Missing model_type in older callers retains .01 rather than selecting an unrelated recipe.
- Case normalization on server settings matches production dispatch.
- FedOpt server optimizer settings remain independent of the client rate.
- Old published contracts are never rewritten; differing round rates refuse before training.
- Phone and desktop runtime results are checked separately; same data does not imply cross-runtime numerical identity.

## Ledger

- RED: new rate tests fail in four ResNet server/plan cases; twelve legacy/history checks pass. GREEN: 223 focused tests pass; full runtime 644 pass, 30 slow deselected.
- Same-seed CPU controls reproduce all six outcomes exactly. The .01 rate yields final aggregate training-diagnostic loss 8249706.25 (Prox) and 286.4884 (Opt); .001 yields .332129 and .647579. The .0001 controls remain bounded. No official test selection.
- Source 4914e16 synchronized to g14. Existing APK/core unchanged. Fresh FedProx run c285565b-a22f-471c-877c-9951e8e54648, client rate .001, mu .1, two 64-example participants; native production completed and original-runtime proof is confirmed below.
- Native proof scope for this configuration-only change: both original phone contributions and both desktop contributions replay on their original runtime, aggregate reconstructed exactly. Existing changed-label/rate/mu proof remains attributable to the earlier .01 runs; new counterfactual native controls are not claimed.

- Fresh production FedProx run completed: phone reported loss .998004/.799230, compute400009/407648ms. Independent official test reload4333/10000=43.33%, mean CE1.77117. All four updates reconstruct the registered checkpoint exactly; original g14 replays exact and changed-label control differs. Both original native rounds are exact; final verification is recorded below. No matched mixed-runtime seed comparison to the old run.
- Diagnostic review found two minor guards: enforce exactly two participants and finite logits/loss including overflow. Corrected with RED/GREEN; six final controls remain numerically exact. Final runtime650 passed,30 slow deselected. All30 phone files unchanged; installed APK hash verified unchanged.

- Final original native replay exited0: both phone rounds max_abs_diff0 and bitwise_equal1. All four original contributions and registered aggregate verified. Public checked receipt retained at research/results/on-device/resnet-fedprox-client-rate-2026-10-10.json. Original APK and transfer hashes match; isolated phone replay fixtures removed. No new native counterfactual control or FedOpt phone claim.

- Final whole-slice review: no remaining correctness findings. Independent focused tests, guarded source/input/result/log hashes, both aggregate reconstructions, all four original-runtime contribution replays, official-test reload, APK/binary/core binding and dataset preservation verified. Source code and curated receipts committed locally; synchronized to g14. No push or merge.
