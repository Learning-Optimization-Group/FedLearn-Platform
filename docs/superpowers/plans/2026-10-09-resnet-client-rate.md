# ResNet client-rate stabilization

Base: 956b31a. Continue on the isolated feature branch with local commits only.

Goal: diagnose the exploding adaptive ResNet updates, select a bounded client-rate correction, and verify it on desktop and vivo.

## Design and limits

Compare .01, .001 (the existing base recipe rate), and .0001 on the same initial checkpoint, two 64-example partitions, two rounds and identical recorded seeds. Use production Python client fitting and aggregation; FedOpt server Adam remains .01 and FedProx mu remains .1. Offline counterfactual contracts and resolver constants are explicitly overridden only within the diagnostic process. No official test data informs selection. Diagnostic training accuracy cannot establish generalization.

If .001 removes the explosion, make only CIFAR_RESNET18 adaptive client rates recipe-aware in the Torch-free shared settings module. Server round settings and execution plans must agree. Keep other recipes at their existing values until independently diagnosed. Stored old contracts remain unchanged; incompatible client/server settings must continue failing closed. NIH adaptive support remains unavailable.

## Work and verification

- [ ] Reproduce the .01 failure and complete all six bounded desktop controls; retain checked provenance.
- [ ] RED/GREEN tests for ResNet server/plan agreement, legacy recipe settings and contract mismatch refusal. Implement the narrow shared resolver and run runtime regression checks.
- [ ] Commit and synchronize verified code to g14; use the existing installed native runtime and 64-image snapshot for a fresh two-round mixed qualification. Preserve phone datasets.
- [ ] Check accepted updates, aggregate reconstruction, independent reload/evaluation and original-runtime replay. Retain evidence and limitations in the guide and public receipt.
- [ ] One fresh whole-slice review, corrections, local commits and clean-state checks. No push or merge.

## Review focus

- Missing model_type in older callers retains .01 rather than selecting an unrelated recipe.
- Case normalization on server settings matches production dispatch.
- FedOpt server optimizer settings remain independent of the client rate.
- Old published contracts are never rewritten; differing round rates refuse before training.
- Phone and desktop runtime results are checked separately; same data does not imply cross-runtime numerical identity.
