# Pretrained NIH, recorded FedOpt settings, adaptive ResNet

Base: 22dadcf. Work stays on the existing isolated feature branch; no push or merge.

## Scope and decisions

1. Add optional `fedOpt` settings to starts: variant (adam/yogi), serverLearningRate, beta1, beta2, tau. Omitted settings retain the existing Adam/1.0/.9/.99/.001 behavior. Reject ignored settings on other strategies and non-finite/out-of-range values. Persist resolved settings on each run, expose them in its manifest and spawn from that record. The client learning rate remains the existing contract value. Dashboard exposes server rate and variant with accurate help.
2. Run a frozen ImageNet ResNet18 NIH research baseline on the previously verified patient-separated packages: 512 pooled training examples, 256 enriched validation examples and the retained proportional reference. Repeat grayscale into RGB and use recorded ImageNet normalization at 224 pixels. Freeze backbone including BatchNorm state, extract features once, train only a seeded two-class linear head for ten epochs at .001; no validation-based selection. Record weights hash, transform, package hashes, all epochs and checkpoint reload. This benchmark does not create an Android NIH pretrained contract or establish clinical usefulness.
3. Qualify adaptive ResNet using two rounds on vivo and g14. FedOpt uses an explicitly preselected .01 server rate. FedProx needs two or more local batches so its proximal term can affect updates. Preserve existing phone datasets and retain exact original-native/Python contribution replay, aggregate replay, buffer handling, controls and independent final evaluation.

## Verification and completion

- RED/GREEN tests for settings validation, persisted defaults/custom settings, manifest and argv, real Python strategy selection, dashboard submission and field removal after switching strategy.
- Frozen-backbone tests assert evaluation mode, unchanged state, transform and train-only head updates; source binding must continue refusing leakage.
- Run appropriate backend, frontend and runtime suites; live mixed runs must complete with all original contribution/aggregate replays matching.
- Retain public receipts with honest prevalence and efficacy limits; source and private artifacts remain attributable to actual runs.
- One fresh final review for the entire slice; correct significant findings and commit locally. Sync only verified commits to g14.

## Ledger

- Hardware: g14 reachable over Tailscale; vivo authorized. Restored reverse tunnel; Metro and API healthy.
- FedOpt and frozen NIH baseline implementation: focused tests pass. Backend/full frontend/runtime verification in progress.
- Ruling: expose Adam and Yogi only, because the production aggregator implements those variants; another variant requires new implementation and tests.
- NIH preflight: existing tensors are normalized by .5/.5; explicitly undo this before RGB ImageNet normalization. Refusal log retained.

- Verification: backend 1172 tests / 167 suites, no failures; coverage passes. Frontend 138 tests, lint and build pass (explicit local development API origin). Runtime 628 passed, 30 slow deselected; initial sandbox tokenizer-network failure retried successfully.
- NIH fixed ten-epoch frozen baseline completed: balanced validation AUROC .619263, proportional reference .604084 (six positives); checkpoint hash and reload verified. Source ac1a495.
- ResNet: imported 64 phone images through the native picker; importer/Python snapshot IDs match. Two mixed FedOpt rounds running at saved server rate .01; live results pending.

- Final code review: no correctness blockers or important defects. Independent focused runtime 12, UI 32 and framework 18 assertions passed; NIH package/disjointness/report checks passed. Live ResNet evidence still pending.
- Final: minor (deferred): pretrained weight verification uses official eight-character hash prefix; public receipt retains the full actual SHA-256.
- Final: minor (deferred): dashboard test verifies settings removal after success/reset; explicit manual method switching and Yogi submission deserve additional coverage.

- ResNet/FedOpt production completed: run 528b2460-6501-4401-b240-dd7f650656b1, four accepted 64-example updates, registered aggregate exact and both g14 contributions bitwise exact. Phone compute 389060/385256 ms; losses 32108.27 / 48104680.875. Independent official CIFAR evaluation 1000/10000 (10%). Reduced server rate .01 does not stabilize current client rate .01; no useful-learning claim. Original native replay running.
- Stopped unused staged FedProx run f4562651-c9f2-4d8f-ac2b-24eb805e8c8b before it could expire; fresh sequential run 07827dcb-aa1d-48d5-9b3e-fe28bd5386d2 is training. Its original four-update recording window starts after FedOpt capture.
- Phone preservation: all 27 previous dataset hashes match; exactly three new files for the 64-image native-imported snapshot.

- Both sequential production ResNet runs completed. FedProx run 07827dcb-aa1d-48d5-9b3e-fe28bd5386d2: all four updates aggregate exactly; both original g14 contributions bitwise exact; official test reload 1000/10000 (10%). All four original desktop contributions now proved.
- Both original native round-one contributions match bitwise; changed-label controls differ. Remaining original native round-two/rate/zero-mu checks still running. Isolated native replays use immutable per-strategy fixtures; the independently derived Prox round-two parent matched full aggregation and its original final reference was added atomically.
- CPU scheduling: isolated replay processes migrated to dedicated performance cores 5 and 6 after initial comparisons, without runtime/input changes or timing-comparison claims. Private affinity receipt retained.
- Tailscale fresh connection check required renewed authentication; user completed it and access was verified. Existing training tunnel remained active throughout.

- Original vivo FedOpt runtime verification completed with exit 0: both round states bitwise exact, label/rate controls differ, finite arrays. Complete log retained locally. Prox zero-mu/final round checks remain.

- Completion: original vivo Prox replay exited 0; both round states bitwise exact and changed labels/rates/zero-mu controls differ. All eight original contributions and both registered aggregates now proved; public checked receipt retained at research/results/on-device/resnet-adaptive-2026-10-09.json. Same initial model, different round seeds. All 27 previous phone files unchanged plus three expected imported files.
- All three tasks complete. Code review had no correctness blockers; both deferred minors remain recorded. Curated reports and guide updates checked against retained artifacts. No push or merge.
