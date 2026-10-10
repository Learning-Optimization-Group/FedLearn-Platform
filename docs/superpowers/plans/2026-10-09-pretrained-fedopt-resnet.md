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
- Ruling: expose Adam and Yogi only, because the production aggregator implements those variants.
- NIH preflight: existing tensors are normalized by .5/.5; explicitly undo this before RGB ImageNet normalization. Refusal log retained.

- Verification: backend 1172 tests / 167 suites, no failures; coverage passes. Frontend 138 tests, lint and build pass (explicit local development API origin). Runtime 628 passed, 30 slow deselected; initial sandbox tokenizer-network failure retried successfully.
- NIH fixed ten-epoch frozen baseline completed: balanced validation AUROC .619263, proportional reference .604084 (six positives); checkpoint hash and reload verified. Source ac1a495.
- ResNet: imported 64 phone images through the native picker; importer/Python snapshot IDs match. Two mixed FedOpt rounds running at saved server rate .01; live results pending.

- Final code review: no correctness blockers or important defects. Independent focused runtime 12, UI 32 and framework 18 assertions passed; NIH package/disjointness/report checks passed. Live ResNet evidence still pending.
- Final: minor (deferred): pretrained weight verification uses official eight-character hash prefix; public receipt retains the full actual SHA-256.
- Final: minor (deferred): dashboard test verifies settings removal after success/reset; explicit manual method switching and Yogi submission deserve additional coverage.
