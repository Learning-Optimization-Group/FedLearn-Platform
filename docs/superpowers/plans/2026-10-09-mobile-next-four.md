# Mobile next four implementation plan

Goal: deliver all four authorized slices with evidence.
Spec: ../specs/2026-10-09-mobile-next-four.md.
Execution: inline; one fresh whole-branch correctness review at the end.

- [x] ResNet: prepare/import new32-image112px snapshots with disjoint contents;
  create project and retain initial checkpoint; two-round mixed run; exact
  original-runtime/BatchNorm and aggregate replay; record evidence and support
  flag. Existing qualification helpers in .temp/live-mobile provide the pattern.
- [x] NIH: add research/benchmarks/nih_learning.py and runtime tests for safe
  snapshot/provenance binding, finite binary metrics and reproducible training.
  Prepare128+128 train/512 validation samples via verified NIH manifests; run
  centralized three-epoch1e-3/1e-4 baselines on g14; compare actual three-round
  platform FedAvg using the same pooled training data. Record every result.
- [x] Strategies: test Python rate overrides, native multi-step Adam proximal
  parity/mutations, closed matrices and old-runtime refusal before code changes.
  Edit client.py, execution_plan.py, TrainableExecutorchModel and FederatedLoop,
  runtime capability bridge/gate, three readers/corpus and NIH guards/UI if
  enabled. Run focused/full suites, rebuild same-signed APK preserving snapshots,
  qualify mixed multi-batch ECG under both methods and replay original runtimes.
- [x] XGBoost: add framework/benchmarks/xgboost_iris preparation/evaluation and
  meaningful split/schema/replay tests. Default binary Iris; real platform CPU
  run with Mac/g14, external heldout and centralized comparison; record exact
  receipts, inference mapping, provenance and limits.
- [ ] Final review/fixes, appropriate full checks, clean local commits and
  g14 fast-forward, update guides and root mobile-status-context. No merge/push.

Review focus: leakage/duplicate rows; integer BatchNorm buffers; optimizer
anchor/rate and reset per round; old-native refusal; data/label mapping; unclear
statistical claims. Every substantive failure gets a regression before its fix.

## Rulings and progress

Ruling: preserve existing worktree rather than create another branch; it is clean
at cc9a6c9 and contains the qualified artifacts/tooling. User authorized all four
steps; continue without further permission gates for routine implementation.
Ruling: recorded small samples qualify execution; larger NIH samples diagnose
learning. Neither establishes clinical performance or full dataset convergence.


Evidence: NIH pooled 256-row controls and three federated rounds completed;
all six CPU contributions and registered aggregate replay bitwise. Validation
missed all six positives; lower-rate/reset controls did not establish a useful
classifier. Iris real platform four-round CPU run, eight trees, original local
and aggregate replay, inference and two pooled controls completed; 18/19 test
correct remains a small execution benchmark.

Evidence: ECG FedProx and corrected FedOpt completed two rounds on vivo/g14,
300 rows/client, three batches/round. Exact original native/Python and aggregate
replays passed, as did phone changed-label/rate/zero-mu controls. Same-signed
APK preserved 27 dataset files. Full runtime 593 passed; backend 1167 passed
and coverage gate passed; framework 1359 passed; mobile 658 passed; native
CTest passed. Final catalog checks/proto mirrors passed.

Ruling: NIH remains FedAvg/Robust only; extending its backend/UI strategy guards
is optional to this slice and lacks image live evidence — cost: NIH adaptive
strategies require another qualification.
Ruling: binary Iris is the real-data default; no clinical feature schema was
provided — cost: this proves collaborative tree execution, not clinical value.
Ruling: qualify default FedOpt settings as execution only; its server rate 1.0
causes severe loss growth. Preserve the failure and rerun after fixing Python's
finite-loss batch skips — cost: predictive benefit remains unproven.
Ruling: FedProx was staged before FedOpt completed; remove the earlier FedOpt
phone contribution from its window by the previously verified exact blob hash,
then require four unique cohort/round contributions and exact registry aggregate.
Cost if misbound: the aggregate/replay assertions fail rather than qualify it.

Commits: eb8b7f2 native Adam proximal/rates/matrices; d312334 asynchronous
acknowledgment and race/failure regression fixes; 337a3ae selected-optimizer
rate cross-check; 04f4674 finite contracted batches. Existing tracked ECG source
was restored from its exact pre-qualification backup after all live runs.

ResNet evidence: all four original runtime contributions replay bitwise,
BatchNorm float buffers included and integer counters preserved; original
wrong-label controls differ. Repeat after async upload fix completed and its
four-update aggregate reconstructs the registry artifact exactly. Independent
reload on 10,000 official test images gives 3485 correct (34.85%). The repeat's
full phone telemetry and individual replay are not claimed. Catalog cleared
only after actual original native proof. Curated receipts retain source/APK
and artifact/contract identities.
