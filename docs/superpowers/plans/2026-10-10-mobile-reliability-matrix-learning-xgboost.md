# Mobile reliability, qualification matrix, learning and XGBoost

Goal: complete the four requested slices with measured evidence, regressions for
substantive failures and explicit support limits.

Base: fb9c2b7; existing isolated feat/xgboost-mobile-foundations worktree.
Execution: inline, one fresh final review. Keep local commits and clean g14 sync;
no push/main merge. Preserve root user edits, installed datasets, services and
unrelated runs. New g14 files stay under /home/anurag/codebase/FedLearn_Platform.

## Authorized design

The user approved the four proposed slices with “do 1 to 4” and explicitly chose
breast-cancer measurements. Routine implementation and qualification are
authorized. Unsupported cells stay fail-closed until qualified. Model quality
claims remain separate from execution support. Replay each original runtime;
do not claim numerical equality across phone and Python runtimes.

## Task 1: interruption reliability

- [x] Trace training checkpoints, enrollment/rejoin, provider and foreground
  service lifecycle. Establish a green full mobile baseline.
- [x] Create isolated TinyNet three-round fixture runs via normal API. Use actual
  phone UI enrollment/start and verify service and accepted-update records.
  Delay the second participant to hold a known round during interventions.
- [x] Exercise screen off/background and temporary gRPC route loss, restore the
  route and verify accepted round IDs/counts without duplicate submissions.
  A brief screen-off check does not qualify forced Doze or long suspension.
- [x] Exercise Stop/rejoin and server stop on isolated runs; check cancellation,
  service removal, bounded loop exit, submitted-round preservation and server
  process cleanup. Restore forwarding and preserve datasets.
- [x] Reproduce failures with regressions before minimal fixes; run full mobile
  tests/types/lint and native checks for native changes. Retain actual API,
  device/recorder evidence, timestamps, source/APK identities and limitations.

## Task 2: remaining support matrix

- [ ] Enumerate execution-plan/native/mobile recipe/strategy/arm combinations
  against curated live receipts. Distinguish supported gaps from deliberate
  refusals: subset arms, image DeComFL and iOS are not silently enabled.
- [ ] Qualify supported gaps using prepared data and preserved runtime, beginning
  with missing Robust real-model and pneumonia adaptive cases. Bind contracts,
  contributors, original-runtime replays and aggregate/checkpoint hashes.
- [ ] Audit packaged desktop bundle, enrollment and local-data handoff. Qualify
  an actual packaged-client run with its required bundle; fix concrete gaps with
  tests. Python CLI execution does not count as packaged-client evidence.
- [ ] Retain a machine-readable matrix and guide with evidence, refusal or concrete
  missing prerequisite for every cell.

## Task 3: learning diagnostics

- [ ] Bind existing NIH patient/image-disjoint packages and CIFAR train partitions;
  official test data stays outside hyperparameter selection.
- [ ] Diagnose class-wise failure, preprocessing, target mapping and pretrained
  baseline. Run fixed longer ResNet training controls with heldout validation and
  matched initial state/seed, preserving complete trajectories.
- [ ] Run a bounded NIH follow-up justified by the diagnosis. Use validation for
  selection, disclose separate reference evaluation, record budgets/package hashes
  and independently reload final checkpoints.
- [ ] Report improvements or lack of improvement; no promise of useful NIH
  classification or clinical efficacy from these small research samples.

## Task 4: breast-cancer XGBoost

- [ ] Trace four-feature assumptions through project creation, contract, CSV,
  delta validation, aggregation and artifact inference.
- [ ] Add an explicit bounded ordered thirty-feature Wisconsin Diagnostic Breast
  Cancer schema, preserving four-feature Iris behavior and contracts. Bind the
  schema to the initial model, every delta, CSV header and inference width.
- [ ] Prepare deterministic client/train/validation/test splits with source identity,
  feature/label mapping and duplicate-group separation. No client rows on server.
- [ ] Write regressions before implementation for reordered/duplicate/missing
  features, wrong-width model/delta/inference, non-finite values, malformed trees
  and legacy compatibility.
- [ ] Qualify authenticated collaborative Mac/g14 training, original local/aggregate
  replay, external heldout metrics, fixed centralized tree-budget controls,
  registered artifact reload and normal inference. Android tree training remains
  explicitly refused.

## Final verification

- [ ] Appropriate full checks for changed components, proto mirrors, one fresh
  whole-slice review/fixes, curated receipts/docs, local commits and clean g14
  synchronization. Update ignored status context with actual HEADs and next work.

## Ledger

- g14 HEAD fb9c2b7 clean; vivo authorized, battery100% USB-powered. API8084 and
  Metro8088 healthy; existing same-signed APK retained.
- Ruling: use the established worktree and inline execution; cost: upstream
  integration remains separate from qualification.
- Ruling: start short lifecycle checks with TinyNet to control round transitions;
  image-model lifecycle claims require longer compute-duration evidence.
- User selected breast-cancer measurements; ordered30-feature extension authorized.


- Task1: mobile baseline658passed. Actual UI enrollment/start confirmed provider
  run1ae4c570-1e76-4aa1-90d0-04c3be7ae73e, phone partition1, g14partition0.
  Debug warning overlay was dismissed; a transient hierarchy-export failure was
  bypassed using the observed screenshot Start control. No direct-loop substitute.
- Reproduced failure: foreground service active and power Asleep, but no phone
  round2 contribution within60seconds after server reachedround2. Foregrounding
  immediately producedround2. This is failed background qualification, retained.
- Independent probe while backgrounded: native getServerStatus resolves, JS
  setTimeout1000 does not after3seconds; foregrounding releases the timer.
  React Native0.80 JavaTimerManager confirms activity-pause suppression.
- Actual Stop resets provider to notJoined and removes foreground service. Stopped
  only identity-checked isolated project; owned g14 gated participant terminated.
- New pacing-delay regressions failed first: JS8 cases (missing delay API), native4
  cases (missing scheduler class). Native scheduling/cancellation implementation
  passed full mobile666tests/lint/types and Android unit suite. Preflight refusal
  regression is now being added before changing the older-runtime guard.
- Remaining Task1: updated same-signed APK, preservation and fresh full interruption,
  cancellation/rejoin qualification. Tasks2–4 remain pending; no broad completion.

- Fixed background and actual socket-outage tests completed. Network proxy closes
  an active phone connection and rejects reconnects; six-second outage produces
  actual transient errors then recovery. Each completed run has six unique updates.
  ADB reverse removal alone does not kill established sockets; retain that ineffective
  probe without a network-recovery claim. Normal mapping restored and proxy stopped.
- Stop/rejoin test completed: stop69ms, service absent, checkpoint1 retained; same-run
  restart skips already accepted round1, finishes rounds2/3, six unique updates and
  final checkpoint3. Mounted provider methods used for network/rejoin, without replacing
  loop operations; touch automation is not claimed for those cases.
- Terminal telemetry follow-up: two regressions failed on staleTRAINING display after
  actual terminal status/backend-ended confirmation. Hook retires polling/clears old
  status. Ruling: use a termination notification, not invented full RPC counters,
  because RoundOps intentionally exposes only state/round. Cost: Home omits expired
  progress; exact completed state remains in project API and retained receipts.
