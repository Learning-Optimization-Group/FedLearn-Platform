# Corrected-rate ResNet FedOpt qualification

Base: f95f8a6. Existing isolated feature branch; local commits only, no push or merge.

Goal: qualify the corrected ResNet client rate under FedOpt on vivo and g14.

## Fixed design

Two rounds, two clients, 64 preserved CIFAR training images per client, two batches
of 32. Client Adam .001; recorded server Adam .01, beta1 .9, beta2 .99, tau .001.
These values precede official-test evaluation and match the bounded CPU control.
Existing installed APK/core and prepared snapshots remain unchanged. No production
code changes are planned. Replay original contributions on their own runtime,
reconstruct the registered aggregate, check BN buffers/integer counters, reload
on the complete official test set, verify dataset preservation and retain hashes.

## Work

- [x] Fresh platform run and phone-first enrollment; desktop strategy/contract
  must both be FedOpt. Capture recording offset before any updates.
- [x] Finish production, retain completed state/phone results, reconstruct both
  aggregate parents and registered checkpoint, replay both desktop contributions.
- [x] Replay both phone contributions against the installed original core and
  verify exit0/cleanup; independent official-test evaluation and preservation.
- [x] Public receipt/guide/context updates and one fresh whole-slice review.
- Local evidence commit and clean g14 synchronization: record actual completion
  and resulting HEAD in ignored docs/mobile-status-context.md after this commit.

## Limits

Two rounds establish bounded execution rather than convergence. Original-runtime
bitwise replay does not assert cross-runtime numerical identity or full wire
metadata equality. No new native counterfactual controls are claimed. Earlier
control proof stays attributable to its .01 runs. Compare initial-state hashes and
round seeds explicitly; historical or cross-strategy scores cannot establish
algorithm superiority. Cached qualification timings are not fresh probe timings.

## Ledger

- g14 HEAD f95f8a6 clean; vivo authorized; API and Metro accessible through the
  existing reverse forwarding. Phone prior Prox run completed.
- Ruling: the private launch helper inherited a Prox literal; derive its strategy
  from the requested qualification before staging. Production sources unchanged.
- New FedOpt project/run staging is underway; final run identity and settings
  will be retained from the actual manifest before enrollment.

- Fresh run 3a05d0d6-6856-4afe-ad38-f4714223a10d; project f9a10ff6-3a06-4669-a783-9ab56bbceee8. Manifest and round settings are FedOpt, client .001, recorded server Adam .01/.9/.99/.001. Phone owns partition0, desktop1. Private launch argv independently checked as FedOpt.

- Interrupted first attempt retained separately: run3a05d0d6-6856-4afe-ad38-f4714223a10d,
  two64-example round1 contributions accepted. Phone reported loss .859298,
  compute393578ms; server round1 evaluation36.06%. Full two-round qualification
  did not complete and is not claimed.
- Root cause evidence: macOS power log records host sleep/darkwake while local
  API/gRPC/Metro forwarding stalled. Desktop reports stream ping timeout; phone
  status deadlines and network rejoin failures; result collector times out too.
  After wake, forwarded/local health paths respond. Stopped only this run after
  preserving accepted updates, phone state and server/power logs.
- Ruling: do not resume an expired round2 deadline or present a partial run as
  qualification. Prepare a fresh run with fixed settings and keep host awake.
  Temporary caffeinate guard started; user's lid-open/power-ready reply is pending
  because closed-lid sleep can override the guard. No fresh dependent run started.

- Independent original g14 round1 replay is bitwise exact (64 examples); changed
  label control differs. Latest final artifact returns404 after stopping. Public
  incomplete-run receipt prepared at research/results/on-device/resnet-fedopt-host-sleep-interruption-2026-10-10.json.
- Mac/private and g14 attempt folders renamed to resnet-rate-fedopt-interrupted-sleep;
  fresh resnet_rate_awake_setup.py uses a distinct awake-host project name.
  Existing forwarding works after wake. Runtime sources unchanged; focused54 pass.
- Remaining: receive lid-open/power-ready reply, start fresh fixed-setting run,
  then complete original native/Python replays, aggregate and official-test reload,
  preservation, final review, local evidence commit and g14 sync.

- User confirmed ready/lid open; battery92%, temporary90-minute keep-awake guard
  asserts idle system/display sleep prevention. Fresh run97b09e5d-8492-4793-b7b0-9d372eb2231c
  is active with unchanged .001 client and .01 server Adam settings. Phone-first
  partition0/desktop1 verified; source remains f95f8a6. No rate/data changes.
- Private collector now retries up to five transient CDP read failures and keeps
  its last actual state; it never substitutes a completed outcome. Original
  interrupted attempt remains separate and is excluded from completion evidence.

- Awake attempt round1: both updates accepted; phone reportedloss .783977,
  compute392046ms. Server round1 evaluation34.68%; desktop round2 already
  accepted. Actual intermediate phone callbacks lagged server acceptance; tracked
  progress against the recorder rather than treating stale UI state as failure.


- Completed fresh run 97b09e5d-8492-4793-b7b0-9d372eb2231c, project
  79a757e2-3cc0-48f9-b764-e4a1ae721eb0. Actual API status is COMPLETED.
  Phone losses .783977/.835260, compute 392046/389329 ms; all four accepted
  updates have 64 examples. Both original phone and g14 contributions replay
  bitwise, both aggregate parents and final checkpoint reconstruct exactly,
  float BN variances are finite/nonnegative and integer counters unchanged.
- Independent official test reload: 3383/10000 (33.83%), mean CE 2.265242.
  Exact registered NPZ hash retained. Initial state matches earlier FedOpt and
  corrected Prox; both seeds differ. No algorithm superiority claim.
- All 30 phone dataset hashes equal the pre-run map. Installed APK and transferred
  native replay binary hashes match; replay process and isolated files absent.
  Native wrapper, desktop replay and heldout reload each exited 0; collector
  exit 0. Proof logs and raw hash bindings retained with separate interruption.
- Verification: 54 focused rate/settings/strategy tests passed; all five proto
  mirrors match. Production sources unchanged from f95f8a6. Original replay
  evidence covers arrays only, with no fresh cached-probe timing or new native
  counterfactual controls. The default server rate 1.0 remains unqualified.
- Ruling costs: correcting the private Prox launch literal would cost a wrong
  strategy qualification if mistaken; deriving the requested strategy and checking
  manifest/argv removes that ambiguity. Restarting after host sleep costs another
  fixed two-round run, while reusing the partial attempt would falsely qualify it.
- Final review: no blocking or minor findings. Independently verified all 28 raw
  bindings, 102 float arrays and 20 integer counters, all 10000 official test
  images with the same confusion matrix/mean CE, data/package/native identities,
  recorded completion, and fresh post-run preservation/cleanup. Separate selected
  review checks: 52 tests pass, proto mirrors match, staged diff check clean.
  Original phone/g14 replay logs were inspected, not independently rerun in review.
- The task-specific temporary keep-awake guard expired; its PID is absent.
  Existing services/forwarding remain. Branch/worktree kept for further mobile
  work; no push or merge. Ruling: retain the established local branch workflow
  rather than request integration again; the cost is deferred upstream integration.
- Remaining external bookkeeping: create the local evidence commit, fast-forward
  the clean g14 checkout, and retain actual HEAD/clean checks in the status context.
