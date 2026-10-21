# Simulation Lab implementation plan

**Execution:** inline under the continuous capstone mandate, after fixed-study
capture, reconciliation and report. No implementer delegation. One fresh read-only
review per independently rejectable code slice; fixes have reproduced regressions.

**Goal:** Run, stop, reopen, compare and export real bounded NIH simulations from the
existing authenticated FedLearn dashboard, with honest data/privacy/cost semantics.

**Architecture:** A narrow Python worker owns resolver/core execution and durable
round evidence. A Spring simulation service persists job ownership/state and owns
one bounded subprocess. The React Lab consumes typed APIs and actual completed
rounds; retained fixed-study results are a separate read-only entry.

**Tech stack:** Python3.12/PyTorch/NumPy, Java21/Spring/Flyway/PostgreSQL, Node24/
React/Vite/Recharts, existing cookie auth/Ember components, homelab Playwright suite.

**Spec:** `docs/superpowers/specs/2026-10-20-simulation-lab-design.md`.

## Global constraints

- Finish the immutable fixed-study capture/audit/report first. No tracked source
  edits or runtime/regression/browser checks alongside its measured execution.
- Dedicated cached-feature NIH simulation, validation only for new runs; natural
  prevalence, whole-patient IID and full accepted cohort; mean/trimmed mean only.
- Choices/defaults exactly as the spec table; seed 0–1,000,000,000, clients10/80/1000,
  rounds1–20(default3), epochs1–7(default1), LR0.003,batch 256,C=0.05/0.4.
- Heads linear/MLP8/128/512; none/centralε1/4/8,δ1e-5,replace-one; clean/own-update
  sign-flip−4/own-data label-flip with malicious fraction0.2; trim ratio0.2.
- One active process, no sweep queue,180s whole-process deadline, CPU4 threads;
 2GiB engine-completion high-water RSS qualification observation, not enforced cap.
- Owner/admin authorization plus per-user job ownership; cookies/CSRF as existing;
  no commands, paths, patient rows, checkpoints or predictions supplied/exposed by UI.
- Research-record export is explicitly not DP-safe. No automatic robustness theorem,
  patient privacy, clinical usefulness, best-round winner or physical-cost claims.
- Reuse engine/core numerics/callbacks, safe argument-list launch and ProcessTrees;
  do not alter old protocols, metrics, raw records or physical FL project contracts.
- No publication/messages/push/merge; local tested commits and private backup only.

## Review focus

1. API failure, logout/unmount and late poll/stop must never hand ownership to a new
   user or orphan/restart an existing training process; Task2/3 tests own these cases.
2. Missing/corrupted/wrong cache and unsupported JSON must refuse before scoring,
   and no GUI choice may evaluate official test or expose server paths; Task1/2.
3. Forced death or stop in/after the final round must preserve actual completed
   work and distinguish full qualification from partial/stopped/error; Task1/2.
4. Two concurrent starts and backend restart/PID reuse must not create another
   worker or kill a foreign process; Task2 real process/race tests.
5. Comparison/export/replay must preserve provenance, show recipe differences and
   privacy/cost scope, and not fabricate live data or successful outcomes; Task3/4.

### Task 1: Bounded Python worker and evidence contract

**Files:** create `framework/src/fedlearn/lab_worker.py`,
`framework/src/fedlearn/simulation/lab_contract.py`,
`framework/tests/test_simulation_lab_worker.py`,
`framework/tests/test_simulation_lab_contract.py`,
`framework/tests/test_simulation_lab_release_boundaries.py`.

**Consumes:** existing `prepare_core_config`, `run_core(on_round,should_stop)`,
cache loader and existing finite/tagged encoding contract; no numerical change.
Keep the production worker independent of the uninstalled benchmark package.
**Produces:**
- `resolve_choices(choices: dict, *, cache_path: Path, report_sha256: str) -> dict`
  returns `config`, `config_hash`, `display_recipe`; strict whitelisted input table.
- `run_worker(config_path: Path, output: Path, *, report_sha256: str,
  run_id: str, parent_stream, deadline_seconds: float=180) -> int`; validates fresh owned output,
  setsCPU4, invokes the core with explicit cohort/validation split, persists evidence.
- Choice JSON keys: `workloadId`(literal`nih-qualified`), `head`(linear/mlp8/mlp128/
  mlp512), `numClients`, `rounds`, `localEpochs`, `aggregation`, `clipNorm`, `privacy`,
  `epsilon`, `behavior`(clean/sign_flip/label_flip), `seed`; all optional except the
  enclosing object, with spec defaults and epsilon 4 even when noDP. Reject any other
  key. The resolved config name is fixed`simulation-lab`; runUUID is separate so
  repeated identical choices have the same config hash.
- CLI `python -m fedlearn.lab_worker` resolve/run modes use trusted server-configured
  paths, never browser commands. The lightweight entrypoint starts monitors before
  importing the simulation package/Torch; historical package imports remain unchanged.
  Run requires `--run-id`canonicalUUID and a fresh`execution/`directory beneath the
  API-owned UUID job directory. Input config remains beside it, outside its fresh
  output. Fixed output files: `progress.json`,`record.json`,`execution.json`, plus
  bounded diagnostics; no browser-provided path is read from a progress snapshot.
- Version1 progress snapshots: `version`, `run_id`, `config_hash`, `state`,
  `completed_rounds`, `requested_rounds`, `rounds` (actual completed events only),
  `record_encoding`, `record_file`, `error_code`, `error_message`. Relative filenames
  are fixed within the owned directory; no browser file access from these values.
- Library-owned `write_research_record(path: Path, record: dict) -> str` in
  `lab_contract.py` preserves the existing `json`/`nonfinite_tagged_v1` byte/schema
  contract. Pin historical finite/nonfinite fixtures in tests; do not change old
  benchmark helper source or import it from an installed production worker.
- Record encodings `json` and `nonfinite_tagged_v1`; terminal completed only after
  complete core qualification. Exit0completed,2failed,3stopped,124deadline.

- [x] Write failing contract tests: defaults/each allowed choice, reject unknown keys,
  booleans-as-integers, NaN/Infinity/ranges, test split/path/matrix/oracle/localDP
  injection; same choices resolve same hash and all mandated core fields match.
- [x] Run focused tests with `PYTHONPATH=src python -m pytest -q --no-cov
  tests/test_simulation_lab_contract.py`; Expected: absent API RED.
- [x] Implement the typed resolver/display projection and tests GREEN. Server workload
  path/pin never come from submitted choices. Do not hash a display-only substitute.
- [x] Write real subprocess worker regressions with explicit small synthetic cache:
  real2 round run/progress/marker; preflight rejection before engine; cancellation
  after actualround1; postflight failure retains record; parent pipe EOF and deadline;
  failure beforefirst/afterlast callback; abrupt kill retains start/completed events;
  nonfinite tagged evidence and no successful marker for stopped/timeout paths.
- [x] Run worker tests to observe RED, then implement atomic evidence, core callbacks,
  independent deadline/parent detection, bounded diagnostics and precise terminal
  classification. Reuse exact integer/finite/config/cache/core checks.
- [x] Verify focused GREEN, one read-only code review and RED→GREEN substantive fixes;
  full framework tests/coverage with thread exceptions as errors. Expected: allpass,
  coverage≥existing73%gate and no actual NIH classifier claim from synthetic tests.
- [x] Commit `feat(simulation): add bounded dashboard worker contract`.

### Task 2: Owned persistent Spring simulation jobs

**Files:** create `.../simulation/SimulationRun.java`, `SimulationStatus.java`,
`SimulationRunRepository.java`, `SimulationDtos.java`, `SimulationProcessRunner.java`,
`SimulationService.java`, `SimulationController.java`, `SimulationProperties.java`,
`SimulationStartupReconciler.java` under
`backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/`;
create `backend/fl-platform-api/src/main/resources/db/migration/V33__simulation_runs.sql`
(check latest migration before execution; advance number if another authorized change
usedV33); corresponding tests under `src/test/.../simulation/`; document server config.

**Consumes:** Task1 worker schema/CLI and resolver identity; existing authenticated
user IDs(Long), role security and ProcessTrees. Existing Spring/DataSource is reused.
**Produces:** endpoints/types consumed by Task3:
- `GET /api/simulations/workloads` returns `workloads:[{id, ready, reason, label,
  representation, reportSha256,splitSummary, supportedChoices, limits}]`.
- `POST /resolve` request `{"choices":{...}}` with exactlythat top-level key; response `{workloadId,configHash,recipe,limits}`.
- `POST /runs` uses the same strict choices wrapper;202 `{id,status,configHash,recipe,createdAt}`;
  conflict409 with generic busy code, owned activeID only when caller can see it.
- `GET /runs?page=0&size=20` bounded≤100, own rows unless explicit admin scope;
  `{content,totalElements,page,size}` with concise run summaries.
- `GET /runs/{uuid}` `{id, status, configHash, recipe, createdAt, startedAt, endedAt,
  completedRounds, requestedRounds, rounds, finalSummary, qualification, recordAvailable,
  recordEncoding, errorCode, errorMessage}`. Missing fields are null, not fabricated0.
- `POST /runs/{uuid}/stop` returns same detail; idempotent, terminal states immutable.
- `GET /runs/{uuid}/record` attachment diagnostic JSON, otherwise409notready;
  foreign/unknown404, anonymous401, ordinaryUSER403 for all endpoints.
- Status vocabulary ACCEPTED, STARTING, RUNNING, STOPPING, COMPLETED, STOPPED, FAILED,
  TIMED_OUT, INTERRUPTED; every DTO includes explicit research/simulated-cost scope.

- [ ] Write RED controller/schema/migration/repository tests for every role/ownership
  operation, safe filename/attachment, pagination and persisted metadata; fields named
  as above, FKcreated_by toexisting users, no overload of physicalFL runs.
- [ ] Implement migration/entity/DTO/properties and controller ownership boundaries.
  Configured interpreter/source/cache/pin/storage are server-only; missing/unqualified
  data returns unavailable metadata and refuses launch before scoring.
- [ ] Write RED service/process integration cases: concurrentstarts admitexactlyone;
  failedlaunch persistsFAILED; realchild completes2 rounds; stopduringround drains;
  deadline/shutdown/parentdeath/restart; PIDreuse/foreignprocess neverkilled;
  malformed/foreign/repeated/backwards snapshot rejected; terminalstop raceimmutable.
- [ ] Implement `SimulationService.resolve/start/list/get/stop/download`, process
  admission/persistence/watchdog/output bounds, monotonic progress projection and
  exact PID/start-identity reconciliation. Drain children before freeing the slot.
  On unresolved oldownership, refuse another launch and retain an explicit error.
- [ ] Verify no authsecret passed to child and no browser paths/commands accepted;
  every exported record binds ownedjob/config and correct complete/partial encoding.
- [ ] Focused backend tests RED→GREEN, one read-only review/fixes, then
  `SPRING_PROFILES_ACTIVE=test ./gradlew test jacocoTestCoverageVerification` with
  Docker/Testcontainers. Expected: tests/coverage pass and realworker lifecycle proven.
- [ ] Commit `feat(simulation): manage owned dashboard simulation runs`.

### Task 3: Existing dashboard workflow and comparisons

21 October: source implementation and 179-test/lint/build gates pass. The single
read-only review and regression-first Important fixes are finished. Actual authenticated
browser/worker, keyboard/focus and responsive acceptance now also pass. The browser
checks use real jobs and retained records, separately from component fixtures.

**Files:** create `frontend/src/components/redesign/SimulationLab.tsx`,
`SimulationLab.test.tsx`, `SimulationRunView.tsx`, `SimulationCompare.tsx`,
`frontend/src/services/simulationService.ts` and contract tests;
modify `frontend/src/App.tsx`, `.../redesign/Sidebar.tsx` and route/nav regressions.

**Consumes:** Task2 exact DTOs/endpoints viaexisting axiosConfig(cookieauth), Task1
choice limits/display semantics; existing tokens/Card/Button/Recharts components.
**Produces:** owner/admin `/simulations`, selected`?run=<uuid>` restored onrefresh,
form→resolve→start→actualprogress→terminal/reopen/export and 2–4run comparison.

- [x] Write RED route/nav/service cases: owner/admin allowed, USER gated, requests
  conform toDTOs and existing CSRF/cookie handling, no bearer-token substitute.
- [x] Write RED meaningful component cases: missingworkload/busy/errors, exactbounds,
  actualcompletedround count/noelapsedfakeprogress, stopremainsSTOPPINGuntilverified,
  completedvstimedoutvstopped, restoreselectedrun and noduplicatePOSTonrefresh,
  abortignore lateoldselection/session polls, foreignrecord refusal, exportfilename.
- [x] Implement form/readiness/resolve/preset/recent-run UI and bounded polling with
  cleanup onroute/auth changes. Unmount/logout never cancelsserverownedwork.
  Defaultshortdemo is exploratory; E7/R20preset is explicit and still bounded.
- [x] Write RED comparisons: onlysameworkloadpin/split,2–4terminalruns, final-round
  endpoints, explicitdiffs/partialstate, no primaryCI/winner/bestround; diagnostic
  download labelledresearchdata, costchart distinguishes modeledbytes fromelapsed.
- [x] Implement comparison/result panes and a read-only link to independently
  verified fixed-study report/curated aggregates; never expose private IDs/rawmodels.
- [x] Verify keyboard labels/focus, busy states, responsive layout using existing
  Ember design language. One read-only review/fixes; run `npm run lint`,
  `npm run test:coverage`, `npm run build`. Expected: allpass, no realdata inference
  implied by component fixtures. Commit `feat(frontend): add Simulation Lab workflow`.

### Task 4: Actual browser evidence and final demonstration

**Files:** create `~/code/homelab/fedlearn/tests/simulation-lab.spec.ts`; update its
README for configured NIH source/pin; curated screenshots/demo instructions under
`docs/artifacts/capstone/`, integration instructions in current capstone plan/TODO.

**Consumes:** actual Task1–3 stack and prepared pretrained NIH cache, retained
independently checked study aggregates. No intercepted canned API/training results.
**Produces:** reproducible realbrowser run/reopen/export/stop/ownership evidence,
source-bound diagnostic record and labelled replay fallback; final demo script.

- [x] Read current homelab instructions. `~/code/homelab/AGENTS.md` was absent during
  this plan's preparation; search once for its replacement and follow the existing
  fedlearnREADME/config if stillabsent. Do not invent missing-file instructions.
- [x] Add owner-authenticated tests tagged`@core @simulation-lab`, plus a second
  user's denied read/stop/export. Target the capstone worktree usingFEDLEARN_REPO;
  verify runningbackend/frontend source first, since reuseExistingServer may attach
  an unrelated rootcheckout. Never reuse foreign liveapp state or credentials.
- [x] Drive realUI start→atleastoneactualround→completed→refresh/reopen→download;
  compare two retained differentsettings, verify displayed actualconfig/metric values.
  Stopanotheractualrun and assert noownedchild survives; verify unavailableworkload
  and visiblylabelledreplay, without rewriting frozenstudy/caches or publicdata.
- [x] Run `FEDLEARN_REPO=<capstone-worktree> npx playwright test
  --config playwright.simulation.config.ts tests/simulation-lab.spec.ts --project=core --grep @simulation-lab` inhomelab/fedlearn.
  Expected: pass onactualstack with matchingrecord/source; no trainingretries to hide
  failures. Use precise failures tofix relevant code with regression/fullgates.
- [x] Render/inspect curatedscreenshots and reconcile all shownvalues withretained
  records; keepauthcookies/rawtraces/privatepaths outsidecommitted evidence.
- [ ] Write a short reproducible demo/replayfallback guide; update finalreport,
  presentation and TODO fromactualevidence. Verify privateOrin backupbeforecleanup.
- [ ] Commit eachrepo's ownedchanges locally, preservinguseredits; no push/merge.

21 October: enabled and unavailable phases passed; responsive recheck reused all
three retained job IDs without retraining. Source/record identities reconciled.
Live/replay guide and aggregate receipt written. The working report Markdown/PDF
now include the verified checkpoint; all eight PDF pages are visually checked.
Final presentation integration, new off-device preservation and full rehearsal
remain open; Task4 is not closed.

## Plan self-review and handoff

Self-review resolved an interface gap: explicitrunUUID is separate from fixedname
configuration identity, public camelCase choice keys map to resolvedPythonfields,
and record serialization belongs in the installed library rather than an uninstalled
benchmark helper. Historical encoding fixtures pin compatibility without rewriting
the old source-bound benchmark entrypoints.
Dependencies agree: Pythonworker version 1/config/evidence → SpringDTOs/lifecycle →
Reactservice/form/results → actualbrowser/runtime. Each ReviewFocus belongs to a
specified regression group. Allglobalsettings/scopes map to Task1 orTask2, product
semantics toTask3, actualdemonstrability toTask4. Serverconfiguration/docs/tests are
included with their deliverable, not separate scaffolding tasks. No mathematical
or numerical engine rewrite is planned.

The user selected dashboard integration; the design's fresh builder check found
nothing important left. Inline execution is retained under the continuous mandate.
This plan is ready for review and later execution; current fixed-study Task3 remains
the immediate priority. Do not start this plan's runtime work during study capture.
