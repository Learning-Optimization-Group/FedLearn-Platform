# Simulation Lab in the FedLearn dashboard

20 October2026. Design prepared while the frozen confirmatory study captures data.
No implementation or runtime checks occur during that capture. The study's source,
recipes and outputs remain unchanged. The user selected integration into the existing
FedLearn dashboard rather than a separate application. Other choices below are
implementation decisions under the continuing capstone mandate, not additional
user preferences. A fresh read-only builder check returned “nothing important left”.

## Intent and success

A project owner should be able to run a bounded, reproducible federated simulation
on the prepared NIH workload, see actual round outcomes, stop work, reopen a run,
and download its diagnostic record. An advisor should be able to see how privacy,
aggregation and malicious-client settings affect the same learning problem and
understand the evidence's limits. The demonstration must execute the actual simulator;
a separately selected saved-result replay is clearly labelled.

Use the existing React/Vite dashboard, Spring API, cookie authentication, role gates,
Ember components/tokens, Recharts and Python simulation engine. Do not introduce a
second frontend, an unauthenticated local service or a new training implementation.
No phone, g14, Jetson training service or cloud account is needed for cached-feature
simulation. Real data preparation remains a separate server-admin task.

Success is an actual UI-to-API-to-worker run with validated configuration, a resolved
recipe identity, complete observed rounds, useful metrics, working cancellation,
reopen/export and an honest comparison. A static dashboard, mocked chart, progress
counter or replay presented as live training does not pass.

## Scope

Required:
- Prepared NIH No Finding/any-finding workload, natural prevalence, whole-patient IID
  simulated clients, explicit validation evaluation and margin-based AUROC.
- Mean/trimmed mean, matched noDP/centralDP, clean/sign-flip/label-flip, four head sizes.
- Single bounded run, lifecycle and ownership checks, live completed-round updates,
  final results, research-record export and comparison of two to four retained runs.
- Read-only entry to the independently verified capstone report/aggregate results.
- Real browser verification of the meaningful flows and graceful missing-data/errors.

Outside this slice: image upload/preparation from the browser, arbitrary server paths
or commands, protocol/matrix editing, custom Python, localDP, oracle attacks, dynamic
accepted sets, end-to-end image-network training, physical-device orchestration,
automatic experiment search and publishing a privacy-protected model.
The frozen360-cell study is never launched or modified by this UI. Demonstration runs
are labelled exploratory and evaluate validation only; official-test study outcomes
are viewed as retained, independently verified results.

## User flow

Workspace navigation adds **Simulation Lab** for PROJECT_OWNER and PLATFORM_ADMIN.
Route `/simulations` uses existing ProtectedRoute/RoleRoute/LayoutV2. Ordinary USER
accounts do not see it and cannot access its API. Existing Benchmarks remains unchanged.

The page has a quiet workload/readiness summary, a compact configuration form, and a
recent-runs list. The selected run opens its progress/results pane without losing
configuration. Use ordinary labels, visible units, useful empty/error states and
keyboard-accessible controls; preserve existing desktop/responsive conventions.

1. **Choose settings.** Explain that clients are simulated patient groups and features
   are frozen. Show prepared split counts/prevalence without patient/image identifiers.
2. **Review and start.** Display the resolved experimental choices and their identity;
   start is unavailable when the workload is unqualified or a worker already runs.
   Starting returns a persistent run ID, not an optimistic completed result.
3. **Observe.** Show actual last completed round/R, status, elapsed process time,
   AUROC/loss/calibration curves, accepted-client counts, modeled upload/download bytes
   and cumulative accounted epsilon when applicable. No estimated quality/progress.
4. **Stop.** Show stopping until termination is verified. Keep partial rounds and the
   reason; never mark a stopped/timed-out run successfully qualified.
5. **Inspect/reopen.** Completed runs show final-round metrics, actual counts, timing
   scopes, provenance and configuration. Reload/navigation do not create another job
   or silently cancel server-owned work. Restore the selected ID from the route/query.
6. **Compare.** Select2–4 retained terminal runs with the same workload pin/evaluation
   split. Show differing recipe fields beside final metrics/curves. No winner test,
   primary confidence interval or best-round selection from these exploratory runs.
7. **Export.** Download the owned diagnostic JSON with its config/seeds/qualification
   or partial/failure metadata. State that it is a research record, not a DP-safe export.
8. **View study.** Open the verified fixed-study report/aggregate view, clearly distinct
   from live exploratory runs and with failed/unrun counts visible.

Do not show internal filesystem paths, worker commands or credentials in the form.
Show facts that affect decisions: prepared representation, evaluation split, privacy
unit/scope, observed versus modeled costs, complete versus partial execution and
live versus replay. No raw patient IDs, images, final prediction arrays or model
checkpoints are exposed by the initial Lab API.

## Bounded settings and defaults

The backend owns the workload reference, cache path and expected report pin. The
client supplies only typed choices; unknown fields, nonfinite values and unsupported
combinations fail before training. Python's resolver/core policy remains authoritative.
The UI's validation is additional usability, not the execution gate.

| Field | Supported initial choices | Default |
| --- | --- | --- |
| Workload | Qualified pretrained NIH cache, server-configured | That workload |
| Evaluation | Validation only for new runs | Validation |
| Head | linear, MLP8,MLP128,MLP512 | linear |
| Simulated clients |10,80,1000; full cohort each round |10 |
| Rounds | Integer 1–20 |3 |
| Local epochs | Integer 1–7 |1 |
| Learning rate |0.003, fixed in this slice |0.003 |
| Batch size |256, fixed |256 |
| Aggregation |mean, trimmed_mean with ratio0.2 |mean |
| Joint L2 clip |0.05,0.4 |0.05 |
| Privacy |none, central epsilon 1/4/8;delta 1e-5,replace-one |none |
| Behavior |clean, own-update sign flip−4, own-data binary label flip |clean |
| Malicious fraction |0 forclean,0.2 forselected attack |0 |
| Seed |Integer 0–1,000,000,000; displayed/reused explicitly |42 |

Defaults are for a quick live demonstration, not the selected main-study learning
schedule or a clinically useful model. Put the full E7/R20 study recipe in an explicit
preset, retaining bounded runtime and the exploratory label. Choosing a private
preset does not imply that every diagnostic field or downloaded record is protected.

One active worker per API instance; no queued sweep. A concurrent start returns409
with the current owned-run link when visible, otherwise a generic busy state.
Per-worker wall deadline180s includes startup and cache loading, CPU4 threads. Observed
engine-completion RSS over 2GiB rejects qualification; it is not an enforced process-
tree/late-artifact cap. A run can time out under valid settings; preserve that outcome.

## API, storage and ownership

Use `/api/simulations` with class-level owner/admin authorization and current user
identity from existing authentication. Per-run reads/stops/exports require ownership,
except administrators may inspect platform runs. Foreign/unknown IDs return404.
Use UUIDs and server-owned storage, never user-provided paths or shell command text.

Proposed endpoints:
- `GET /workloads`: configured readiness, label/representation/split/count/pin summary,
  supported choices and limits; no local path or record-derived performance.
- `POST /resolve`: typed choices to displayable resolved recipe plus config hash and
  server workload reference. No training or feature scoring. Paths stay server-side.
- `POST /runs`: validates/resolves again, atomically admits one job, persists it and
  starts one child. Return202 with ID/status/config; backend is the final authority.
- `GET /runs`: paginated owned runs (admin can use explicit platform scope), bounded
  page size and stable ordering; return concise summaries, not all logs/records.
- `GET /runs/{id}`: state, last completed-round data, recipe/provenance/qualification
  summaries and final/error information as available. Frontend polls while nonterminal.
- `POST /runs/{id}/stop`: idempotent stop request with current state, owner/admin only.
- `GET /runs/{id}/record`: authenticated diagnostic download when a complete/partial
  record exists; otherwise a precise not-ready response. No arbitrary file-serving API.

Add a Flyway migration and a dedicated simulation-run entity/repository for ownership,
status/timestamps, choices/resolved identity, progress summary and retained-result
metadata. Keep large raw event/log/model files in a configured private run-storage
root under UUID directories. Persist the record format/encoding and completion state;
use the existing tagged representation for any returned nonfinite failure evidence.
No patient-row data is stored in the UI database. Do not overload physical FL project
or benchmark entities with a different execution/qualification contract.

State transitions: accepted→starting→running→completed; stopping→stopped;
starting/running→failed/timed_out/interrupted. Persist before launch and record failed
launches. `completed` requires a full passing core record. Terminal state is immutable;
a stop arriving after successful completion returns completed. Progress is monotonic
and bound to the same job/config, never accepted from a browser-supplied event.

Browser disconnect, page unmount, logout and refresh stop polling but do not change
server-owned jobs. Another user's session cannot adopt or stop a former user's run.
The service's global admission lock is independent of any React component lifecycle.

## Python worker and process lifecycle

The lightweight `python -m fedlearn.lab_worker` entrypoint starts ownership and
deadline monitoring before heavy imports. It calls `prepare_core_config`/`run_core` with explicit declared
client count, validation split, expected pretrained cache pin and CPU4 threads.
It uses the existing `on_round`/`should_stop` interfaces; no numerical engine change
is needed. Emit/persist only real completed rounds and final returned evidence.
Save `CoreQualificationError.record` on postflight/stop failure and preserve progress
on hard termination. Only successful full postflight adds the qualification marker.

Launch a fixed interpreter/module command as an argument list, with a minimal allowed
environment, no web-auth secret and a server-owned config/output path. Separate
structured progress from library chatter; handle output with bounded storage/reading
so a noisy process cannot exhaust memory or block indefinitely. No pickle execution.

Reuse the existing ProcessTrees lifecycle utility and add explicit waiting/ownership
checks. Record PID plus process start identity, not PID alone. The service handles
explicit stop, deadline, application shutdown and worker failure, and drains the owned
process tree before freeing the active slot. Cooperative cancellation occurs at round
boundaries, followed by bounded forced termination if necessary. Preserve every
completed round and classify partial work without a successful marker.

Worker lifetime must remain bounded even if the API dies: detect parent stdin/pipe
closure, request cooperative stop, and retain an independent wall deadline. On API
restart reconcile persisted nonterminal jobs against the exact recorded process
identity, terminate only a demonstrably owned surviving worker, and mark interrupted.
Do not start another worker while old ownership is unresolved; do not kill by a reused
PID or assume a stale database row proves that a process is dead.

A server-owned per-run progress snapshot/event file is written atomically by the
worker after real round completion; the API can serve its validated projection.
Do not fabricate round callbacks by sleeping or update progress from expected duration.
All output paths are beneath the owned run directory, with no symlink/path traversal.
The main study's private raw artifacts are never mutable through Lab endpoints.

## Evidence and verification

Python: exact resolver bounds/pretrained pin/split before scoring; real small synthetic
cache execution through the actual worker; actual round events/full qualification;
stop after a real round, timeout/abrupt-parent loss, failed postflight retaining record,
malformed input, output boundaries and process/thread recipe. Synthetic utility is
labelled control evidence, never passed off as NIH performance.

Backend: controller authorization and per-user ownership for every operation; concurrent
start admits exactlyone process; launch/exit/stop/deadline races; process-tree drain;
restart reconciliation/PID reuse; malformed/foreign worker progress; persisted states,
config bindings, encoding and safe exports; database migration with Testcontainers.

Frontend: actual service contracts, route/sidebar role gates, validation and busy/error
states, start/stop/refresh/reopen/compare/export behavior; no stale polling updates
from an old selection/session; keyboard/labels and actual completed-round charts.
Run frontend lint, coverage and production build; backend tests/coverage require Docker.

Browser E2E lives in `~/code/homelab/fedlearn/`, following its own instructions. Drive
actual login/dashboard/API/Python, not intercepted canned responses. Verify live
start→rounds→completed→reopen→export, cancellation with no surviving child, a second
user denied access, unavailable workload and a labelled replay fallback. Collect
screenshots and records without exposing secrets/identifiers in committed evidence.

Final demo: use a short validation run and a matched saved comparison plus the verified
main-study report. Document hardware/runtime and measured versus modeled cost. Keep
a replay fallback but explicitly label it. A smooth UI does not replace the written
report, independent real-study audit or scientific conclusions.

## Implementation order and unresolved checks

1. Finish fixed-study capture and independent reconciliation/report first. All current
   design/script drafts stay ignored; no tracked edits or runtime checks during capture.
2. Builder check complete with no material questions. Preserve the user-chosen
   dashboard direction and use the implementation plan for the remaining interfaces.
3. Plan/implement the bounded Python worker/lifecycle contract and backend ownership/
   persistence, with actual process regressions and one bounded review per rejectable slice.
4. Implement the existing-style dashboard flow with real contracts, then full stack
   and browser verification. Update the checklist from actual evidence.

The original Oct5 draft was located and inspected in the root checkout after being
absent at that path in this worktree. It explicitly leaves the GUI itself to a
separate item11 design. This spec follows the current consolidated plan and frozen
NIH core first; the existing engine retains its wider algorithms/faults. Do not claim
that the initial Lab UI exposes all those broader historical options. Read-only
artifact hashes and the detailed alignment note are preserved privately.

Implementation-plan decisions: use a separate canonical UUID run identity, leaving
the resolved recipe name fixed so repeated identical choices share a config hash.
Use a strict choices-object API and server-only cache/path mapping. The plan records
public field names, worker evidence filenames and backend DTOs. These settle builder
interfaces without changing the user-selected integration or scientific protocol.
