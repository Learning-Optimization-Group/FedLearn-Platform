# Simulation Lab API and server configuration

20 October 2026. The API is a separate simulation surface in the existing Spring
application, using its PostgreSQL database, cookie authentication and platform
roles. It does not create physical FL projects, clients or training servers. The
dashboard workflow and actual NIH browser demonstration remain separate gates.

## Server configuration

Simulation execution is disabled by default. Configure these properties through a
private Spring configuration file or deployment settings. Values are server owned;
HTTP requests cannot supply an interpreter, command, dataset path or output path.

```properties
app.simulation.enabled=true
app.simulation.interpreter=${CAPSTONE_LAB_PYTHON}
app.simulation.source-root=${CAPSTONE_LAB_MODULE_ROOT}
app.simulation.cache=${CAPSTONE_LAB_CACHE}
app.simulation.report-sha256=${CAPSTONE_LAB_FEATURE_PIN}
app.simulation.storage=${CAPSTONE_LAB_JOB_ROOT}
app.simulation.deadline-seconds=180
app.simulation.reconcile-on-startup=true
# Optional: the capstone Effusion R0 bundle (train + development selection only)
app.simulation.effusion-bundle=${CAPSTONE_LAB_EFFUSION_BUNDLE}
app.simulation.effusion-report-sha256=${CAPSTONE_LAB_EFFUSION_PIN}
```

**Effusion workload (`nih-effusion-r0`).**
- The two optional settings enable a second workload, built by
  `repro/build_effusion_bundle.py`. When they are unset, `/simulations/workloads` lists it
  as unavailable.
- The bundle holds standardised R0 features and exact-token Effusion labels for train and
  the development selection half. It has no holdout and no official-test rows; the loader
  re-verifies every hash and refuses anything else.
- Its recipes are the measured presets `effusion-clean` (LR 0.1, clip 1.0) and
  `effusion-dp` (LR 0.03, clip 0.05). Each fixes its own clip, so requests omit `clipNorm`.
  The historical workload keeps the `standard` recipe.
- A Lab run with seed 20261101, 80 clients, 20 rounds, 7 epochs and `effusion-clean`
  reproduces the capstone A7 run exactly: per-round AUROC difference 0.0 and the same
  final digest.

Use an absolute executable path and a module root containing the installed
`fedlearn` package or the framework's `src` directory. The cache must pass the
existing integrity/pretrained checks and match the independently selected report
SHA256. The private writable job root must be outside public/static serving.
Flyway V33 creates `simulation_runs`, with an owner foreign key to `users`.

Each job has its own UUID directory. Its input configuration, bounded process log
and fixed `execution` evidence files remain private. Only explicit path-free
projections are persisted for HTTP responses and research-record downloads.

The worker receives a minimal environment: module root, four-thread settings,
offline feature settings and a fixed system search path. Backend authentication,
database credentials, home-directory settings and arbitrary environment values are
not inherited. Resolver stdout carries JSON; bounded stderr is drained separately.

## HTTP contract

All routes are under `/api/simulations`. Anonymous requests return 401; ordinary
USER accounts receive 403. PROJECT_OWNER and PLATFORM_ADMIN accounts may use the
Lab. Other users' runs appear absent (404) to owners. Administrators may read/stop
individual runs; listing all users' runs requires explicit `scope=platform`.

| Method and route | Result |
| --- | --- |
| GET `/workloads` | Readiness, bound report pin, aggregate split counts, supported choices and effective limits |
| POST `/resolve` | Canonical configuration hash and display recipe, without starting training |
| POST `/runs` | 202 with persisted run ID, state, hash, recipe and creation time |
| GET `/runs?page=0&size=20&scope=mine` | Owned paginated history; size 1–100 |
| GET `/runs/{uuid}` | Retained recipe, actual completed rounds, terminal outcome and available projected record |
| POST `/runs/{uuid}/stop` | Idempotent request; STOPPING until process/tree drainage; terminal outcomes immutable |
| GET `/runs/{uuid}/record` | Owned JSON attachment; 409 while no retained record is available |

Resolve and start accept exactly `{"choices":{...}}`. The input stream is bounded
to 64 KiB before parsing; duplicate keys, trailing JSON and nonfinite constants are
refused. Choice types and numerical bounds are enforced by the canonical Python
resolver. Every failed resolver exit maps to a controlled validation response;
private tracebacks and library warnings are not HTTP messages.

Choices and defaults are in the [worker contract](simulation-lab-worker-contract.md).
Each response includes the research scope. Missing metrics/timestamps are null.
Costs retain their modeled/deployment-time meaning; CPU time is a local measurement.
Downloads are explicitly **diagnostic research records, not DP-safe releases**.

The retained projection includes the path-free resolved settings and derived
seeds, modeled-network assumptions, available runtime/source/cache fingerprints,
aggregate split counts, observed per-round participation, measured engine wall
time/RSS with their scopes, modeled totals and controlled terminal metadata.
Unavailable values remain null. Hostnames, filesystem paths, per-client mappings,
patient identities, arrays and arbitrary private metadata are excluded. Downloads
remain available from PostgreSQL without rereading private worker artifacts.

## Process ownership and restart behavior

One active worker is admitted per application instance; there is no job queue.
The intended deployment has one simulation-serving backend process. Distributed
admission across multiple backend replicas is not implemented.

The row is committed as ACCEPTED, then STARTING before process launch. The API
holds the worker's stdin pipe open and persists its PID, OS process start identity
and observed executable. A separate supervisor enforces the effective whole-process
deadline (at most 180 seconds), including interpreter startup. Stop/deadline signals
are followed by forced termination after two seconds if needed. Tracked descendants
must exit before admission is released. This supervises the trusted worker; it is
not containment against an adversarial process that deliberately escapes ancestry.

Progress must match the run/configuration identity, preserve completed round
prefixes and move forward. A completed outcome additionally requires actual exit 0,
consistent terminal execution evidence, the exact launch configuration, all requested
rounds, finite encoding, core qualification and the observed RSS qualification.
A complete round count alone cannot produce COMPLETED. Partial/tagged research
evidence remains explicitly unqualified.

Before configuration resolution, the real worker can stop, time out or lose its
owner with a null configuration hash and zero requested/completed rounds. Such a
snapshot is accepted only for the exact run, with empty rounds, no record and no
previously resolved identity; it can never qualify as completed. The first stop
cause is retained during process drainage, so a later deadline does not relabel an
earlier explicit stop.

On startup, unfinished rows are reconciled. The service only terminates a live
process whose PID/start/executable/argument/storage identity matches. A reused PID
is left alone. An ambiguous launch window or unidentifiable process blocks further
launches with `OWNERSHIP_UNRESOLVED`; it is never treated as successful completion.
An administrator must independently establish and clear the old process ownership
before repairing that retained state. There is no automatic resume or checkpoint
recovery. Normal shutdown drains owned work and retains INTERRUPTED.

Backend death closes the parent pipe. The Python worker independently requests
interruption and applies its own bounded stop/deadline behavior. Neither this nor
the 2 GiB engine-completion RSS observation is a hard process-tree memory cap or a
guarantee against host power loss, filesystem failure or uninterruptible native code.

## Verification

The 20 October checkpoint passed all **1,207 backend tests**, with no skips,
including 32 simulation checks, and the coverage gate at **84.09% line coverage**.
The initial full run had one admin-login HTTP EOF failure; eight isolated admin
checks and a subsequent instrumented full run passed without a code workaround.
The cause of that intermittent transport failure remains unresolved. It must not
be described as fixed. Dashboard/browser acceptance is still outstanding.

Controller checks use real cookie login and PostgreSQL. Fault controls use actual
Python subprocesses with explicit synthetic contracts. The opt-in numerical-worker
checks additionally execute two real engine rounds on schema-valid synthetic
features, and kill an isolated Java owner to verify parent-pipe interruption. These
establish execution behavior, not NIH classification quality.

```sh
SIMULATION_REAL_WORKER_PYTHON="$CAPSTONE_PYTHON" \
  SPRING_PROFILES_ACTIVE=test ./gradlew test jacocoTestCoverageVerification
```

Use Java 21 and Docker. The opt-in interpreter needs the framework development
dependencies, including pytest for the synthetic cache fixture. Normal Java CI can
run process-contract controls with system Python 3; the numerical-worker checks are
skipped unless the interpreter variable is explicitly set.

One minor remains deferred: busy responses currently omit an active-run link even
for its owner or an administrator. They disclose no foreign ID. Use owned history
to reopen a run after a lost start response.
