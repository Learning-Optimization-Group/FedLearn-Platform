# 04 - Federated Orchestration (FlServerManager)

The `FlServerManager` is the most operationally complex class in the Spring Boot backend. It is responsible for bridging the gap between the stateless Java REST API and the stateful, heavily computational Python Federated Learning servers.

The application has **one** supported execution path: the FL server runs as a **local child process** on the same host as the backend. This is true everywhere — development, the `ec2demo` profile, and the hardened single-VM `production` profile. There is no cloud-execution mode; see [The ECS/Fargate path (removed, fail-closed)](#the-ecsfargate-path-removed-fail-closed) below.

## Local Execution (the FlServerProcessRunner seam)

`startServerForProject(...)` performs its policy gates and then delegates to `startLocalServer(...)`, the only real path in the class. The raw `ProcessBuilder` mechanics no longer live in `FlServerManager` itself — since **DA-8** they sit behind the `FlServerProcessRunner` seam, whose default implementation is `LocalProcessFlServerRunner`. The manager keeps all *policy* (argv construction, environment scrubbing, port reservation, run-state persistence, the startup probe, log broadcasting); the runner does nothing but apply the env customizer, merge stderr into stdout, set the working directory, and start the process. That split is what makes the spawn path unit-testable with a fake runner.

### Process Lifecycle
1. **Port Allocation:** The manager scans the configured port range — `fl.server.port-range.start`/`.end`, defaulting to **50000–50010** — and reserves the first port it can actually bind with a probe `ServerSocket`. Reserved ports are tracked in a set guarded by a lock, so two concurrent project starts cannot pick the same port between probe-close and the Python child's bind.
2. **Command Construction:** It builds a command array around the FL-server shell wrapper, resolved from `python.script.fl-server.path` (default `../../fl-runtime/run_fl_server.sh`; `.bat` on Windows). Federation over Text (FoT) selects its own wrapper, `python.script.fot-server.path` (a `@Value` default in the manager — it is not declared in `application.properties`). The wrapper indirection is what lets the same Java code work on macOS, Linux, and Windows.

   Three conditional flag groups hang off the project's own configuration:
   - **`--training-arm <ARM>`**, emitted **only when the project's arm is not `FULL`**. That
     asymmetry is deliberate: every pre-arm spawn's argv stays byte-identical to what it was, and
     both `fl_server.py` and `client.py` resolve an omitted arm to `FULL`. Whether the *recipe*
     supports the arm is validated by `recipes.validate_arm()` on the Python side — the catalog is
     the authority; the `TrainingArm` enum and the `V22`/`V23` CHECK only bound the vocabulary on
     this side. See [03 - Project Management Lifecycle](03_project_management.md).
   - **`--aggregation FFA_LORA --task-type <…>`** for `LLM_LORA` projects.
   - **the `--dp-*` group** when DP is enabled (SE-11). The flag names are a pinned contract with
     `fl_server.py`'s argparse. Creation already validated completeness, but the spawn seam
     re-checks so a null knob can never reach the argv as the literal string `"null"`
     (SE-10 fail-closed); every value is a typed number formatted with `String.valueOf`, never a
     raw string.
3. **Environment Construction:** The child environment is **rebuilt from an allowlist** rather than inherited (SE-17), so no backend secret — DB password, web-auth JWT secret, CORS config — reaches the FL server. Only OS/runtime essentials plus the `FEDLEARN_*` namespace survive; the manager then sets the explicit per-run variables:
   ```java
   env.keySet().removeIf(key -> !isAllowedChildEnvKey(key));
   env.put("FEDLEARN_INTERNAL_API_KEY", internalApiKey);
   env.put("FEDLEARN_INTERNAL_RUN_TOKEN", internalRunToken);  // SE-7: scoped to (projectId, runId)
   env.put("FEDLEARN_BACKEND_URL", backendInternalUrl);
   ```
   The per-run token (SE-7) is minted fresh for each start and is the only credential the child can present on `/api/internal/**` — a leaked run token can mutate only its own project. TLS enforcement (SE-2) and client-auth enforcement (SE-1/SE-7) are passed as explicit toggles, both off by default.
4. **Execution and Tracking:** The runner starts the process and returns a `SpawnedFlProcess`. The manager stores `process.toHandle()` in a `ConcurrentHashMap<UUID, ProcessHandle>` keyed by the `projectId` — a `ProcessHandle` rather than a `Process` because a restarted JVM can only ever re-adopt a *handle* to a child that outlived a backend crash (BA-3).
5. **Identity Persistence:** The child's PID, start instant, reserved port, and run-token hash are recorded on the active `Run` (BA-3, migrations `V15`/`V16`), so a startup reconciler can distinguish a still-live FL server from a dead — or PID-reused — one. If that persistence fails, the spawn **fails closed**: the child is killed rather than left as an unreconcilable orphan.

### Crash recovery: `StartupReconciler` (BA-3)

The in-memory `ConcurrentHashMap` is lost when the JVM dies, so a backend crash would otherwise orphan
every FL server: the children keep running and holding gRPC ports while their runs sit forever in a
non-terminal state with no handle to stop them. `bootstrap/StartupReconciler` closes that on boot. For
each still-in-flight run it compares the recorded PID **and** the recorded OS start instant against
the live process table (`ProcessProbe`), then either:

- **re-adopts** the run — PID live *and* start instant matches, so the server genuinely survived the
  restart. It is tracked again (a later `/stop` can terminate it) and its internal token is rehydrated
  into `RunTokenRegistry` from `runs.internal_token_hash` so its result callbacks keep authenticating;
  or
- **reaps** it — the process is dead, the PID was never recorded, or the live PID belongs to a
  *different* process (PID reuse). The run is reset to `FAILED` and the project returns to idle. A
  PID-reused process is never killed: it is not ours, and our own server is already gone with its port
  already free.

Pairing the start instant with the PID is what makes PID reuse safe to detect; a recycled PID on an
unrelated process cannot share the original's start time. Reconciliation never throws out of its
runner — a failure here must not stop the application booting — and the last pass's summary is
exposed as an actuator health-indicator detail.

Boot reconciliation only fires once, so the same class carries a **periodic stuck-run sweep**
(`app.fl.stuck-run-sweep.interval-ms`, default 60 s, flag-gated and fail-soft) for the other half of
the problem: an FL server that dies while the *backend* stays up skips the terminal callback and
leaves its run `RUNNING` forever. The sweep uses a **two-pass debounce** — a `RUNNING` run whose child
is no longer tracked is reaped only if it was also untracked on the previous pass — so a run that is
merely mid-completion or mid-stop settles to its terminal status first and a `FAILED` verdict cannot
clobber a `COMPLETED` one. Only `RUNNING` is swept; `PENDING`/`STARTING` runs are legitimately
mid-launch (a slow LLM start can sit in `STARTING` for minutes).

### Output Redirection
Because a spawned local process does not automatically print to the parent's console, the manager creates a dedicated daemon thread `fl-server-stdout-{id}`. This thread continuously reads from the child's `InputStream`.

Every line read is passed to the `WebSocketService` to be broadcast to the UI. If the process crashes during startup — the probe window is `fl.server.startup-probe-seconds`, default **3 seconds** — an exception is thrown containing the captured `stdout` to help developers debug.

## The ECS/Fargate path (removed, fail-closed)

**There is no cloud-execution path in this build.** The backend carries no AWS SDK dependency and no ECS orchestration code. The only surviving trace is a single property, `ecs.cluster-name=${ECS_CLUSTER_NAME:}` (blank by default), which exists purely so that setting it can be **rejected**. `ecs.task-definition`, `ecs.subnets`, `ecs.security-groups` and friends do not exist as properties at all.

Two fail-closed gates enforce this:

1. **Boot-time (OP-14, the real gate).** `FlOrchestrationModeValidator` carries a `@PostConstruct` check that throws `IllegalStateException` when `ecs.cluster-name` is non-blank. It has **no `@Profile` annotation**, so it gates in *every* profile — ECS is unsupported everywhere. The blank default is the supported single-VM path and always boots. The point is to move the failure to startup: an operator who wires `ECS_CLUSTER_NAME` should be told immediately, not discover the gap mid-federation.
   ```
   OP-14: ecs.cluster-name is set ("...") but ECS/Fargate FL-server orchestration is not
   implemented (tracked as OP-12). This build supports the hardened single-VM architecture
   only — FL servers run as local processes. Unset ECS_CLUSTER_NAME to boot.
   ```
2. **Runtime backstop.** `FlServerManager.startServerForProject(...)` still throws `UnsupportedOperationException` on the same branch. A correctly-booted app can never reach it; it exists so that no route around the validator can record a project as `RUNNING` on an unreachable port while a real task leaks.

> **Historical note (past tense — not usable guidance).** An AWS Fargate orchestration path *was* implemented once, in `1239dda refactor(backend): migrate to AWS Fargate orchestration`. The AWS SDK was subsequently removed (`9124b62`), which deleted that implementation along with it; `8d5dfdc` (OP-14) then added the boot-time validator to make the remaining configuration fail closed. None of that code is in the repository today, and none of it would compile without the SDK. Managed-task orchestration is **deferred to OP-12** and will, when built, arrive as an alternative `FlServerProcessRunner` implementation behind the DA-8 seam rather than as inline SDK calls.

### The supported deployed architecture
The `production` profile is the **hardened single-VM** profile — it is not, and never should be described as, an "ECS Fargate profile". FL servers run as local Python processes on the same VM as the backend, exactly as in development, with the hardening (secret scrubbing, fail-closed boot checks, run-token scoping) layered on top.

## Graceful Shutdown
To prevent orphaned processes, the manager hooks into the Spring Application Context lifecycle via the `@PreDestroy` annotation. When the Java backend is shutting down, it iterates through the `ConcurrentHashMap<UUID, ProcessHandle>` of running servers and calls `destroyForcibly()` on each live child. The same teardown (`stopServerForProject`) backs the stop endpoint, which additionally evicts the run's internal token from the `RunTokenRegistry` (SE-7) before terminating the child.
