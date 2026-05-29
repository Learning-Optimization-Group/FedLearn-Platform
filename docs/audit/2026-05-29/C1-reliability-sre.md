# C1 — Reliability / Fault-Tolerance / SRE for FL Workloads

**Date:** 2026-05-29
**Scope:** Failure modes, checkpointing/resumability, round-recovery, FL-server HA & orchestration, disaster recovery, graceful degradation, and SLO/SLI definitions for long-running FL jobs.
**Builds on:** `docs/audit/2026-05-27/01-backend.md` (C4 race, H4 stdout wedge, H5 SIGKILL, M2 port cap, M9 in-memory broker), `03-framework.md` (H1 heartbeat-death invisibility, H5 unbounded streaming), `04-observability.md` (dead `RoundResult` pipeline, cardinality risks). This report **extends** those — it does not re-litigate them. Where I cite a prior finding I name it (e.g. *prior 01-backend C4*).

---

## 0. Executive framing

FedLearn's reliability posture today is that of a **single-process research harness wearing a web UI**. The "FL server" is a Python process spawned by Spring Boot via `ProcessBuilder` (`FlowerServerManager.startLocalServer():199`), tracked in an in-memory `ConcurrentHashMap<UUID, Process>` (`:85`), with **all federated state — global model, per-round seed history, per-round gradient history, per-client participation cursor — held in Python heap and lost on exit.** A run is a fixed `for round in range(1, num_rounds+1)` loop (`fl_server.py` → `framework/.../server.py:137`) that **blocks indefinitely** waiting for an exact client count, with no round timeout, no quorum degradation, no checkpoint, and no resume. The model is written to disk **once, destructively, at the very end** (`fl_server.py:545` `save_path = args.model_path`).

For a research demo this is acceptable. For a **production startup running multi-hour healthcare/edge federations across churning clients**, the expected outcome of any single fault (server OOM, JVM redeploy, one client dropping) is **total loss of the run**. This is the single largest production-readiness gap in the platform and the area where v2 must be most opinionated.

The good news: the DeComFL protocol is **structurally cheap to make resumable** — its entire reconstructable state per round is `(seeds[K][P], avg_gradients[K][P])`, i.e. `O(K·P)` scalars, not the model. The recovery primitive already exists (`rebuild_model`, `get_rebuild_history`). v2's job is to **persist that ledger durably and make the orchestrator stateless**, not to invent new ML machinery.

---

## 1. Failure-mode inventory (evidence-grounded)

I walked each failure the assignment named, plus the ones the code revealed. Severity is operational impact on a live multi-hour run.

### F1 — FL server crash mid-run → **total, silent run loss** (CRITICAL)

- Global model lives only in `coordinator._global_model_params` (`coordinator.py:26`) and `DeComFL.global_params_flat` (`decomfl_strategy.py:78`); both are Python heap.
- The only persistence is the **terminal** save at `fl_server.py:543-545` and the **terminal** results POST loop at `fl_server.py:561-587`. Both run *after* the `range(num_rounds)` loop exits normally. **A crash at round 4/5 writes nothing** — no model, no metrics for rounds 1-4.
- `server.py:172-175` catches `Exception`, logs it, calls `signal_stop()`, returns `([], {})` — so even a caught exception discards all `history`.
- There is **no `torch.save` anywhere in `framework/src/fedlearn/server/`** (verified by grep). No checkpoint cadence exists.
- **Consequence:** a 6-hour LLM federation that dies at hour 5 restarts at round 1. For a paying customer this is unacceptable.

### F2 — Client disconnect/reconnect & the `rebuild_model` replay — **partially robust, but fragile and unbounded** (HIGH)

The DeComFL replay is the platform's headline recovery mechanism. It is genuinely clever and is the right primitive — but the audit must be honest about its limits.

**What works:** When a client rejoins, `GetDeComFLConfig` (`grpc_servicer.py:299`) calls `strategy.get_rebuild_history(client_id, current_round)` (`decomfl_strategy.py:128`), which returns every missed round's `(seeds, avg_gradients)` between `client_last_round[client_id]+1` and `current_round`. The client replays them deterministically in `rebuild_model` (`decomfl_client.py:71-119`), regenerating each perturbation from its seed and applying `x -= (η/P)·Σ g·z`. Because communication is `O(K·P)` scalars regardless of model size (DeComFL's core property), this is bandwidth-cheap.

**Where it breaks:**

1. **Replay state is in-memory and server-lifetime-scoped.** `seed_history` and `gradient_history` are dicts on the strategy object (`decomfl_strategy.py:66-67`). If the **server** restarts, every entry is gone. A client that reconnects after a server restart gets `rebuild_history = []` and silently resumes from a **wrong (re-initialized) model** — divergence with no error. Client-side resilience is undermined by server-side amnesia.
2. **Replay cost is `O(missed_rounds · K · P · model_dim)` of `torch.randn` on the client.** `rebuild_model` regenerates a full-dimension perturbation per `(k,p)` per missed round (`decomfl_client.py:103`). For a 7B-param model with K=10, P=20, after a 50-round absence that is `50·200 = 10,000` full-model RNG draws + accumulations on a possibly-mobile client. The paper's *communication* is dimension-free; the *reconstruction compute* is not. On a phone (`fed-mobile` C++ port) this can exceed the round budget and thermally throttle.
3. **The `client_last_round` cursor only advances on successful aggregation** (`decomfl_strategy.py:174`, inside `aggregate_fit`). A client whose upload was *received* but whose round failed to aggregate has an ambiguous cursor. Combined with *prior 03-framework H1* (heartbeat death invisible), a client can be marked dead, have its late update rejected as stale (`coordinator.py:228-234`), yet its cursor never advanced — leading to an ever-growing replay window.
4. **No replay integrity check.** `rebuild_model` trusts the history blindly; there is no hash/version of the model the client is rebuilding *toward*, so a seed-schedule mismatch (e.g. *prior 03-framework C3* compression-flag drift, or an RNG-device difference between `cuda` server and `cpu`/Metal client — note `_generate_perturbation` uses `self.device` on the server (`decomfl_strategy.py:212`) but the client uses its own device in `generate_perturbation`) produces a **silently divergent model**, not a failure.

### F3 — `startServerForProject` wait-3s-then-throw + the duplicate-spawn race (HIGH; *partially mitigated since prior C4*)

- `FlowerServerManager.startLocalServer():229` does `process.waitFor(startupProbeSeconds, SECONDS)` (default 3s, now configurable — improvement since prior audit). If Python exits inside 3s it throws with captured stdout (`:238`). **But a slow boot that takes 4s "succeeds" even if it then crashes at 4s** — the backend reports RUNNING, the row says RUNNING, no client ever connects, and nothing notices. The probe proves "didn't crash in 3s," not "healthy."
- **The C4 logical race is only half-fixed.** `FlowerServerManager` now has a `reservedPorts` set + `synchronized(portReservationLock)` in `findFreePort()` (`:337-353`), which prevents two starts binding the *same port*. **But `ProjectService.startServerForProject` is still not `@Transactional`** (verified — `:147` has no annotation, unlike `createProject:92`, `stopServerForProject:198`). Two concurrent `POST /start` both read `isServerRunning()==false` (`:172`), both proceed, both call `stopServerForProject` then spawn — producing **two FL servers on two different ports for one project**, the second overwriting `runningServers.put(projectId,…)` (`:200`, still a plain `put`, not `putIfAbsent`) and **orphaning the first process** (alive, bound, unreachable by `/stop`). Prior C4's recommended pessimistic row-lock + `putIfAbsent` is **not yet applied at the service layer.**

### F4 — JVM restart loses the process map → **phantom-RUNNING projects + orphan processes** (CRITICAL)

- `runningServers` is a non-persistent field (`:85`). `@PreDestroy stopAllOnShutdown()` (`:303`) only fires on a **graceful** context close — it does nothing for `SIGKILL`, OOM-killer, ECS task replacement, `kill -9`, or a crashed JVM.
- There is **no `@PostConstruct` / `ApplicationReadyEvent` reconciliation** (verified by grep — only `BootstrapRunner` runs at startup, for admin seeding). On restart the map is empty, yet `Project.status` rows still say `RUNNING` (set at `ProjectService.startServerForProject:182`, never reset on crash).
- **Consequence:** every prior FL Python process is now an **orphan** — still alive, still bound to its `50000-50010` port, still possibly accepting clients, but invisible to the backend. `/stop` returns "no running server." The port pool (only 11 wide, *prior 01-backend M2*) leaks. The DB lies about run state. This is a classic split-brain.

### F5 — No round-level timeout / quorum → **one slow or dead client wedges the run forever** (CRITICAL)

- A round completes **only** when `len(self._client_updates_received) >= self.clients_per_round` (`coordinator.py:80` / `:253`). `wait_for_round_to_complete()` (`coordinator.py:42-46`) loops on a 1s `Event.wait` and breaks **only** on `stop_requested`.
- `heartbeat_timeout = 300` (`coordinator.py:34`) and `is_client_alive` / `get_active_clients` **exist but are never consulted by the round loop** (verified — only `update_client_heartbeat` is called, from `grpc_servicer.py:247`). The liveness machinery is dead code.
- `should_stop` is **hardcoded `False`** (`coordinator.py:165`) — the server has a wire field to tell a client to stop (`HeartbeatResponse.should_stop`, proto:101; client honors it at `grpc_client.py:267`) but **never sets it.** The graceful-drain path is unreachable.
- **Consequence:** in a 100-client cross-device federation where `clients_per_round=100`, a single client crashing mid-round hangs the **entire run indefinitely**, holding a process + port until a human notices and hits `/stop`. There is no partial aggregation, no min-quorum, no straggler deadline.

### F6 — DR / backup posture: **none** (HIGH)

- Global model artifact is overwritten in place (`fl_server.py:545`) — no versioning, no S3, no off-host copy. *prior 03-framework M4* notes peak `2× model size` memory on the download path; a 14 GB LLM model on the host's local disk with no replication is a single-disk-failure-from-permanent-loss situation.
- H2 file-mode DB in `ec2demo` (`CLAUDE.md` profile table) holds `Project`, `RoundResult`, `audit_events`. No documented backup/restore for it in the SRE sense (it is a single file on one EC2 instance).

### F7 — stdout-pipe wedge & STOMP backpressure can stall the FL process (MEDIUM; *prior 01-backend H4 / 04-observability Risk 2*)

Still relevant to reliability: the FL server's only liveness signal to the browser is stdout over a daemon reader thread → `WebSocketService.sendLogs` → STOMP. If the reader dies or a slow WS subscriber back-pressures, the child's ~64KB stdout pipe fills and the **Python FL server blocks on `write()` mid-round** — a reliability fault disguised as a logging concern. Prior audit flagged the mechanism; I flag the **reliability blast-radius**: log transport must never be in the FL-progress critical path.

---

## 2. How mature ML-training / FL platforms handle long-job reliability (research)

I anchored v2 on three reference designs.

### NVIDIA FLARE — the gold standard for FL server HA

FLARE solves exactly this problem and v2 should mirror its model conceptually (without adopting FLARE itself — platform invariant: custom framework, no Flower; FLARE is a separate question for *B2 tech-stack*).

- **Overseer + hot/standby Service Providers (SPs).** An Overseer is the authoritative registry of which FL server is "hot"; it heartbeats all entities every 5s and, on missed heartbeats, promotes a cold standby — **automatic cutover, no human in the loop.** ([FLARE HA docs](https://nvflare.readthedocs.io/en/2.6/programming_guide/high_availability.html))
- **Snapshot-based job continuation on shared storage.** The Controller takes a snapshot **after each round** (e.g. Scatter-and-Gather); on cutover the new SP restores `FLComponent` states + `FLContext` + job workspace from the **latest snapshot on storage shared by all SPs**. ([FLARE HA docs](https://nvflare.readthedocs.io/en/2.6/programming_guide/high_availability.html))
- **The honest caveat FLARE documents:** *"If the Controller didn't create additional snapshots, then the job will be executed from the beginning after the SP cutover."* — i.e. **HA without per-round checkpointing only buys you a cold restart.** This validates my F1/F4 verdict: HA orchestration and round-level checkpointing are **two separate requirements**; you need both.

**Direct lesson for FedLearn:** FedLearn today has neither the Overseer (no external registry; the JVM *is* the registry, and it forgets on restart — F4) nor the snapshot persistor (no per-round checkpoint — F1). Both are mandatory for v2.

### Flower — checkpoint via the evaluate hook

Flower's idiom is to checkpoint inside the server-side `evaluate`/strategy callback after each round (`model.save_pretrained(f"{save_path}/peft_{server_round}")`). ([Flower paper](https://arxiv.org/pdf/2007.14390), [Flower FT FedAvg + checkpointing](https://medium.com/mitb-for-all/how-to-train-your-llm-simultaneously-with-10-different-teachers-ba556d4ed2c1)). FedLearn already has the perfect seam: `coordinator._trigger_decomfl_aggregation_and_evaluation()` and `strategy.evaluate()` (`decomfl_strategy.py:242`) run **once per round, under the round lock** — that is precisely where a checkpoint write belongs. The literature also covers FL-specific checkpoint strategies for heterogeneous fault-prone nodes ([Electronics 2024, 13(6):1007](https://doi.org/10.3390/electronics13061007)).

### Goodput / ETTR — the right SLI vocabulary for long jobs

Google Cloud's **Goodput** framing is the correct mental model for FL-run reliability SLIs: `Goodput = useful_compute_time / (useful_compute_time + idle + checkpoint + recovery_overhead)`; related: **Effective Training Time Ratio (ETTR)**. ([Goodput metric](https://cloud.google.com/blog/products/ai-machine-learning/goodput-metric-as-measure-of-ml-productivity), [Elastic training & checkpointing](https://cloud.google.com/blog/products/ai-machine-learning/elastic-training-and-optimized-checkpointing-improve-ml-goodput)). For FedLearn the analogue is **round goodput** = `rounds_committed / (rounds_committed + rounds_lost_to_faults + recovery_rounds)`. Nebius and others frame fault-tolerant clusters around minimizing restart-from-checkpoint cost ([Nebius reliable clusters](https://nebius.com/blog/posts/how-we-build-reliable-clusters)).

### Kubernetes JobSet / Pod Failure Policy — the orchestration substitute for `ProcessBuilder`

The industry answer to "spawn a long-running training process and supervise it" is **JobSet** (k8s-native API for distributed ML/HPC) with **Pod Failure Policy** to distinguish retriable from terminal failures, and **in-place pod restart** (k8s v1.35) / watcher-sidecar patterns to restart a worker from its last checkpoint without rescheduling. ([JobSet](https://jobset.sigs.k8s.io/docs/overview/), [Pod Failure Policy GA](https://kubernetes.io/blog/2024/08/19/kubernetes-1-31-pod-failure-policy-for-jobs-goes-ga/), [in-place restart v1.35](https://kubernetes.io/blog/2026/01/02/kubernetes-v1-35-restart-all-containers/)). This is the v2 replacement for the `ConcurrentHashMap<UUID,Process>` orchestrator — the supervisor becomes the control plane, not the JVM heap.

---

## 3. v2 design

I separate this into the **5 layers** a reliable long-job platform needs. Each is calibrated to a startup (seed → Series-A), not hyperscale; I flag the tier where each becomes necessary.

### 3.1 Checkpointing & resumability (the foundation — do this first)

The reconstructable state of a DeComFL run is tiny. Persist a **per-round checkpoint** consisting of two parts:

**(a) The round ledger (small, the source of truth for replay):**
```
checkpoint/{project_id}/round_{r}.json   (or a single append-only ledger)
  round_number, seeds[K][P], avg_gradients[K][P],
  client_last_round snapshot, loss, metrics, model_hash, strategy_config_hash
```
This is `O(K·P)` floats/ints + a hash — kilobytes per round. It makes `seed_history`/`gradient_history`/`client_last_round` **durable**, which directly fixes F2.1 (server-restart amnesia kills client replay) and F1 (no checkpoint).

**(b) The global model artifact (large, periodic, for fast warm-start):**
- Write `global_model_round_{r}.safetensors` to **object storage (S3/MinIO)**, not destructively over `args.model_path`. Keep the last *N* + every *M*-th round.
- Use **`safetensors`**, not `torch.save`/pickle (aligns with *prior 03-framework C2/C3* anti-pickle stance; `weights_only=True` is the documented invariant). I have not verified safetensors handles every model class FedLearn ships; flag as a build-time check.

**Where to write it (exact seam):** inside `_trigger_decomfl_aggregation_and_evaluation()` (`coordinator.py:278-302`), **after** `self.strategy.gradient_history[...] = avg_gradients` and `evaluate(...)`, **before** `self.current_round += 1` — i.e. under `_lock`, so the checkpoint is atomic with the round commit. The ledger write must be **fsync'd / `PutObject`-confirmed before the round counter advances**; otherwise a crash between advance and persist loses a round (the same ordering bug FLARE avoids).

**Cadence:** ledger **every round** (cheap); model artifact every round at seed tier, every M rounds at scale (cost-tunable). This is the classic checkpoint-frequency vs recovery-cost tradeoff that ML-goodput literature optimizes ([elastic checkpointing](https://cloud.google.com/blog/products/ai-machine-learning/elastic-training-and-optimized-checkpointing-improve-ml-goodput)).

**Resume:** on FL-server start, if a checkpoint exists for `project_id`, hydrate `global_params_flat`, `seed_history`, `gradient_history`, `client_last_round` from the ledger and set `current_round = last_committed + 1`. The fixed `range(1, num_rounds+1)` loop in `server.py:137` must become `range(resume_round, num_rounds+1)`.

> **Verdict — checkpointing subsystem: rebuild** (it does not exist). One-line: *there is zero per-round persistence; build a ledger + object-store artifact write into the existing per-round aggregation seam.*

### 3.2 Round-recovery & graceful degradation (fix the wedge)

Three changes, all in the coordinator/server loop:

1. **Straggler deadline + min-quorum.** Replace the "wait for exactly `clients_per_round`" gate (`coordinator.py:80,253`) with: complete the round when **either** all expected clients reported **or** a per-round `round_deadline` elapses **and** `len(updates) >= min_fit_clients`. `min_fit_clients` already exists (`strategy.py` / `decomfl_strategy.py:53`) but is not used as a quorum floor in the loop. This makes F5 survivable: a dead client degrades the round instead of hanging it. (FedAvg weights renormalize naturally; DeComFL averages over `num_clients` actually received — `_calculate_average_gradients:326` already divides by `len(results)`, so partial quorum is mathematically fine.)
2. **Wire the dead-client check.** Call `get_active_clients()` / `is_client_alive()` (currently dead, `coordinator.py:169-198`) from the round loop to drive the deadline decision and to emit a `client_dropped` event. Connect *prior 03-framework H1*: a `threading.Event` set on N consecutive heartbeat failures should let the **client** abandon a doomed round rather than upload a stale result.
3. **Make `should_stop` real** (`coordinator.py:165`). On `/stop`, on `stop_requested`, or on a client that must abandon (server has moved on), set `should_stop=True` so the client drains gracefully instead of being `destroyForcibly()`-orphaned. This is the cooperative-shutdown channel that today is hardcoded off.

> **Verdict — round-recovery / degradation: rebuild.** One-line: *the liveness machinery exists but is unwired; the round loop has no timeout or quorum, so any single straggler is a run-killer.*

### 3.3 FL-server HA & orchestration (replace the `ProcessBuilder` model)

The `ConcurrentHashMap<UUID,Process>` + `ProcessBuilder` model is the structural source of F3 and F4 and is the platform's stated scaling cliff (*prior 01-backend, memory 231/1387*). v2 by tier:

- **Seed tier (single host, today + hardening):** keep spawn-as-process **but make the JVM a stateless supervisor over durable state.**
  - **Reconciliation on boot:** add an `ApplicationReadyEvent` listener in `FlowerServerManager` that scans `Project.status='RUNNING'` rows, probes each recorded `serverPort` (and/or checks a `pid`/`lease` table), and either re-adopts a live process or marks the project `INTERRUPTED` and triggers checkpoint-resume. This directly fixes F4 (phantom-RUNNING + orphans).
  - **Persist the process registry:** replace the pure in-memory map with a DB-backed lease (`fl_server_instances(project_id, host, port, pid, lease_expires_at)`), refreshed by a heartbeat. The map becomes a cache, not the source of truth.
  - **Health probe, not a 3s sleep:** replace `waitFor(3s)` (F3) with an actual readiness check — poll the gRPC `GetServerStatus` (the RPC already exists, proto `GetServerStatusRequest`) until it returns `WAITING_FOR_CLIENTS` or a timeout. "Booted and serving" beats "didn't crash in 3s."
  - **Apply prior C4 fully:** `@Transactional` + pessimistic lock on the project row in `ProjectService.startServerForProject:147`, and `putIfAbsent` at `FlowerServerManager:200`.
- **Series-A tier (HA orchestration):** move FL servers onto **Kubernetes JobSet** (or a thin custom Operator) — one Job per run, with **Pod Failure Policy** to separate retriable (OOM, node drain → restart-from-checkpoint) from terminal (config error → fail the run) faults ([Pod Failure Policy](https://kubernetes.io/blog/2024/08/19/kubernetes-1-31-pod-failure-policy-for-jobs-goes-ga/), [JobSet](https://jobset.sigs.k8s.io/docs/overview/)). The backend issues a declarative Job spec instead of `pb.start()`; the FL pod resumes from its checkpoint (§3.1) on restart. This is the proper successor to the ECS-Fargate path that already exists half-built in `startEcsFargateServer():103` (note: that path currently returns `Optional.empty()` and is **not tracked in any map at all** — it has *no* lifecycle management, so on the Fargate path `/stop` and shutdown teardown do nothing; F4 is *worse* on Fargate today).
- **The Overseer question:** a FLARE-style hot/standby *FL-server* is **overkill for a startup** — a single FL server per run, made cheaply resumable via checkpoints, recovers in seconds and costs nothing while idle. The HA that matters is **backend (control-plane) HA + durable checkpoints**, not warm-standby FL servers. Reserve hot/standby for the hyperscale tier or a specific enterprise SLA. (This is where FedLearn can be *leaner* than FLARE: DeComFL's tiny state makes cold-resume fast enough that warm standby is unjustified spend.)

> **Verdict — FL-server orchestration: refactor (seed) → rebuild (Series-A).** One-line: *spawn-as-JVM-child is salvageable only if the JVM becomes stateless over a durable registry + checkpoints; at scale, replace with k8s JobSet + Pod Failure Policy.*

### 3.4 Disaster recovery

- **Model artifacts → object storage with versioning** (S3 versioned bucket / MinIO). Stop the destructive in-place overwrite (`fl_server.py:545`). RPO = one round; RTO = checkpoint-load time (seconds–minutes for non-LLM, minutes for LLM).
- **Control-plane DB:** migrate off single-file H2 to **Postgres with PITR / automated snapshots** (RDS or equivalent) — defer the *engine* choice to *B6 scale-cost*, but the **DR requirement is non-negotiable** because `RoundResult` + `audit_events` + the proposed `fl_server_instances` lease table become recovery-critical.
- **Checkpoint ledger** co-located with model artifacts in object storage; it is the true DR asset (kilobytes, replayable).
- **Documented restore runbook** + a periodic **restore drill** (restore latest checkpoint into a fresh FL server, verify `model_hash` matches) — backups you never test are not backups.

> **Verdict — DR: rebuild.** One-line: *no off-host copy, destructive model writes, single-file DB; needs object-store artifacts + Postgres PITR + a tested restore path.*

### 3.5 SLO/SLI for long-running FL jobs

Two SLO families — **platform** (control plane) and **per-run** (the thing customers actually care about). The latter is the differentiator and maps to *04-observability* dashboards B/C.

**Platform (control-plane) SLOs**
| SLI | Definition | Proposed SLO (Series-A) |
|---|---|---|
| API availability | `1 - 5xx_rate` on `/api/**` | 99.5% monthly |
| Run-start success | started servers passing readiness probe / start attempts | ≥ 99% |
| STOMP log-stream availability | live-log delivery uptime | 99% (best-effort; never in FL critical path — F7) |

**Per-run (FL-job) SLOs — the headline**
| SLI | Definition (source) | Proposed SLO |
|---|---|---|
| **Round goodput** | `rounds_committed / (rounds_committed + rounds_lost_to_faults + recovery_rounds)` — FL analogue of ML Goodput ([Goodput](https://cloud.google.com/blog/products/ai-machine-learning/goodput-metric-as-measure-of-ml-productivity)) | ≥ 95% |
| **Run completion** | runs reaching `num_rounds` (or convergence) without manual intervention | ≥ 99% (with checkpoint-resume counted as success) |
| **Recovery time (RTO)** | wall-clock from fault detection → resumed at last checkpoint | ≤ 2 min (non-LLM), ≤ 10 min (LLM) |
| **Round-progress lost (RPO)** | committed rounds discarded by a fault | ≤ 1 round |
| **Straggler-bounded round latency** | p95 round duration vs `round_deadline` | p95 ≤ deadline; ≤ 5% rounds hit deadline |
| **Client churn tolerance** | rounds completed with quorum despite dropouts / total rounds | ≥ 99% |

**Error budget policy:** burning the round-goodput budget for a customer's run pages on-call and **freezes risky deploys** (the F4 redeploy-loses-the-run failure is itself a budget burner — this couples deploy hygiene to FL reliability). All of these are computable from the *prior 04-observability* metric inventory once the dead `RoundResult` pipeline is wired and the new `rounds_lost_total{reason}` / `recovery_rounds_total` counters are added.

---

## 4. Decision table (verdicts)

| Module / subsystem | Verdict | One-line rationale |
|---|---|---|
| Per-round checkpoint / resume | **rebuild** | Does not exist; DeComFL state is tiny and the per-round seam is already there — build a durable ledger + object-store artifact. |
| DeComFL `rebuild_model` replay (client) | **salvage** | Correct primitive and bandwidth-cheap; salvage as-is but back it with durable server-side history and a model-hash integrity check. |
| `seed_history`/`gradient_history` persistence | **rebuild** | In-memory dicts; server restart wipes replay state and silently diverges reconnecting clients. |
| Round loop (`server.py` fixed range + `wait_for_round_to_complete`) | **rebuild** | No timeout, no quorum; one straggler hangs the run forever; dead liveness machinery. |
| `FlowerServerManager` local-spawn lifecycle | **refactor** | Keep spawn-as-process at seed tier but make JVM stateless over a durable registry + add boot reconciliation, readiness probe, full C4 fix. |
| `FlowerServerManager` ECS-Fargate path | **rebuild** | Fire-and-forget `runTask` with no tracking, no stop, no reconciliation — worse than the local path; replace with JobSet/Operator + checkpoint-resume. |
| JVM in-memory `runningServers` map as source of truth | **kill** | Single point of amnesia (F4); replace with a DB-backed lease table; map becomes a cache only. |
| `should_stop` / cooperative shutdown channel | **refactor** | Wire exists end-to-end but server hardcodes `False`; make it real for graceful drain. |
| Destructive in-place model save (`fl_server.py:545`) | **kill** | Overwrites input, no versioning, no off-host copy; replace with versioned object-store artifacts. |
| Single-file H2 DB for run state | **rebuild** | No PITR/backup; becomes recovery-critical once lease + results tables land — move to Postgres with snapshots. |
| Terminal-only results POST (`fl_server.py:561-587`) | **refactor** | Move per-round POST into the per-round aggregation seam so a crash doesn't lose all metrics. |
| SLO/SLI definitions for FL runs | **rebuild** | None exist; adopt round-goodput / RTO / RPO / churn-tolerance with an error-budget deploy freeze. |

---

## 5. Prioritized recommendations

**P0 — stop losing runs (weeks, mostly framework + one backend seam)**
1. Per-round **durable ledger** (`seeds`, `avg_gradients`, `client_last_round`, `loss`, `model_hash`) written under `_lock` before round advance in `_trigger_decomfl_aggregation_and_evaluation` (`coordinator.py:300`). Fixes F1 + F2.1.
2. **Resume on boot**: hydrate strategy state + `current_round` from the ledger; change `server.py:137` range to start at `resume_round`.
3. **Backend boot reconciliation** (`ApplicationReadyEvent` in `FlowerServerManager`): reconcile `status='RUNNING'` rows ↔ live processes; mark orphans `INTERRUPTED` and resume. Fixes F4.
4. **Round deadline + min-quorum** in the round loop; wire `is_client_alive`. Fixes F5.

**P1 — make recovery trustworthy & DR real**
5. **Versioned model artifacts to object storage**; kill the in-place overwrite. Fixes F6.
6. **Model-hash + strategy-config-hash** in the ledger and a client-side replay integrity check. Hardens F2.4.
7. **Readiness probe** via `GetServerStatus` replaces the 3s sleep; **full C4 fix** (`@Transactional` + row lock + `putIfAbsent`). Fixes F3.
8. **Make `should_stop` real** + the heartbeat-failure `threading.Event` (joins *prior 03-framework H1*). Cooperative drain.

**P2 — scale & SLO maturity (Series-A)**
9. **DB-backed FL-server lease table**; kill the in-memory-map-as-truth.
10. **k8s JobSet + Pod Failure Policy** orchestration replacing `ProcessBuilder`/ECS-fire-and-forget; FL pods resume from checkpoint.
11. **Postgres + PITR**; tested restore drill (restore checkpoint → verify `model_hash`).
12. **SLO instrumentation**: `rounds_lost_total{reason}`, `recovery_rounds_total`, round-goodput; error-budget deploy-freeze policy. Builds on *04-observability*.

---

## 6. Open uncertainties (flagged, not papered over)

- **Replay-compute ceiling on mobile.** I have *not* benchmarked `rebuild_model` cost for large `missed_rounds × K × P × model_dim` on the `fed-mobile` C++ libtorch path. The dimension-free claim is about *communication*, not *reconstruction compute*; for LLM-scale models on edge clients this may need a **periodic full-model anchor checkpoint** the client can warm-start from to bound replay depth. Cross-check with *A6 mobile* and *B1 paper-alignment*.
- **Server/client RNG-device parity.** `_generate_perturbation` uses `self.device` server-side (`decomfl_strategy.py:212`); `torch.randn` with a `Generator` is documented as reproducible **per device**, but I have not verified bit-exact parity across CUDA server vs CPU/Metal/Android client. If it diverges, replay (and rebuild integrity) is unsound. Needs an explicit cross-device determinism test (overlaps *C3 reproducibility*).
- **safetensors coverage** for every model class FedLearn federates — assert at build time before committing to it as the artifact format.
- **Async coordinator.** `async_coordinator.py` (with commented-out `pika`/RabbitMQ) hints at a planned async/queue path; I assessed the **active synchronous** coordinator. If the async path is revived, its reliability model (at-least-once delivery, dedupe) must be re-audited separately.

---

## 7. Sources

- NVIDIA FLARE — High Availability & Server Failover: https://nvflare.readthedocs.io/en/2.6/programming_guide/high_availability.html
- Flower: A Friendly Federated Learning Framework (paper): https://arxiv.org/pdf/2007.14390
- Flower federated fine-tuning + per-round checkpointing: https://medium.com/mitb-for-all/how-to-train-your-llm-simultaneously-with-10-different-teachers-ba556d4ed2c1
- An Efficient Checkpoint Strategy for FL on Heterogeneous Fault-Prone Nodes (Electronics 2024, 13(6):1007): https://doi.org/10.3390/electronics13061007
- Google Cloud — Goodput as a measure of ML productivity: https://cloud.google.com/blog/products/ai-machine-learning/goodput-metric-as-measure-of-ml-productivity
- Google Cloud — Elastic training & optimized checkpointing improve ML Goodput: https://cloud.google.com/blog/products/ai-machine-learning/elastic-training-and-optimized-checkpointing-improve-ml-goodput
- Nebius — Building reliable clusters for distributed AI workloads: https://nebius.com/blog/posts/how-we-build-reliable-clusters
- Kubernetes JobSet (k8s-native distributed ML/HPC API): https://jobset.sigs.k8s.io/docs/overview/
- Kubernetes — Pod Failure Policy for Jobs GA (1.31): https://kubernetes.io/blog/2024/08/19/kubernetes-1-31-pod-failure-policy-for-jobs-goes-ga/
- Kubernetes v1.35 — in-place Pod restart: https://kubernetes.io/blog/2026/01/02/kubernetes-v1-35-restart-all-containers/
