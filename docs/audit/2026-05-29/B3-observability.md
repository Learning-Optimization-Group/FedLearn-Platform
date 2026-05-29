# B3 — Observability (Platform + FL-Run / ML)

**Date:** 2026-05-29 · **Branch:** `main-clean` · **Scope:** Greenfield v2 observability design for a production-grade startup.
**Builds on:** `docs/audit/2026-05-27/04-observability.md` (cited inline as **[prior §N]**). This report does **not** restate the prior inventory; it re-verifies the load-bearing claims, corrects two, adds the mobile + DeComFL + dead-code dimensions the prior pass missed, and converts the recommendations into a calibrated v2 stack with phasing, cost, and verdicts.

---

## 0. TL;DR for the synthesizer

1. **The single highest-leverage fix is a 30-line Python HTTP callback.** The entire FL-run telemetry pipeline already exists end-to-end *except the producer*: `RoundResult` entity → `ResultsController` POST `/api/internal/results/{projectId}` → STOMP `/topic/results/{projectId}` → `recharts` loss/accuracy chart in `ResultsModal.tsx`. The coordinator already computes the metrics (`coordinator.py:108` `self.latest_metrics = {"loss": loss, **metrics}`) and the backend already passes the callback URL + key to the Python child (`FlowerServerManager.java:188-191`). **Nobody POSTs.** The dashboard chart is permanently empty. This is the user's emphasized ask and it is ~1 day of work. **Verdict: salvage the pipeline, build the missing caller.**
2. **Platform observability is at zero-and-a-half.** Backend has `spring-boot-starter-actuator` but **no Micrometer registry** (`build.gradle:31`), exposing only `health,loggers` (`application.properties:53`). Framework has 6 `opentelemetry-*` packages + `prometheus_client` pinned (`requirements.txt:37-44`) and **imported in zero `src/` files** (verified). These are paid-for, unused dependencies. **Verdict: refactor (wire what's pinned).**
3. **There is no correlation ID anywhere.** No `traceparent` crosses JVM → Python → gRPC client → mobile. Build W3C Trace Context propagation; the gRPC metadata channel is the natural carrier (the contract is `fedlearn.v1`).
4. **Mobile is a fully siloed observability island.** The C++ core computes `loss`/`accuracy`/`step` locally (`FederatedLoop.cpp:49-54`) and renders them to an in-app `MetricsDisplay.jsx`, but **no telemetry returns to the platform.** Server-side dashboards cannot see mobile client health.
5. **For DeComFL specifically, "bytes transferred" is the headline KPI** (the paper's entire thesis is O(K·P) communication independent of model dimension). The existing `RoundResult` schema has *no* communication-cost column. Without it, the platform cannot demonstrate its own differentiator.
6. **Correction to prior audit:** the prior report instrumented `coordinator.py` line numbers (`start_round` at 36-40, aggregation at 87-119) that do not match the current file, and did not flag that `async_coordinator.py` (RabbitMQ/`pika`) is **dead code** (commented out at `server.py:8-10`; no `pika` in `requirements.txt`). Instrumenting the live synchronous path (`coordinator.py` + `grpc_servicer.py`) is correct; do not instrument the RabbitMQ path.

---

## 1. Re-verification of the prior audit's claims (and two corrections)

| Prior claim | Status | Evidence (this pass) |
|---|---|---|
| Backend exposes only `health,loggers`; Micrometer not on classpath | **CONFIRMED** | `application.properties:53` `management.endpoints.web.exposure.include=health,loggers`; `build.gradle:31` only `spring-boot-starter-actuator`, no `micrometer-registry-*`. |
| Framework OTel/Prometheus pinned but imported nowhere | **CONFIRMED** | `requirements.txt:37-44` (`opentelemetry-api/sdk/exporter-prometheus/proto/semantic-conventions==…`, `prometheus_client`, `opencensus`); `grep -rn "import opentelemetry\|import prometheus\|import structlog\|import mlflow\|import wandb" src/` → **0 hits**. |
| `RoundResult` pipeline exists end-to-end but no Python caller POSTs | **CONFIRMED + EXTENDED** | `ResultsController.java:38-59` writes the row + `webSocketService.sendResultUpdate`; `WebSocketService.java:63-67` rebroadcasts on `/topic/results/{id}`; **the consumer also exists** — `frontend/.../redesign/ResultsModal.tsx:21,39-46` plots `loss`/`accuracy` via `recharts`, `DashboardV2.tsx:153` subscribes `/topic/results/*`. `grep` for any HTTP caller to `internal/results` in `framework/` + `client-docker/` → **0 hits.** |
| Coordinator computes metrics but they're discarded | **NEW — CONFIRMED** | `coordinator.py:108,291` sets `latest_metrics`; `server.py:151` reads `get_latest_metrics()` and only `logging.info(...)`-s it into a `history` list discarded at process end (`server.py:153-165`). The data exists in-process; it is never exported. |
| Callback plumbing is already wired | **NEW — CONFIRMED** | `FlowerServerManager.java:188-191` injects `FEDLEARN_INTERNAL_API_KEY` + `FEDLEARN_BACKEND_URL`; re-added in `buildEnvOverrides` at `:386-388`. The Python child has everything it needs to POST today. |
| Prior instrumentation line numbers for `coordinator.py` | **CORRECTION** | Prior report cited `start_round()` at 36-40 and aggregation at 87-119. Current file: `start_round` begins at `coordinator.py:34`; the synchronous aggregation path runs through `_trigger_aggregation_and_evaluation` referenced from `coordinator.py:100-114` and a second DeComFL path at `:272-298`. Re-anchor instrumentation to the live file before implementation. |
| RabbitMQ async path | **NEW — DEAD CODE** | `async_coordinator.py` exists (RabbitMQ `ResultConsumer`) but is commented out at `server.py:8-10`; no `pika`/`amqp` in `requirements.txt`. Do **not** build observability for it; flag for removal (cross-ref A3/B2). |

**Net:** the prior audit's core diagnosis holds and its stack table is largely right. This report keeps the prior stack, corrects the line anchors, and adds the four dimensions it missed (mobile island, DeComFL comms KPI, dead RabbitMQ path, the *consumer-already-exists* fact that makes the producer fix a 1-day win rather than a feature).

---

## 2. Current state, condensed (delta over prior §1)

- **Backend:** SLF4J/Logback, profile-gated patterns (prior §1 still accurate). No MDC, no `traceparent`, no Micrometer. `WebSocketService.persistLog` (`:91`) writes a `ServerLog` row per stdout line, feature-flagged, **unbounded growth** (prior §7.5 — still true).
- **Framework:** root-logger `JSONFormatter` (`server.py:20-35`) emits `{"timestamp","level","message","stackTrace"?}` — but **carries no `project_id`, `round_idx`, `client_id`, or `traceparent`**. The backend log-persistence parser keys on exactly this shape, so any enrichment must be additive.
- **Frontend:** `logger.ts` is a `console.*` chokepoint explicitly designed for "later wiring to Sentry / Datadog RUM" (`logger.ts:7`). `recharts` present; `ResultsModal` ready; no RUM, no `web-vitals`, no client-side `traceparent`.
- **Desktop:** `electron-log` with rotation + local-only crash dumps (prior §1 accurate).
- **Mobile (`origin/fed-mobile`):** native C++ DeComFL core. `FederatedLoop.cpp:49-54` computes/holds `loss`, `accuracy`, `step`, `total`; surfaces via `setStatus` to RN `MetricsDisplay.jsx`. **No outbound telemetry to the platform** — not over gRPC metadata, not over HTTP. The proto carries training params but no metrics-report RPC for the server to record client health.

---

## 3. The two layers, and why they must stay separate

The user's emphasis is **FL-run observability** ("observability of the projects/runs users create"). That is a *product* surface for the researcher/operator who launched a run. It is **not** the same as platform SRE observability, and conflating them is the classic mistake (prior §7.6 flagged the "split-brain" risk). Concretely:

| | **FL-run / ML observability** (product) | **Platform observability** (SRE) |
|---|---|---|
| Audience | Run owner, researcher, the customer | On-call engineer, you |
| Question | "Is *my run* converging? Which client is dragging it? How much did it cost in bytes?" | "Is the *platform* healthy? p99 latency, JVM heap, are FL processes leaking?" |
| Time horizon | Run lifetime + permanent comparison history | 15–30d rolling |
| Backend | **MLflow** (history) + **STOMP live feed** (now) | **Prometheus** (metrics) + **Loki** (logs) + **Tempo** (traces) |
| Cardinality | per-run, per-client (high — push to MLflow, not Prometheus) | per-endpoint, per-service (low) |

These are different stores on purpose. Prometheus is wrong for per-client-per-round ML history (cardinality explosion — prior §7.1 correctly flagged this). MLflow is wrong for live p99 alerting. The contract belongs documented on the run dashboard's first panel (prior §7.6).

---

## 4. v2 observability stack (one choice per layer, with rationale + cost)

This keeps the prior §3 table where it was right and tightens the ML layer.

| Layer | Choice | Rationale | Why not the alternative |
|---|---|---|---|
| Metrics backend | **Prometheus** (pull/scrape) | FL servers are short-lived per-project processes; Python `prometheus_client` already pinned. | Push-gateway only for the truly ephemeral; see §6 risk. |
| Dashboards | **Grafana** | Single pane over Prom + Loki + Tempo; FL run dashboards as committed JSON. | — |
| Logs | **Grafana Loki** + **Alloy** (successor to Promtail) | Cheap, label-indexed; matches the existing JSON log shape. | ELK/OpenSearch = JVM + storage overkill at startup scale. |
| Traces | **OpenTelemetry** (W3C Trace Context) → **Tempo** | One propagation standard across JVM/Python/TS; gRPC metadata carries `traceparent` natively ([OTel gRPC metadata](https://oneuptime.com/blog/post/2026-02-06-grpc-metadata-trace-propagation/view)). | — |
| Spring metrics | **Micrometer** + `micrometer-registry-prometheus` on the existing actuator | Idiomatic; zero-config HTTP/JVM/JDBC/STOMP metrics; exposes `/actuator/prometheus`. | — |
| Spring tracing | **`opentelemetry-javaagent.jar`** (no code change) | Auto-instruments Servlet, JDBC, **gRPC**, RestTemplate; the agent injects `traceparent` for free. | Manual SDK = more code, same result. |
| Python metrics | **`prometheus_client`** (already pinned) | Direct histogram/counter API; `/metrics` on `gRPC_port + 1000`. | — |
| Python tracing | **`opentelemetry-sdk`** + **`opentelemetry-instrumentation-grpc`** (already pinned family) | Server interceptor extracts `traceparent` from the JVM-spawned env; client/servicer spans chain into the JVM trace. | — |
| Python logging | **`structlog`** with stdlib bridge, **keeping the current JSON keys** | Bind `project_id`/`round_idx`/`client_id`/`trace_id` contextually; backend log parser stays compatible (additive keys). | — |
| **Experiment tracking** | **MLflow (self-hosted)** as v2 floor; **W&B optional managed add-on** later | MLflow is Apache-2.0, fully self-hostable, $0 licence, infra ~$100–500/mo small-team ([ZenML](https://www.zenml.io/blog/mlflow-vs-weights-and-biases), [Uplatz](https://uplatz.com/blog/the-2025-mlops-landscape-a-comparative-analysis-of-mlflow-weights-biases-and-neptune/)). For a startup that may run **healthcare/pneumonia** federations, self-hosted (data never leaves VPC) is a near-requirement. | **W&B** self-hosting is enterprise-only ($2k–5k/mo Dedicated Cloud); great UX but vendor lock-in + cost + data-residency friction this early ([ZenML](https://www.zenml.io/blog/weights-and-biases-alternatives)). Keep W&B as a per-customer opt-in, not the floor. |
| Frontend RUM | **Sentry (self-hosted or SaaS dev tier)** + `web-vitals` via `logger.ts:7` chokepoint | The chokepoint is already there by design. | — |
| Desktop errors | **`@sentry/electron`** (main + renderer); leave `electron-log` | Sentry hooks electron-log; preserves local dumps. | — |
| **Mobile telemetry** | **OTel C++ / lightweight metrics-report RPC** + Sentry React Native | Mobile is currently invisible to the platform (§3, §7). | — |
| Transport | **OpenTelemetry Collector** (OTLP gRPC) → fan-out to Prom/Loki/Tempo | One pipeline, swappable backends, no vendor lock-in. | — |

### Why MLflow over a custom store (vs prior, which only asserted it)
The prior audit listed MLflow without defending it against "just use Postgres + the RoundResult table." The defense: the `RoundResult` table is the right place for the *live operational* per-round row (it feeds STOMP), but it cannot answer "compare run A vs run B's hyperparameters and artifacts" without you re-implementing the model registry, parameter logging, artifact store, and run-comparison UI that MLflow gives for free. For a startup whose product *is* FL experiments, buying that surface for $0 (Apache-2.0) is correct. Keep both: `RoundResult` = live feed, MLflow = comparison/lineage. (Cross-ref **C3 reproducibility** — MLflow is also the run-lineage substrate there.)

---

## 5. Correlation IDs across JVM → Python → gRPC client → mobile (the missing thread)

This is the part the platform has *nothing* of today and the part that makes incident triage possible. Design:

```
Browser (Sentry trace) ──HTTP /start──▶ Spring Boot
        traceparent header (OTel javaagent on servlet)
                                              │  span: "start project"
                                              ▼
              FlowerServerManager.startServerForProject
              inject into ProcessBuilder env (FlowerServerManager.java:188 area):
                  TRACEPARENT = <current span context, W3C format>
                  OTEL_RESOURCE_ATTRIBUTES = service.name=fl-server,project.id=<uuid>
                  (PROJECT_ID already present at :375)
                                              │
                                              ▼
              python fl_server.py  ── OTel SDK extracts TRACEPARENT from env
                  root span "fl-run {project_id}" parented to the JVM span
                                              │ gRPC server interceptor
                                              ▼  traceparent in gRPC metadata
              gRPC client (desktop / docker / mobile C++)
                  client interceptor injects/continues traceparent
```

Concrete edits (re-anchored to current files):
- **`FlowerServerManager.java`** — in the env-build block (`:188-191`, mirrored in `buildEnvOverrides` `:372-388`) add `TRACEPARENT` (serialize the active span via the OTel `TextMapPropagator`) and `OTEL_RESOURCE_ATTRIBUTES`. `PROJECT_ID` already there.
- **`server.py:20-35`** — replace ad-hoc `JSONFormatter` with structlog JSON renderer; at startup, `extract()` `TRACEPARENT` from `os.environ` into the OTel context so the run's root span chains to the JVM span. Bind `project_id` (from `PROJECT_ID`) into the logging context.
- **`grpc_servicer.py`** — register `opentelemetry.instrumentation.grpc` **server interceptor before `grpc.server(...)` is constructed** (`server.py:71` area). Prior §7.4 correctly flagged the import-order no-op; keep that warning. gRPC metadata is the W3C-standard carrier ([OTel docs](https://opentelemetry.io/docs/concepts/context-propagation/)).
- **Mobile** — the C++ gRPC client (`FederatedLoop.cpp`) currently sends no trace context. v2: add an interceptor that injects `traceparent` into outgoing metadata so a mobile client's rounds appear inside the same server trace.
- **gRPC-plaintext caveat (audit item #37):** `traceparent` and any baggage travel over the same insecure channel as weights. Fine for a Tailscale-meshed demo; document alongside the gRPC-TLS item and do not put PII in baggage. (Cross-ref **B4 security**.)

**Result:** one `trace_id` joins a browser click → the spawned process → every client RPC → mobile, queryable in Tempo and stamped on every Loki log line.

---

## 6. FL-run / ML observability — the emphasized layer, designed concretely

### 6.1 Three delivery channels, each for its purpose (extends prior §4.3)

1. **Live feed → STOMP `/topic/results/{projectId}` (build the missing producer — P0).**
   Add an HTTP callback in the framework. At the end of each round's aggregation/eval (the live path around `coordinator.py:100-114`, DeComFL path `:272-298`), POST the enriched `RoundResultDto` to `${FEDLEARN_BACKEND_URL}/api/internal/results/${PROJECT_ID}` with header `X-Internal-Api-Key: ${FEDLEARN_INTERNAL_API_KEY}`. Both env vars already injected (`FlowerServerManager.java:188-191`). The consumer chart already renders (`ResultsModal.tsx`). **This is the user's ask, ~1 day, no schema migration needed for loss/accuracy.**
   *Implementation note:* use a stdlib `urllib.request` POST (no new dep) or `httpx` if added; wrap in try/except so a telemetry failure never crashes the run. Fire from a non-blocking path so it can't stall `wait_for_round_to_complete`.

2. **Live ops metrics → Prometheus `/metrics` on the FL process.**
   `prometheus_client.start_http_server(grpc_port + 1000)` inside `server.start_server()` after `grpc_server.start()` (`server.py:135` area). Prometheus discovers targets via a static file written by `FlowerServerManager.startLocalServer` listing live `(project_id, port)` pairs (prior §4 channel 1 — keep). Low-cardinality, per-project.

3. **History / comparison → MLflow.**
   `mlflow.set_experiment(project_id)`; `mlflow.log_params(hyperparams)` once at run start; `mlflow.log_metrics({...}, step=round)` per round; log the final model as an artifact. This is the researcher's surface and the **C3 reproducibility** substrate.

### 6.2 Metric inventory (corrected + DeComFL-aware)

Keep prior §4's inventory but **add the communication-cost dimension that DeComFL's entire value proposition depends on**, and split FedAvg vs DeComFL byte accounting:

```
# Convergence (Gauge per project)
fedlearn_round_loss{project_id}
fedlearn_round_accuracy{project_id}

# Round mechanics
fedlearn_round_duration_seconds          Histogram {project_id, strategy}
fedlearn_aggregation_seconds             Histogram {project_id, strategy}
fedlearn_rounds_completed_total          Counter   {project_id, status}
fedlearn_round_clients_active            Gauge     {project_id}

# COMMUNICATION COST — DeComFL's headline KPI (NEW, schema-impacting)
fedlearn_uplink_bytes_total              Counter   {project_id, strategy, direction}
fedlearn_downlink_bytes_total            Counter   {project_id, strategy, direction}
fedlearn_decomfl_scalars_transmitted     Counter   {project_id}   # K*P scalars/round — the O(K*P) claim
fedlearn_model_param_count               Gauge     {project_id}   # P (dimension) — for the dimension-free comparison

# Client telemetry (bound cardinality — see §7)
fedlearn_client_compute_seconds          Histogram {project_id}            # NOT per client_id
fedlearn_client_progress_ratio           Gauge     {project_id, client_id} # bounded by active clients
fedlearn_client_last_heartbeat_age_seconds Gauge   {project_id, client_id}

# gRPC (via OTel instrumentation, free)
fedlearn_grpc_request_duration_seconds   Histogram {project_id, rpc}
```

The **`fedlearn_decomfl_scalars_transmitted` vs `fedlearn_model_param_count`** pair is what lets a dashboard *prove* the paper's claim: scalars/round stays O(K·P) while param count P can be in the millions. The DeComFL strategy already has K (`num_local_steps`) and P (`num_perturbations`) and per-round gradient scalars (`decomfl_strategy.py:58,171`), so the counter increments are trivial to add at the aggregation site. This is the metric that turns the platform's research differentiator into a customer-facing number — neither the prior audit nor the existing schema captured it. **Schema impact:** add a nullable `comm_bytes`/`scalars_transmitted` to `RoundResult` via a new `V6__add_comm_cost_to_round_result.sql` (Flyway-owned, per the invariant).

### 6.3 Client contribution & drift (the harder ML-observability asks)

The user asked for "client contribution and drift." Honest scoping:

- **Contribution (per-client value).** The literature standard is **Shapley-value-based** contribution, but exact SV requires re-running FL over all client subsets — infeasible online ([arXiv 2505.23246](https://arxiv.org/pdf/2505.23246), [arXiv 2502.17526](https://arxiv.org/pdf/2502.17526)). **Recommendation:** v2 ships a *cheap proxy* first — per-client marginal accuracy delta and update-norm contribution logged to MLflow per round — and treats Shapley/Maverick-aware valuation ([arXiv 2405.12590](https://arxiv.org/pdf/2405.12590)) as a **later research feature**, not a launch requirement. **Uncertainty flagged:** for DeComFL the clients send *gradient scalars*, not weight deltas, so update-norm contribution must be defined over the projected-gradient magnitude, not parameter L2. This needs B1's paper-alignment input before implementation — do not fabricate the formula.
- **Drift.** Two distinct things: (a) **client-vs-global drift** (how far a client's update pulls from the aggregate — proxy: cosine of client gradient vs aggregated gradient, computable at the aggregation site for both strategies); (b) **data/concept drift over time** (needs a held-out eval set; ties to **C2 data-engineering**). Ship (a) as a per-round MLflow metric in v2; defer (b).

### 6.4 Mobile — close the island
Add a server-recorded client-health signal: either (i) piggyback `loss`/`accuracy`/`step` (already in `FederatedLoop.cpp:49-54`) onto the existing heartbeat RPC metadata, or (ii) a small `ReportClientMetrics` RPC. Then mobile clients appear in Dashboard C alongside desktop/docker clients. Without this, a customer running a 500-phone federation has zero server-side visibility into device health. (Cross-ref **A6 mobile**, **B1 paper-alignment** for the proto change to `fedlearn.v1`.)

---

## 7. Three concrete dashboards (Grafana JSON-as-code, committed)

### Dashboard A — Platform Overview (SRE audience)
*Source: Micrometer/Prometheus + Loki.*
- Request rate by endpoint — `rate(http_server_requests_seconds_count[1m])`.
- p50/p95/p99 latency — `histogram_quantile(0.99, http_server_requests_seconds_bucket)`.
- 5xx error rate — `rate(http_server_requests_seconds_count{status=~"5.."}[5m])`.
- JVM heap + GC pause — `jvm_memory_used_bytes`, `jvm_gc_pause_seconds`.
- **Active FL processes** gauge — new `fedlearn_active_projects` from `runningServers.size()` (`FlowerServerManager`).
- **Orphaned/leaked process detector** — processes in `runningServers` whose `/metrics` hasn't been scraped in N scrapes (catches the §C1 zombie-process failure mode).
- STOMP session count — `simp_message_broker_sessions` (Spring built-in via Micrometer).

### Dashboard B — Per-Run FL (audience: run owner; Grafana var `$project_id`)
*This is the user's emphasized surface. Source: Prometheus live + MLflow link-out.*
- Rounds completed / target — stat panel.
- **Loss & accuracy convergence curves** — `fedlearn_round_loss` / `fedlearn_round_accuracy` over rounds. (Mirrors `ResultsModal.tsx` so the in-app and Grafana views agree.)
- **Communication cost panel (DeComFL hero)** — `fedlearn_uplink_bytes_total` + `fedlearn_decomfl_scalars_transmitted` plotted against `fedlearn_model_param_count`, with a derived "bytes-per-round vs equivalent FedAvg full-model bytes" stat to *visualize the savings*. This panel is the product's competitive proof.
- Round duration heatmap — `fedlearn_round_duration_seconds_bucket`.
- Comm-vs-compute split — `fedlearn_aggregation_seconds` stacked under `fedlearn_round_duration_seconds`.
- Active clients gauge.
- Live log panel — Loki `{project_id="$project_id"}` (or keep the STOMP stream in-app; pick one — see §8 risk).

### Dashboard C — Client Telemetry (audience: debugging stragglers; vars `$project_id` + `$client_id`)
*Source: heartbeat-driven gauges + MLflow per-client history.*
- Per-client compute time per round (aggregated histogram at project level + per-client gauge).
- Progress ratio over time — heartbeat-driven `fedlearn_client_progress_ratio`.
- **Straggler table** — clients sorted by `fedlearn_client_last_heartbeat_age_seconds` (catches the client that dropped mid-round and is stalling aggregation).
- **Contribution panel** (MLflow link-out) — per-client marginal accuracy delta / gradient-cosine drift (§6.3), with a banner noting these are proxies, not exact Shapley.
- **Mobile vs desktop vs docker split** — once §6.4 lands, `count by (client_type)`.

---

## 8. Risks (extends prior §7; keeps the still-valid ones, adds new)

| # | Risk | Status | Mitigation |
|---|---|---|---|
| 1 | **`client_id` cardinality explosion** in Prometheus | prior §7.1 still valid | Per-client detail → MLflow; Prometheus keeps `client_id` only on bounded heartbeat/progress gauges. |
| 2 | **STOMP backpressure** — log shipping on the stdout-reader daemon thread can block the Python process | prior §7.2 still valid | Bounded queue (drop-oldest) on `convertAndSend`, **or** move bulk logs to Loki and keep STOMP for the low-rate `RoundResult` feed only. Recommend the latter: STOMP carries ≤1 msg/round (sub-Hz), Loki carries the firehose. |
| 3 | **`/actuator/prometheus` accidentally public** over the ALB | prior §7.3 still valid | Separate `management.server.port` bound internal-only; keep out of `SecurityConfig` `permitAll`. |
| 4 | **OTel gRPC instrument-before-server import order** | prior §7.4 still valid | Interceptor registration before `grpc.server(...)`; assert in a smoke test. |
| 5 | **`server_logs` / `audit_events` unbounded growth** | prior §7.5 still valid | TTL job (`DELETE WHERE timestamp < now()-30d`) or partitioning, Flyway-managed, before turning up volume. |
| 6 | **MLflow vs Prometheus split-brain** confuses users | prior §7.6 still valid | Document the contract on Dashboard B's first text panel: Prometheus = "healthy now", MLflow = "compare runs". |
| 7 | **`traceparent` over plaintext gRPC** (item #37) | prior §7.7 still valid | No PII in baggage; document with the gRPC-TLS work (B4). |
| 8 | **`.eml` reset tokens** | prior §7.8 still valid | Profile wiring already enforces SMTP outside dev; add defense-in-depth WARN if dev adapter loads elsewhere. |
| 9 | **Telemetry callback can crash/stall the run** | **NEW** | The POST producer (§6.1) must be best-effort: try/except, short timeout, off the round-completion critical path. A monitoring failure must never fail a training run. |
| 10 | **Instrumenting dead code** (`async_coordinator.py` RabbitMQ path) | **NEW** | Only instrument the live synchronous `coordinator.py`/`grpc_servicer.py` path. Flag the RabbitMQ file for deletion (A3/B2). |
| 11 | **Mobile remains invisible** if §6.4 is skipped | **NEW** | A federation dominated by phones would have zero server-side device health. Make the client-metrics RPC a v2 requirement, not optional. |
| 12 | **Cardinality of `strategy` + `rpc` labels** if more strategies/RPCs are added | **NEW (low)** | Bounded today (2 strategies, ~6 RPCs); keep an eye as the proto grows. |

---

## 9. Phased plan (calibrated to startup runway)

**P0 — Make the emphasized feature actually work (days, not weeks).**
1. Framework HTTP callback → `/api/internal/results/{id}` (loss/accuracy). Dashboard chart lights up. *No new infra.*
2. `V6` Flyway migration: add `comm_bytes` / `scalars_transmitted` to `RoundResult`; extend `RoundResultDto`; populate from the DeComFL aggregation site. The communication-cost story becomes demoable.

**P1 — Platform observability floor.**
3. Micrometer Prometheus registry + `/actuator/prometheus` on an internal `management.server.port`.
4. `prometheus_client` `/metrics` on the FL process + the §6.2 inventory.
5. OTel Java agent + `TRACEPARENT` env propagation through `ProcessBuilder`.
6. structlog enrichment (`project_id`/`round_idx`/`trace_id`) — additive keys.
7. Compose/Helm: Prometheus + Grafana + Loki + Tempo + OTel Collector. Three dashboards as committed JSON.

**P2 — Depth & product polish.**
8. MLflow self-hosted; wire `log_params`/`log_metrics`/artifact at `coordinator.py` round-end.
9. Sentry (frontend via `logger.ts:7`, desktop main+renderer).
10. Per-client contribution proxy + gradient-cosine drift → MLflow (needs B1 sign-off on the DeComFL scalar formula).

**P3 — Mobile + advanced.**
11. Mobile client-metrics RPC + OTel C++ gRPC interceptor (proto change → B1/A6).
12. Shapley/Maverick contribution as a research feature (not launch-blocking).

Effort: P0 ≈ 2–3 days; P1 ≈ 1.5 weeks; P2 ≈ 1 week; P3 ≈ 1 week + proto coordination. Roughly the prior audit's 3–4 week estimate, reordered so the user's emphasized win lands first.

---

## 10. Verdicts on existing observability assets

| Asset | Verdict | One-line rationale |
|---|---|---|
| `RoundResult` → `ResultsController` → STOMP → `ResultsModal` pipeline | **salvage** | Fully built end-to-end *including the recharts consumer*; only the Python producer is missing — a ~1-day fix, highest leverage on the board. |
| `coordinator.latest_metrics` (`coordinator.py:108,291`) | **salvage** | Metrics already computed in-process; just export them via the §6.1 callback. |
| Framework `opentelemetry-*` + `prometheus_client` pinned deps | **refactor (wire them)** | Paid-for, imported nowhere; the v2 plan activates exactly these versions. |
| Backend Actuator (`health,loggers` only) | **refactor** | Add Micrometer registry + Prometheus endpoint on an internal port; keep the profile-gated logging. |
| Framework root-logger `JSONFormatter` (`server.py:20-35`) | **refactor** | Keep the JSON shape (backend parser depends on it); migrate to structlog and add correlation keys. |
| `WebSocketService.persistLog` → `ServerLog` table | **refactor** | Useful, but unbounded; add TTL/partitioning and consider Loki for the firehose, STOMP for the low-rate result feed. |
| `async_coordinator.py` (RabbitMQ `ResultConsumer`) | **kill** | Dead code — commented out at `server.py:8-10`, no `pika` dependency; do not build observability around it. |
| Frontend `logger.ts` chokepoint | **salvage** | Purpose-built for RUM wiring (`:7`); plug Sentry + web-vitals here. |
| Desktop `electron-log` + local crash dumps | **salvage** | Keep; add Sentry hooks for remote shipping. |
| Mobile telemetry path | **rebuild (build new)** | None exists server-side; the C++ core has the data (`FederatedLoop.cpp:49-54`) but no pipeline back — design the client-metrics RPC. |
| No correlation-ID system | **rebuild (build new)** | Nothing today; design W3C Trace Context JVM→Python→gRPC→mobile per §5. |
| Experiment tracking | **rebuild (build new = MLflow)** | None today; MLflow self-hosted is the $0/Apache-2.0/data-resident floor; W&B a later managed opt-in. |

---

## 11. Sources

- Prior audit: `docs/audit/2026-05-27/04-observability.md`.
- MLflow vs W&B (open-source/self-host/pricing): [ZenML — MLflow vs W&B](https://www.zenml.io/blog/mlflow-vs-weights-and-biases), [ZenML — W&B alternatives](https://www.zenml.io/blog/weights-and-biases-alternatives), [Uplatz 2025 MLOps landscape](https://uplatz.com/blog/the-2025-mlops-landscape-a-comparative-analysis-of-mlflow-weights-biases-and-neptune/).
- OTel context propagation / gRPC metadata: [OpenTelemetry — Context propagation](https://opentelemetry.io/docs/concepts/context-propagation/), [gRPC metadata trace propagation](https://oneuptime.com/blog/post/2026-02-06-grpc-metadata-trace-propagation/view).
- FL client contribution / Shapley feasibility: [arXiv 2505.23246](https://arxiv.org/pdf/2505.23246), [arXiv 2502.17526 (FedSV)](https://arxiv.org/pdf/2502.17526), [arXiv 2405.12590 (Maverick-aware SV)](https://arxiv.org/pdf/2405.12590).
- Codebase evidence cited inline by `file:line` against `main-clean` and `origin/fed-mobile`.
