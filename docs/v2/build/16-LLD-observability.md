# 16 — Low-Level Design (LLD): Observability Stack

**Document type:** Production build specification — Low-Level Design (LLD) for one unit.
**Unit:** the **observability stack** — two layers: (1) the platform Site-Reliability-Engineering (SRE) layer (Micrometer → Prometheus on an internal management port; Grafana + Loki + Tempo + OpenTelemetry (OTel) Collector; `structlog` structured Python logs; World-Wide-Web-Consortium (W3C) `traceparent` propagated across the Java Virtual Machine (JVM) → spawned Python → gRPC client → mobile), and (2) the Federated-Learning-run / Machine-Learning (FL-run / ML) layer (per-round and per-client training telemetry, convergence curves, communication-cost tracking, MLflow experiment tracking, the incremental per-round `RoundResult` Hypertext-Transfer-Protocol (HTTP) callback).
**Audience:** a mid-sized local Large Language Model (LLM, ~30 billion parameters, e.g. Qwen/Llama 32B on an Apple M4 Max). Every interface, metric name, span name, log field, environment variable, command, and failure path below is **pre-decided**. Implement the bodies; do not redesign the contracts or invent metric/span/topic names not listed here.
**Status:** build-authoritative for v2 (version 2). Conforms to and never contradicts the foundation docs: `01-ARCHITECTURE-HLD.md`, `02-TECH-STACK.md`, `03-DATA-MODEL.md`, `04-API-CONTRACTS.md`. Where this LLD references those docs it cites them by exact section.
**Date authored:** 2026-05-29.
**Source of truth:** the v2 audit synthesis at `/home/anurag/codebase/FedLearn-Platform/docs/audit/2026-05-29/README.md`, plus the assigned depth reports `B3-observability.md`, `B7-standards-dx.md`, `C1-reliability-sre.md` in the same directory. Every existing-code claim cites `path:line` against `main-clean`; every external/market claim cites a source Uniform Resource Locator (URL); pinned versions come from `02-TECH-STACK.md §20`.

---

## 0. How to read this document (and the abbreviation key)

The first time an acronym appears it is written in full followed by the short form in parentheses; thereafter the short form is used. The complete glossary is at the end of this section.

**Normative keywords:** "MUST" = a build constraint no code may violate; "SHOULD" = a strong default; "MUST NOT" = forbidden. These mirror the foundation docs.

**This unit's relationship to the foundation docs.** This LLD is the **only** document that specifies the concrete metric names, span names, log field names, the three Grafana dashboard layouts, the `docker-compose` for the local observability stack, the OTel Collector pipeline configuration, and the wiring of the pinned-but-unused `opentelemetry-*`/`prometheus_client` Python dependencies (`B3-observability.md:11,24`). It **consumes** the contracts already locked in `04-API-CONTRACTS.md`:
- §5 — the internal callback `POST /api/internal/runs/{runId}/results` and its `RoundResultDto` shape (the incremental per-round producer this unit owns).
- §11 — the STOMP (Simple Text Oriented Messaging Protocol) topics `/topic/results/{projectId}`, `/topic/logs/{projectId}`, `/topic/status/{projectId}`, `/topic/runs/{projectId}` and their payload shapes (which carry `traceId`).
- §12 — the standard error envelope (carries `traceId`).
- §13 — the per-run scoped token and its env var injection (`FEDLEARN_RUN_TOKEN`, `TRACEPARENT`).
- §14 — the W3C `traceparent` propagation contract (the carrier at each hop).
- §10 — the gRPC `fedlearn.v2` contract, specifically the `ReportClientMetrics` RPC and the `bytes_received` accounting fields this unit reads.

It **references** `03-DATA-MODEL.md §5.2` for the `round_results` table columns (`uplink_bytes`, `downlink_bytes`, `scalars_transmitted`, `gpu_utilization`, timing columns) that back the communication-cost telemetry, and `fl_runs.mlflow_run_id` for the MLflow link-out.

**Full glossary (every acronym used in this document):**

| Short form | Full form |
|---|---|
| LLD | Low-Level Design |
| SRE | Site Reliability Engineering |
| FL | Federated Learning |
| ML | Machine Learning |
| OTel | OpenTelemetry |
| W3C | World-Wide-Web Consortium |
| JVM | Java Virtual Machine |
| API | Application Programming Interface |
| HTTP / HTTPS | HyperText Transfer Protocol / HTTP Secure |
| REST | Representational State Transfer |
| STOMP | Simple Text Oriented Messaging Protocol |
| WS | WebSocket |
| gRPC | Google Remote Procedure Call |
| RPC | Remote Procedure Call |
| UUID | Universally Unique Identifier |
| MDC | Mapped Diagnostic Context (SLF4J/Logback per-thread key-value store) |
| SLF4J | Simple Logging Facade for Java |
| OTLP | OpenTelemetry Protocol |
| PromQL | Prometheus Query Language |
| LogQL | Loki Query Language (the Grafana-Loki query language) |
| TSDB | Time-Series Database |
| MLflow | (Machine-Learning lifecycle tool; not an acronym) |
| Loki | (Grafana log aggregation system; not an acronym) |
| Tempo | (Grafana distributed-tracing backend; not an acronym) |
| Grafana | (the Grafana visualization tool; not an acronym) |
| Micrometer | (the JVM metrics facade; not an acronym) |
| `structlog` | (the Python structured-logging library; not an acronym) |
| Alloy | (Grafana's log/telemetry collection agent, successor to Promtail; not an acronym) |
| DeComFL | Dimension-Free Communication Federated Learning (the platform's zeroth-order FL strategy; `04-API-CONTRACTS.md §0` flags the v1 "Decomposed" expansion as wrong per the paper) |
| FedAvg | Federated Averaging |
| ZO | Zeroth-Order (optimization) |
| KPI | Key Performance Indicator |
| SLO / SLI | Service-Level Objective / Service-Level Indicator |
| ETTR | Effective Training Time Ratio |
| RTO / RPO | Recovery Time Objective / Recovery Point Objective |
| DP | Differential Privacy |
| DLG | Deep Leakage from Gradients |
| PII | Personally Identifiable Information |
| S3 | (AWS) Simple Storage Service |
| MinIO | (an S3-compatible self-hosted object store; not an acronym) |
| RUM | Real User Monitoring |
| SDK | Software Development Kit |
| CI | Continuous Integration |
| YAML | YAML Ain't Markup Language |
| JSON | JavaScript Object Notation |
| ALB | (AWS) Application Load Balancer |
| EOL | End Of Life |
| ECS | (AWS) Elastic Container Service |
| k8s | Kubernetes |
| HMAC | Hash-based Message Authentication Code |

---

## 1. Purpose & single responsibility

The observability stack is the cross-cutting subsystem that **emits, propagates, stores, and visualizes the platform's metrics, logs, and traces, and the FL-run training telemetry, so that (a) an on-call engineer can answer "is the platform healthy?" and (b) a run owner can answer "is my run converging, which client is dragging it, and how much did it cost in bytes?" — with one `trace_id` joining a browser click → the JVM control plane → the spawned Python FL server → every client gRPC call → the mobile client.**

It owns exactly five concerns:
1. **Platform metrics** — Micrometer registry → Prometheus on an **internal** management port (`B3-observability.md:218`, risk #3); Python `prometheus_client` `/metrics` on the FL process.
2. **Structured logging** — `structlog` on the Python FL server (keeping the existing JSON key shape so the backend log parser stays compatible) and SLF4J Mapped Diagnostic Context (MDC) on the JVM, both binding `project_id` / `round_idx` / `trace_id` / `client_id`.
3. **Distributed tracing** — OTel W3C `traceparent` propagation across JVM → spawned-Python-env → gRPC metadata → mobile, fanned to Tempo via the OTel Collector.
4. **FL-run telemetry** — the **incremental per-round** `RoundResult` HTTP callback to `POST /api/internal/runs/{runId}/results` (the v1 producer that never fired — `B3-observability.md:10`), the communication-cost wedge (`uplink_bytes`/`downlink_bytes`/`scalars_transmitted`), per-client metrics via the gRPC `ReportClientMetrics` RPC, and MLflow experiment tracking.
5. **Visualization** — three committed Grafana dashboards (Platform Overview, Per-Run FL, Client Telemetry) and the local `docker-compose` observability stack.

**It does NOT own:** the FL algorithm (Python FL-framework LLD `11-`), the run lifecycle / lease / reconciler (orchestration-substrate LLD `12-`), the identity/authorization layer or the per-run-token mint/validate logic (control-plane / security LLDs `10-`/`18-`), the artifact-store internals (data-and-artifact LLD `17-`), nor the persistence of `round_results` rows (that is the internal-callbacks controller in LLD `10-`; this unit produces the POST that fills them). This unit **emits** telemetry and **wires** the stores; the row writes and the run state machine belong to other units.

---

## 2. Position in the system — dependencies & interfaces

### 2.1 What this unit depends on (CONSUMES)

| Dependency | What this unit needs from it | Contract reference |
|---|---|---|
| Internal-callbacks controller (control-plane LLD 10) | The HTTP endpoint `POST /api/internal/runs/{runId}/results` that accepts the `RoundResultDto`; persists `round_results`; rebroadcasts to STOMP | `04-API-CONTRACTS.md §5`, §5.1 (`RoundResultDto`) |
| Orchestration substrate (LLD 12) | Injects the env vars this unit reads: `FEDLEARN_RUN_ID`, `FEDLEARN_RUN_TOKEN`, `FEDLEARN_BACKEND_URL`, `FEDLEARN_PROJECT_ID`, `TRACEPARENT`, `OTEL_*` | `04-API-CONTRACTS.md §13` env table, §14 (the JVM→process hop) |
| Per-run scoped token (security LLD 18) | The `flrun_<...>` token the FL server sets as `Authorization: Bearer` on every callback | `04-API-CONTRACTS.md §13` |
| `round_results` table | The columns the telemetry fields map to (`uplink_bytes`, `downlink_bytes`, `scalars_transmitted`, `gpu_utilization`, `round_started_at`, `round_ended_at`) | `03-DATA-MODEL.md §5.2` |
| `fl_runs` table | `mlflow_run_id` (the MLflow link-out column); `org_id`/`project_id` for label/log enrichment | `03-DATA-MODEL.md §5.2` |
| gRPC `fedlearn.v2` contract | `ReportClientMetrics` RPC (per-client telemetry), `bytes_received` on upload/download responses (comm-cost accounting), `traceparent` gRPC metadata key | `04-API-CONTRACTS.md §10` (§10.2 proto, §10.3 framing rules) |
| STOMP relay / WebSocketService (control-plane LLD 10) | Broadcasts the structured `LogLinePayload` / `RoundResultPayload` (carrying `traceId`) | `04-API-CONTRACTS.md §11`, §11.1 |
| OTel Java agent + Micrometer tracing bridge | Auto-instruments Servlet/JDBC/gRPC and originates the JVM span whose context is serialized into the spawned process env | `02-TECH-STACK.md §20`, `B3-observability.md:72` |

### 2.2 What depends on this unit (EXPOSES)

| Consumer | What it calls / receives | Contract reference |
|---|---|---|
| FL server round loop (Python FL-framework LLD 11) | `TelemetryEmitter.emit_round(...)` (fires the incremental POST + Prometheus + MLflow + STOMP-via-backend), `bind_trace_context()`, `start_metrics_server()` | this doc §5, §6 |
| FL client (Python / Docker / mobile C++) | Calls the gRPC `ReportClientMetrics` RPC; the client OTel interceptor continues `traceparent` in outgoing metadata | `04-API-CONTRACTS.md §10`, this doc §6.5 |
| Grafana | Reads Prometheus (metrics), Loki (logs), Tempo (traces) datasources; renders the three committed dashboards | this doc §7 |
| On-call engineer / run owner | The three dashboards; the error-envelope `traceId` → Tempo deep link | `04-API-CONTRACTS.md §12`, this doc §7 |
| Prometheus | Scrapes `/actuator/prometheus` (JVM, internal port) and the FL process `/metrics` (per-run) | this doc §6.2, §8 |
| OTel Collector | Receives OTLP spans from the JVM agent and the Python SDK; fans out to Tempo | this doc §6.3, §8 |

### 2.3 Position diagram (where this unit sits)

```
 Browser (OTel web SDK sets traceparent)
   │  HTTP /api/projects/{id}/runs  (traceparent header, §14)
   ▼
 ┌──────────────────────── JVM CONTROL PLANE ───────────────────────────┐
 │  Micrometer registry ─▶ /actuator/prometheus  (INTERNAL mgmt port)    │ ──scrape──▶ Prometheus
 │  OTel Java agent: span "POST /api/projects/{id}/runs"                  │ ──OTLP────▶ OTel Collector ─▶ Tempo
 │  SLF4J + MDC {project_id, run_id, trace_id}                           │ ──Alloy───▶ Loki
 │  Orchestration substrate (LLD 12) serializes span → env TRACEPARENT   │
 └────────────────────────────────┬─────────────────────────────────────┘
                                  │ spawn executor with TRACEPARENT + FEDLEARN_* env (§13/§14)
                                  ▼
 ┌──────────────────────── PYTHON FL SERVER ────────────────────────────┐
 │  observability/  (THIS UNIT, framework side)                          │
 │   tracing.py     extract(TRACEPARENT) → root span "fl-run {run_id}"   │ ──OTLP────▶ OTel Collector ─▶ Tempo
 │   metrics.py     prometheus_client /metrics on grpc_port+1000         │ ──scrape──▶ Prometheus
 │   logging_setup  structlog JSON: project_id/round_idx/trace_id        │ ──Alloy───▶ Loki
 │   telemetry.py   emit_round(): POST /api/internal/runs/{runId}/results│ ──HTTP───▶ Internal-callbacks ctrl (LLD 10) ─▶ round_results + STOMP
 │   mlflow_sink.py log_params/log_metrics/log_artifact                  │ ──HTTP───▶ MLflow tracking server
 │   grpc OTel server interceptor: traceparent in metadata               │
 └────────────────────────────────┬─────────────────────────────────────┘
                                  │ gRPC metadata traceparent (§14)
                                  ▼
        gRPC clients (desktop / docker / mobile C++) — client interceptor continues traceparent;
        clients call ReportClientMetrics(run_id, round, loss, accuracy, client_type, compute_ms) (§10)
```

---

## 3. Tech stack for this unit (pinned versions + one-line reasoning)

All versions are copied from `02-TECH-STACK.md §20` / §24.8; `verify-before-use` (VBU) entries MUST be resolved to an exact value before pinning. Do not substitute alternatives.

### 3.1 Platform layer (JVM side)

| Technology | Pinned version | One-line reasoning |
|---|---|---|
| Micrometer + `micrometer-registry-prometheus` | Spring Boot `3.5.14` BOM-managed (VBU; do not override) | Idiomatic zero-config JVM/HTTP/JDBC/STOMP metrics on the existing actuator (`B3-observability.md:71`); v1 had the actuator but no registry (`B3-observability.md:23`). |
| `micrometer-tracing-bridge-otel` | Boot `3.5.14` BOM-managed (VBU) | Bridges Micrometer spans to the OTel context so the agent and Micrometer agree (`02-TECH-STACK.md §2.1`). |
| OTel Java agent (`opentelemetry-javaagent.jar`) | latest stable (VBU; pin the exact jar in CI) | Auto-instruments Servlet/JDBC/**gRPC**/RestTemplate and injects `traceparent` with no code change (`B3-observability.md:72`). |
| Prometheus | `3.12.0` | Pull/scrape metrics store; FL servers are short-lived per-run processes (`B3-observability.md:67`). |
| Grafana | `13.0.1` | Single pane over Prometheus + Loki + Tempo; dashboards committed as JSON (`B3-observability.md:68`). |
| Loki | `3.7.2` | Cheap label-indexed log store matching the existing JSON log shape (`B3-observability.md:69`). |
| Grafana Alloy (log/telemetry agent) | latest stable (VBU) | Successor to Promtail; ships container/file logs to Loki (`B3-observability.md:69`). |
| Tempo | `3.0` (**breaking config changes** — read its migration notes before pinning) | Trace backend behind the OTel Collector (`B3-observability.md:70`, `02-TECH-STACK.md §20`). |
| OpenTelemetry Collector | `0.153.0` | One OTLP pipeline, swappable backends, no vendor lock-in (`B3-observability.md:80`). |

### 3.2 FL-run / framework layer (Python side)

| Technology | Pinned version | One-line reasoning |
|---|---|---|
| `prometheus_client` | already pinned in `framework/requirements.txt:37-44` (VBU exact; **wire it — it is imported nowhere today**) | Direct counter/histogram/gauge API; `/metrics` on `grpc_port + 1000` (`B3-observability.md:73`, `:11`). |
| `opentelemetry-api` / `opentelemetry-sdk` | already pinned (`requirements.txt:37-44`) (VBU exact) | Extract `TRACEPARENT` from env; create the run root span (`B3-observability.md:74`). |
| `opentelemetry-instrumentation-grpc` | already pinned family (VBU exact) | gRPC server/client interceptors carry `traceparent` in metadata (`B3-observability.md:74`). |
| `opentelemetry-exporter-otlp` | VBU exact (pin alongside the sdk version) | Exports spans over OTLP gRPC to the Collector. |
| `structlog` | VBU exact (add to `requirements.txt`; **new dep**) | Bind `project_id`/`round_idx`/`client_id`/`trace_id` contextually while keeping the current JSON keys additive (`B3-observability.md:75`). |
| MLflow (client + self-hosted server) | `3.12.0` | `$0`, Apache-2.0, data-resident experiment/lineage store; the C3 reproducibility substrate (`B3-observability.md:76`, `02-TECH-STACK.md §8`). |

> **Pinning discipline (`02-TECH-STACK.md §0.1`):** pin every dependency to an **exact** version (no `^`/`>=`). The v1 `framework/requirements.txt` had range pins flagged as a reproducibility hole (`B7-standards-dx.md:201`). The `opentelemetry-*` and `prometheus_client` packages are **already pinned but imported in zero `src/` files** (`B3-observability.md:11,24`); this unit activates exactly those pins — do not bump them while wiring. Also trim the dead `opentelemetry.*` mypy override in `framework/pyproject.toml` once the real imports exist (`B7-standards-dx.md:181,250`).

> **License flags (`02-TECH-STACK.md §20`):** Grafana, Loki, and Tempo are AGPLv3 — self-hosting internally is fine; redistributing a **modified** Grafana/Loki/Tempo triggers copyleft. Prometheus, OTel Collector, Micrometer, MLflow are Apache-2.0. Do not modify-and-redistribute the AGPLv3 components without legal review.

---

## 4. Module / file structure (exact directory trees, one-line responsibility per file)

This unit spans two deployable trees (the JVM backend and the Python framework) plus a repo-root `observability/` directory for the local stack. Author exactly these files.

### 4.1 JVM backend (`backend/fl-platform-api/`)

```
backend/fl-platform-api/
  src/main/java/com/federated/fl_platform_api/observability/
    ObservabilityConfig.java          # @Configuration: register the Prometheus MeterRegistry + common tags (org, instance)
    FlMetrics.java                    # central MeterRegistry holder; defines the 4 JVM-side fedlearn_* meters (§6.1)
    TraceContextLogFilter.java        # OncePerRequestFilter: copy trace_id + project_id into SLF4J MDC for every request
    StompMetricsConfig.java           # exposes simp_* STOMP session/broker metrics via Micrometer
  src/main/resources/
    application.properties            # mgmt port + actuator exposure (§8) — EDIT, do not create
    logback-spring.xml                # JSON encoder binding MDC keys {trace_id, project_id, run_id} (§6.4)
```

### 4.2 Python FL framework (`framework/src/fedlearn/observability/`)

```
framework/src/fedlearn/observability/
  __init__.py            # re-exports: init_observability(), TelemetryEmitter, bind_round, bind_client
  config.py              # ObservabilitySettings dataclass: reads FEDLEARN_*/OTEL_* env (§8); single source of config
  logging_setup.py       # configure_structlog(): JSON renderer KEEPING {timestamp,level,message,stackTrace} + additive keys
  tracing.py             # init_tracer(): extract(TRACEPARENT)->root span "fl-run {run_id}"; OTLP exporter; helpers
  metrics.py             # PrometheusMetrics: defines every fedlearn_* metric (§6.2); start_metrics_server(port)
  telemetry.py           # TelemetryEmitter.emit_round(): incremental POST + Prometheus + MLflow + per-round span
  callback_client.py     # ResultCallbackClient: best-effort HTTP POST to /api/internal/runs/{runId}/results (§5)
  mlflow_sink.py         # MlflowSink: set_experiment/log_params/log_metrics/log_artifact; resolves run_id (§6.6)
  grpc_interceptors.py   # server + client OTel gRPC interceptors (traceparent in metadata, §10.3); registered BEFORE grpc.server()
  comm_cost.py           # CommCostAccumulator: sums uplink/downlink bytes + scalars_transmitted per round (§6.2)
```

> **Why a dedicated package, not inline edits to `server.py`/`coordinator.py`:** `B3-observability.md:113-114,131` re-anchors instrumentation to the live synchronous path (`server.py`, `coordinator.py`, `grpc_servicer.py`) but the *logic* must live in one importable, testable package so the round loop calls `emitter.emit_round(...)` at the seam rather than carrying urllib/Prometheus/MLflow code inline. This also makes the telemetry independently unit-testable (§10) and keeps the dead RabbitMQ `async_coordinator.py` path uninstrumented (`B3-observability.md:15,225`, risk #10).

### 4.3 Repo-root local observability stack (`observability/`)

```
observability/
  docker-compose.observability.yml   # Prometheus + Grafana + Loki + Tempo + OTel Collector + MLflow + MinIO (§8.3)
  prometheus/
    prometheus.yml                    # scrape configs: actuator (internal) + file_sd for live FL (project_id,port)
    targets/fl-servers.json           # file_sd target list written by the substrate (LLD 12) as runs start/stop
  otel-collector/
    config.yaml                       # OTLP receivers -> batch processor -> Tempo exporter (§8.2)
  loki/
    loki-config.yaml                  # single-binary Loki config (filesystem store for local; object store in prod)
  alloy/
    config.alloy                      # scrape docker/file logs -> push to Loki with labels {service,project_id}
  tempo/
    tempo.yaml                        # Tempo 3.0 config (note breaking changes vs 2.x)
  grafana/
    provisioning/datasources/datasources.yml   # Prometheus + Loki + Tempo + (MLflow link via dashboard, not datasource)
    provisioning/dashboards/dashboards.yml      # auto-load the 3 committed dashboards from /dashboards
    dashboards/platform-overview.json           # Dashboard A (§7.1) — committed JSON-as-code
    dashboards/per-run-fl.json                  # Dashboard B (§7.2) — committed JSON-as-code
    dashboards/client-telemetry.json            # Dashboard C (§7.3) — committed JSON-as-code
```

---

## 5. Key interfaces & type signatures (FULL signatures)

### 5.1 Python — `ObservabilitySettings` (config.py)

```python
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class ObservabilitySettings:
    run_id: str                  # FEDLEARN_RUN_ID (UUID string); REQUIRED
    project_id: str              # FEDLEARN_PROJECT_ID (UUID string); REQUIRED
    backend_url: str             # FEDLEARN_BACKEND_URL base, e.g. "https://api.internal:8443"; REQUIRED
    run_token: str               # FEDLEARN_RUN_TOKEN ("flrun_<...>"); REQUIRED for callbacks
    traceparent: Optional[str]   # TRACEPARENT (W3C); may be None for dev/local
    grpc_port: int               # the FL server gRPC port; metrics server binds grpc_port + 1000
    otel_collector_endpoint: str # OTEL_EXPORTER_OTLP_ENDPOINT, default "http://localhost:4317"
    mlflow_tracking_uri: str     # MLFLOW_TRACKING_URI, default "http://localhost:5000"
    callback_timeout_seconds: float = 2.0   # short timeout — telemetry MUST NOT stall the round (C1 F7, B3 risk #9)
    enabled: bool = True         # FEDLEARN_OBSERVABILITY_ENABLED; false in unit tests

    @staticmethod
    def from_env() -> "ObservabilitySettings": ...   # reads os.environ; raises ValueError if a REQUIRED var is missing
```

### 5.2 Python — `init_observability()` (the one entrypoint the FL server calls)

```python
def init_observability(settings: ObservabilitySettings) -> "TelemetryEmitter":
    """
    Call ONCE at FL-server startup, BEFORE grpc.server(...) is constructed
    (so the OTel gRPC server interceptor is registered first — B3 risk #4, C1).
    Order (MUST be exactly this):
      1. configure_structlog(settings)                 # logging first, so later steps log structured
      2. tracer = init_tracer(settings)                # extract(TRACEPARENT) -> root span "fl-run {run_id}"
      3. metrics = PrometheusMetrics(); metrics.start_metrics_server(settings.grpc_port + 1000)
      4. mlflow_sink = MlflowSink(settings)             # set_experiment(project_id); start/resume run
      5. return TelemetryEmitter(settings, metrics, mlflow_sink, tracer)
    Returns the emitter the round loop calls. Idempotent: a second call is a no-op (logs WARN).
    """
```

### 5.3 Python — `TelemetryEmitter` (telemetry.py) — the FL-run telemetry surface

```python
from typing import Optional

class TelemetryEmitter:
    def __init__(self, settings, metrics, mlflow_sink, tracer): ...

    def emit_round(
        self,
        *,
        server_round: int,
        loss: Optional[float],
        accuracy: Optional[float],
        val_loss: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        gpu_utilization: Optional[float] = None,
        uplink_bytes: Optional[int] = None,
        downlink_bytes: Optional[int] = None,
        scalars_transmitted: Optional[int] = None,   # DeComFL: K*P scalars this round
        model_param_count: Optional[int] = None,     # model dimension d (for the dimension-free comparison)
        round_duration_seconds: Optional[float] = None,
        aggregation_seconds: Optional[float] = None,
        active_clients: Optional[int] = None,
        strategy: str = "DeComFL",                    # "FedAvg" | "DeComFL" — Prometheus label only
    ) -> None:
        """
        Called ONCE per round at the aggregation seam (coordinator.py, under the round lock,
        AFTER aggregation/eval, BEFORE current_round advances). Best-effort and non-blocking:
          1. Update Prometheus gauges/counters/histograms (§6.2) — in-process, cannot fail the run.
          2. mlflow_sink.log_metrics({...}, step=server_round) — wrapped in try/except.
          3. callback_client.post_round(RoundResultDto) — short timeout, try/except (§5.4).
          4. Open a child span "fl-round {server_round}" with the byte/scalar attributes (§6.3).
        A failure in ANY step logs a WARN and returns; it MUST NOT raise into the round loop
        (C1 F7 reliability blast-radius; B3 risk #9).
        """

    def emit_run_finished(self, *, final_status: str, total_rounds: int,
                          final_model_sha256: Optional[str],
                          error_message: Optional[str] = None) -> None:
        """End-of-run: log final MLflow tags + the final model artifact; POST .../finished is owned by
        the orchestration substrate (LLD 12), not here — this only writes the MLflow side."""

    def record_client_metrics(self, *, client_id: str, round: int, loss: float, accuracy: float,
                              client_type: str, compute_ms: int) -> None:
        """Called by the gRPC servicer when a client invokes ReportClientMetrics (§6.5)."""
```

### 5.4 Python — `ResultCallbackClient` (callback_client.py) — the incremental producer

```python
class ResultCallbackClient:
    def __init__(self, settings: ObservabilitySettings): ...

    def post_round(self, dto: dict) -> bool:
        """
        POST `dto` (a RoundResultDto, 04-API-CONTRACTS.md §5.1) as JSON to
        `${backend_url}/api/internal/runs/${run_id}/results`
        with headers:
          Authorization: Bearer ${run_token}          # the per-run scoped token (§13), NOT the v1 global key
          Content-Type:  application/json
          traceparent:   <current span context>        # so the callback hop joins the same trace (§14)
        Uses stdlib urllib.request with timeout=callback_timeout_seconds (no new dependency — B3 §6.1).
        Returns True on HTTP 202; on any exception/timeout/non-2xx logs WARN and returns False.
        NEVER raises. The expected success is 202 Accepted (the controller persists then rebroadcasts STOMP).
        """
```

The `RoundResultDto` JSON shape (EXACT — `04-API-CONTRACTS.md §5.1`; keys are camelCase on the wire, mapped to the snake_case `round_results` columns by the controller):

```jsonc
{
  "serverRound": 7,            // -> round_results.round_idx
  "loss": 0.2314, "accuracy": 0.9012,
  "gpuUtilization": 0.0,
  "uplinkBytes": 240, "downlinkBytes": 240,   // -> uplink_bytes, downlink_bytes
  "scalarsTransmitted": 50,                   // -> scalars_transmitted (DeComFL wedge)
  "modelParamCount": 66000000,
  "roundDurationSeconds": 4.2, "aggregationSeconds": 0.1,
  "activeClients": 3
}
```

### 5.5 Python — `PrometheusMetrics` (metrics.py)

```python
from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, start_http_server

class PrometheusMetrics:
    def __init__(self, registry: CollectorRegistry | None = None): ...
    # Every metric is created in __init__ with the EXACT names/labels in §6.2.
    def start_metrics_server(self, port: int) -> None:
        """prometheus_client.start_http_server(port) — call AFTER grpc_server.start() (server.py ~:135).
        port = grpc_port + 1000 (B3 §6.1 channel 2). Idempotent; second call logs WARN."""
```

### 5.6 Python — gRPC OTel interceptors (grpc_interceptors.py)

```python
import grpc
from opentelemetry.instrumentation.grpc import server_interceptor, client_interceptor

def build_server_interceptor() -> grpc.ServerInterceptor:
    """Returns the OTel server interceptor. MUST be passed to grpc.server(interceptors=[...])
    BEFORE the server is started (B3 risk #4 / C1 — the v1 import-order no-op)."""

def build_client_interceptor() -> grpc.UnaryUnaryClientInterceptor:
    """For the Python FL client; continues traceparent in outgoing gRPC metadata (§14)."""
```

### 5.7 JVM — `FlMetrics` (FlMetrics.java)

```java
package com.federated.fl_platform_api.observability;

import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.stereotype.Component;
import java.util.concurrent.atomic.AtomicInteger;

@Component
public class FlMetrics {
    private final AtomicInteger activeProjects = new AtomicInteger(0);   // fedlearn_active_projects (§6.1)
    private final AtomicInteger orphanCandidates = new AtomicInteger(0); // fedlearn_orphaned_processes (§6.1)

    public FlMetrics(MeterRegistry registry) {
        Gauge.builder("fedlearn_active_projects", activeProjects, AtomicInteger::get)
             .description("FL runs the supervisor believes are active").register(registry);
        Gauge.builder("fedlearn_orphaned_processes", orphanCandidates, AtomicInteger::get)
             .description("Tracked FL processes whose /metrics has not been scraped in N intervals")
             .register(registry);
    }
    public void setActiveProjects(int n) { activeProjects.set(n); }
    public void setOrphanCandidates(int n) { orphanCandidates.set(n); }
}
```

> The reconciler (orchestration-substrate LLD 12) **calls** `setActiveProjects`/`setOrphanCandidates`; it owns the count, this unit owns the meter. JVM HTTP/JVM-heap/GC/JDBC/STOMP metrics are emitted automatically by Micrometer + the OTel Java agent — no Java code needed (`B3-observability.md:184-190`).

---

## 6. Core algorithms & flows

### 6.1 JVM-side metric inventory (Micrometer; mostly free, four custom)

| Metric (Prometheus name) | Type | Labels | Source |
|---|---|---|---|
| `http_server_requests_seconds_count` / `_bucket` | Counter / Histogram | `uri`, `status`, `method` | Micrometer web auto-instrumentation (free) |
| `jvm_memory_used_bytes`, `jvm_gc_pause_seconds` | Gauge / Summary | `area`, `id` | Micrometer JVM binder (free) |
| `simp_message_broker_*` (STOMP session/broker) | Gauge/Counter | — | Micrometer Spring Messaging (free; `B3-observability.md:190`) |
| `fedlearn_active_projects` | Gauge | — | `FlMetrics.setActiveProjects` (reconciler-driven) |
| `fedlearn_orphaned_processes` | Gauge | — | `FlMetrics.setOrphanCandidates` (catches the C1 F4 zombie-process mode) |

These four-plus-free metrics back **Dashboard A** (§7.1). The orphan detector exists specifically to surface the C1-F4 phantom-RUNNING split-brain (`C1-reliability-sre.md:49-53`, `B3-observability.md:189`).

### 6.2 Python-side metric inventory (`prometheus_client`; EXACT names — do not rename)

Copied verbatim from `B3-observability.md §6.2`. Cardinality discipline (`02-TECH-STACK.md §20` "cardinality discipline", `B3-observability.md:216`): **`client_id` appears only on bounded heartbeat/progress gauges, never on histograms** — per-client detail goes to MLflow, not Prometheus.

```
# --- Convergence (Gauge per run; labelled project_id) ---
fedlearn_round_loss{project_id}                       Gauge
fedlearn_round_accuracy{project_id}                   Gauge

# --- Round mechanics ---
fedlearn_round_duration_seconds{project_id,strategy}  Histogram
fedlearn_aggregation_seconds{project_id,strategy}     Histogram
fedlearn_rounds_completed_total{project_id,status}    Counter
fedlearn_round_clients_active{project_id}             Gauge

# --- COMMUNICATION COST — DeComFL's headline KPI (the bandwidth wedge) ---
fedlearn_uplink_bytes_total{project_id,strategy}      Counter
fedlearn_downlink_bytes_total{project_id,strategy}    Counter
fedlearn_decomfl_scalars_transmitted{project_id}      Counter   # K*P scalars/round — the O(K*P) proof
fedlearn_model_param_count{project_id}                Gauge     # model dimension d (for dimension-free comparison)

# --- Client telemetry (BOUNDED cardinality) ---
fedlearn_client_compute_seconds{project_id}                       Histogram  # NOT per client_id
fedlearn_client_progress_ratio{project_id,client_id}              Gauge      # bounded by active clients
fedlearn_client_last_heartbeat_age_seconds{project_id,client_id}  Gauge

# --- gRPC (via OTel instrumentation, free) ---
fedlearn_grpc_request_duration_seconds{project_id,rpc}            Histogram

# --- Reliability SLI counters (C1 §3.5 — new) ---
fedlearn_rounds_lost_total{project_id,reason}                     Counter
fedlearn_recovery_rounds_total{project_id}                        Counter
```

> **Why `fedlearn_decomfl_scalars_transmitted` paired with `fedlearn_model_param_count`:** this pair lets a dashboard *prove* the DeComFL thesis — scalars/round stays O(K·P) while param count `d` can be in the millions (`B3-observability.md:154-155`). The DeComFL strategy already has `K` (`num_local_steps`) and `P` (`num_perturbations`) and per-round gradient scalars, so the counter increments are trivial at the aggregation site (`B3-observability.md:166`). The byte counts are read off the gRPC `SubmitGradientScalarsResponse.bytes_received` / `SubmitModelUpdateResponse.bytes_received` fields (`04-API-CONTRACTS.md §10.2`). `fedlearn_rounds_lost_total` / `fedlearn_recovery_rounds_total` are the C1 round-goodput SLI inputs (`C1-reliability-sre.md:176,183`).

### 6.3 Span inventory (OTel; EXACT span names + attributes)

One `trace_id` joins all hops (`04-API-CONTRACTS.md §14`). Span names are fixed strings; attributes use OTel semantic-convention-style keys plus `fedlearn.*` custom keys.

| Span name | Created by | Parent | Key attributes |
|---|---|---|---|
| `POST /api/projects/{projectId}/runs` | OTel Java agent (Servlet) | browser root span (if `traceparent` present) | `http.method`, `http.route`, `fedlearn.project_id`, `fedlearn.org_id` |
| `fl-run {run_id}` | Python `tracing.init_tracer()` (extract from env `TRACEPARENT`) | the JVM launch span | `fedlearn.run_id`, `fedlearn.project_id`, `fedlearn.strategy`, `service.name=fl-server` |
| `fl-round {server_round}` | `TelemetryEmitter.emit_round()` | `fl-run {run_id}` | `fedlearn.round_idx`, `fedlearn.active_clients`, `fedlearn.uplink_bytes`, `fedlearn.downlink_bytes`, `fedlearn.scalars_transmitted` |
| `grpc {rpc_name}` (e.g. `grpc SubmitGradientScalars`) | OTel gRPC server interceptor | `fl-round` (or `fl-run`) | `rpc.system=grpc`, `rpc.method`, `fedlearn.client_id`, `fedlearn.client_type` |
| `grpc {rpc_name}` (client side, incl. mobile) | OTel gRPC client interceptor | continues server context via metadata | same; mobile rounds appear inside the server trace |

`OTEL_RESOURCE_ATTRIBUTES` set by the substrate at launch (`B3-observability.md:99`): `service.name=fl-server,fedlearn.project_id=<uuid>,fedlearn.run_id=<uuid>`.

### 6.4 Log field inventory (`structlog` JSON; keep the v1 keys, add the new ones)

The v1 root-logger `JSONFormatter` (`framework .../server.py:20-35`) emits `{"timestamp","level","message","stackTrace"?}` and the backend log-persistence parser keys on exactly that shape, so enrichment **MUST be additive** (`B3-observability.md:38`). `structlog` keeps those four keys and binds:

| Field | Type | Bound when | Source |
|---|---|---|---|
| `timestamp` | ISO-8601 string | always (v1 key) | `structlog.processors.TimeStamper(fmt="iso")` |
| `level` | string | always (v1 key) | log level |
| `message` | string | always (v1 key) | the log message |
| `stackTrace` | string \| absent | on exceptions (v1 key) | `structlog.processors.format_exc_info` |
| `project_id` | UUID string | bound at startup (from `FEDLEARN_PROJECT_ID`) | `logging_setup.bind_global` |
| `run_id` | UUID string | bound at startup (from `FEDLEARN_RUN_ID`) | `logging_setup.bind_global` |
| `round_idx` | int \| absent | bound inside the round loop via `bind_round(r)` | `structlog.contextvars` |
| `client_id` | string \| absent | bound inside per-client gRPC handlers via `bind_client(id)` | `structlog.contextvars` |
| `trace_id` | 32-hex string \| absent | bound from the active OTel span context | `tracing.current_trace_id()` |

On the JVM side, `logback-spring.xml` uses a JSON encoder that emits the same `trace_id` / `project_id` / `run_id` from SLF4J MDC (populated by `TraceContextLogFilter` and the orchestration code). The STOMP `LogLinePayload` / `RunEventPayload` carry `traceId` (`04-API-CONTRACTS.md §11.1`), so an in-app log line and a Loki log line share the id.

### 6.5 The incremental per-round telemetry flow (the emphasized fix)

This is the single highest-leverage fix in the audit: the v1 pipeline existed end-to-end **except the producer**, and even the existing producer batched after the run finished, leaving the chart empty during training (`B3-observability.md:10`, `04-API-CONTRACTS.md §5` "per-round POST is incremental, not batched"). The flow:

```
coordinator round loop (Python FL server, synchronous path)
  ┌────────────────────────────────────────────────────────────────────────┐
  │  for r in range(resume_round, num_rounds+1):          # C1: resume-aware │
  │     run_round(r) -> aggregate -> evaluate             # under _lock      │
  │     comm = comm_cost.snapshot_round(r)                # uplink/downlink/scalars
  │     emitter.emit_round(                                                   │
  │        server_round=r, loss=loss, accuracy=acc,                          │
  │        uplink_bytes=comm.uplink, downlink_bytes=comm.downlink,           │
  │        scalars_transmitted=comm.scalars, model_param_count=d,           │
  │        round_duration_seconds=dt, aggregation_seconds=agg_dt,           │
  │        active_clients=n, strategy=strategy_name)      # ONE call/round   │
  └───────────────────────────────┬──────────────────────────────────────────┘
                                  │ inside emit_round (best-effort, ordered):
       (1) Prometheus update ─────┤  in-process, cannot fail the run
       (2) MLflow log_metrics ────┤  try/except, WARN on failure
       (3) HTTP POST ─────────────┼──▶ POST ${BACKEND}/api/internal/runs/${run_id}/results
                                  │        Authorization: Bearer ${run_token}; traceparent: <span>
                                  │        body = RoundResultDto (§5.4)   timeout=2s, try/except
       (4) child span "fl-round r"┘
                                  ▼
   Internal-callbacks controller (LLD 10): 202 ─▶ persist round_results row
                                                ─▶ WebSocketService -> STOMP /topic/results/{projectId}
                                                ─▶ frontend recharts chart lights up PER ROUND
```

**Sequence diagram (cross-component, one round):**

```
FL round loop      TelemetryEmitter     Prometheus   MLflow    Internal-ctrl(JVM)    STOMP/browser
     │                   │                  │           │             │                   │
     │ emit_round(r,...) │                  │           │             │                   │
     │──────────────────▶│                  │           │             │                   │
     │                   │ set gauges/inc   │           │             │                   │
     │                   │─────────────────▶│           │             │                   │
     │                   │ log_metrics(step=r)          │             │                   │
     │                   │─────────────────────────────▶│             │                   │
     │                   │ POST .../runs/{id}/results (Bearer flrun_, traceparent)         │
     │                   │────────────────────────────────────────────▶│                  │
     │                   │                  │           │     persist round_results        │
     │                   │                  │           │     send /topic/results/{proj}   │
     │                   │                  │           │             │──────────────────▶ │  (chart updates)
     │                   │◀───────────────────────────────────────────│ 202 Accepted      │
     │◀──────────────────│  (returns None; WARN-only on any failure)                       │
```

### 6.6 MLflow experiment-tracking flow (`mlflow_sink.py`)

```python
class MlflowSink:
    def __init__(self, settings: ObservabilitySettings):
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.project_id)        # experiment == project (B3 §6.1 channel 3)
        # Resume the run if fl_runs.mlflow_run_id was pre-created by the substrate; else start one.
        self._run = mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID") or None)
        # The substrate (LLD 12) writes the resulting run id back into fl_runs.mlflow_run_id (03-DATA-MODEL §5.2).

    def log_params(self, params: dict) -> None: ...      # ONCE at run start: seed, K, P, eta, mu, num_rounds, strategy
    def log_metrics(self, metrics: dict, step: int) -> None: ...  # PER round: loss/accuracy/uplink/scalars/...
    def log_artifact(self, local_path: str) -> None: ...  # final model + per-round checkpoint pointers
```

> **Why MLflow AND Prometheus AND `round_results` (three stores, on purpose):** Prometheus answers "healthy now" (low cardinality, 15–30 d retention); MLflow answers "compare run A vs run B and inspect lineage" (high per-client/per-round cardinality); the `round_results` table feeds the **live** STOMP chart. They are different stores for different questions; conflating them is the "split-brain" mistake the audit names (`B3-observability.md:47,57,83`, risk #6). `fl_runs.mlflow_run_id` is the link-out from the operational row to the MLflow run.

### 6.7 W3C `traceparent` propagation algorithm (the carrier at each hop)

The format is locked (`04-API-CONTRACTS.md §14`): `version "-" trace-id "-" parent-id "-" trace-flags`, e.g. `00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01`. The carrier per hop (this unit implements the Python and gRPC hops; the substrate implements the env-serialize hop; the agent implements the JVM hop):

```
Hop                          Carrier                         Who implements it
browser -> JVM               HTTP header `traceparent`       frontend OTel web SDK (frontend LLD 13) / OTel Java agent originates root if absent
JVM span -> Python process   env var TRACEPARENT             orchestration substrate (LLD 12) serializes via OTel TextMapPropagator (§14)
Python process root span     extract(os.environ["TRACEPARENT"])  THIS UNIT: tracing.init_tracer()
FL server -> client (gRPC)   gRPC metadata key `traceparent` THIS UNIT: grpc_interceptors server/client interceptor
client -> mobile             gRPC metadata key `traceparent` mobile C++ gRPC interceptor (mobile LLD) — continues same context
any hop -> logs              structlog/MDC field `trace_id`  THIS UNIT (Python) + TraceContextLogFilter (JVM)
any error response           error envelope `traceId`        GlobalExceptionHandler (control-plane LLD 10)
```

Python extraction (the two-line `extract()` `B3-observability.md:113` promises):

```python
# tracing.py
from opentelemetry import trace, context as otel_context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

def init_tracer(settings):
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(
        OTLPSpanExporter(endpoint=settings.otel_collector_endpoint, insecure=True)))  # TLS outside dev (§14 caveat)
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer("fedlearn.fl-server")
    parent_ctx = None
    if settings.traceparent:
        carrier = {"traceparent": settings.traceparent}
        parent_ctx = TraceContextTextMapPropagator().extract(carrier=carrier)
    root = tracer.start_span(f"fl-run {settings.run_id}", context=parent_ctx)
    root.set_attribute("fedlearn.run_id", settings.run_id)
    root.set_attribute("fedlearn.project_id", settings.project_id)
    otel_context.attach(trace.set_span_in_context(root))   # make the run span the ambient context
    return tracer
```

> **Caveat (locked, MUST honor — `04-API-CONTRACTS.md §14`, `B3-observability.md:116`):** gRPC is plaintext only in `dev`; in all other profiles `traceparent` and baggage travel over TLS+mTLS. **Never put PII in baggage** — `traceparent` carries only opaque ids. The OTLP exporter `insecure=True` above is the dev default; outside dev set TLS on the exporter.

---

## 7. Data it owns

### 7.1 In-memory / on-disk structures owned by this unit

| Structure | Where | Shape | Lifetime |
|---|---|---|---|
| Prometheus registry + meters | Python FL process | `CollectorRegistry` with the §6.2 metrics | FL-process lifetime |
| `structlog` context vars | Python FL process | `{project_id, run_id, round_idx?, client_id?, trace_id?}` | per-thread / per-round |
| OTel TracerProvider + `fl-run` root span | Python FL process | OTel SDK objects | FL-process lifetime |
| `CommCostAccumulator` | Python FL process (`comm_cost.py`) | `{round_idx: {uplink:int, downlink:int, scalars:int}}` | reset per round |
| Micrometer `MeterRegistry` + `FlMetrics` gauges | JVM | `fedlearn_active_projects`, `fedlearn_orphaned_processes` + free meters | JVM lifetime |
| SLF4J MDC keys | JVM (per request/thread) | `{trace_id, project_id, run_id}` | per request/thread |
| Prometheus `file_sd` target list | `observability/prometheus/targets/fl-servers.json` | `[{ "labels": {"project_id": "...","run_id":"..."}, "targets": ["host:port+1000"] }]` | written by the substrate as runs start/stop |
| Grafana dashboard JSON | `observability/grafana/dashboards/*.json` | committed JSON-as-code (3 files) | versioned in git |

### 7.2 Persistent tables this unit READS / WRITES-THROUGH (it owns none directly)

This unit does **not** own any table; it produces the data that the internal-callbacks controller (LLD 10) writes. The relevant columns it populates **through** the `RoundResultDto` POST (defined in `03-DATA-MODEL.md §5.2`, table `round_results`):

| Column (`round_results`) | Type | Filled from `RoundResultDto` field |
|---|---|---|
| `round_idx` | INTEGER | `serverRound` |
| `loss`, `accuracy`, `val_loss`, `val_accuracy` | DOUBLE PRECISION | `loss`, `accuracy`, (val\_\* optional) |
| `num_clients_reported` / mapped via `activeClients` | INTEGER | `activeClients` |
| `uplink_bytes`, `downlink_bytes` | BIGINT | `uplinkBytes`, `downlinkBytes` |
| `scalars_transmitted` | BIGINT | `scalarsTransmitted` (DeComFL wedge) |
| `gpu_utilization` | DOUBLE PRECISION | `gpuUtilization` |
| `round_started_at`, `round_ended_at` | TIMESTAMPTZ | derived by the controller from receipt time / `roundDurationSeconds` |

It also **reads** `fl_runs.mlflow_run_id` (UUID/text) for the MLflow link-out and `fl_runs.org_id`/`project_id` for log/label enrichment. The `round_results` `UNIQUE (fl_run_id, round_idx)` constraint (`03-DATA-MODEL.md §5.2`) makes the incremental POST **idempotent** — a retried POST for the same round does not duplicate a row (the controller upserts on conflict).

---

## 8. Configuration & environment variables

### 8.1 Environment variables READ by the Python observability package

| Env var | Type | Default | Profile/mode | Source |
|---|---|---|---|---|
| `FEDLEARN_RUN_ID` | UUID string | (required) | all | substrate injects (`04-API-CONTRACTS.md §13`) |
| `FEDLEARN_PROJECT_ID` | UUID string | (required) | all | substrate injects (§13) |
| `FEDLEARN_BACKEND_URL` | URL string | (required) | all | substrate injects; HTTPS/VPC-internal outside dev (§13) |
| `FEDLEARN_RUN_TOKEN` | string `flrun_<...>` | (required) | all | substrate injects; the per-run scoped token (§13) |
| `TRACEPARENT` | W3C string | unset | all (may be unset in dev/local) | substrate serializes the launch span (§14) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | URL | `http://localhost:4317` | all | OTel SDK standard |
| `OTEL_RESOURCE_ATTRIBUTES` | k=v,k=v | `service.name=fl-server,...` | all | substrate sets (`B3-observability.md:99`) |
| `MLFLOW_TRACKING_URI` | URL | `http://localhost:5000` | all | this unit / deployment |
| `MLFLOW_RUN_ID` | string | unset | all | substrate may pre-create the MLflow run (§6.6) |
| `FEDLEARN_METRICS_PORT_OFFSET` | int | `1000` | all | `grpc_port + offset` for `/metrics` (`B3-observability.md:131`) |
| `FEDLEARN_OBSERVABILITY_ENABLED` | bool | `true` | `false` in unit tests | this unit |
| `FEDLEARN_CALLBACK_TIMEOUT_SECONDS` | float | `2.0` | all | short — telemetry MUST NOT stall a round (C1 F7) |

### 8.2 JVM `application.properties` keys (EDIT, do not invent)

```properties
# --- Actuator + Micrometer Prometheus on an INTERNAL management port (B3 risk #3) ---
management.server.port=9090
management.endpoints.web.exposure.include=health,info,prometheus,loggers
management.endpoint.prometheus.enabled=true
management.metrics.tags.application=fl-platform-api
# --- Micrometer tracing bridge to OTel; sample everything in dev, ratio in prod ---
management.tracing.sampling.probability=1.0
management.otlp.tracing.endpoint=http://localhost:4318/v1/traces
```

> **Why a separate `management.server.port=9090` (not the app `8081`):** `/actuator/prometheus` MUST NOT be reachable over the public ALB (`B3-observability.md:218`, risk #3; `02-TECH-STACK.md §20`). Bind the management port internal-only and keep it out of the `SecurityConfig` `permitAll` list. The OTel Java agent is attached via `-javaagent:opentelemetry-javaagent.jar` with `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_SERVICE_NAME=fl-platform-api` env (no code change — `B3-observability.md:72`).

### 8.3 The local observability `docker-compose`

`observability/docker-compose.observability.yml` (pinned images per `02-TECH-STACK.md §20`; resolve VBU tags before committing):

```yaml
# Local observability stack for FedLearn v2. Run: docker compose -f docker-compose.observability.yml up -d
services:
  prometheus:
    image: prom/prometheus:v3.12.0
    command: ["--config.file=/etc/prometheus/prometheus.yml"]
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./prometheus/targets:/etc/prometheus/targets:ro   # file_sd: live FL (project_id,port)
    ports: ["9091:9090"]                                  # host 9091 to avoid clashing with mgmt 9090

  loki:
    image: grafana/loki:3.7.2
    command: ["-config.file=/etc/loki/loki-config.yaml"]
    volumes: ["./loki/loki-config.yaml:/etc/loki/loki-config.yaml:ro"]
    ports: ["3100:3100"]

  alloy:
    image: grafana/alloy:latest          # VBU: pin exact tag
    command: ["run", "/etc/alloy/config.alloy"]
    volumes:
      - ./alloy/config.alloy:/etc/alloy/config.alloy:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro      # tail container logs -> Loki
    depends_on: [loki]

  tempo:
    image: grafana/tempo:3.0            # NOTE: 3.0 has breaking config changes vs 2.x — read migration notes
    command: ["-config.file=/etc/tempo/tempo.yaml"]
    volumes: ["./tempo/tempo.yaml:/etc/tempo/tempo.yaml:ro"]
    ports: ["3200:3200"]

  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.153.0
    command: ["--config=/etc/otelcol/config.yaml"]
    volumes: ["./otel-collector/config.yaml:/etc/otelcol/config.yaml:ro"]
    ports: ["4317:4317", "4318:4318"]    # OTLP gRPC + HTTP
    depends_on: [tempo]

  grafana:
    image: grafana/grafana:13.0.1
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true   # LOCAL ONLY; never in a deployed env
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
    ports: ["3000:3000"]
    depends_on: [prometheus, loki, tempo]

  minio:
    image: minio/minio:latest            # VBU: pin exact tag/digest; MLflow artifact store (S3-compatible)
    command: ["server", "/data", "--console-address", ":9001"]
    environment: ["MINIO_ROOT_USER=minioadmin", "MINIO_ROOT_PASSWORD=minioadmin"]
    ports: ["9000:9000", "9001:9001"]

  mlflow:
    image: ghcr.io/mlflow/mlflow:v3.12.0
    command: >
      mlflow server --host 0.0.0.0 --port 5000
      --backend-store-uri sqlite:////mlflow/mlflow.db
      --default-artifact-root s3://mlflow/
    environment:
      - MLFLOW_S3_ENDPOINT_URL=http://minio:9000
      - AWS_ACCESS_KEY_ID=minioadmin
      - AWS_SECRET_ACCESS_KEY=minioadmin
    ports: ["5000:5000"]
    depends_on: [minio]
```

The OTel Collector pipeline (`observability/otel-collector/config.yaml`):

```yaml
receivers:
  otlp:
    protocols:
      grpc: { endpoint: 0.0.0.0:4317 }
      http: { endpoint: 0.0.0.0:4318 }
processors:
  batch: {}
exporters:
  otlp/tempo:
    endpoint: tempo:4317
    tls: { insecure: true }    # LOCAL ONLY; TLS outside dev
service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/tempo]
```

Prometheus scrape config (`observability/prometheus/prometheus.yml`):

```yaml
global: { scrape_interval: 15s }
scrape_configs:
  - job_name: fl-platform-api          # JVM actuator on the INTERNAL mgmt port
    metrics_path: /actuator/prometheus
    static_configs:
      - targets: ["host.docker.internal:9090"]
  - job_name: fl-servers               # per-run Python /metrics, discovered via file_sd
    file_sd_configs:
      - files: ["/etc/prometheus/targets/fl-servers.json"]
```

> **Why `file_sd` for FL servers (not static or push-gateway):** FL servers are short-lived per-run processes on dynamic ports (`B3-observability.md:67,131`). The orchestration substrate (LLD 12) rewrites `targets/fl-servers.json` as runs start/stop; Prometheus reloads it without restart. Push-gateway is reserved for "the truly ephemeral" and is not the default (`B3-observability.md:67`).

---

## 9. Error handling & edge cases (enumerate real failure modes + exact handling)

| # | Failure mode | Exact handling |
|---|---|---|
| E1 | **Telemetry POST fails / times out / backend down** | `ResultCallbackClient.post_round` catches all exceptions, logs WARN with `run_id`+`round_idx`, returns `False`. The round loop NEVER sees an exception (C1 F7; `B3-observability.md` risk #9). The Prometheus update already happened in-process, so the data is not wholly lost. |
| E2 | **STOMP subscriber backpressure stalls the FL process** | Bulk logs go to Loki via Alloy (the firehose); STOMP carries only the ≤1-msg/round `RoundResult` feed (sub-Hz) — `B3-observability.md:217`, risk #2. The stdout reader thread on the JVM uses a bounded drop-oldest queue (control-plane LLD 10). Log transport is NEVER on the FL-progress critical path (C1 F7). |
| E3 | **`/actuator/prometheus` reachable publicly** | Bound to `management.server.port=9090` internal-only; excluded from `SecurityConfig` `permitAll`; not routed by the public ALB (`B3-observability.md:218`, risk #3). A CI/infra check asserts the management port is not in the ALB target group. |
| E4 | **OTel gRPC interceptor registered after `grpc.server(...)`** (no-op trace) | `init_observability()` MUST run before `grpc.server(...)` (§5.2); a smoke test asserts a span is produced for a sample RPC (`B3-observability.md:219`, risk #4; `test_grpc_interceptor_registered`, §10). |
| E5 | **`client_id` cardinality explosion in Prometheus** | `client_id` is a label ONLY on `fedlearn_client_progress_ratio` and `fedlearn_client_last_heartbeat_age_seconds` (bounded by active clients); per-client history goes to MLflow (`B3-observability.md:216`, risk #1; `02-TECH-STACK.md §20`). Histograms never carry `client_id`. |
| E6 | **`server_logs` / `audit_events` unbounded growth** | A Flyway-managed TTL job (`DELETE WHERE timestamp < now()-30d`) or partitioning runs before volume turns up (`B3-observability.md:220`, risk #5). This LLD flags it; the cron/job is owned by the data LLD. |
| E7 | **`TRACEPARENT` absent (dev/local launch)** | `init_tracer` creates a fresh root `fl-run {run_id}` span with no parent (the `if settings.traceparent` guard in §6.7). Traces are still produced, just not joined to a browser span. |
| E8 | **MLflow server unreachable** | `MlflowSink` methods wrap `mlflow.*` calls in try/except, log WARN, and continue. A run MUST complete even if experiment tracking is down (same best-effort contract as E1). |
| E9 | **Instrumenting dead code** (`async_coordinator.py` RabbitMQ path) | Do NOT instrument it — it is commented out at `server.py:8-10`, no `pika` dep (`B3-observability.md:15,225`, risk #10). Only the live synchronous `coordinator.py`/`grpc_servicer.py` path is wired. |
| E10 | **Mobile client invisible to the platform** | Mobile MUST call the gRPC `ReportClientMetrics` RPC (`04-API-CONTRACTS.md §10.2`); `TelemetryEmitter.record_client_metrics` feeds the per-client gauges + MLflow so mobile appears in Dashboard C (`B3-observability.md:175-176`, risk #11). Skipping it leaves a phone federation with zero server-side device health. |
| E11 | **PII leaking into trace baggage over plaintext gRPC** | `traceparent` carries only opaque ids; NO PII in baggage (`04-API-CONTRACTS.md §14` caveat, `B3-observability.md:222`, risk #7). Code review + a test asserting baggage keys are an allowlist. |
| E12 | **Duplicate round POST after a retry** | The `round_results` `UNIQUE (fl_run_id, round_idx)` constraint makes the POST idempotent (`03-DATA-MODEL.md §5.2`); the controller upserts on conflict, so a retried `emit_round` does not duplicate a row (§7.2). |
| E13 | **Required env var missing at FL-server start** | `ObservabilitySettings.from_env()` raises `ValueError` naming the missing var; the FL server fails fast at startup (better than silently disabling telemetry mid-run). In unit tests set `FEDLEARN_OBSERVABILITY_ENABLED=false` to bypass. |

---

## 10. Testing strategy

**Frameworks:** Python — `pytest` (`framework/tests/`, runs without GPU per project conventions); JVM — JUnit 5 via `./gradlew test` (the `test` profile, in-memory H2, Flyway disabled). Frontend dashboard rendering is out of scope for this unit (frontend LLD 13 owns recharts tests).

| Test name | Layer | Asserts |
|---|---|---|
| `test_settings_from_env_requires_run_id` | Python | `ObservabilitySettings.from_env()` raises `ValueError` when `FEDLEARN_RUN_ID` is unset (E13). |
| `test_structlog_keeps_v1_keys` | Python | A log line still contains `timestamp`/`level`/`message`, plus additive `project_id`/`run_id`/`trace_id` (§6.4) — the backend parser stays compatible (`B3-observability.md:38`). |
| `test_tracer_extracts_traceparent` | Python | Given `TRACEPARENT=00-<32hex>-<16hex>-01`, the `fl-run` span's `trace_id` equals the carrier trace-id (§6.7). |
| `test_tracer_no_traceparent_makes_root` | Python | With `TRACEPARENT` unset, a root `fl-run` span is still created (E7). |
| `test_emit_round_updates_prometheus` | Python | After `emit_round(server_round=3, loss=0.5, accuracy=0.9, scalars_transmitted=50)`, `fedlearn_round_loss` == 0.5 and `fedlearn_decomfl_scalars_transmitted` increased by 50 (§6.2). |
| `test_emit_round_posts_dto` | Python | `emit_round` calls `ResultCallbackClient.post_round` exactly once with a body whose `serverRound`/`uplinkBytes`/`scalarsTransmitted` match the args, and header `Authorization: Bearer flrun_...` is set (§5.4). |
| `test_callback_never_raises_on_backend_down` | Python | With the backend URL unreachable, `emit_round` returns `None` (does not raise) and logs WARN (E1, risk #9). |
| `test_callback_short_timeout` | Python | The POST uses `timeout == callback_timeout_seconds` (2.0); a slow endpoint does not block beyond it (E1, C1 F7). |
| `test_round_results_idempotent_on_retry` | JVM (integration, Testcontainers PG) | Two POSTs for `(fl_run_id, round_idx)` yield one `round_results` row (E12, `UNIQUE` constraint). |
| `test_grpc_interceptor_registered` | Python | The OTel server interceptor is in `grpc.server(interceptors=...)` and a sample RPC produces a span (E4, risk #4). |
| `test_client_id_not_on_histograms` | Python | No histogram metric in `metrics.py` declares a `client_id` label (E5, risk #1). |
| `test_mlflow_sink_best_effort` | Python | With an unreachable MLflow URI, `log_metrics` logs WARN and does not raise (E8). |
| `test_record_client_metrics_feeds_gauges` | Python | `record_client_metrics(client_id="m1", client_type="mobile", ...)` sets `fedlearn_client_progress_ratio{client_id="m1"}` (E10). |
| `ObservabilityConfigTest.prometheusEndpointOnMgmtPort` | JVM | `/actuator/prometheus` responds on `management.server.port`, and is NOT exposed on the app port / public security chain (E3, risk #3). |
| `FlMetricsTest.activeProjectsGauge` | JVM | `setActiveProjects(2)` makes `fedlearn_active_projects` report 2 (§5.7, §6.1). |
| `dashboards_are_valid_json` | CI (any) | Each of the three `observability/grafana/dashboards/*.json` parses as JSON and references only metric names present in §6.2/§6.1 (guards against renamed metrics drifting from dashboards). |

---

## 11. Build & run (verify this unit in isolation)

```bash
# --- 1. Stand up the local observability stack (repo root) ---
cd observability
docker compose -f docker-compose.observability.yml up -d
# Verify each backend is up:
#   Grafana   http://localhost:3000  (anonymous, local only)
#   Prometheus http://localhost:9091
#   Tempo     http://localhost:3200/ready
#   Loki      http://localhost:3100/ready
#   MLflow    http://localhost:5000

# --- 2. Python framework: install deps incl. the newly-wired observability stack ---
cd ../framework
pip install -e .            # picks up structlog + the now-USED opentelemetry-*/prometheus_client pins
pytest tests/ -k observability   # runs the §10 Python tests (no GPU needed)

# --- 3. Smoke the FL-process metrics + trace export locally ---
export FEDLEARN_OBSERVABILITY_ENABLED=true
export FEDLEARN_RUN_ID=$(uuidgen) FEDLEARN_PROJECT_ID=$(uuidgen)
export FEDLEARN_BACKEND_URL=http://localhost:8081 FEDLEARN_RUN_TOKEN=flrun_dev
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 MLFLOW_TRACKING_URI=http://localhost:5000
python run_local_test.py    # the local end-to-end smoke (project convention)
curl -s http://localhost:51000/metrics | grep fedlearn_   # grpc_port(50000)+1000; sees the §6.2 metrics

# --- 4. JVM backend: actuator + Prometheus on the internal mgmt port ---
cd ../backend/fl-platform-api
SPRING_PROFILES_ACTIVE=dev ./gradlew bootRun
curl -s http://localhost:9090/actuator/prometheus | grep -E 'http_server_requests|jvm_memory_used|fedlearn_active_projects'
curl -s http://localhost:8081/actuator/prometheus   # MUST be 404/forbidden — NOT on the app port (E3)
SPRING_PROFILES_ACTIVE=test ./gradlew test --tests "com.federated.fl_platform_api.observability.*"

# --- 5. Validate the committed dashboards parse and reference only known metrics ---
for f in observability/grafana/dashboards/*.json; do python -c "import json,sys; json.load(open('$f'))"; done
```

**Done-condition for the unit in isolation:** (a) `curl .../metrics | grep fedlearn_` shows the §6.2 names on the FL process; (b) `curl :9090/actuator/prometheus` shows JVM + `fedlearn_active_projects` and `:8081/actuator/prometheus` is not reachable; (c) a local FL round produces a `fl-run`/`fl-round` span visible in Tempo and a row visible in MLflow; (d) all §10 tests pass.

---

## 12. Reasoning & alternatives (why this design; what was rejected, cited to audit findings)

1. **Two layers kept strictly separate (platform SRE vs FL-run/ML).** Conflating them is the classic "split-brain" mistake (`B3-observability.md:47,57,83`, risk #6). Prometheus is wrong for per-client-per-round ML history (cardinality explosion — `B3-observability.md:55,216`); MLflow is wrong for live p99 alerting. **Rejected:** one store for everything (cheaper to build, but the cardinality bill and the wrong-tool-for-the-question problem make it a false economy). **Chosen:** Prometheus = "healthy now", MLflow = "compare runs", `round_results`+STOMP = "live chart", documented on Dashboard B's first panel.

2. **The incremental per-round POST is the headline fix.** The entire FL-telemetry pipeline existed end-to-end *except the producer*, and even then it batched after the run (`B3-observability.md:10`, `04-API-CONTRACTS.md §5`). **Rejected:** batch-at-end (v1 behavior — the chart is empty during the multi-hour run, useless for the run owner). **Chosen:** one best-effort POST per round at the aggregation seam, idempotent via `UNIQUE (fl_run_id, round_idx)`, off the critical path (C1 F7). This is the user's emphasized ask and ~1 day of producer work (`B3-observability.md:10`).

3. **Wire the pinned-but-unused deps rather than add new ones.** `opentelemetry-*` and `prometheus_client` are already pinned and imported nowhere (`B3-observability.md:11,24`) — paid-for, unused. **Rejected:** add a different vendor SDK (Datadog) — cost-prohibitive at hyperscale host counts (`02-TECH-STACK.md §20`, `B6-scale-cost.md:139`). **Chosen:** activate exactly the pinned versions; the only new Python dep is `structlog`, which keeps the v1 JSON log keys so the backend parser stays compatible (additive — `B3-observability.md:38,75`).

4. **`structlog` keeping the v1 JSON shape, not a clean-slate format.** The backend log-persistence parser keys on `{timestamp,level,message,stackTrace}` (`B3-observability.md:38`). **Rejected:** redesign the log schema (breaks the parser, forces a coordinated two-sided change). **Chosen:** additive enrichment (`project_id`/`round_idx`/`trace_id`/`client_id`) so the change is one-sided and safe.

5. **W3C `traceparent` over OTel, env var for the JVM→process hop.** v1 had no correlation id anywhere (`B3-observability.md:12`, `:87`). The FL server is launched by three backends (k8s Job, ECS RunTask, dev process); an env var is the one carrier all three support uniformly, and `TRACEPARENT` is the OTel SDK's standard extraction path (`04-API-CONTRACTS.md §14`, `B3-observability.md:97-99`). **Rejected:** a CLI flag or a file (not uniform across the three launchers; more custom parsing). **Chosen:** env `TRACEPARENT` + gRPC metadata, the W3C-standard carriers.

6. **Prometheus on a separate internal management port.** **Rejected:** expose `/actuator/prometheus` on the app port (`:8081`) behind the ALB — that leaks internal metrics publicly (`B3-observability.md:218`, risk #3). **Chosen:** `management.server.port=9090`, internal-only.

7. **MLflow self-hosted, not W&B.** MLflow is Apache-2.0, `$0`, data-resident — a near-requirement for healthcare/pneumonia federations (`B3-observability.md:76`, `02-TECH-STACK.md §8`). **Rejected:** Weights & Biases (self-hosting is enterprise-only, $2k–5k/mo, vendor lock-in, data-residency friction). **Chosen:** MLflow self-hosted as the v2 floor; W&B a later per-customer opt-in.

8. **Communication-cost metrics from day one.** DeComFL's entire thesis is O(K·P) communication independent of model dimension `d`; v1's schema had no comm-cost column so the platform could not demonstrate its own differentiator (`B3-observability.md:14,154-155`). **Rejected:** defer the comm-cost panel as a "nice to have" (loses the customer-facing proof of the wedge). **Chosen:** `uplink_bytes`/`downlink_bytes`/`scalars_transmitted`/`model_param_count` shipped now, backing the Dashboard B hero panel.

9. **Bounded `client_id` cardinality + per-client detail to MLflow.** "Label cardinality *is* the observability bill" (`02-TECH-STACK.md §20`, `B6-scale-cost.md:119`). **Rejected:** `client_id` on histograms (unbounded cost). **Chosen:** `client_id` only on two bounded heartbeat/progress gauges (`B3-observability.md:216`, risk #1).

10. **Reliability SLI counters added (`fedlearn_rounds_lost_total`, `fedlearn_recovery_rounds_total`).** C1 defines round-goodput as the headline per-run SLI (`C1-reliability-sre.md:176,183`), computable only once these counters exist. **Rejected:** ship only convergence metrics (cannot compute goodput/RTO/RPO). **Chosen:** add the two reason-labelled counters so the C1 error-budget policy is measurable.

11. **Do NOT instrument the dead RabbitMQ path.** `async_coordinator.py` is commented out, no `pika` dep (`B3-observability.md:15,225`, risk #10). **Rejected:** instrument both paths "to be safe" (wasted effort on dead code, plus it implies the path is alive). **Chosen:** instrument only the live synchronous `coordinator.py`/`grpc_servicer.py` path; flag the RabbitMQ file for deletion.

---

## 13. Build task checklist for the ~30B local model (ordered, dependency-aware)

Each task is one file/feature with a concrete done-condition. Do them in this order.

1. **`framework/src/fedlearn/observability/config.py`** — implement `ObservabilitySettings` + `from_env()`. **Done when** `test_settings_from_env_requires_run_id` passes.
2. **`logging_setup.py`** — `configure_structlog()` keeping the v1 JSON keys + additive `project_id`/`run_id`/`round_idx`/`client_id`/`trace_id`; add `bind_round(r)` / `bind_client(id)` / `bind_global(...)`. **Done when** `test_structlog_keeps_v1_keys` passes. (Add `structlog` to `requirements.txt`, exact pin.)
3. **`tracing.py`** — `init_tracer()` extracting `TRACEPARENT` → `fl-run {run_id}` root span with OTLP exporter; `current_trace_id()` helper. **Done when** `test_tracer_extracts_traceparent` and `test_tracer_no_traceparent_makes_root` pass.
4. **`metrics.py`** — `PrometheusMetrics` declaring every §6.2 metric with EXACT names/labels; `start_metrics_server(port)`. **Done when** `test_emit_round_updates_prometheus` (partial) and `test_client_id_not_on_histograms` pass.
5. **`comm_cost.py`** — `CommCostAccumulator` summing uplink/downlink/scalars per round from the gRPC `bytes_received` fields; `snapshot_round(r)`. **Done when** a unit test confirms per-round reset and correct sums.
6. **`callback_client.py`** — `ResultCallbackClient.post_round(dto)` (stdlib urllib, Bearer `flrun_` header, `traceparent` header, 2 s timeout, never raises). **Done when** `test_emit_round_posts_dto`, `test_callback_never_raises_on_backend_down`, `test_callback_short_timeout` pass.
7. **`mlflow_sink.py`** — `MlflowSink` (`set_experiment(project_id)`, `log_params`/`log_metrics`/`log_artifact`, best-effort). **Done when** `test_mlflow_sink_best_effort` passes.
8. **`telemetry.py`** — `TelemetryEmitter.emit_round(...)` orchestrating Prometheus → MLflow → POST → child span (best-effort, ordered per §6.5); `record_client_metrics(...)`; `emit_run_finished(...)`. **Done when** `test_record_client_metrics_feeds_gauges` and the full `test_emit_round_*` suite pass.
9. **`grpc_interceptors.py`** — `build_server_interceptor()` / `build_client_interceptor()`; ensure registration before `grpc.server(...)`. **Done when** `test_grpc_interceptor_registered` passes.
10. **`observability/__init__.py`** — `init_observability(settings)` calling steps 2→8 in the §5.2 order; idempotent. **Done when** importing and calling it twice is a no-op-with-WARN and returns a `TelemetryEmitter`.
11. **Wire the FL server seam (edit `framework .../server.py`, `coordinator.py`, `grpc_servicer.py`)** — call `init_observability()` before `grpc.server(...)`; call `emitter.emit_round(...)` once per round at the aggregation seam (live synchronous path only, NOT `async_coordinator.py`); make the round loop resume-aware (`range(resume_round, ...)`). **Done when** `run_local_test.py` produces a per-round POST and a `fl-round` span in Tempo.
12. **`backend .../observability/ObservabilityConfig.java` + `FlMetrics.java`** — register the Prometheus `MeterRegistry`; declare `fedlearn_active_projects` / `fedlearn_orphaned_processes`. **Done when** `FlMetricsTest.activeProjectsGauge` passes.
13. **`backend .../observability/TraceContextLogFilter.java` + `StompMetricsConfig.java`** — copy `trace_id`/`project_id` into MDC; expose `simp_*` metrics. **Done when** a request log line carries `trace_id` from MDC.
14. **`backend .../resources/application.properties` + `logback-spring.xml`** — set `management.server.port=9090`, expose `prometheus`, JSON log encoder binding MDC keys. **Done when** `ObservabilityConfigTest.prometheusEndpointOnMgmtPort` passes (mgmt port serves it, app port does not).
15. **`observability/otel-collector/config.yaml`, `prometheus/prometheus.yml`, `loki/`, `alloy/`, `tempo/tempo.yaml`** — the five backend configs (§8.3). **Done when** `docker compose up -d` brings all services to ready.
16. **`observability/docker-compose.observability.yml`** — the full local stack incl. MLflow + MinIO. **Done when** §11 step 1 verification curls all return ready.
17. **`observability/grafana/provisioning/*` + the three `dashboards/*.json`** — Dashboard A (§7.1), B (§7.2), C (§7.3) as committed JSON, plus datasource provisioning. **Done when** `dashboards_are_valid_json` passes and all three render in local Grafana against a smoke run.
18. **CI wiring** — add the Python observability tests to `framework.yml` and the JVM tests to `backend.yml`; add the `dashboards_are_valid_json` check (`B7-standards-dx.md §5.1`). Trim the dead `opentelemetry.*` mypy override in `pyproject.toml` now that real imports exist (`B7-standards-dx.md:181`). **Done when** PR CI runs and passes these gates.

---

## 7. (appendix) Three Grafana dashboard layouts (ASCII)

> Placed here as a referenced appendix; §4.3 / §10 / §13 cite "§7.1/§7.2/§7.3". Each dashboard is committed as JSON-as-code (`B3-observability.md:180`). PromQL/LogQL queries are the exact ones from `B3-observability.md §7`.

### 7.1 Dashboard A — Platform Overview (audience: on-call engineer / SRE)

*Source: Micrometer/Prometheus + Loki.*

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  DASHBOARD A — PLATFORM OVERVIEW                          [datasource: Prom]   │
├────────────────────────────┬───────────────────────────────────────────────── ┤
│ Request rate by endpoint   │ p50 / p95 / p99 latency                           │
│ rate(http_server_requests_ │ histogram_quantile(0.99,                          │
│   seconds_count[1m])        │   http_server_requests_seconds_bucket)            │
├────────────────────────────┼─────────────────────────────────────────────────┤
│ 5xx error rate             │ JVM heap + GC pause                               │
│ rate(http_server_requests_ │ jvm_memory_used_bytes ; jvm_gc_pause_seconds      │
│   _count{status=~"5.."}[5m])│                                                   │
├────────────────────────────┼─────────────────────────────────────────────────┤
│ Active FL processes (stat) │ Orphaned/leaked process detector (stat, RED if>0) │
│ fedlearn_active_projects    │ fedlearn_orphaned_processes  (catches C1 F4)      │
├────────────────────────────┴─────────────────────────────────────────────────┤
│ STOMP session count: simp_message_broker_sessions                              │
├────────────────────────────────────────────────────────────────────────────── ┤
│ Live error logs (Loki):  {service="fl-platform-api"} | level="ERROR"           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Dashboard B — Per-Run FL (audience: run owner; template var `$project_id`)

*Source: Prometheus live + MLflow link-out. This is the user's emphasized surface.*

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  DASHBOARD B — PER-RUN FL          var: $project_id      [Prom + MLflow link]  │
│  TEXT PANEL: "Prometheus = healthy NOW; MLflow = compare runs"  (risk #6)      │
├────────────────────────────┬─────────────────────────────────────────────────┤
│ Rounds completed / target  │ Loss & accuracy convergence curves                │
│ fedlearn_rounds_completed_  │ fedlearn_round_loss{project_id="$project_id"}     │
│   total / target (stat)     │ fedlearn_round_accuracy{project_id="$project_id"} │
│                            │ (mirrors in-app recharts ResultsModal)            │
├────────────────────────────┴─────────────────────────────────────────────────┤
│  *** COMMUNICATION-COST PANEL — the DeComFL hero (competitive proof) ***       │
│  fedlearn_uplink_bytes_total + fedlearn_decomfl_scalars_transmitted            │
│  plotted vs fedlearn_model_param_count ; derived stat:                         │
│  "bytes/round vs equivalent FedAvg full-model bytes"  → visualizes the savings │
├────────────────────────────┬─────────────────────────────────────────────────┤
│ Round duration heatmap     │ Comm-vs-compute split (stacked)                   │
│ fedlearn_round_duration_    │ fedlearn_aggregation_seconds under                │
│   seconds_bucket            │   fedlearn_round_duration_seconds                 │
├────────────────────────────┼─────────────────────────────────────────────────┤
│ Active clients (gauge)     │ Round goodput (C1 SLI)                            │
│ fedlearn_round_clients_     │ committed / (committed + rounds_lost + recovery)  │
│   active                    │ from fedlearn_rounds_lost_total / _recovery_      │
├────────────────────────────┴─────────────────────────────────────────────────┤
│ Live logs (Loki): {project_id="$project_id"}   |   [Open run in MLflow ↗]      │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Dashboard C — Client Telemetry (audience: debugging stragglers; vars `$project_id` + `$client_id`)

*Source: heartbeat-driven gauges + MLflow per-client history.*

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  DASHBOARD C — CLIENT TELEMETRY     vars: $project_id, $client_id              │
├────────────────────────────┬─────────────────────────────────────────────────┤
│ Per-client compute / round │ Progress ratio over time                          │
│ fedlearn_client_compute_    │ fedlearn_client_progress_ratio                    │
│   seconds (project-level    │   {project_id="$project_id",client_id="$client_id"}│
│   histogram) + per-client   │                                                   │
├────────────────────────────┴─────────────────────────────────────────────────┤
│ STRAGGLER TABLE — clients sorted by                                            │
│   fedlearn_client_last_heartbeat_age_seconds{project_id="$project_id"} DESC    │
│   (catches the client that dropped mid-round and is stalling aggregation)      │
├────────────────────────────┬─────────────────────────────────────────────────┤
│ Contribution panel         │ Mobile vs desktop vs docker split                 │
│  (MLflow link-out: per-     │ count by (client_type)  (once ReportClientMetrics │
│   client marginal-accuracy  │   lands — §6.5 / E10)                             │
│   / gradient-cosine drift)  │                                                   │
│  BANNER: "proxies, not exact Shapley" (B3 §6.3)                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

> **Contribution/drift scoping (`B3-observability.md §6.3`):** v2 ships a *cheap proxy* — per-client marginal-accuracy delta + gradient-cosine drift logged to MLflow per round — NOT exact Shapley-value contribution (infeasible online). **Uncertainty flagged (do not fabricate):** for DeComFL the clients send *gradient scalars*, not weight deltas, so the update-norm/contribution formula must be defined over projected-gradient magnitude; this needs the paper-alignment (B1) sign-off before implementation (`B3-observability.md:172`). The banner makes the proxy nature explicit to the viewer.

---

*End of 16-LLD-observability.md. Conforms to `02-TECH-STACK.md §20` (pinned versions), `03-DATA-MODEL.md §5.2` (`round_results` columns, `fl_runs.mlflow_run_id`), and `04-API-CONTRACTS.md §5/§10/§11/§12/§13/§14` (callback, gRPC, STOMP, error-envelope, token, traceparent contracts). All existing-code claims cite `path:line` against `main-clean`; all design decisions cite the assigned audit reports `B3-observability.md`, `B7-standards-dx.md`, `C1-reliability-sre.md` under `docs/audit/2026-05-29/`. Authorship is human-only.*
