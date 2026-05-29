# B2 — Tech Stack & Architecture: Build-vs-Adopt for FedLearn v2

**Date:** 2026-05-29
**Agent:** B2 (Tech-stack & architecture)
**Scope:** Production tech stack + architecture for the FL platform. Build-vs-adopt landscape (Flower / NVIDIA FLARE / OpenMined PySyft / FedML-TensorOpera / Apheris / managed). What to keep CUSTOM vs ADOPT. Concrete v2 component stack.
**Builds on:** `docs/audit/2026-05-27/{01-backend,03-framework,05-tooling-ci}.md` and `docs/audit/2026-05-29/00-DESIGN.md`. Cites and extends; does not duplicate.

---

## 0. Executive summary

The v1 platform conflates **two layers that production FL frameworks separate**: a *control plane* (users, projects, RBAC, audit, results — the Spring Boot app, which is genuinely valuable) and an *FL orchestration substrate* (the Python `fl_server.py` that Spring spawns per-project via `ProcessBuilder`). The control plane is salvageable and largely good. The substrate is where the platform reinvents — poorly — what Flower (SuperLink/SuperNode) and NVIDIA FLARE (SCP/CCP) already do at production grade: long-running multi-tenant servers, multi-run job scheduling, mTLS, no per-job port allocation.

The **one true differentiator** is **DeComFL** — zeroth-order, O(1)-per-round communication (ICLR 2025, RIT/Yang). No off-the-shelf framework ships it. Parameter chunking and the parallel-heartbeat dual-stub are *infrastructure papering over a self-inflicted problem*: DeComFL's entire thesis is that you transmit ~1MB total regardless of model size, which makes the >300MB chunker and the long-round heartbeat **largely moot for DeComFL runs** — they exist only for the FedAvg path. That reframing is the core build-vs-adopt decision.

**Recommended posture:** *Keep the control plane (Spring Boot, refactored). Keep DeComFL custom. Stop spawning per-project Python processes — adopt a long-running FL substrate.* The defensible v2 wedge is "the managed control plane + observability + the DeComFL algorithm," **not** a hand-rolled gRPC FL runtime that is strictly behind Flower/FLARE on every axis except DeComFL.

A critical caveat flagged up front (extends C4 in `00-DESIGN.md`): the **mobile native C++ FL core** (libtorch + raw gRPC, `origin/fed-mobile:mobile_client/`) is a hard constraint on any "adopt Flower/FLARE" decision. Flower's `SuperNode` and FLARE's `CCP` are Python client runtimes; **neither has a first-class C++/JNI/ObjC++ on-device SDK**. This is the single biggest reason the gRPC contract (`fedlearn.v1`) likely stays custom — see §4.3.

---

## 1. What v1 actually is (grounding the decision)

| Layer | v1 implementation | Evidence |
|---|---|---|
| Control plane | Spring Boot 3.4.5, Java 21, Gradle; REST + STOMP; owns users/projects/results/identity (V1–V5) | `backend/fl-platform-api/build.gradle:3,8` |
| FL substrate | Per-project Python process spawned by `ProcessBuilder`, tracked in a `ConcurrentHashMap<UUID,Process>` | `flower/FlowerServerManager.java:85,182,200` |
| Port model | One TCP port per running project, range **50000–50010 (11 ports max)** | `application.properties:125-126`; `FlowerServerManager.java:70,73,337` |
| FL algorithms | Custom FedAvg + **DeComFL** (~776 LOC across 4 files) | `decomfl_strategy.py` (255), `zeroth_order.py` (130), `decomfl_client.py` (230), `async_coordinator.py` (161) |
| Wire protocol | Custom gRPC, `package fedlearn.v1`; chunked upload (`CHUNK_SIZE` default 4MB / 50MB on stream) | `protos/fedlearn.proto:3`; `serializer.py:26-27`; `grpc_client.py:192` |
| Artifact store | **None.** Results live in the DB; S3 is a TODO comment only | `ProjectService.java:283,308`; `ProjectController.java:101` |
| Clients | Python (`framework`), Docker (`client-docker`), Electron (`fedlearn-desktop`), **native C++ mobile** (`fed-mobile`) | `00-DESIGN.md §3` |

**The scaling cliff** (named in `00-DESIGN.md` A1): the substrate is `1 OS process + 1 TCP port per concurrent project`, capped at 11, with `findFreePort()` racing under concurrent `/start` (backend audit C4). This is not a tuning problem; it is an architectural mismatch. Production FL substrates are *one long-running server multiplexing many runs* (§2).

---

## 2. Build-vs-adopt landscape (2025–2026)

### 2.1 Comparison matrix

| Framework | License | Maturity | What it provides | What it locks you into |
|---|---|---|---|---|
| **Flower (`flwr`)** | Apache-2.0 | PyPI "5 - Production/Stable"; v2.x; Oxford origin; large community | SuperLink (long-running server) + SuperNode (long-running client), **multi-run / multi-tenancy on one SuperLink, no extra ports per job**, Process-Mode isolation, automated TLS cert provisioning, Docker + Helm, FLARE-runtime interop, framework-agnostic (PyTorch/TF/HF) | Flower's `Driver`/`Fleet` API + ServerApp/ClientApp programming model; **Python client SDK** (no native C++ on-device); its serialization (`flwr.common.Parameters`) |
| **NVIDIA FLARE** | Apache-2.0 | v2.6/2.7; `nvflare` on PyPI; arXiv 2210.13291; healthcare-proven | SCP/CCP control processes, **Job Scheduler + multi-job (jobs share clients/server, no extra ports)**, **mTLS via PKI, federated authorization, built-in audit logs**, sim→prod path, K8s-native "coming soon" | FLARE's job/workflow API (Controller/Executor, FLContext, Shareable); enterprise support via NVIDIA AI Enterprise; heaviest framework |
| **OpenMined PySyft** | Apache-2.0 | Research-oriented; 501(c)(3) non-profit; active 2025–26 but repeated API churn historically | Privacy primitives (secure aggregation, additive secret sharing, DP), data-owner/scientist model, `KotlinSyft`/`SwiftSyft` on-device workers | Syft's compute-graph abstraction; weaker production runtime story; **not** an orchestration substrate for arbitrary PyTorch FL |
| **FedML / TensorOpera** | Open-source core Apache-2.0; **platform (MLOps + Launch scheduler) is proprietary SaaS** | Company since 2022 (rebrand 2024), ~$20M+ raised, 2000+ devs, 10+ enterprise | Open-source `fedml` lib + cross-cloud scheduler ("Launch") + managed MLOps + FedLLM | **Control-plane lock-in**: the valuable part (MLOps, scheduler, monitoring) is the SaaS, exactly the layer this startup wants to *own* |
| **Apheris** | Proprietary, commercial | Series A ($8.25M, Jan 2025); J&J, Roche, AISB network; revenue 4× since late-2023 PMF | Managed federated-computing platform for **life-sciences**; governance, data residency, network operations | Full vertical SaaS lock-in; pricing not public; a *competitor*, not a dependency |
| **Managed/commercial (Owkin, Rhino, NVIDIA AI Enterprise FLARE)** | Proprietary / vertical | Healthcare/pharma vertical SaaS | End-to-end managed FL networks | Vertical lock-in; not a substrate to build on |

Sources: Flower license/maturity/architecture — [PyPI flwr](https://pypi.org/project/flwr/), [Flower architecture](https://flower.ai/docs/framework/explanation-flower-architecture.html), [Flower enterprise patterns (Dec 2025)](https://flower.ai/blog/2025-12-02-enterprise-grade-federated-ai/), [Red Hat: scaling Flower with OCM (Mar 2026)](https://developers.redhat.com/articles/2026/03/05/how-scale-enterprise-federated-ai-flower-and-ocm). FLARE — [GitHub NVFlare 2.6](https://github.com/NVIDIA/NVFlare/tree/2.6), [FLARE system architecture](https://nvflare.readthedocs.io/en/main/system_architecture/system_architecture.html), [FLARE multi-job](https://nvflare.readthedocs.io/en/2.6.0/user_guide/flower_integration/flare_multi_job_architecture.html), [arXiv 2210.13291](https://arxiv.org/abs/2210.13291). Flower↔FLARE interop — [NVIDIA blog](https://developer.nvidia.com/blog/supercharging-the-federated-learning-ecosystem-by-integrating-flower-and-nvidia-flare/), [arXiv 2407.00031](https://arxiv.org/html/2407.00031v2). PySyft — [OpenMined blog](https://openmined.org/blog/fl-in-10-lines-of-code-with-pysyft/), [KotlinSyft](https://github.com/OpenMined/KotlinSyft). FedML/TensorOpera — [GitHub FedML-AI/FedML](https://github.com/FedML-AI/FedML), [TensorOpera Federate docs](https://docs.tensoropera.ai/federate). Apheris — [TechCrunch (Jan 2025)](https://techcrunch.com/2025/01/02/apheris-rethinks-the-ai-data-bottleneck-in-life-science-with-federated-computing/), [SiliconANGLE](https://siliconangle.com/2025/01/02/apheris-raises-8-25m-healthcare-focused-federated-ai-platform/).

### 2.2 The decisive architectural fact

Both production-grade open frameworks already solve v1's biggest cliff. From the research:

- **Flower:** "multiple ServerApps and ClientApps can run within the same federation … a single long-running SuperLink and multiple long-running SuperNodes … sometimes referred to as multi-tenancy or multi-job" — and explicitly "without requiring extra open ports on the server host" ([Flower architecture](https://flower.ai/docs/framework/explanation-flower-architecture.html), [NVIDIA interop blog](https://developer.nvidia.com/blog/supercharging-the-federated-learning-ecosystem-by-integrating-flower-and-nvidia-flare/)).
- **FLARE:** the SCP "schedule, deploy, monitor, and abort jobs" and "creates separate processes for the job" — i.e. job isolation **without** a control-plane-managed port range ([FLARE multi-job](https://nvflare.readthedocs.io/en/2.6.0/user_guide/flower_integration/flare_multi_job_architecture.html)).

v1's `ProcessBuilder`-per-project + 11-port range (`FlowerServerManager.java:200`, `application.properties:126`) is a worse reimplementation of exactly this. **This is the strongest single argument to adopt a substrate** — it's not a feature gap, it's a foundational one.

### 2.3 Why "no `flwr` dependency" is a defensible *invariant* but not a *moat*

CLAUDE.md mandates no `flwr` dependency, and `03-framework.md` H6 flags that `client-docker` *already* violates this via `flwr_datasets` (`client.py:85`, `requirements.txt:7-8`). Two distinct things are conflated under "no Flower":

1. **No `flwr-datasets`** at runtime (a *data-partitioning* helper) — agreed, drop it (H6); HuggingFace `datasets` Dirichlet split replaces it. This is a contamination bug, not an architecture choice.
2. **No Flower *orchestration substrate*** — this is the load-bearing decision and deserves to be re-examined for v2, not treated as religion. The honest reason to keep the substrate custom is **not** "Flower bad" — it is **the native C++ mobile client** and **DeComFL's non-standard scalar-only protocol**, neither of which fits Flower's Python-SDK + `Parameters` model cleanly (§4).

---

## 3. What to keep CUSTOM (the differentiators)

### 3.1 DeComFL zeroth-order — **the actual moat. KEEP CUSTOM.**

DeComFL reduces per-round communication from O(d) to O(1) by sending "only a constant number of scalar values … regardless of the dimension d," transmitting "only around 1MB of data in total … to fine-tune a model with billions of parameters" ([arXiv 2405.15861](https://arxiv.org/abs/2405.15861), [ICLR 2025 OpenReview](https://openreview.net/forum?id=omrLHFzC37), [ZidongLiu/DeComFL](https://github.com/ZidongLiu/DeComFL)). **No off-the-shelf framework (Flower, FLARE, PySyft, FedML) ships DeComFL or ZO-FL as a built-in strategy** (confirmed: search returned only the paper/repo, no framework integration). This is the genuine, paper-backed, IP-attached differentiator (C4 in `00-DESIGN.md` raises the RIT IP-ownership question — material and out of B2 scope, flagged for B4/C4).

**Verdict: salvage the algorithm, refactor the implementation.** The *idea* is the moat; the *code* has correctness debt: the global-RNG mutation `np.random.seed(seed)` clobbers co-resident servers (`decomfl_strategy.py:82`, audit M5) and the O(KP·N) Python-loop aggregation (`decomfl_strategy.py:330`, audit H4) must be vectorized before "dimension-free" is defensible at scale. In v2, DeComFL should be a **pluggable strategy module** behind whatever substrate is chosen, not entangled with the gRPC transport.

### 3.2 Parameter chunking (>300MB) — **REFACTOR, scope it correctly.**

Chunking exists for *large dense tensor transfer* on the FedAvg path (`serializer.py:26-27`, `grpc_client.py:192`). It is **architecturally irrelevant to DeComFL**, whose payload is O(1) scalars — DeComFL never transmits a 300MB model. So:

- It is a *FedAvg-only* concern. If FedAvg is the secondary path (it should be — DeComFL is the differentiator), chunking is a transport-layer detail, not a platform feature.
- It is *table stakes*, not a moat: Flower and FLARE both handle large-model transfer (Flower large-model streaming, FLARE production deployments train large models). If v2 ever adopts a substrate for the FedAvg path, **chunking comes for free**.
- The current implementation is **broken on upload** (`03-framework.md` C1 — `chunks_to_parameters` raises `KeyError 'parameters'`, with a failing unit test already proving it). Any v2 "keep chunking" decision must first fix C1.

**Verdict: refactor** into the transport layer; do not market as a differentiator; fix C1 first.

### 3.3 Parallel heartbeat (dual gRPC stubs) — **REFACTOR; partially obviated.**

The dual-stub design (training stub blocks during `fit()`, heartbeat stub on a parallel thread so the server doesn't time out long rounds) is a real engineering solution to a real problem on the **FedAvg path** (long synchronous rounds). But:

- DeComFL rounds are *short* (forward passes + scalar exchange), so the "long-round timeout" problem the heartbeat solves is **mostly a FedAvg artifact**.
- The implementation has the H1 flaw (`03-framework.md`): heartbeat-thread death is invisible to the training thread; the channel can die mid-`fit()` and the server silently marks the client dead, then rejects the late upload as stale (`coordinator.py:60-65`).
- Both Flower (SuperNode keep-alive) and FLARE (CCP heartbeat to SCP) provide liveness out of the box. So this is *also* table stakes that v1 hand-rolled.

**Verdict: refactor** — keep the mechanism for the FedAvg path, fix H1 (shared `threading.Event`), but do not treat it as a differentiator. If a substrate is adopted for FedAvg, its built-in liveness replaces this entirely.

### 3.4 The honest differentiator stack

| Claimed differentiator | Reality | v2 disposition |
|---|---|---|
| DeComFL ZO O(1) comms | Genuine, paper-backed, unique | **KEEP CUSTOM** (refactor impl) |
| Parameter chunking | Table stakes; FedAvg-only; currently broken | **REFACTOR** into transport; fix C1 |
| Parallel heartbeat | Table stakes; FedAvg-only; has liveness bug | **REFACTOR**; fix H1 |
| Per-project process spawn | Anti-pattern; substrates do this better | **KILL** |
| Custom gRPC `fedlearn.v1` | Justified *only* by native C++ mobile + DeComFL scalars | **SALVAGE** (govern with `buf`) |
| Control plane (auth/RBAC/audit/identity) | Real product value | **SALVAGE / REFACTOR** |

---

## 4. The v2 architecture decision

### 4.1 The two-plane split

Production FL platforms (FLARE's SCP/CCP, Flower's SuperLink/SuperNode) all separate **control plane** from **FL substrate**. v2 must too:

```
                ┌─────────────────────────── CONTROL PLANE (own this) ──────────────┐
  Browser/SPA ──┤ Spring Boot: auth (cookie JWT), org/project RBAC, audit, results,  │
  Desktop/Mobile│  run-launch API, observability fan-in, artifact-store front        │
                └───────────────┬───────────────────────────────────────────────────┘
                                │ gRPC (Spring gRPC 1.0 GA) — internal, mTLS
                ┌───────────────▼─────────────── FL SUBSTRATE (decide: adopt vs custom) ┐
                │  Long-running FL server(s) multiplexing runs (NOT 1 process/project)  │
                │   Strategies: DeComFL (custom) | FedAvg/FedProx (custom or adopted)   │
                └───────────────┬───────────────────────────────────────────────────────┘
                                │ gRPC fedlearn.v1 (custom proto, buf-governed) — WAN, mTLS
       Python / Docker / Electron / native C++ mobile clients
```

### 4.2 The substrate decision — three options

**Option A — Adopt Flower SuperLink/SuperNode as the substrate; run DeComFL as a custom Flower `Strategy`.**
- Pros: kills the spawn cliff for free; gets multi-run, TLS provisioning, Helm/Docker, large-model handling, liveness — all the table-stakes work the audit flags as broken or missing.
- Cons (decisive): **violates the stated "no `flwr`" invariant** at the substrate level; **no native C++ on-device SuperNode** — the mobile core (`mobile_client/shared/src/*.cpp`) cannot be a Flower client without a C++ gRPC↔Flower-protocol shim that does not exist; DeComFL's scalar protocol is awkward inside Flower's `Parameters` model.

**Option B — Adopt NVIDIA FLARE as the substrate.**
- Pros: strongest enterprise security posture out of the box (mTLS via PKI, federated authz, built-in audit logs — directly answers B4 concerns), proven in healthcare (relevant to the pneumonia demo and likely vertical).
- Cons: heaviest framework; FLARE's Controller/Executor/Shareable job model is a large adoption surface; **same native-C++-client gap**; risks coupling the startup to NVIDIA's ecosystem and AI-Enterprise support model.

**Option C — Keep a custom substrate, but rebuild it as a long-running multi-tenant server (steal the SuperLink/SCP pattern, not the dependency).**
- Pros: preserves the custom `fedlearn.v1` proto the native C++ mobile client already speaks; keeps DeComFL first-class; honors the invariant; no new heavyweight dependency.
- Cons: the startup keeps maintaining FL-runtime plumbing (liveness, TLS, job scheduling, isolation) that Flower/FLARE maintain for free — *opportunity cost*, and the audit shows v1 is bad at exactly this plumbing.

**Recommendation: Option C, but explicitly a rebuild, and benchmark-gated against Option A.**

Rationale: the **native C++ mobile FL core is the constraint that tips it.** `00-DESIGN.md §3` confirms the mobile client is a real, substantial libtorch+gRPC C++ implementation of DeComFL — it speaks `fedlearn.v1` directly. Adopting Flower/FLARE would orphan it or force a C++↔Flower shim that is *more* work than fixing the custom substrate. Combined with DeComFL's non-standard scalar protocol, the custom proto earns its keep. **But the per-project `ProcessBuilder` spawn model must die** (verdict: kill): v2's substrate is **one long-running gRPC server per node, multiplexing runs by `run_id`**, exactly like SuperLink. The control plane's `/start` becomes "submit a run to the substrate," not "fork a Python process and grab a port."

> **Skeptic-bait, stated honestly:** Option C is the higher-maintenance path and reasonable engineers would pick A. The decision hinges entirely on (a) the native C++ mobile client being non-negotiable and (b) DeComFL being the product. If either weakens — if mobile is cut, or if FedAvg becomes the primary product — **flip to Option A (Flower) immediately**; the table-stakes savings dominate. I recommend a **2-week spike**: implement DeComFL as a Flower `Strategy` and measure whether the mobile C++ client can be bridged. Let evidence, not the invariant, settle it.

### 4.3 Why the custom gRPC proto stays (and how to govern it)

The proto is consumed by **four languages** (Java backend `option java_package = "com.fedlearn.v1"` at `fedlearn.proto:5`, Python framework, TS desktop, **C++ mobile**), and `00-DESIGN.md §3` already found **proto drift**: the mobile `shared/proto/` copy has a malformed `SubmitModelUpdate(SubmitModelUpdateReque…)` and diverges from canonical `fedlearn.v1`. This is a governance failure, not a reason to abandon the custom proto.

**Adopt `buf`** as the single source of truth: one `buf.gen.yaml` generates Java/Python/TS/C++ from one canonical `.proto`; `buf lint` + `buf breaking` gate PRs; optionally publish to the Buf Schema Registry so each client `npm/pip/gradle install`s a generated package instead of vendoring drifting copies ([Buf](https://buf.build/), [Connect RPC codegen](https://connectrpc.com/docs/web/generating-code/), [multi-language codegen guide](https://oneuptime.com/blog/post/2026-01-08-grpc-code-generation-multiple-languages/view)). This directly answers B7's "proto codegen across 4 languages" item and eliminates the drift class. **Verdict: salvage proto + adopt buf tooling.**

---

## 5. Concrete v2 component stack

| Concern | v1 | v2 recommendation | Rationale / trade-off |
|---|---|---|---|
| **Control-plane lang/framework** | Spring Boot 3.4.5, Java 21 | **Keep Spring Boot (bump to 3.5.x / Boot 4 line), Java 21+** | The auth/RBAC/audit/V1–V5 identity investment is real and good (`01-backend.md`); rewriting in Go/FastAPI throws away the most valuable, least-broken layer. Spring gRPC 1.0 GA now gives native gRPC to the substrate without a REST bridge ([Spring gRPC GA](https://piotrminkowski.com/2025/12/15/grpc-spring/)). |
| **FL orchestration substrate** | `ProcessBuilder`-per-project, 11 ports | **Long-running multi-tenant custom server (Option C); spike Flower (Option A) in parallel** | §4.2. Kills the spawn cliff (`FlowerServerManager.java:200`) and the 11-port cap (`application.properties:126`). Substrate multiplexes runs by `run_id`. |
| **FL algorithms** | Custom FedAvg + DeComFL | **DeComFL custom (refactor); FedAvg custom or delegated** | §3.1. DeComFL is the moat; FedAvg is table stakes. |
| **gRPC / proto tooling** | hand-run `protoc`, vendored copies, drift | **Custom `fedlearn.v1` proto + `buf` (lint/breaking/codegen/BSR)** | §4.3. Single source of truth across Java/Python/TS/C++; fixes the mobile proto drift. |
| **Datastore** | H2 (dev/ec2demo file-mode), Postgres driver present | **PostgreSQL (managed: RDS/Aurore or Neon/Supabase at seed tier)** | Flyway already targets Postgres (`flyway-database-postgresql` at `build.gradle:46`); H2 is a POC crutch. `01-backend.md` correctly recommends Testcontainers-Postgres to surface H2↔PG divergence. Keep Flyway-owned schema invariant. |
| **Streaming / message layer (logs)** | In-memory STOMP broker, single replica | **Keep STOMP/WS at the edge; back it with Redis Pub/Sub (or NATS) for multi-replica fan-out** | `01-backend.md` M9: the in-memory broker can't route STOMP user-destinations across replicas. Redis-backed relay (or NATS) unblocks horizontal scale of the control plane without a Kafka-sized commitment at seed stage. |
| **Run/telemetry event bus** | None (`RoundResult` telemetry pipeline is empty per `00-DESIGN.md` B3) | **NATS JetStream (seed) → Kafka (only if volume demands)** for per-round/per-client FL telemetry | Decouples substrate→control-plane telemetry; durable replay for the observability story (defer detail to B3). NATS is operationally lighter than Kafka for a startup. |
| **Artifact / model store** | **None** (DB blobs + S3 TODO) | **S3-compatible object store (AWS S3, or MinIO self-host); content-addressed model checkpoints** | `ProjectService.java:283,308` already anticipates this. Critical for C3 (reproducibility) and DeComFL's checkpoint/round-recovery (C1 reliability). Models never belong in Postgres rows. |
| **Experiment tracking** | none | **MLflow (self-hosted)** over W&B | `03-framework.md` and `04-observability.md` already recommend MLflow for the on-prem/federated bias; defer to B3. |
| **Compute orchestration** | bare EC2, Python processes | **Substrate on K8s (managed: EKS/GKE) at Series-A; Docker Compose / single VM at seed** | Flower Helm charts / FLARE K8s-soon both assume this; defer cost tiers to B6. |
| **Mobile FL** | native C++ libtorch+gRPC | **Keep native C++ core; reconcile proto via buf; lift `mobile_client/` subtree later** | §4.3 + `00-DESIGN.md §3`; detail deferred to A6/B5. |

### 5.1 Trade-offs called out explicitly

- **Spring Boot vs a rewrite:** A startup's instinct is "rewrite the slow Java in Go/Python." Wrong here — the Java *control plane* is the working, valuable part; the *substrate* (Python) is the problem, and that's already Python. Don't rewrite the healthy organ. Java's enterprise auth/RBAC/audit story (Spring Security, the V5 identity layers) is hard to reproduce quickly in FastAPI/Go.
- **Custom substrate vs Flower:** higher maintenance (you own liveness/TLS/scheduling) traded for native-C++-mobile + DeComFL fit. Re-evaluate every quarter; the moment mobile or DeComFL stops being core, adopt Flower.
- **NATS vs Kafka:** NATS JetStream is the right seed-stage choice (lighter ops); Kafka only if FL-telemetry volume or multi-consumer fan-out demands it. Don't pay the Kafka tax early.
- **MinIO vs S3:** MinIO matches the on-prem/federated bias and avoids egress cost for self-hosted customers; S3 for the managed SaaS. Object store either way — never DB blobs.

---

## 6. Decision table (salvage / refactor / rebuild / kill)

| Module / subsystem | Verdict | One-line rationale |
|---|---|---|
| Control plane (Spring Boot app) | **salvage** | Auth/RBAC/audit/V5 identity are the most valuable, least-broken layer; keep and harden per `01-backend.md`. |
| FL substrate (`ProcessBuilder`-per-project model) | **kill** | 1 process + 1 of 11 ports per project is an architectural anti-pattern that Flower/FLARE solve with one long-running multi-run server. |
| FL substrate (as a *concept*, rebuilt long-running) | **rebuild** | Replace spawn model with a SuperLink-style multi-tenant server keyed on `run_id`; benchmark-gate against adopting Flower. |
| DeComFL algorithm (the idea/IP) | **salvage** | The only genuine, paper-backed differentiator; no framework ships it. |
| DeComFL implementation (`decomfl_strategy.py`) | **refactor** | Vectorize O(KP·N) loop (H4), fix global-RNG clobber (M5), make it a pluggable strategy. |
| Parameter chunking | **refactor** | Table-stakes FedAvg-only transport detail, currently broken (C1); fix and demote from "feature." |
| Parallel heartbeat (dual stubs) | **refactor** | Real for FedAvg, mostly moot for DeComFL, has a liveness bug (H1); keep + fix, don't market it. |
| Custom `fedlearn.v1` proto | **salvage** | Justified by native C++ mobile + DeComFL scalars; govern with `buf` to kill drift. |
| Proto codegen / vendored copies | **rebuild** | Drift (mobile copy diverges); replace ad-hoc protoc with `buf` single-source-of-truth. |
| `flwr-datasets` runtime dep | **kill** | Contamination bug (H6); replace with HF `datasets` Dirichlet split. |
| Datastore (H2 file-mode) | **rebuild** | POC crutch; move to managed Postgres (Flyway already targets it). |
| STOMP in-memory broker | **refactor** | Works single-replica; back with Redis/NATS relay for horizontal scale (M9). |
| Artifact store | **rebuild** (greenfield) | Doesn't exist; add S3/MinIO object store — prerequisite for reproducibility + checkpointing. |
| Run-telemetry bus | **rebuild** (greenfield) | Empty today; add NATS JetStream for FL-run observability (hand to B3). |

---

## 7. Prioritized recommendations

**P0 — settle the substrate question before any v2 code.**
1. Run the **2-week Flower-strategy spike** (§4.2): implement DeComFL as a `flwr` `Strategy`, attempt a C++-mobile↔Flower bridge. Decision criterion: *can the native C++ client participate without a bespoke shim?* If yes → Option A; if no → Option C. This single experiment de-risks the whole v2.
2. **Kill the `ProcessBuilder`-per-project model** regardless of A/C — design the substrate as one long-running multi-run server (`run_id`-keyed), removing the `findFreePort()` race (C4) and the 11-port cap (M2) by construction.

**P1 — lock the contracts.**
3. Adopt **`buf`** as proto single-source-of-truth; generate Java/Python/TS/C++; gate `buf lint`/`buf breaking` in CI. Reconcile the mobile proto drift first.
4. Stand up **managed Postgres** + Testcontainers-PG tests; retire H2 outside the `test` profile (keep the Flyway invariant).
5. Add the **S3/MinIO artifact store** + content-addressed checkpoints (unblocks C1 reliability and C3 reproducibility).

**P2 — scale-out plumbing.**
6. Back STOMP with **Redis/NATS** relay (M9) so the control plane scales past one replica.
7. Stand up **NATS JetStream** telemetry bus + MLflow (coordinate with B3).
8. Refactor DeComFL impl (H4 vectorization, M5 RNG) and fix chunking (C1) + heartbeat (H1) on the FedAvg path.

**Cross-cutting flags (hand-offs):**
- **RIT IP ownership of DeComFL** is the gating commercial question for "DeComFL is our moat" → **C4 / B4**.
- Per-tier cost modeling of substrate-on-K8s vs single-VM → **B6**.
- Native-vs-Electron-vs-shared-C++-core for desktop, and the mobile subtree lift → **B5 / A6**.
- mTLS for the WAN gRPC (audit item #37) — FLARE-style PKI is the reference; → **B4**.

---

## 8. Uncertainty / things I could not verify

- **Flower native C++ client:** I found no first-class C++/JNI/ObjC++ SuperNode SDK in the Flower docs surfaced; I am *fairly* but not *certain* there is none as of the searched material. The Option-C recommendation hinges partly on this — the §7-P0 spike is designed precisely to confirm it before committing. (Flagged uncertain rather than asserted.)
- **FedML/TensorOpera exact license split:** the open-source `fedml` core is Apache-2.0; the *platform/MLOps/Launch* layer is the proprietary SaaS. I did not open the LICENSE file directly; treat the "platform is proprietary" claim as high-confidence-from-positioning, not file-verified.
- **Apheris pricing:** not public; flagged as such.
- DeComFL convergence/communication guarantees are from the paper ([arXiv 2405.15861](https://arxiv.org/abs/2405.15861)); whether the *v1 implementation* preserves them is **B1's** question, not B2's — I deliberately do not re-litigate it.
