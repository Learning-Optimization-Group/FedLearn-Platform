# B6 — Scale, Cost & Infrastructure Economics

**Date:** 2026-05-29
**Agent:** B6 (scale / cost / infra economics)
**Branch:** `main-clean`
**Builds on:** [`2026-05-27/01-backend.md`](../2026-05-27/01-backend.md) (FlowerServerManager scaling cliff, M2 11-port cap, M9 in-memory STOMP broker), [`2026-05-27/04-observability.md`](../2026-05-27/04-observability.md) (observability stack + cardinality risks → cost), and the [`00-DESIGN.md`](00-DESIGN.md) tiered-sizing ask.

> **Scope of this report:** dollar-grade tiered sizing (seed → Series-A → hyperscale), the unit economics of the *FL-server-per-project* primitive at each tier, DB / compute / storage / egress / observability cost models, and adoption-curve benchmarking. I do **not** re-litigate the correctness bugs the per-unit audits own; I cost the architecture they describe.

---

## 0. TL;DR

| Tier | Orgs / Active users | Concurrent FL projects | Model-size band | All-in infra $/mo (est.) | Dominant cost driver |
|---|---|---|---|---|---|
| **Seed** | 1–25 orgs / ~100 users | **≤ 8 concurrent** (today capped at **11** by the port range) | 0.1M–25M params (CNN/MLP/small ViT) | **$350 – $900** | Always-on backend + RDS baseline (FL compute is bursty/cheap) |
| **Series-A** | 25–500 orgs / ~5k users | 50–300 concurrent | 1M–1.3B params (incl. LLM fine-tune) | **$4k – $14k** | FL-server compute (Fargate/EKS task-hours) + DB |
| **Hyperscale** | 500+ orgs / 50k+ users | 1k–10k concurrent | up to 7B+ params | **$60k – $250k+** | FL-server GPU compute + control-plane DB + observability cardinality |

**The single largest cost lever is the strategy mix, not the cloud bill.** DeComFL transmits **~1 MB total** between server and client to fine-tune a *billion-parameter* model ([arXiv 2405.15861](https://arxiv.org/abs/2405.15861)); a full-parameter FedAvg round on the same model moves **~2.6 GB per client per round** (1.3B params × 2 bytes fp16 × 1 download + ~ same up). At AWS internet egress of **$0.09/GB** for the first 10 TB/mo ([S3 pricing](https://aws.amazon.com/s3/pricing/)), a 100-round / 100-client LLM run costs **~$23,000 in egress under FedAvg vs ~$0.0009 under DeComFL** — a *six-to-seven-order-of-magnitude* swing. **DeComFL is the product's cost moat; the cost model below treats it as a first-class pricing tier, not a research curiosity.**

---

## 1. The unit that must change: FL-server-per-project

### 1.1 What exists today

The platform's compute primitive is *one FL server process per running project*. Two code paths already exist in `FlowerServerManager.startServerForProject()` (`backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/flower/FlowerServerManager.java:94-101`):

1. **Local `ProcessBuilder` path** (`startLocalServer`, line 150) — spawns `bash run_fl_server.sh` on a port from a **fixed range `50000–50010`** (`application.properties:125-126`), tracked in a `ConcurrentHashMap<UUID, Process>` (`FlowerServerManager.java:85`).
2. **ECS Fargate `RunTask` path** (`startEcsFargateServer`, line 103) — already coded, gated by `ecs.cluster-name` being non-blank. Each project becomes a Fargate task via `EcsClient.runTask(...)` (line 133), env-injected with project config.

**This is further along than the 00-DESIGN framing implies** — the scheduler abstraction is half-built. The Fargate path is the right shape; it just needs hardening (it's part of the *unfinished* `production` profile per CLAUDE.md) and a capacity/cost-control layer.

### 1.2 The scaling cliff (confirmed, extended)

The 2026-05-27 backend audit flagged **M2: "single instance can never run more than 11 concurrent projects"** (`01-backend.md:80`). I extend that with the cost-relevant consequences:

- **Port range is the hard ceiling on the local path.** 11 ports → 11 concurrent projects *per backend instance*, regardless of host size. This is a config artifact, not a resource limit — but it caps seed-tier throughput.
- **Process lifetime ≠ request lifetime.** An FL run lasts **hours** (C1 reliability audit territory). The spawned process pins a port and host memory for the entire run. On a single EC2 host, 11 long-lived Python servers each loading a model into RAM is the real constraint, not the port count.
- **No bin-packing, no autoscaling, no queue.** `startServerForProject` either finds a free port or fails. There is no admission control, no "your run is queued" state, no scale-to-zero. At seed tier this is fine; at Series-A it's a reliability *and* cost problem (over-provision to avoid rejections = waste).
- **In-memory STOMP broker caps horizontal scale** (`01-backend.md:M9`, `WebSocketConfig:39`). You cannot simply run N backend replicas behind an ALB, because `/topic/logs/{projectId}` user-destinations won't route between replicas. This forces vertical scaling of the backend until a relay broker (RabbitMQ/ActiveMQ STOMP relay, or Redis-backed) lands.

**Verdict on the orchestration primitive: `refactor` (seed) → `rebuild` (Series-A+).** The per-project-server *concept* is sound and maps cleanly to k8s Jobs / ECS RunTask. The *local ProcessBuilder implementation* must be killed for any tier above seed.

### 1.3 Target orchestration by tier

| Tier | FL-server runtime | Why |
|---|---|---|
| **Seed** | `ProcessBuilder` on a single right-sized EC2/Fargate-service host, **port range raised to 50000–50100** (100 concurrent), reader-thread hazard fixed (`01-backend.md:H4`). | Cheapest possible. No orchestrator tax. 100 ports >> 8 expected concurrent. |
| **Series-A** | **ECS Fargate `RunTask` per project** (the path already in `startEcsFargateServer`). One task = one FL server. Scale-to-zero by definition: you pay only while a run is active. | No control-plane fee (vs EKS $73/mo). Per-second billing, 1-min minimum ([Fargate pricing](https://aws.amazon.com/fargate/pricing/)). Ideal for bursty hours-long runs. |
| **Hyperscale** | **EKS with Karpenter** (or Fargate for CPU runs + a GPU node pool for ≥1B-param fine-tunes). FL server = a k8s `Job`; GPU runs land on `g5`/`g6` nodes via nodeSelector. | Bin-packing many small CPU FL servers onto fewer nodes beats Fargate's per-task overhead above ~$5k/mo of task-hours; GPU scheduling needs k8s device plugins Fargate doesn't offer. |

**The Fargate-vs-EKS crossover** is well-documented: Fargate has no control-plane fee but a per-task premium; EKS adds a flat **$73/mo** control-plane (standard support) plus node cost but bin-packs ([EKS pricing](https://aws.amazon.com/eks/pricing/), [Vantage Fargate analysis](https://www.vantage.sh/blog/fargate-pricing)). For ≤ ~30–50 steady concurrent tasks, Fargate wins on simplicity and cost; above that, EKS+Karpenter wins.

---

## 2. The cost moat: DeComFL communication economics

This is the most important section for a *startup* pitch, so I quantify it precisely against the code.

### 2.1 What the code actually transmits

`decomfl_strategy.py` confirms the data plane: clients send **gradient scalars `[K][P]`** plus the strategy regenerates perturbations from **seeds** (`decomfl_strategy.py:148-208`, `_generate_perturbation` line 210). Defaults are **K=1 local step, P=10 perturbations** (`decomfl_strategy.py:33-34`). So a DeComFL uplink is **~K×P = 10 float32 scalars (~40 bytes) + bookkeeping per client per round** — independent of model dimension `d`. Downlink is seeds + scalar history for `rebuild_model` (`get_rebuild_history`, line 128), still O(K·P·rounds-missed), not O(d).

Contrast the **FedAvg path**, which moves full parameter tensors and is exactly why the platform built **parameter chunking** (`serializer.py`, default chunk **4 MB** — note the code comment says 50 MB but `_DEFAULT_CHUNK_SIZE_MB` defaults to `"4"`, `serializer.py:24-27`) and a **1 GB gRPC message ceiling** (`server.py:90-91`, `grpc_client.py:82-83`).

### 2.2 Egress cost per run (the headline number)

Assume a **1.3B-param** model, fp16 (2 B/param) ≈ **2.6 GB** full-model transfer, a **100-round** run, **100 clients/round**, both up + down.

| Strategy | Bytes / client / round | Total run transfer | Internet egress cost @ $0.09/GB |
|---|---|---|---|
| **FedAvg (full params)** | ~5.2 GB (2.6 GB down + 2.6 GB up) | ~52 TB | **~$4,700** (first-10TB tier; tiers down to $0.07/GB above 100 TB) |
| **DeComFL** | ~80 B (10 scalars up + seeds/history down) | < 1 MB ([paper claim](https://arxiv.org/abs/2405.15861)) | **< $0.001** |

> **Caveat (flagged):** the platform's gRPC is **plaintext over WAN** (audit item #37, `grpc.insecure_channel` at `grpc_client.py:55`; a `secure_channel` path exists unused at line 72). Egress is billed regardless of encryption, so TLS does not change these numbers — but TLS adds ~2–5% overhead and is a security, not cost, decision (see B4).

**Pricing implication:** DeComFL runs are *nearly free to serve* on the data plane. The platform should **price LLM fine-tuning as a premium tier while its marginal egress cost is ~0** — the cost is compute (the client's local forward passes), not network. This is a defensible margin story for the GTM agent (C4).

### 2.3 The asymmetry to watch

DeComFL collapses *network* cost but **not compute**. Zeroth-order estimation needs **P forward passes per step** (`zeroth_order.py`), so a client doing P=10 does ~10× the forward compute of one inference. On the *server*, `aggregate_fit` regenerates each perturbation `z` over the full flattened model for every (client, k, p) tuple (`decomfl_strategy.py:180-197`) — the 2026-05-27 framework audit's "O(KP·N)" concern. **At hyperscale this server-side reconstruction is a CPU/GPU cost, not a network cost** — it moves the bottleneck onto the FL-server task sizing, which the tier tables below account for.

---

## 3. Tiered sizing with concrete cost models

All prices **us-east-1, on-demand, 2026**, ~730 hrs/mo. Reserved/Savings Plans and Graviton (ARM) cut 20–60% — noted where material. These are *infrastructure* estimates; they exclude salaries, third-party SaaS seats, and the GPU cost of clients (clients are user-owned hardware — desktop/Jetson/mobile — by design).

### 3.1 SEED tier — design-partner / research stage

**Profile:** 1–25 orgs, ~100 users, ≤8 concurrent FL projects, models 0.1M–25M params (the MNIST/CIFAR/ViT-tiny band the framework targets today). This is the current `ec2demo` reality, productionized.

| Component | Choice | $/mo | Notes |
|---|---|---|---|
| Backend (Spring Boot) | 1× Fargate service, 1 vCPU / 4 GB, always-on | **~$60** | 1 vCPU = $0.04048/hr ([Fargate](https://aws.amazon.com/fargate/pricing/)) × 730 + 4 GB × $0.004445 × 730 ≈ $42 + $13. Graviton: ~$48. |
| Database | **RDS Postgres `db.t4g.micro/small`, Single-AZ** | **~$25–60** | Kill H2-file-mode (CLAUDE.md notes `ec2demo` still on H2). t4g.small ≈ $0.032/hr. 20 GB gp3 ≈ $2. |
| FL-server compute | `ProcessBuilder` on the **same backend host** OR small Fargate tasks on demand | **~$30–150** | 8 concurrent CPU runs × few hrs/day. Bursty → cheap. Raise port range to ≥50100. |
| Model/artifact storage | **S3 Standard** | **~$5–20** | Even 1,000 × 25M-param fp32 checkpoints (~100 MB each) = 100 GB × $0.023 = $2.30 ([S3](https://aws.amazon.com/s3/pricing/)). |
| Egress | gRPC traffic + dashboard | **~$10–30** | Small models, DeComFL-dominant. First 100 GB/mo free. |
| Observability | **Self-hosted Prometheus+Grafana+Loki on the backend host**, OR Grafana Cloud free tier | **~$0–50** | Grafana Cloud free: 10k series, 50 GB logs ([Grafana pricing](https://monitoringcost.com/grafana-cloud-pricing)). 04-observability's stack fits the free tier at this volume. |
| ALB + misc (NAT, logs, DNS) | | **~$40–80** | ALB ~$22 + NAT GW ~$33 + CloudWatch. NAT is a silent tax — consider a NAT instance at this tier. |
| **Total** | | **~$350 – $900/mo** | |

**Dominant driver:** always-on backend + DB baseline. FL compute is noise. **Optimization:** scale-to-zero is impossible for the backend (it must accept logins), but the DB can be `db.t4g.micro` and observability can be self-hosted on the same box. Do **not** adopt Aurora here — Serverless v2 floor of 0.5 ACU ≈ **$0.06/hr → ~$44/mo just idling** ([Aurora pricing](https://aws.amazon.com/rds/aurora/pricing/)) plus I/O charges, with no benefit at this volume.

### 3.2 SERIES-A tier — multi-tenant production

**Profile:** 25–500 orgs, ~5k users, 50–300 concurrent FL projects, models 1M–1.3B params (LLM fine-tune via DeComFL becomes a real workload). Multi-AZ, real SLOs.

| Component | Choice | $/mo | Notes |
|---|---|---|---|
| Backend | 2–4× Fargate (2 vCPU/8 GB) behind ALB, **STOMP relay broker required** | **~$600–1,200** | Per-replica ~$160. **Blocker:** in-memory broker (`01-backend.md:M9`) must become a RabbitMQ/ActiveMQ STOMP relay before >1 replica — add **Amazon MQ (RabbitMQ) mq.m5.large ~$180/mo** or self-host. |
| Database | **RDS Postgres `db.r6g.large` Multi-AZ** (2 vCPU/16 GB) | **~$330** | $0.225/hr × 730 ([db.r6g.large](https://instances.vantage.sh/aws/rds/db.r6g.large)) × ~2 for Multi-AZ, partly offset by 1-yr RI (−~40%). Add ~$30 storage. |
| FL-server compute | **ECS Fargate RunTask per project**, CPU 2 vCPU/8 GB for ≤25M; 4 vCPU/16 GB for LLM-DeComFL | **~$1,500–6,000** | This becomes the **dominant variable cost**. 300 concurrent × ~3 hr avg run × ~$0.10/hr/task, churned daily. Per-second billing means idle = $0. See §3.4 unit economics. |
| Model/artifact storage | S3 Standard + Intelligent-Tiering | **~$50–300** | 1.3B fp16 checkpoint ≈ 2.6 GB; 5,000 checkpoints ≈ 13 TB × $0.022 ≈ $290. Lifecycle old rounds to S3-IA/Glacier. |
| Egress | DeComFL-dominant → tiny; FedAvg LLM runs → expensive | **~$200–2,000** | **Steer users to DeComFL for large models.** A handful of FedAvg-on-LLM runs can dominate this line (see §2.2). Put VPC endpoints on S3 to kill in-region egress. |
| Observability | Grafana Cloud Pro or self-hosted Prometheus/Loki/Tempo on a dedicated node | **~$300–1,500** | 50–300 concurrent runs × per-round metrics. **Cardinality is the cost knob** (`04-observability.md` Risk 1): keep `client_id` off histograms, send per-client detail to MLflow. Grafana Cloud Pro $19 + ~$8/1k series + $0.50/GB logs ([Grafana](https://monitoringcost.com/grafana-cloud-pricing)). |
| ALB/NAT/WAF/misc | | **~$200–400** | |
| **Total** | | **~$4,000 – $14,000/mo** | |

**Dominant driver:** FL-server compute task-hours, then DB. **Optimizations that matter most here:**
1. **Fargate Spot for interruption-tolerant runs** (−~70%) — but FL runs are stateful; only safe once C1's round-checkpointing/`rebuild_model` recovery exists.
2. **Graviton everywhere** (ARM Fargate is ~21% cheaper: $0.0089944 vs $0.011244 per vCPU-sec, [Fargate pricing](https://aws.amazon.com/fargate/pricing/)). The framework is PyTorch — ARM wheels exist; the Jetson path already proves ARM viability.
3. **Right-size by model band**, not one task size for all.

### 3.3 HYPERSCALE tier — platform business

**Profile:** 500+ orgs, 50k+ users, 1k–10k concurrent projects, up to 7B+ params, GPU-backed server-side ZO reconstruction, global, compliance-grade.

| Component | Choice | $/mo | Notes |
|---|---|---|---|
| Backend | EKS-hosted, 6–20 pods autoscaled, multi-region | **~$3,000–8,000** | EKS control plane $73 + node fleet + relay broker cluster. |
| Database | **Aurora PostgreSQL** (writer + 2 readers, r6g.2xl class) **or Citus/distributed** for control-plane sharding | **~$3,000–12,000** | At this org/audit-event/round-result volume, Aurora's storage auto-scaling + read replicas earn their keep. **Decision point:** if `audit_events` + `round_result` + `server_logs` write volume saturates a single writer, move time-series rows (logs/round-results) **out of Postgres entirely** into Loki/ClickHouse/Timescale and keep Aurora for identity + control plane. **I do not recommend Citus unless a single Aurora writer is provably saturated** — distributed Postgres adds operational cost most platforms never need. |
| FL-server compute | **EKS + Karpenter**: CPU node pool (Graviton) + **GPU node pool (`g5.xlarge`/`g6`)** for ≥1B-param ZO reconstruction | **~$30,000–150,000** | The dominant line. GPU is needed because `aggregate_fit` regenerates perturbations over the full model (§2.3). 7B-param server-side ZO needs GPU memory. Spot GPU + checkpointing is the cost survival strategy. |
| Storage | S3 + Intelligent-Tiering + Glacier lifecycle | **~$2,000–15,000** | 7B fp16 ≈ 14 GB/checkpoint; registry of 100k+ checkpoints → petabyte-class. Tiering and dedup (content-addressed by hash) are mandatory. |
| Egress / networking | CloudFront for dashboard assets, VPC endpoints, cross-region replication | **~$3,000–30,000** | **Still DeComFL-dominated on the FL data plane** — the moat holds. Egress here is mostly dashboard/CDN + cross-region DB replication, *not* model transfer, *if* DeComFL is the default. |
| Observability | Self-hosted Grafana/Mimir/Loki/Tempo cluster (Datadog is cost-prohibitive at this host count) | **~$5,000–25,000** | Datadog $80–120/host/mo ([Datadog cost](https://leanopstech.com/blog/datadog-vs-grafana-cloud-pricing-2026/)) × hundreds of hosts = untenable. **Self-host or use Grafana Cloud with aggressive cardinality control.** This is why 04-observability's cardinality discipline is a *cost* control, not just a hygiene one. |
| **Total** | | **~$60,000 – $250,000+/mo** | Highly GPU-mix-dependent. |

**Dominant driver:** GPU FL-server compute and storage. **The whole hyperscale cost case rests on DeComFL keeping egress flat while model sizes grow** — if the platform lets users default to FedAvg on 7B models, egress alone could rival compute.

### 3.4 Unit economics of one FL-server-per-project

This is the number that decides pricing. **One Series-A-tier CPU run:**

- Fargate 4 vCPU / 16 GB = (4 × $0.04048) + (16 × $0.004445) = **$0.233/hr** (x86) or **~$0.18/hr** (Graviton) ([Fargate pricing](https://aws.amazon.com/fargate/pricing/)).
- A 3-hour, 50-client DeComFL LLM-fine-tune run: **~$0.70 compute + ~$0 egress + negligible storage = ~$0.70 cost-to-serve.**
- The *same* run under FedAvg: **~$0.70 compute + ~$2,400 egress** (50 clients × 100 rounds × 5.2 GB × $0.09). **3,400× cost difference, all network.**

**Pricing recommendation:** meter on **run-hours × model-band**, with DeComFL runs priced at a healthy margin over a ~$1 cost-to-serve and FedAvg-on-large-models either surcharged for egress or quietly steered toward DeComFL. The unit economics *only* work if the orchestrator (a) scales to zero between runs (Fargate/Jobs do; the current always-on ProcessBuilder host does not bin-pack idle) and (b) enforces per-org concurrency quotas so one tenant can't spawn 500 tasks.

---

## 4. Database scaling decision

| Stage | Choice | Rationale |
|---|---|---|
| Seed | **RDS Postgres Single-AZ, t4g class** | Kill H2 (`ec2demo` still uses H2 per CLAUDE.md). Flyway V1–V5 already targets Postgres dialect in the `production` profile (`application-production.properties:22,31`). The migration is config, not schema rework. |
| Series-A | **RDS Postgres Multi-AZ, r6g class, 1× read replica** | Sustained, predictable load → provisioned + 1-yr RI beats Serverless v2 (which has no RI equivalent and bills $0.12/ACU-hr, [Aurora](https://aws.amazon.com/rds/aurora/pricing/)). |
| Hyperscale | **Aurora Postgres (writer + readers)**; **offload time-series tables** | Aurora's decoupled storage + fast replicas fit a read-heavy multi-tenant dashboard. **Do not put `server_logs`/`round_result` growth on the OLTP DB** — both are flagged unbounded (`04-observability.md` Risk 5, `01-backend.md` references `server_logs` no rotation). Route those to Loki/ClickHouse. |

**Citus/distributed Postgres: `kill` (for now).** No evidence the control-plane (orgs/users/projects/memberships — small, bounded tables per V5) needs horizontal sharding. The *growth* is in append-only telemetry, which belongs in a TSDB/log store, not a sharded RDBMS. Revisit only if a single Aurora writer is measurably saturated.

**Aurora Serverless v2 verdict: `refactor`-candidate, not default.** Its idle floor (~$44/mo at 0.5 ACU) makes it *worse* than provisioned RDS for steady load, but it's attractive for **dev/staging environments** that sit idle nights/weekends. Use it there, not in prod.

---

## 5. Adoption-curve benchmarking

Calibrating the tier thresholds against the closest comparable, **Flower Labs** (the leading OSS FL framework — explicitly *not* a dependency here per platform rules, but the best market analog):

- **Flower: 6.9k GitHub stars, 3k+ developer community, 1,000+ dependent projects, adopters incl. Samsung, Nokia Bell Labs, Brave, Banking Circle; universities incl. Oxford, Harvard, MIT, Stanford** ([Felicis](https://www.felicis.com/blog/investing-in-flower), [Flower Series A blog](https://flower.ai/blog/2024-02-15-announcing-series-a/)).
- **Funding: $20M Series A (Feb 2024, led by Felicis), ~$24.1M total, ~$100M valuation** ([Fortune](https://fortune.com/2024/02/15/flower-labs-federated-learning-pioneer-valued-at-100-million-in-venture-capital-funding-round/), [Crunchbase](https://www.crunchbase.com/organization/flower-1de8)).
- **FedML / TensorOpera (Nexus AI):** pivoted from pure FL to a general distributed-training/GenAI cloud; pricing is variable cross-cloud spot-bidding, "~4–5× cheaper than major clouds" ([fedml.ai/pricing](https://fedml.ai/pricing)) — telling that *even the FL incumbents broadened to GenAI compute to monetize*.

**Implications for the tiers:**
1. **The seed→Series-A transition is research-led, not consumer-led.** Comparable FL adoption is dominated by universities + a handful of F500 design partners, not viral self-serve. The seed tier should optimize for **5–25 high-touch design-partner orgs**, not 10k self-serve signups. Concurrency, not user count, gates cost.
2. **The DeComFL/communication-reduction angle is the differentiator Flower doesn't lead with.** Flower's pitch is "FL framework"; this platform's pitch can be "**fine-tune billion-param models over the network for ~$1/run**." That reframes the cost story as the product.
3. **Don't over-provision for hyperscale early.** Flower hit F500 adoption *before* needing hyperscale infra, because FL workloads are bursty and client-side compute is user-owned. The platform's cost curve should stay flat-ish through Series-A precisely because clients bring their own compute.

---

## 6. Cost-control mechanisms the v2 must build (prioritized)

1. **(P0) Per-org concurrency quotas + admission control.** The 11-port cap is accidentally protecting you today; once it's lifted, *nothing* stops one tenant from spawning unbounded FL servers. This is both a cost and a multi-tenant-fairness control. **Without it, the Fargate/EKS bill is unbounded.**
2. **(P0) Scale-to-zero orchestration.** Fargate RunTask / k8s Jobs bill only while running. The current always-on ProcessBuilder host does not bin-pack or release idle capacity. This is the difference between paying for *runs* and paying for *capacity*.
3. **(P1) Default to DeComFL for models >100M params; surcharge or warn on FedAvg-over-WAN for large models.** This is the single biggest egress lever (§2.2). Make it a product default, not a footnote.
4. **(P1) Move `server_logs` / `round_result` / `audit_events` off the OLTP DB and add retention/partitioning** (extends `04-observability.md` Risk 5). Unbounded telemetry on Postgres is a silent storage + IOPS cost.
5. **(P1) Graviton/ARM Fargate + S3 VPC endpoints.** ~21% compute saving + kills in-region S3 egress, for near-zero engineering cost.
6. **(P2) Content-addressed model storage (hash-keyed) + S3 lifecycle tiering.** Dedup identical base models across orgs; lifecycle old round checkpoints to Glacier. Critical at hyperscale where the registry is petabyte-class.
7. **(P2) Observability cardinality budget enforced in code** (`04-observability.md` Risk 1). At hyperscale, label cardinality *is* the observability bill. Treat `client_id`-on-histogram as a cost bug.
8. **(P2) Fargate Spot / Spot GPU for runs — only after round-level checkpointing exists** (C1 dependency). −70% on the dominant compute line, but unsafe until interrupted runs can resume via `rebuild_model`.

---

## 7. Decision table

| Module / subsystem | Verdict | One-line rationale |
|---|---|---|
| FL-server-per-project **concept** | **salvage** | The primitive maps cleanly to Fargate RunTask / k8s Jobs; keep it. |
| **Local `ProcessBuilder`** FL-server launcher | **refactor → kill above seed** | Fine for seed with a raised port range; the 11-port/single-host model is a hard cliff at Series-A. |
| **ECS Fargate `RunTask`** path (`startEcsFargateServer`) | **salvage** | Already coded; the right Series-A primitive. Needs hardening (it's in the unfinished `production` profile) + concurrency quotas. |
| Orchestration at hyperscale | **rebuild** | EKS+Karpenter with CPU+GPU node pools; Fargate doesn't do GPU scheduling for 1B+ ZO reconstruction. |
| **In-memory STOMP broker** | **rebuild** | Single-replica cap (`M9`); blocks horizontal backend scaling — must become a STOMP relay (RabbitMQ/Amazon MQ). |
| **DB: H2-file (ec2demo)** | **kill** | Replace with RDS Postgres now; Flyway already targets Postgres in the prod profile. |
| **DB: RDS Postgres (prod)** | **salvage → Aurora at hyperscale** | Provisioned RDS+RI through Series-A; Aurora only when read-replica fan-out / storage auto-scaling earns it. |
| **Citus / distributed Postgres** | **kill** | No evidence the bounded control-plane tables need sharding; growth is telemetry → TSDB, not RDBMS. |
| **Aurora Serverless v2 for prod OLTP** | **kill** | Idle floor (~$44/mo) + no RI makes it worse than provisioned for steady load; keep it for idle dev/staging only. |
| **DeComFL as a cost tier / default for large models** | **salvage (elevate to product strategy)** | ~6–7 orders of magnitude egress reduction is the cost moat; price it, default to it. |
| **Parameter chunking (FedAvg path)** | **salvage** | Required for the FedAvg-on-large-model case that DeComFL doesn't cover; the cost penalty is egress, addressed by steering + surcharge, not by removing chunking. |
| **`server_logs`/`round_result`/`audit_events` on OLTP** | **refactor** | Unbounded append-only growth belongs in Loki/ClickHouse, not the transactional DB. |
| **Observability stack (self-host Prom/Grafana/Loki)** | **salvage** | Correct and cost-appropriate vs Datadog at scale; gate spend with cardinality budgets. |

---

## 8. Open questions / uncertainty flags

- **GPU need for server-side ZO reconstruction at 7B** is *inferred* from `aggregate_fit`'s full-model perturbation regeneration (`decomfl_strategy.py:180-197`), not benchmarked. The framework audit (A3/B1) should confirm whether server-side reconstruction is the bottleneck and at what model size it forces GPU. My hyperscale GPU line is the largest uncertainty in this report.
- **DeComFL's "1 MB total" figure** is from the paper abstract ([arXiv 2405.15861](https://arxiv.org/abs/2405.15861)); the per-round byte count I derive (K×P scalars + seeds) matches the code's intent but I have **not** measured the actual wire size of the `rebuild_history` downlink for clients that miss many rounds — that downlink is O(rounds-missed × K × P) and could matter for high-churn mobile clients. Flagged for C1/C2.
- **Cloud prices** are us-east-1 on-demand snapshots (2026); all are subject to AWS changes and exclude RI/Savings Plan discounts (20–60%), which materially lower every steady-state line. Treat tier totals as order-of-magnitude, not quotes.
- **Per-second Fargate billing has a 1-minute minimum** ([Fargate pricing](https://aws.amazon.com/fargate/pricing/)); for very short runs the minimum dominates — irrelevant for hours-long FL runs but relevant if the platform ever adds quick eval-only tasks.

---

## Sources

- AWS Fargate pricing — https://aws.amazon.com/fargate/pricing/ ; analysis: https://www.vantage.sh/blog/fargate-pricing
- Amazon Aurora pricing — https://aws.amazon.com/rds/aurora/pricing/
- Amazon RDS pricing — https://aws.amazon.com/rds/pricing/ ; db.r6g.large specs: https://instances.vantage.sh/aws/rds/db.r6g.large
- Amazon S3 pricing — https://aws.amazon.com/s3/pricing/
- Amazon EKS pricing — https://aws.amazon.com/eks/pricing/
- Grafana Cloud pricing — https://monitoringcost.com/grafana-cloud-pricing ; Datadog vs Grafana cost: https://leanopstech.com/blog/datadog-vs-grafana-cloud-pricing-2026/
- DeComFL paper (ICLR 2025) — https://arxiv.org/abs/2405.15861 ; repo: https://github.com/ZidongLiu/DeComFL
- Flower Labs adoption/funding — https://www.felicis.com/blog/investing-in-flower ; https://flower.ai/blog/2024-02-15-announcing-series-a/ ; https://fortune.com/2024/02/15/flower-labs-federated-learning-pioneer-valued-at-100-million-in-venture-capital-funding-round/
- FedML/TensorOpera pricing — https://fedml.ai/pricing
