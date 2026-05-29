# FedLearn Platform — Practical Run-Cost Analysis

**Date:** 2026-05-29
**Branch:** `main-clean`
**Scope:** What it costs, per month, to actually run the FedLearn Platform on a public cloud (AWS — Amazon Web Services), explained for a non-finance reader.
**Companion to:** the DeComFL Correctness design spec (`docs/v2/specs/2026-05-29-decomfl-correctness-design.md`) and the scale/cost research (`docs/audit/2026-05-29/B6-scale-cost.md`).

> **Acronyms used in this document** (first-use expansion, per house rule): FL (Federated Learning), DeComFL (Decomposed Federated Learning), ZO (Zeroth-Order), RNG (Random Number Generator), AWS (Amazon Web Services), EC2 (Elastic Compute Cloud), RDS (Relational Database Service), S3 (Simple Storage Service), EKS (Elastic Kubernetes Service), ECS (Elastic Container Service), ALB (Application Load Balancer), NAT (Network Address Translation), VPC (Virtual Private Cloud), DNS (Domain Name System), DB (Database), vCPU (virtual Central Processing Unit), CPU (Central Processing Unit), GPU (Graphics Processing Unit), GB (Gigabyte), TB (Terabyte), MB (Megabyte), TDD (Test-Driven Development), LLM (Large Language Model), TLS (Transport Layer Security), SLO (Service Level Objective), RI (Reserved Instance), TSDB (Time-Series Database), API (Application Programming Interface), STOMP (Simple Text Oriented Messaging Protocol), WS (WebSocket), gRPC (Google Remote Procedure Call), OPT (Open Pre-trained Transformer), CDN (Content Delivery Network), AZ (Availability Zone), MQ (Message Queue — as in Amazon MQ), WAF (Web Application Firewall), k8s (Kubernetes), I/O (Input/Output), fp16 / fp32 (16-bit / 32-bit floating-point number formats), MNIST (Modified National Institute of Standards and Technology handwritten-digit dataset), CIFAR (Canadian Institute For Advanced Research image dataset), ViT (Vision Transformer), CNN (Convolutional Neural Network), MLP (Multi-Layer Perceptron), IoT (Internet of Things). Brand/product names used as-is (no expansion): AWS Fargate, AWS Graviton (ARM — Advanced RISC Machines — based processors), Amazon Aurora, Spot Instances, Prometheus/Grafana/Loki, Route 53.

---

## 1. What "running the app" actually means

The platform is not one program — it is several pieces that must all be online at once. "The bill" is the sum of keeping each piece running. Here is every piece in plain words.

| Piece | Plain-language description | Is it always on? |
|---|---|---|
| **Control plane** (Spring Boot API) | The brain. A Java web server (`backend/fl-platform-api/`) that handles logins, owns the list of users/organisations/projects, and decides when to start a training run. It must be reachable 24/7 so people can log in. | **Yes** |
| **Spawned FL servers** (Python) | When someone clicks "start" on a project, the control plane launches a separate Python program (`framework/`) that coordinates one training run. Each running project = one of these. They live for **hours**, then exit. | **No — only while a run is active** |
| **Database** (PostgreSQL) | The filing cabinet. Stores accounts, project metadata, audit logs, and per-round results. Must be online whenever the app is. | **Yes** |
| **Object storage** (S3) | The warehouse. Holds model checkpoints and artifacts (large files). You pay for how much you keep and how much you move in/out. | **Yes (storage), pay-per-use** |
| **Dashboard** (React frontend) | The website users see. Static files; cheap to serve. Talks to the control plane and streams live training logs over a WebSocket. | **Yes (but cheap)** |
| **Observability** (metrics/logs) | The dashboard *for operators* — graphs of CPU, error rates, training progress (Prometheus/Grafana/Loki, or a hosted equivalent). You either self-host it (uses the boxes above) or pay a vendor. | **Yes** |
| **Networking glue** | Load balancer (ALB), the NAT gateway that lets private servers reach the internet, DNS, and **outbound bandwidth ("egress")** — the charge for every gigabyte the cloud sends *out* to clients/users. | **Yes** |

**The one structural fact that drives this whole document:** the FL *clients* (the machines doing the actual model training — laptops, Jetson boards, phones) are **user-owned hardware**, by design. The cloud bill does **not** include client compute. What the cloud *does* pay for on the data plane is **bandwidth** — every byte the FL server exchanges with clients is billed as egress. That is exactly the line item DeComFL collapses, and it is why the DeComFL fix (below) is a cost story, not just a correctness story.

---

## 2. Itemized monthly cost — two realistic tiers

These two tiers come straight from the B6 scale/cost research (`docs/audit/2026-05-29/B6-scale-cost.md` §0, §3). All prices are **us-east-1 (N. Virginia), on-demand, 2026 list prices**, ~730 hours/month, and **exclude** discounts (Reserved Instances / Savings Plans typically cut 20–60% off steady-state lines). They also exclude salaries, third-party software seats, and client hardware. Treat each total as an **order-of-magnitude operating estimate, not a quote.**

### Price reference table (sourced)

| Resource | Unit price (2026 list) | Source |
|---|---|---|
| Fargate vCPU (x86) | $0.04048 / vCPU-hour | [AWS Fargate pricing](https://aws.amazon.com/fargate/pricing/) |
| Fargate memory (x86) | $0.004445 / GB-hour | [AWS Fargate pricing](https://aws.amazon.com/fargate/pricing/) |
| Fargate Graviton (ARM) | ~20% cheaper than x86 | [AWS Fargate pricing](https://aws.amazon.com/fargate/pricing/) |
| RDS PostgreSQL `db.t4g.small` (2 vCPU / 2 GB) | $0.032 / hour | [Vantage db.t4g.small](https://instances.vantage.sh/aws/rds/db.t4g.small) |
| RDS PostgreSQL `db.r6g.large` (2 vCPU / 16 GB) | $0.225 / hour | [Vantage db.r6g.large](https://instances.vantage.sh/aws/rds/db.r6g.large) |
| S3 Standard storage | $0.023 / GB-month (first 50 TB) | [AWS S3 pricing](https://aws.amazon.com/s3/pricing/) |
| Internet egress (data out) | $0.09 / GB (first 10 TB), $0.085 (10–50 TB), $0.07 (150 TB+); **first 100 GB/mo free** | [AWS data-transfer pricing 2026](https://leanopstech.com/blog/aws-data-transfer-pricing-2026/), [S3 pricing](https://aws.amazon.com/s3/pricing/) |
| EKS control plane | $0.10 / hour (~$73/mo, standard support) | [AWS EKS pricing](https://aws.amazon.com/eks/pricing/) |

> **Note on Fargate vs raw EC2.** B6 uses AWS Fargate (run-a-container, no server to manage) as the compute primitive for both tiers because the FL-server-per-project workload is bursty and Fargate bills per-second with scale-to-zero. Equivalent raw EC2 instances are cheaper per hour but you pay for them 24/7; for bursty FL runs Fargate's "pay only while running" wins. This analysis follows B6's choice.

---

### Tier 1 — SEED (design-partner / research stage)

**Who this is:** 1–25 organisations, ~100 users, **≤8 concurrent FL projects**, small models (0.1M–25M parameters — the MNIST/CIFAR/small-ViT band the framework targets today). This is the current `ec2demo` setup, productionised. (`B6-scale-cost.md` §3.1.)

| # | Component | What it is, in plain words | Assumed size | Monthly cost | How it's derived |
|---|---|---|---|---|---|
| 1 | **Control plane** (Spring Boot) | The always-on brain | 1 Fargate task, 1 vCPU / 4 GB, 24/7 | **~$42** | (1 × $0.04048 + 4 × $0.004445) × 730 hr ≈ $29.6 + $13.0. Graviton: ~$34. |
| 2 | **Database** | Always-on filing cabinet (PostgreSQL) | `db.t4g.small`, single-AZ, 20 GB | **~$25–30** | $0.032/hr × 730 ≈ $23.4 + ~$2 storage + backups |
| 3 | **FL-server compute** | The Python coordinators that run during training | 8 concurrent CPU runs, a few hours/day each (bursty) | **~$30–150** | Per run ~$0.10–0.25/hr; only billed while running. Bursty → cheap. |
| 4 | **Object storage** (S3) | Warehouse for model checkpoints | ~100 GB of small-model checkpoints | **~$5–20** | 100 GB × $0.023 ≈ $2.30 + request/lifecycle overhead |
| 5 | **Egress** (data out) | Bandwidth to clients + dashboard | Small models, DeComFL-dominant traffic | **~$10–30** | Small; first 100 GB/mo is free |
| 6 | **Observability** | Operator graphs/logs | Self-hosted on the control-plane box, or a free hosted tier | **~$0–50** | Fits a free hosted tier at this volume |
| 7 | **Networking glue** | Load balancer + NAT + DNS + log retention | 1 ALB, 1 NAT gateway, Route 53 | **~$40–80** | ALB ~$22 + NAT gateway ~$33 + misc; NAT is a quiet, fixed tax |
| | **TOTAL** | | | **~$350–900 / mo** | matches B6 §0/§3.1 |

**What dominates the seed bill:** the **always-on baseline** (control plane + database + networking glue), *not* the FL training. Training compute is noise here because runs are short and small. The cheapest honest seed footprint sits near the low end (~$350) if you self-host observability on the same box and use a NAT instance instead of the managed NAT gateway.

**Assumptions stated:** single availability zone (no high-availability replica), no Reserved-Instance discount applied, ≤8 simultaneous runs (well under the current 11-port cap), and the database migrated off H2-file mode to real RDS PostgreSQL (B6 recommends this; `ec2demo` still ships H2 today).

---

### Tier 2 — SERIES-A (multi-tenant production)

**Who this is:** 25–500 organisations, ~5,000 users, **50–300 concurrent FL projects**, models from 1M up to 1.3B parameters (LLM fine-tuning via DeComFL becomes a real workload). Multi-AZ, real reliability targets. (`B6-scale-cost.md` §3.2.)

| # | Component | What it is, in plain words | Assumed size | Monthly cost | How it's derived |
|---|---|---|---|---|---|
| 1 | **Control plane** (Spring Boot) | The brain, now run in multiples behind a load balancer | 2–4 Fargate tasks, 2 vCPU / 8 GB each | **~$600–1,200** | ~$160/replica × 2–4. ([Fargate](https://aws.amazon.com/fargate/pricing/)) |
| 1a | **Message relay broker** | A required add-on: with >1 control-plane copy, live log streaming needs a shared message bus (the current in-memory broker can't route between copies) | Amazon MQ (RabbitMQ) `mq.m5.large`, or self-hosted | **~$180** | B6 §3.2 blocker; needed before multi-replica |
| 2 | **Database** | Filing cabinet with a hot standby for failover | `db.r6g.large` (2 vCPU / 16 GB), Multi-AZ | **~$330** | $0.225/hr × 730 × ~2 (Multi-AZ) ≈ $328, partly offset by RI; + ~$30 storage ([Vantage](https://instances.vantage.sh/aws/rds/db.r6g.large)) |
| 3 | **FL-server compute** | The dominant variable cost — many training runs at once | 50–300 concurrent runs, ~3 hr avg, CPU 2–4 vCPU/8–16 GB, churned daily | **~$1,500–6,000** | ~$0.10–0.23/run-hr × run-hours/mo; scale-to-zero between runs. See §4. |
| 4 | **Object storage** (S3) | Warehouse, now with billion-param checkpoints | ~5,000 checkpoints, ~13 TB | **~$50–300** | 13 TB × $0.022–0.023 ≈ $290; lifecycle old rounds to cheaper tiers |
| 5 | **Egress** (data out) | Bandwidth — DeComFL runs are tiny; FedAvg-on-LLM runs are expensive | DeComFL-dominant, with occasional FedAvg | **~$200–2,000** | A few FedAvg-on-LLM runs can dominate this line (see §3) |
| 6 | **Observability** | Operator graphs/logs at production volume | Hosted Pro tier, or self-hosted on a dedicated node | **~$300–1,500** | Cost knob = metric **cardinality** (keep per-client labels off histograms) |
| 7 | **Networking glue** | Load balancer + NAT + web firewall + DNS | ALB + NAT + WAF | **~$200–400** | |
| | **TOTAL** | | | **~$4,000–14,000 / mo** | matches B6 §0/§3.2 |

**What dominates the Series-A bill:** **FL-server compute task-hours first, then the database.** Unlike seed tier, training is now the biggest line. The two highest-leverage savings are (a) **steering large-model runs to DeComFL** so egress stays flat (§3), and (b) **Graviton/ARM everywhere** (~20% off compute — the framework is PyTorch and ARM wheels exist, proven by the Jetson path).

**Assumptions stated:** Multi-AZ database, 2–4 control-plane replicas (which forces the relay broker line 1a), no Spot discounts (FL runs are stateful; Spot is only safe once round-checkpointing exists — a separate reliability item), and no Reserved-Instance discount applied to the headline (applying a 1-year RI to the DB and steady control plane would pull the total toward the low end).

---

## 3. How fixing DeComFL changes the cost

This is the section that makes the whole platform economically interesting — and it depends entirely on the three DeComFL bugs in the companion spec being fixed.

### 3.1 The mechanism, in plain words

Ordinary federated learning (**FedAvg**) ships the **entire model** between server and client every round. A 125M-parameter model is ~500 MB; you send it down and the client sends its update back up — every round, every client. That is pure egress, and egress is billed per gigabyte.

**DeComFL** does something different. The client and server agree on a shared random seed. From that seed both sides independently regenerate the same random perturbation vector `z` (the RNG-determinism contract — Bug 2 in the spec is precisely about making this regeneration *bit-identical* across devices). The client then computes the model update locally and only has to transmit a **handful of scalar numbers** (the ZO gradient estimates) plus the seeds. The model itself never crosses the wire. Communication becomes **O(K×P)** — a few hundred bytes per client per round — **independent of model size**. (`docs/wikis/framework/06_decomfl.md` §"Communication Comparison".)

The platform's own wiki quantifies this for OPT-125M with K=5 local steps, P=10 perturbations: FedAvg moves 500 MB up + 500 MB down; DeComFL moves ~400 bytes up + ~1.6 KB down — a **~1.25-million-fold reduction** (`docs/wikis/framework/06_decomfl.md` lines 68–74). The DeComFL paper's headline figure is **~1 MB total** to fine-tune a *billion-parameter* model ([arXiv 2405.15861](https://arxiv.org/abs/2405.15861)).

### 3.2 Worked example — one training run, in dollars

Let's price a single concrete run and compare strategies. **Egress price used: $0.09/GB** (first-10-TB tier, [AWS data-transfer pricing 2026](https://leanopstech.com/blog/aws-data-transfer-pricing-2026/)).

**Setup A — OPT-125M (500 MB at fp16/fp32 band per the wiki), 50 clients, 100 rounds:**

| | Per client / round | Total transfer (50 × 100, up+down) | Egress cost @ $0.09/GB |
|---|---|---|---|
| **FedAvg** | ~1.0 GB (500 MB down + 500 MB up) | ~5,000 GB ≈ **4.9 TB** | **~$440** |
| **DeComFL (correct)** | ~2 KB (scalars up + seeds down) | < 0.01 GB | **< $0.001 (effectively $0)** |

**Setup B — 1.3B-param LLM (~2.6 GB at fp16), 100 clients, 100 rounds** (the B6 headline case, `B6-scale-cost.md` §2.2):

| | Per client / round | Total transfer | Egress cost @ $0.09/GB |
|---|---|---|---|
| **FedAvg** | ~5.2 GB (2.6 GB down + 2.6 GB up) | ~52 TB | **~$4,700** (tiered down past 10 TB) |
| **DeComFL (correct)** | ~80 bytes | < 1 MB | **< $0.001** |

So a **single** large-model run under FedAvg can cost **hundreds to thousands of dollars in bandwidth alone**; the same run under correct DeComFL costs a fraction of a cent. B6's per-run unit economics (§3.4) put a 3-hour Series-A DeComFL run at **~$0.70 total cost-to-serve** (compute) versus **~$2,400** for the FedAvg equivalent — a **~3,400× difference, essentially all network.**

### 3.3 Why the *fix* is what unlocks the saving

The saving above is only real if DeComFL actually works. Today it does not, in three independent ways the companion spec fixes (all verified in code):

| Bug | Where (verified) | What it costs you if unfixed |
|---|---|---|
| **Bug 1 — missing 1/P factor** | `framework/src/fedlearn/server/decomfl_strategy.py:197,200` — line 197 divides `delta` by `(num_clients × self.P)`, then line 200 multiplies by `self.P` again (`x_current = x_current - self.eta * delta * self.P`), cancelling it. The client (`decomfl_client.py:208`) correctly uses `(eta/P)·delta`. | The global model takes **P× (10× at default) too large a step** and diverges from every reconnecting client's rebuilt trajectory. The run produces a **wrong model** — you paid the (tiny) DeComFL bandwidth but got a broken result, so you'd fall back to FedAvg and pay full egress. |
| **Bug 2 — device-dependent RNG** | Server generates `z` on `self.device` (`decomfl_strategy.py:210-219`); client does the same on its device (`estimators/zeroth_order.py:45-48`). Seeded `torch.randn` is **not** bit-identical across CPU/CUDA/MPS. | Server and client reconstruct **different `z`** on any GPU server or mixed-device fleet → aggregation is silent garbage. Same failure mode: the cheap-bandwidth run gives a wrong model, forcing a fallback to FedAvg-scale bandwidth. |
| **Bug 3 — serializer asymmetry** | `serializer.py:97` saves a **bare** state-dict (`torch.save(params, buffer)`); `serializer.py:155` loads expecting a **wrapped** dict (`model_data['parameters']`). | **`KeyError` on every chunked upload** — and every model larger than the chunk size (i.e. every transformer/LLM, the exact path DeComFL exists for) takes the chunked path. LLM federations **cannot complete a round at all.** |

**The bottom line for cost:** DeComFL is the platform's bandwidth moat — the thing that turns a ~$4,700 egress run into a ~$0 one. But **a broken DeComFL gives you the worst of both worlds**: either it errors out (Bug 3 — no run completes) or it silently produces a wrong model (Bugs 1 and 2), which in practice means reverting to FedAvg and paying full FedAvg-scale egress. The correctness fix in the companion spec (validated by the TDD test suite T1–T5, run via `cd framework && pytest`) is the precondition for the saving in §3.2 being a real number rather than a brochure claim.

> **Honest caveat (from B6 §2.3, §8):** DeComFL collapses *network* cost, **not compute**. ZO estimation needs P forward passes per local step, so a client doing P=10 does ~10× the forward compute of one inference — but that compute runs on **user-owned client hardware**, off the cloud bill. On the *server*, the cleanup C-1 in the spec (hoisting the `z` generation out of the per-client loop, `O(K·P·N) → O(K·P)`) reduces server-side reconstruction cost; whether a GPU server is needed for very large models (7B+) is *inferred*, not benchmarked (B6 §8 flags this as its largest uncertainty). The bandwidth numbers above are solid; the server-compute numbers at hyperscale are estimates.

---

## 4. What drives the bill — and the cheap-vs-expensive knobs

### The 2–3 dominant cost drivers

1. **Always-on baseline (seed tier) → FL-server compute (Series-A and up).** At seed scale the bill is mostly the control plane + database + networking glue sitting there 24/7; training is noise. As you grow, **FL-server run-hours overtake everything** and become the largest single line. This is the crossover B6 documents (§0).
2. **The database**, consistently the #2 line once you go Multi-AZ. A failover-ready PostgreSQL roughly doubles the single-instance price.
3. **Egress (bandwidth) — but only if you let large-model runs use FedAvg.** With correct DeComFL as the default for big models, this line stays small even as models grow. Let it default to FedAvg-over-the-internet and a *single* run can eclipse a month of compute (§3).

### Cheap knobs (high saving, low effort)

| Knob | Effect | Source |
|---|---|---|
| **Default large models (>100M params) to correct DeComFL** | Turns the egress line from thousands to ~$0 per run; the single biggest lever | B6 §6 (P1); §3 above |
| **Graviton / ARM Fargate** | ~20% off all compute, near-zero engineering cost | [Fargate pricing](https://aws.amazon.com/fargate/pricing/) |
| **Scale-to-zero orchestration** (Fargate RunTask / k8s Jobs) | Pay only while a run is active, not for idle capacity | B6 §6 (P0) |
| **Reserved Instances / Savings Plans** on the always-on lines (control plane, DB) | 20–60% off steady-state | B6 §3 |
| **S3 VPC endpoints + lifecycle tiering** | Kills in-region transfer charges; ages old checkpoints to cheaper storage | B6 §6 |
| **Self-host observability at small scale** | Avoids a per-host monitoring vendor bill | B6 §3.1 |

### Expensive knobs (watch these — they balloon the bill)

| Knob | Why it's expensive | Source |
|---|---|---|
| **FedAvg on large models over the internet** | Egress dominates everything; one run can cost more than a month of compute (§3.2) | B6 §2.2 |
| **No per-org concurrency quota once the 11-port cap is lifted** | One tenant can spawn unbounded FL servers → unbounded Fargate/EKS bill. B6 calls the current port cap an *accidental* cost control | B6 §6 (P0), R10 |
| **GPU FL-server compute at hyperscale** | The dominant line for 1B+ param server-side reconstruction; largest single uncertainty | B6 §3.3, §8 |
| **High-cardinality observability** (per-client labels on metrics) | At scale, label cardinality *is* the monitoring bill | B6 §6 (P2) |
| **Unbounded telemetry on the transactional DB** | `server_logs` / `round_result` growth on PostgreSQL is a silent storage + I/O cost; belongs in a log store / TSDB | B6 §6 (P1) |
| **Aurora Serverless v2 for steady production load** | ~$44/mo idle floor with no Reserved-Instance discount — worse than provisioned RDS for steady load | B6 §3.1, §4 |

---

## 5. One-paragraph summary for a non-technical reader

Running FedLearn is like running a website with an occasional heavy batch job. The fixed monthly cost — keeping the brain (control plane), the filing cabinet (database), and the networking online — is roughly **$350–900/month at a small research scale** and **$4,000–14,000/month at multi-tenant production scale** on AWS list prices (before discounts that typically save 20–60%). The variable cost is training runs, and here the platform has a genuine superpower: **DeComFL** ships only a few bytes per round instead of the whole model, so a large-model training run that would cost **thousands of dollars in bandwidth under ordinary federated learning costs effectively zero**. That superpower is currently broken in three ways (a P× too-large step, a device-dependent random-number bug, and a crash on every large-model upload); fixing them — the subject of the companion design spec — is what turns the bandwidth saving from a slide into an actual line on the bill.

---

## Sources

- AWS Fargate pricing — https://aws.amazon.com/fargate/pricing/
- AWS S3 pricing — https://aws.amazon.com/s3/pricing/
- AWS data-transfer (egress) pricing 2026 — https://leanopstech.com/blog/aws-data-transfer-pricing-2026/
- Amazon RDS for PostgreSQL pricing — https://aws.amazon.com/rds/postgresql/pricing/
- RDS `db.t4g.small` price/specs — https://instances.vantage.sh/aws/rds/db.t4g.small
- RDS `db.r6g.large` price/specs — https://instances.vantage.sh/aws/rds/db.r6g.large
- Amazon EKS pricing — https://aws.amazon.com/eks/pricing/
- DeComFL paper (ICLR 2025) — https://arxiv.org/abs/2405.15861 ; repo: https://github.com/ZidongLiu/DeComFL
- Internal: `docs/audit/2026-05-29/B6-scale-cost.md` (scale/cost research), `docs/audit/2026-05-29/README.md` (master synthesis), `docs/wikis/framework/06_decomfl.md` (communication/bandwidth tables), `docs/v2/specs/2026-05-29-decomfl-correctness-design.md` (correctness spec)
- Code evidence (verified): `framework/src/fedlearn/server/decomfl_strategy.py:197,200,210-219`; `framework/src/fedlearn/client/decomfl_client.py:208`; `framework/src/fedlearn/estimators/zeroth_order.py:45-48`; `framework/src/fedlearn/communication/serializer.py:97,155`
