# 02 — Tech Stack Reference (FedLearn Platform v2)

**Document type:** Production build documentation — the single authoritative tech-stack reference for the v2 (version 2) greenfield rebuild.
**Audience:** A mid-sized (~30B-parameter) local Large Language Model (LLM) implementing the build. This document pre-decides every technology, pins every version, and gives the reasoning behind each choice. **Do not substitute alternatives.** If a version is labeled `verify-before-use`, the implementer MUST run the stated verification command and pin the exact resolved version before writing it into a manifest.
**Date authored:** 2026-05-29.
**Source of truth:** The v2 audit synthesis at `/home/anurag/codebase/FedLearn-Platform/docs/audit/2026-05-29/README.md` and the per-unit reports `B2-tech-stack.md`, `B7-standards-dx.md`, `B5-desktop-strategy.md`, `B6-scale-cost.md` in the same directory.

---

## 0. How to read this document

Every acronym is expanded in full the first time it appears, e.g. "API (Application Programming Interface)". After first expansion the short form is used.

Each layer section gives a table or block with these fields:

| Field | Meaning |
|---|---|
| **Technology** | The exact chosen tool. |
| **Pinned version** | The exact version string to write into the manifest. `verify-before-use` means run the verification command, then pin the resolved value. |
| **What it is** | One plain sentence. |
| **Why chosen** | The reasoning, tied to v2 audit findings where relevant. |
| **Alternatives rejected** | The top one or two considered, and why they lost. |
| **License** | The Open Source Initiative (OSI) license, and any commercial caveat. |

### 0.1 Version-pinning discipline (read before pinning anything)

The v2 audit raised dependency hygiene to **high severity** because v1 ran Spring Boot 3.4.5, which reached open-source End-Of-Life (EOL) on 2025-12-31 — "not 'one minor behind' — it is unsupported" (`B7-standards-dx.md:16`). To prevent a repeat:

1. Pin every dependency to an **exact** version (no floating ranges like `^` or `>=`) in the committed manifest. The v1 `framework/requirements.txt` carried `flwr-datasets>=0.3.0` and `protobuf>=4.21.6,<5.0.0` ranges that the audit flagged as a reproducibility hole (`B7-standards-dx.md:201`).
2. Generate a lockfile with hashes (`uv lock` / `pip-compile --generate-hashes` for Python, `package-lock.json` for Node.js, `Cargo.lock` for Rust, Gradle's dependency-locking for Java).
3. Renovate (see §22) opens the upgrade Pull Requests (PRs); Continuous Integration (CI) gates them. Humans never hand-bump.

### 0.2 Version-freshness note

All "current stable" versions below were verified via web search on **2026-05-29** and each carries a source Uniform Resource Locator (URL). Versions move; before the first build, re-verify each pinned version against the cited source. Where a version could not be web-verified to a single value it is marked `verify-before-use` with the command to resolve it.

---

## 1. Language runtimes

These are the four host languages of the v2 monorepo. The audit names exactly these and only these: Java backend, Python framework, TypeScript (frontend + desktop renderer + mobile React Native layer), and C++ mobile core, plus Rust as the new Tauri desktop command layer (`B7-standards-dx.md:5`, `B5-desktop-strategy.md:109`).

### 1.1 Java 21 (Long-Term-Support)

| Field | Value |
|---|---|
| **Technology** | Eclipse Temurin Java Development Kit (JDK) 21 (OpenJDK build). |
| **Pinned version** | `21.0.7+6` (Temurin 21 LTS, latest LTS patch as of 2026-05; `verify-before-use` — run `sdk list java \| grep tem` or check https://adoptium.net/temurin/releases/?version=21). |
| **What it is** | The Long-Term-Support Java runtime and compiler that the Spring Boot control plane compiles and runs against. |
| **Why chosen** | The audit verdict on the control plane is **SALVAGE** — the Java auth/Role-Based-Access-Control (RBAC)/audit/identity investment is "the most valuable, least-broken layer" (`B2-tech-stack.md:178`, `README.md:41`). v1 already targets Java 21 via Gradle toolchain (`backend/fl-platform-api/build.gradle` declares `JavaLanguageVersion.of(21)`). Java 21 is LTS (supported into 2031), giving virtual threads (Project Loom) for the reconciler/heartbeat fan-in and record patterns for the role enum. |
| **Alternatives rejected** | **Rewrite the backend in Go or FastAPI** — explicitly rejected: "Wrong here — the Java *control plane* is the working, valuable part… Don't rewrite the healthy organ" (`B2-tech-stack.md:178`). **Java 25 LTS** — viable but Spring Boot 3.5 targets Spring Framework 6 / Java 17–21 baseline; staying on 21 avoids coupling the bump to a Framework 7 migration. |
| **License** | GNU General Public License v2 with Classpath Exception (GPLv2+CE) — free for commercial use; Temurin distribution adds no commercial license. |

### 1.2 Python 3.10+ (framework runtime)

| Field | Value |
|---|---|
| **Technology** | CPython. |
| **Pinned version** | `3.12.x` (pin exact, e.g. `3.12.9`; `verify-before-use` via `pyenv install --list \| grep 3.12`). Floor is 3.10 per the locked stack; **3.12 is the recommended pin** for performance and typing, and PyTorch 2.12 ships cp312 wheels for both x86-64 and ARM64. |
| **What it is** | The runtime for the custom Federated-Learning (FL) framework: server, client, FedAvg + DeComFL (Dimension-Free Communication Federated Learning — the v1 wiki's "Decomposed" expansion is wrong per the paper, `B1-paper-alignment.md:33`) strategies, and the Google Remote Procedure Call (gRPC) data plane. |
| **Why chosen** | The framework is **SALVAGE core, rebuild serializer** (`README.md:44`). DeComFL is the product differentiator and is implemented in PyTorch/Python; the FL substrate problem is "already Python… that's already the problem language" so keeping it Python is correct (`B2-tech-stack.md:178`). 3.12 gives faster startup (important because the substrate spawns server processes) and per-interpreter Global Interpreter Lock (GIL) improvements relevant to the dual-heartbeat threads. |
| **Alternatives rejected** | **Python 3.10 (the floor)** — works, but 3.12 is materially faster and the ARM64 wheel matrix for PyTorch 2.12 is healthy, so pin higher. **Python 3.13** — free-threaded build is still experimental for PyTorch; defer. |
| **License** | Python Software Foundation License (PSF) — permissive, commercial-friendly. |

### 1.3 Node.js (frontend / desktop renderer / mobile React Native toolchain)

| Field | Value |
|---|---|
| **Technology** | Node.js Active LTS. |
| **Pinned version** | `24.x` (Active LTS as of 2026-05; pin exact e.g. `24.4.0`). Pin in `.nvmrc` and `.tool-versions` at repo root. The audit explicitly says add `.nvmrc` (node 22) — **supersede that to node 24** because 24 is the current Active LTS and 22 enters Maintenance; `verify-before-use` via https://nodejs.org/en/about/previous-releases. |
| **What it is** | The JavaScript runtime that builds and tests the React frontend, the Tauri renderer bundle, and the mobile React Native TypeScript layer. |
| **Why chosen** | All three TypeScript surfaces (`frontend/`, desktop renderer, mobile RN layer) are the JavaScript/TypeScript triangle the audit identifies (`B7-standards-dx.md:65`). One pinned Node version across them removes the version drift the audit flagged (`B7-standards-dx.md:189`). |
| **Alternatives rejected** | **Node 22 (the audit's literal suggestion)** — was correct when written but is now Maintenance LTS; pin 24. **Node 26** — not yet LTS until October 2026 (per Node release schedule); do not pin a Current line to production. **Bun / Deno** — not in the locked stack; Vite 6 + Vitest assume Node. |
| **License** | MIT-style (Node.js is under the MIT License plus bundled-dependency licenses) — permissive. |

### 1.4 Rust (Tauri v2 desktop command layer)

| Field | Value |
|---|---|
| **Technology** | Rust (stable channel, via `rustup`). |
| **Pinned version** | `1.87.x` stable (`verify-before-use` via `rustup show` / https://forge.rust-lang.org/; pin in `rust-toolchain.toml` as `channel = "1.87.0"`). Tauri 2.x requires a recent stable Rust; pin the exact toolchain so CI is reproducible. |
| **What it is** | The systems language for the small Tauri v2 "command layer" that spawns the FL-client subprocess, talks to the Docker socket via `bollard`, and holds the auth token in the Operating System (OS) keychain. |
| **Why chosen** | The desktop verdict is **REBUILD shell → Tauri v2** (`README.md:54`, `B5-desktop-strategy.md:14`). Tauri's privileged process is Rust; the audit scopes this surface as "*small* (spawn/kill a child, stream stdout, one Docker call, one keychain call) — hundreds of lines, not thousands" (`B5-desktop-strategy.md:107`). Rust is mandatory for Tauri; there is no choice of language here, only of version. |
| **Alternatives rejected** | **Keep Electron (Node/TypeScript main process)** — rejected because Tauri "makes the C5 auto-update Remote-Code-Execution (RCE) class *structurally impossible*" via its mandatory signed updater (`B5-desktop-strategy.md:101,111`). **Per-OS native (Swift/WinUI/GTK)** — **KILL**: "3x UI maintenance for an orchestrator UI; zero payoff" (`B5-desktop-strategy.md:186`). |
| **License** | Dual MIT / Apache-2.0 — permissive. |

---

## 2. Control plane — Spring Boot 3.5 (LTS line) + Gradle

### 2.1 Spring Boot

| Field | Value |
|---|---|
| **Technology** | Spring Boot. |
| **Pinned version** | `3.5.14` (latest 3.5 patch, released 2026-04-23; the 3.5 line is the final 3.x minor and the recommended branch for Spring Framework 6 compatibility). Source: https://spring.io/blog/2026/04/23/spring-boot-3-5-14-available-now/ and https://endoflife.date/spring-boot. **Caveat:** OSS patches for 3.5 continue through **2026-06-30**, after which 3.5 needs commercial extended support OR a move to the Spring Boot 4.0 / Framework 7 line. Pin 3.5.14 now; track the 4.0 migration as a separate planned task. `verify-before-use` for any newer 3.5 patch. |
| **What it is** | The Java application framework hosting the Representational-State-Transfer (REST) API, the Simple-Text-Oriented-Messaging-Protocol (STOMP) over WebSocket channel, security, and persistence. |
| **Why chosen** | Salvage the control plane and **bump off the EOL 3.4.5** (`README.md:41`, `B7-standards-dx.md:16`). v1 runs 3.4.5 (verified: `backend/fl-platform-api/build.gradle` has `id 'org.springframework.boot' version '3.4.5'`), which is unsupported. 3.5 LTS is the lowest-risk supported target that keeps the existing Spring Security cookie-Java-Web-Token (JWT) posture intact. |
| **Alternatives rejected** | **Spring Boot 4.0 (Framework 7)** — newer but a larger migration surface (Framework 7 dependency churn); the audit recommends "target the current supported line" (`B7-standards-dx.md:197`) and 3.5 is the conservative supported choice for a salvage. Revisit 4.0 after the v2 launch. **Stay on 3.4.5** — rejected: it is EOL, a security exposure (`B7-standards-dx.md:197`). |
| **License** | Apache-2.0. |

**Spring Boot starters to declare (all version-managed by the Boot Bill-Of-Materials (BOM), so no explicit version):**
`spring-boot-starter-web`, `spring-boot-starter-security`, `spring-boot-starter-data-jpa`, `spring-boot-starter-validation`, `spring-boot-starter-websocket`, `spring-boot-starter-actuator`, `spring-boot-starter-oauth2-resource-server` (only if needed), and `spring-boot-starter-test`. These mirror v1's working set (verified in `backend/fl-platform-api/build.gradle`). Add `io.micrometer:micrometer-registry-prometheus` and `io.micrometer:micrometer-tracing-bridge-otel` for observability (§18).

**Auth dependencies (pin explicitly — not in the Boot BOM):**

| Dependency | Pinned version | Note |
|---|---|---|
| `io.jsonwebtoken:jjwt-api` | `0.12.5` | Carried from v1 (verified in build.gradle); `verify-before-use` for a newer 0.12.x. |
| `io.jsonwebtoken:jjwt-impl` | `0.12.5` | runtimeOnly. |
| `io.jsonwebtoken:jjwt-jackson` | `0.12.5` | runtimeOnly. |

**Auth contract (do not deviate):** cookie-only HttpOnly JWT — the cookie carries the token as `HttpOnly` + `SameSite=Lax` + `Secure`; the frontend sends `withCredentials: true`; **no `Authorization: Bearer` header anywhere, no JavaScript-readable token, no `localStorage`**. The audit rates this posture "textbook-correct" and **SALVAGE** (`README.md:119`). STOMP over WebSocket reuses the same cookie via a handshake interceptor.

### 2.2 Spring gRPC (internal control-plane ↔ substrate channel)

| Field | Value |
|---|---|
| **Technology** | Spring gRPC. |
| **Pinned version** | `verify-before-use` — Spring gRPC reached General Availability (GA) 1.0 in late 2025 (`B2-tech-stack.md:164` cites the GA). Pin the latest 1.x; resolve via https://github.com/spring-projects/spring-grpc/releases. |
| **What it is** | The native gRPC integration for Spring Boot, used for the internal mTLS (mutual Transport-Layer-Security) channel between the JVM control plane and the FL substrate. |
| **Why chosen** | "Spring gRPC 1.0 GA now gives native gRPC to the substrate without a REST bridge" (`B2-tech-stack.md:164`). The v2 reference architecture shows the JVM talking to the substrate over internal mTLS gRPC (`README.md:62-75`). |
| **Alternatives rejected** | **Hand-rolled gRPC + a REST shim** — more code, what v1 effectively did (JSON/STOMP to the FL server). Spring gRPC removes the bridge. |
| **License** | Apache-2.0. |

### 2.3 Gradle (build tool, wrapper committed)

| Field | Value |
|---|---|
| **Technology** | Gradle (via the committed wrapper `gradlew`). |
| **Pinned version** | `9.5.1` (latest stable, 2026-05-09). Source: https://gradle.org/releases/ and https://docs.gradle.org/current/release-notes.html. Pin in `gradle/wrapper/gradle-wrapper.properties`. `verify-before-use` for a newer 9.x. Note: a JVM 17–26 runs Gradle, so Java 21 is fully supported. |
| **What it is** | The Java build tool that compiles, tests, and packages the Spring Boot fat JAR. |
| **Why chosen** | Project invariant: "Gradle wrapper is committed; do not switch the backend to Maven" (root build conventions). The audit assumes `./gradlew` as the backend CI entrypoint (`B7-standards-dx.md:63`). |
| **Alternatives rejected** | **Maven** — explicitly forbidden by project convention. **Bazel for the backend** — rejected for the whole repo (`B7-standards-dx.md:53`, see §23). |
| **License** | Apache-2.0. |

**Backend static-analysis plugins to add (verdict REBUILD — none exist in v1, `README.md:143`):** Spotless + palantir-java-format (formatting), Checkstyle (Google base), SpotBugs + find-sec-bugs (bug/security), JaCoCo (coverage gate `minimum = 0.70`, ratchet quarterly), and **ArchUnit** to "lock `controller→service→repository` and forbid `ProcessBuilder` outside the FL-lifecycle package" (`B7-standards-dx.md:180`). Pin each `verify-before-use`.

---

## 3. Wire contract — Protocol Buffers + gRPC + buf

The gRPC contract is shared across four languages (Java backend, Python framework, TypeScript, C++ mobile) and the v1 copies have already drifted — the mobile copy has `SubmitModelUpdateReque` where canonical has `SubmitModelUpdateRequest` (`B7-standards-dx.md:43,80`). The package moves to `fedlearn.v2` for v2 (locked stack).

### 3.1 Protocol Buffers (protobuf)

| Field | Value |
|---|---|
| **Technology** | Protocol Buffers (the schema language + runtime). |
| **Pinned version** | protobuf runtime: pin the version the buf-generated stubs require (do NOT carry v1's phantom `protobuf>=4.21.6,<5.0.0` pin, which existed only for a non-existent Flower dependency — `B7-standards-dx.md:201`). After buf codegen, pin the exact `protobuf` Python wheel (`verify-before-use` via `buf generate` then `pip show protobuf`). |
| **What it is** | The Interface-Definition-Language (IDL) and binary serialization format for the FL wire protocol (`package fedlearn.v2`). |
| **Why chosen** | The custom proto is justified by the native C++ mobile client and DeComFL's scalar-only protocol; verdict **SALVAGE** the proto, **govern with buf** (`B2-tech-stack.md:107,156`). Neither Flower nor NVIDIA FLARE has a first-class C++ on-device client, so the custom proto "earns its keep" (`B2-tech-stack.md:148`). |
| **Alternatives rejected** | **Adopt Flower's `flwr.common.Parameters` serialization** — rejected; it cannot represent DeComFL's scalar protocol cleanly and has no C++ client path (`B2-tech-stack.md:136`). |
| **License** | BSD-3-Clause. |

### 3.2 gRPC runtime libraries

| Field | Value |
|---|---|
| **Technology** | gRPC (per-language runtime). |
| **Pinned version** | Python `grpcio` + `grpcio-tools`: `verify-before-use` (pin the exact pair, e.g. `1.6x.x`, matched to the protobuf runtime). C++ mobile: gRPC runtime cross-compiled for Android arm64 / iOS — v1 pins `v1.62.0` in `mobile_client/scripts/build_grpc_android.sh` (`B7-standards-dx.md:79`); **buf generates the *stubs* but NOT the linked C++ runtime** (`B7-standards-dx.md:93`), so the runtime cross-compile script stays, pinned `verify-before-use` to a current gRPC C++ release. |
| **What it is** | The Remote-Procedure-Call transport carrying scalars + seeds (DeComFL) or chunked tensors (FedAvg) between FL server and clients. |
| **Why chosen** | gRPC is the v1 transport and is preserved. **Transport security:** v2 defaults to TLS + mTLS with identity bound to the certificate Common-Name (CN) — the audit found "gRPC plaintext over WAN" with full TLS+mTLS *already coded but unused* (`README.md:25`, risk R6 `README.md:186`). Default-secure in v2; do not ship `grpc.insecure_channel` as the default. |
| **Alternatives rejected** | **Connect / connect-go style HTTP-Router transport** — not in the locked stack; gRPC is mandated for the mobile C++ path. |
| **License** | Apache-2.0. |

### 3.3 buf (proto single-source-of-truth + breaking-change gate)

| Field | Value |
|---|---|
| **Technology** | buf CLI (Command-Line Interface). |
| **Pinned version** | `1.70.0` (latest 1.x as of 2026-05; buf stays v1 and backward-compatible — no v2 planned). Source: https://github.com/bufbuild/buf/releases. `verify-before-use` for a newer 1.x. In CI use `bufbuild/buf-setup-action` pinned to its latest tag. |
| **What it is** | A Protobuf toolchain that lints, detects breaking changes, generates multi-language stubs, and manages the schema, replacing hand-run `protoc`. |
| **Why chosen** | **REFACTOR → buf** (`README.md:144`); it is the "highest-value DX rebuild" (`B7-standards-dx.md:71`). `buf breaking` "catches the exact class of drift that produced `SubmitModelUpdateReque`" (`B7-standards-dx.md:88`). One `buf.gen.yaml` generates Python/Java/TS/C++ from one canonical `fedlearn/v2/fedlearn.proto`. |
| **Alternatives rejected** | **Hand-run `grpcio-tools` + vendored copies (v1's model)** — produced the drift; verdict **REBUILD** (`README.md:144`, `B7-standards-dx.md:28`). |
| **License** | Apache-2.0 (buf CLI). |

**Canonical layout (from `B7-standards-dx.md:96-102`):**
```
proto/
  buf.yaml                      # module + lint + breaking config
  buf.gen.yaml                  # managed mode + per-language plugin outputs
  fedlearn/v2/fedlearn.proto    # the ONE canonical file (package fedlearn.v2)
```
Use **managed mode** to keep `java_package` out of the `.proto` (`B7-standards-dx.md:89`). **Spike-gate the C++ leg** before committing mobile to buf-generated C++ — buf does not produce the linked gRPC C++ runtime (`B7-standards-dx.md:93`). CI job `proto.yml` runs `buf lint` + `buf breaking` (against the trunk) + a `buf generate` freshness check that fails if the working tree differs (`B7-standards-dx.md:125,150`).

---

## 4. FL framework — PyTorch + custom strategies

### 4.1 PyTorch

| Field | Value |
|---|---|
| **Technology** | PyTorch (`torch`). |
| **Pinned version** | `2.12.0` (latest stable, released 2026-05-13). Source: https://pypi.org/project/torch/ and https://github.com/pytorch/pytorch/releases. Pin exact; `verify-before-use` and confirm cp312 wheels exist for x86-64 (server) and ARM64 (Jetson / Apple). |
| **What it is** | The deep-learning tensor library implementing the model, FedAvg averaging, and DeComFL zeroth-order perturbations on the server and Python client. |
| **Why chosen** | The framework is custom PyTorch with **no Flower/`flwr` dependency** (locked stack, project invariant). DeComFL and FedAvg are implemented directly against `torch`. |
| **Alternatives rejected** | **Adopt Flower (`flwr`) or NVIDIA FLARE (`nvflare`) as the substrate** — rejected because neither ships DeComFL and neither has a native C++ on-device client (`B2-tech-stack.md:18,75,148`). **TensorFlow** — not in the stack; the codebase is PyTorch. |
| **License** | BSD-3-Clause. |
| **ARM64 note** | PyTorch publishes ARM64 wheels; the Jetson path uses the NVIDIA L4T (Linux-4-Tegra) base image `nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3` and **must not** pass `--runtime nvidia` (project hardware invariant; salvaged in `README.md:129`). Do not assume the host PyPI wheel works on Jetson — use the L4T image's torch. |

**DeComFL correctness fixes (already specified at `docs/v2/specs` and `docs/v2/plans`; restated so this doc is self-contained):**
1. **`1/P` averaging factor** — server aggregation must divide by `P` (number of perturbations). v1 dropped it, so "the global model steps P× (10× at default) too far and diverges" (`README.md:18`, risk R2 `README.md:182`).
2. **CPU-canonical Random-Number-Generator (RNG)** — all perturbations are generated on CPU with a canonical generator; PyTorch "does not guarantee cross-device RNG parity" so a CUDA/MPS server silently corrupts aggregation against CPU clients (`README.md:18`, risk R3 `README.md:183`). A golden-vector parity test gates this in CI.
3. **Serializer symmetry** — the chunked/streaming upload path must wrap the payload symmetrically; v1 `KeyError`s on `'parameters'` so "LLM federations cannot complete a round" (`README.md:19`, risk R4 `README.md:184`).

**Preserve these two custom features:** **parameter chunking** for models > 300 MB (FedAvg path only — DeComFL transmits ~1 MB total regardless of model size, `B2-tech-stack.md:14`); **dual heartbeat** — a training gRPC stub that blocks during `fit()` plus a parallel heartbeat stub on its own thread so the server does not time the client out during long rounds (`B2-tech-stack.md:89-97`). Keep both behind the substrate abstraction.

### 4.2 safetensors (model checkpoint codec)

| Field | Value |
|---|---|
| **Technology** | `safetensors` (Hugging Face). |
| **Pinned version** | `verify-before-use` — pin latest stable (e.g. `0.4.x`) via `pip index versions safetensors`. |
| **What it is** | A safe (no arbitrary code execution), fast tensor serialization format for model checkpoints. |
| **Why chosen** | The serializer rebuild calls for a "typed/safetensors codec" (`README.md:44,104`). It avoids `pickle`'s code-execution risk and the stale-split trap the audit flagged for the v1 pickle cache (`README.md:195`). Checkpoints are content-addressed by SHA-256 and stored in S3/MinIO (§7). |
| **Alternatives rejected** | **Python `pickle`** — rejected: versioning/integrity hole, **REFACTOR → npz+sha256 / safetensors** (`README.md:140,195`). |
| **License** | Apache-2.0. |

### 4.3 Dataset partitioning (own the Dirichlet split; remove flwr-datasets)

| Field | Value |
|---|---|
| **Technology** | Custom `DataSource` + `Partitioner` interface over Hugging Face `datasets` + NumPy. |
| **Pinned version** | `datasets`: `verify-before-use` (latest stable). `numpy`: `verify-before-use` (pin a 1.26+/2.x consistent with the torch build). |
| **What it is** | One interface that loads a dataset and produces a non-Independent-and-Identically-Distributed (non-IID) Dirichlet partition, replacing the four forked `dirichlet_split` copies and the `flwr-datasets` dependency. |
| **Why chosen** | Verdict **REBUILD + KILL flwr_datasets + collapse 4 forks** (`README.md:110`). `flwr-datasets` is a contamination of the "no Flower" invariant and bundle bloat (`B2-tech-stack.md:66`, `README.md:161`). Partitions are keyed on content hashes in the V6 registry (§9). |
| **Alternatives rejected** | **Keep `flwr-datasets`** — KILL (`README.md:110`). It is Apache-2.0 so not a legal risk, but it violates the platform's own "no Flower" hygiene and adds bloat. |
| **License** | `datasets` Apache-2.0; NumPy BSD-3-Clause. |

---

## 5. Online-Transaction-Processing (OLTP) datastore — PostgreSQL + Flyway

### 5.1 PostgreSQL (managed, AWS RDS)

| Field | Value |
|---|---|
| **Technology** | PostgreSQL on Amazon Relational-Database-Service (RDS). |
| **Pinned version** | PostgreSQL `17.10` (latest 17.x minor as of 2026-05-14; 17 is a current major with support to ~2029). Source: https://www.postgresql.org/about/news/postgresql-184-1710-1614-1518-and-1423-released-3297/. `verify-before-use` against the RDS-available 17.x minor. Pin the RDS engine version explicitly in infrastructure-as-code. |
| **What it is** | The transactional relational database owning users, organizations, projects, memberships, `fl_runs`, the dataset registry, and audit events. |
| **Why chosen** | Verdict **REBUILD → managed Postgres** (`README.md:46`, `B6-scale-cost.md:160`). v1 ran H2 file-mode outside dev/test — "a POC crutch" (`B2-tech-stack.md:168`). Flyway already targets the Postgres dialect (v1 `build.gradle` declares `flyway-database-postgresql`). The control-plane tables are **bounded** (orgs/users/projects per the V5 identity model), so no sharding is needed. **Must fix `audit_events.metadata` from `CLOB` → `TEXT`/`JSONB` before cutover** (`README.md:169`). |
| **Alternatives rejected** | **Citus / distributed Postgres** — **KILL**: "No evidence the bounded control-plane tables need sharding; growth is telemetry → Time-Series-Database (TSDB)" (`B6-scale-cost.md:164,209`). **Aurora PostgreSQL** — deferred to hyperscale only; "Aurora Serverless v2 idle floor (~$44/mo at 0.5 ACU) makes it worse than provisioned RDS for steady load" (`B6-scale-cost.md:106,210`). **H2** — KILL outside dev/test (`README.md:108`). |
| **License** | PostgreSQL License (permissive, BSD-like). |

**Tiered sizing (from `B6-scale-cost.md`):** seed → `db.t4g.micro/small` Single-Availability-Zone (~$25–60/mo); Series-A → `db.r6g.large` Multi-AZ + 1 read replica (~$330/mo); hyperscale → Aurora writer + readers, and **route `server_logs`/`round_results`/`audit_events` time-series growth off the OLTP DB into Loki/ClickHouse** (`B6-scale-cost.md:135,162`).

### 5.2 Flyway (schema migrations)

| Field | Value |
|---|---|
| **Technology** | Flyway (community edition) + `flyway-database-postgresql`. |
| **Pinned version** | `verify-before-use` — pin the exact Flyway version managed by Spring Boot 3.5's BOM (Flyway 10+/11+); do not override unless needed. |
| **What it is** | The versioned SQL migration tool that **owns the schema**; JPA (Java Persistence API) runs in `validate`-only mode. |
| **Why chosen** | Project invariant: "Schema is owned by Flyway, not JPA" (salvaged, `README.md:199`). New entity fields require a new `V{n}__*.sql` migration; never rely on `ddl-auto=update`. The dataset/partition registry is **Flyway V6** (§9). |
| **Alternatives rejected** | **Liquibase** — not in the stack; v1 standardized on Flyway. **JPA `ddl-auto`** — forbidden; migrations are the source of truth. |
| **License** | Apache-2.0 (Flyway Community). |

**Test-profile rule (do not change):** the `test` profile uses in-memory H2 with Hibernate `create-drop` and **Flyway disabled**; migrations validate against `dev`/`ec2demo`/`production` only. CI runs PG-backed integration tests via **Testcontainers** to surface H2↔Postgres divergence (`B2-tech-stack.md:168`).

---

## 6. (reserved — see §5) 

*(Section intentionally merged into §5; numbering preserved so cross-references in sibling docs stay stable.)*

---

## 7. Artifact / model store — S3 or MinIO (content-addressed)

| Field | Value |
|---|---|
| **Technology** | Amazon Simple-Storage-Service (S3) in the managed Software-as-a-Service (SaaS); MinIO for self-hosted / on-premise. |
| **Pinned version** | S3 is a managed service (no version). MinIO: `verify-before-use` — pin the exact MinIO server image tag and `mc` client. AWS Java/Python SDKs: Java `software.amazon.awssdk:s3` `verify-before-use` (v1 pins `ecs:2.25.11` — align the S3 SDK to a current 2.x); Python `boto3` `verify-before-use`. |
| **What it is** | An object store holding model checkpoints and dataset artifacts, **content-addressed by SHA-256**, wired to `fl_runs`. |
| **Why chosen** | Verdict **REBUILD** — "Does not exist; only S3 TODOs" in v1 (`README.md:47,109`). "Models never belong in Postgres rows" (`B2-tech-stack.md:171`). Content-addressing enables reproducibility (the determinism manifest references artifact hashes) and per-round checkpoint/resume (`README.md:47,103`). MinIO matches the on-premise/federated bias and avoids egress cost for self-hosted customers (`B2-tech-stack.md:181`). |
| **Alternatives rejected** | **Database blobs (v1's effective state)** — rejected; bloats the OLTP DB and is not content-addressed. **Git-Large-File-Storage** — not an object store; wrong tool. |
| **License** | S3 proprietary managed service; MinIO is **GNU Affero General Public License v3 (AGPLv3)** — *flag*: AGPLv3 has copyleft network-use obligations; for a self-hosted artifact store used internally this is generally acceptable, but legal review is required before redistributing a modified MinIO. The managed-SaaS path uses S3 and sidesteps this. |

---

## 8. Experiment / run lineage — MLflow (self-hosted)

| Field | Value |
|---|---|
| **Technology** | MLflow (self-hosted Model Registry + tracking). |
| **Pinned version** | `3.12.0` (latest as of 2026-05; the 3.x line is current). Source: https://mlflow.org/releases/ and https://github.com/mlflow/mlflow/releases. `verify-before-use` (the search surfaced 3.10–3.12.0; pin the exact resolved version). |
| **What it is** | An open-source platform that tracks runs, parameters, metrics, and registers model versions. |
| **Why chosen** | Verdict **REBUILD → MLflow self-hosted** (`README.md:49`); chosen "over Weights & Biases" for the "on-prem/federated bias" and because it is "$0, data-resident" (`B2-tech-stack.md:172`, `README.md:49`). Each `FlRun` writes a **determinism manifest** (seed, hyperparameters, library/dataset/model hashes) plus content-addressed artifacts to MLflow + S3. |
| **Alternatives rejected** | **Weights & Biases (W&B)** — rejected: SaaS, not data-resident, not free at scale (`B2-tech-stack.md:172`). |
| **License** | Apache-2.0. |

`FlRun` is a backend aggregate; the **determinism manifest** is the reproducibility contract (verdict REBUILD — "Zero exist" in v1, `README.md:113,194`). The manifest plus golden-vector parity tests (Python ↔ C++) gate determinism in CI.

---

## 9. Dataset / partition registry (Flyway V6)

| Field | Value |
|---|---|
| **Technology** | Three Flyway-migrated tables — `datasets`, `dataset_versions`, `partition_recipes` — keyed on content hashes, plus the custom `DataSource`/`Partitioner` interface (§4.3). |
| **Pinned version** | Migration file `V6__dataset_registry.sql` (the version number is the pin). |
| **What it is** | A relational registry recording dataset identity, immutable versions, and the recipe (e.g. Dirichlet alpha, client count, seed) used to partition data — all content-hash-keyed. |
| **Why chosen** | Verdict **REBUILD** — "No registry/lineage; pickle cache → content-addressed npz" (`README.md:110`). Removes `flwr-datasets`, collapses the four `dirichlet_split` forks into one interface, and gives the determinism manifest a stable dataset hash to reference. |
| **Alternatives rejected** | **Pickle split cache (v1)** — REFACTOR → content-addressed npz+sha256: "Versioning/integrity, not RCE" (`README.md:140`). |
| **License** | N/A (own schema). |

---

## 10. Real-time channel — STOMP-over-WebSocket + relay broker

| Field | Value |
|---|---|
| **Technology** | Spring's STOMP-over-WebSocket simple broker at the edge, backed by a Redis or RabbitMQ STOMP relay once multi-replica. |
| **Pinned version** | STOMP/WebSocket: provided by `spring-boot-starter-websocket` (Boot BOM-managed). Frontend client `@stomp/stompjs` `^7.1.1` (carried from v1, verified in `frontend/package.json`); pin exact for v2. Relay broker — see §11 for Redis/RabbitMQ versions. |
| **What it is** | The bidirectional channel that streams FL-server log lines and round telemetry to the browser on `/topic/logs/{projectId}` and `/topic/results/{projectId}` (topics are keyed on `projectId`; the payload carries `runId` — `04-API-CONTRACTS.md §11`). |
| **Why chosen** | Verdict **REFACTOR** — "one-line `enableStompBrokerRelay` swap once the backend is multi-replica" (`README.md:50,117`, `B6-scale-cost.md:206`). The in-memory simple broker cannot route STOMP user-destinations across replicas, capping horizontal scale (`B6-scale-cost.md:42`). |
| **Alternatives rejected** | **Kafka as the live channel** — rejected at seed/Series-A; "Don't pay the Kafka tax early" (`B2-tech-stack.md:180`). **Server-Sent-Events** — would lose the existing STOMP client and the bidirectional contract. |
| **License** | Spring Apache-2.0; `@stomp/stompjs` Apache-2.0. |

---

## 11. Relay broker — Redis or RabbitMQ (Amazon MQ)

| Field | Value |
|---|---|
| **Technology** | RabbitMQ (with the STOMP plugin) **or** Redis Pub/Sub, fronting the STOMP relay. **Recommendation: RabbitMQ** because Spring's `enableStompBrokerRelay` speaks STOMP natively to a RabbitMQ relay; Redis is the lighter alternative if a custom relay is acceptable. |
| **Pinned version** | RabbitMQ `4.3.1` (latest stable as of 2026-05; source https://github.com/rabbitmq/rabbitmq-server/releases) — on AWS use **Amazon MQ for RabbitMQ** (`mq.m5.large` ≈ $180/mo at Series-A, `B6-scale-cost.md:114`). Redis `7.4.9` (latest 7.4.x, 2026-05-05; source https://github.com/redis/redis/releases) — on AWS use Amazon ElastiCache. `verify-before-use` for both. |
| **What it is** | A message broker that fans STOMP messages out across multiple control-plane replicas. |
| **Why chosen** | Required only "once multi-replica" (`README.md:50`). RabbitMQ's STOMP plugin is the direct fit for Spring's relay; the audit names "RabbitMQ (Amazon MQ)" and "Redis" as the two backings (`README.md:50,117`, `B6-scale-cost.md:114`). |
| **Alternatives rejected** | **NATS JetStream** — B2 floated it for a *telemetry bus* (`B2-tech-stack.md:170`), but the locked stack specifies **Redis or RabbitMQ** for the STOMP relay; do not introduce NATS. **Kafka** — too heavy for seed/Series-A (`B2-tech-stack.md:180`). |
| **License** | RabbitMQ — Mozilla Public License 2.0 (MPL-2.0). Redis — **flag**: Redis relicensed to RSALv2/SSPLv1 (dual, non-OSI) for 7.4+; for managed ElastiCache this is AWS's concern, but if self-hosting Redis 7.4+ confirm the license terms, or use the BSD-licensed **Valkey** fork as a drop-in (verify-before-use). RabbitMQ's MPL-2.0 avoids this entirely — another reason to prefer RabbitMQ for the relay. |

---

## 12. Frontend core — React 19 + Vite 6 + TypeScript

### 12.1 React

| Field | Value |
|---|---|
| **Technology** | React (with React-DOM). |
| **Pinned version** | `19.x` — pin exact (v1 uses `^19.0.0`, verified in `frontend/package.json`; resolve the latest 19 patch via `npm view react version`, `verify-before-use`). |
| **What it is** | The component library for the dashboard Single-Page-Application (SPA). |
| **Why chosen** | Verdict **SALVAGE core, refactor** — "Right tool; no Server-Side-Rendering (SSR) need" (`README.md:53,118`). |
| **Alternatives rejected** | **Next.js** — explicitly rejected: "no Next.js; no SSR need" (locked stack, `README.md:53`). The dashboard is an authenticated SPA; SSR adds complexity for no benefit. |
| **License** | MIT. |

### 12.2 Vite

| Field | Value |
|---|---|
| **Technology** | Vite (build tool + dev server) with `@vitejs/plugin-react`. |
| **Pinned version** | `6.x` — pin exact (v1 uses `^6.3.1`, verified in `frontend/package.json`; `verify-before-use` for the latest 6 patch). |
| **What it is** | The frontend bundler and dev server; its `--mode` flags mirror Spring profiles 1:1. |
| **Why chosen** | Carried from v1; the mode↔profile mapping is load-bearing (`.env.{development,ec2demo,production}` committed). `strictPort: true` on `:5173` must stay — backend Cross-Origin-Resource-Sharing (CORS) allowlist is keyed on that port (project invariant). |
| **Alternatives rejected** | **Webpack / Create-React-App** — slower, EOL tooling; v1 already moved to Vite. |
| **License** | MIT. |

### 12.3 TypeScript

| Field | Value |
|---|---|
| **Technology** | TypeScript (`tsc` in strict mode). |
| **Pinned version** | `5.x` — pin exact (v1 uses `^5.7.2`, verified in `frontend/package.json`; `verify-before-use` for the latest 5 patch). Unify to **one** TS version across `frontend` + desktop renderer + mobile RN layer (`B7-standards-dx.md:189`). |
| **What it is** | The typed superset of JavaScript for all three TS surfaces. |
| **Why chosen** | Locked stack; the audit additionally requires bumping ESLint configs to `tseslint.configs.strictTypeChecked` (`B7-standards-dx.md:182`). V5 role types are shared across surfaces. |
| **Alternatives rejected** | **Plain JavaScript** — rejected; the V5 role-type contract and Zod wire boundary need static types. |
| **License** | Apache-2.0. |

---

## 13. Frontend server-state — TanStack Query + Zod

### 13.1 TanStack Query

| Field | Value |
|---|---|
| **Technology** | TanStack Query (`@tanstack/react-query`). |
| **Pinned version** | `5.100.14` (latest 5.x as of 2026-05). Source: https://www.npmjs.com/package/@tanstack/react-query. `verify-before-use` for a newer 5.x. |
| **What it is** | A server-state cache/fetch library for React that deduplicates and caches API calls. |
| **Why chosen** | Verdict **REFACTOR → TanStack Query** — "Kills duplicate fetch triads" in v1 (`README.md:123`, `B2-tech-stack.md` lineage). Manages the REST server-state; STOMP handles the live push channel. |
| **Alternatives rejected** | **Redux Toolkit Query** — heavier and Redux-coupled. **SWR** — fewer features for cache invalidation around the run lifecycle. |
| **License** | MIT. |

### 13.2 Zod (wire-boundary validation)

| Field | Value |
|---|---|
| **Technology** | Zod. |
| **Pinned version** | `4.4.3` (latest as of 2026-05; Zod v4 is stable, ~14× faster string parsing). Source: https://www.npmjs.com/package/zod and https://zod.dev/v4. `verify-before-use` for a newer 4.x. |
| **What it is** | A TypeScript-first runtime schema validator. |
| **Why chosen** | Locked stack: "Zod validation at the wire boundary" (`README.md:53`). Every API/STOMP payload is parsed through a Zod schema at the boundary so malformed or drifted server responses fail loudly, not silently. Pairs with the V5 role types. |
| **Alternatives rejected** | **Yup / io-ts** — Zod has the strongest TS inference and is the locked choice. |
| **License** | MIT. |

---

## 14. Frontend design system — Tailwind v4 + shadcn/ui + OKLCH tokens

### 14.1 Tailwind CSS

| Field | Value |
|---|---|
| **Technology** | Tailwind CSS v4 (with `@tailwindcss/vite` + `@tailwindcss/postcss`). |
| **Pinned version** | `4.x` — pin exact (v1 uses `^4.1.12`, verified in `frontend/package.json`; `verify-before-use` for the latest 4 patch). |
| **What it is** | A utility-first CSS framework; v4 uses the `@theme` directive for design tokens. |
| **Why chosen** | Carried from v1 and required by shadcn/ui's v4 support (which is "ready… with React 19", search-confirmed). The `@theme` directive is where the OKLCH tokens live. |
| **Alternatives rejected** | **Tailwind v3** — superseded; shadcn/ui's current generator targets v4. **CSS Modules / styled-components** — not in the stack. |
| **License** | MIT. |

### 14.2 shadcn/ui

| Field | Value |
|---|---|
| **Technology** | shadcn/ui (copy-in component generator over Radix primitives + Tailwind). |
| **Pinned version** | Not a runtime dependency — it is a CLI that copies component source into the repo. Pin the CLI invocation `verify-before-use` (`npx shadcn@latest` resolves the current generator). Pin the underlying Radix primitive versions in `package.json` once generated. |
| **What it is** | A set of accessible, unstyled-by-default React components you copy into your codebase and own. |
| **Why chosen** | Verdict **REBUILD** the design system — "one OKLCH token package → shadcn/ui (web + desktop) + react-native-reusables/NativeWind (mobile)" (`README.md:55,145`). Used on **both web and the Tauri desktop renderer** (shared React). One brand, replacing v1's three disjoint palettes. |
| **Alternatives rejected** | **Material UI / Chakra** — heavier, opinionated theming that fights the OKLCH token approach; shadcn/ui is the locked choice and the component source is owned in-repo. |
| **License** | MIT (generated component code is yours; Radix primitives are MIT). |

### 14.3 OKLCH color-token package

| Field | Value |
|---|---|
| **Technology** | One internal OKLCH (Oklab Lightness-Chroma-Hue) color-token package, seeded from the existing web `theme.css`. |
| **Pinned version** | Internal package, versioned with the monorepo (no external pin). Consumed by web shadcn, desktop shadcn, and mobile NativeWind. |
| **What it is** | A single source of brand color tokens expressed in the OKLCH color space, emitted as Tailwind `@theme` variables (web/desktop) and NativeWind tokens (mobile). |
| **Why chosen** | Verdict **REBUILD** — v1 had "3 palettes; one OKLCH token pkg" (`README.md:145`). OKLCH gives perceptually uniform lightness for accessible contrast across surfaces. One brand; retire the FedMob/Desktop sub-brands (`README.md:146`). |
| **Alternatives rejected** | **Hex/HSL tokens (v1's unthemed Bootstrap hex on mobile)** — REBUILD; not perceptually uniform (`README.md:132`). |
| **License** | N/A (own package). |

---

## 15. Frontend testing — Vitest + Playwright + MSW

| Field | Value |
|---|---|
| **Technology** | Vitest (unit/component), Playwright (End-to-End / E2E), MSW (Mock Service Worker, network mocking). |
| **Pinned version** | Vitest: `3.x` `verify-before-use` (`npm view vitest version`). Playwright: `1.x` `verify-before-use` (`npm view @playwright/test version`). MSW: `2.14.6` (latest as of 2026-05; source https://www.npmjs.com/package/msw) — `verify-before-use` for a newer 2.x. |
| **What it is** | Vitest runs unit/component tests in the Vite pipeline; Playwright drives real browsers for E2E; MSW intercepts network calls to mock the API/STOMP boundary. |
| **Why chosen** | Verdict **REBUILD / stand-up** — "Zero tests on auth/STOMP/role-gates" in v1 (`README.md:122`). The locked stack names exactly Vitest + Playwright + MSW. Coverage starts by *measuring* (no fail-under yet because v1 has zero tests), then gates at 40% once auth/STOMP paths are covered (`B7-standards-dx.md:162`). |
| **Alternatives rejected** | **Jest** — Vitest integrates natively with Vite and is faster; the locked stack specifies Vitest for the frontend. (The desktop unit layer historically used Jest; v2 desktop renderer aligns to Vitest.) **Cypress** — Playwright has better multi-browser + parallelism and is the locked choice. |
| **License** | Vitest MIT; Playwright Apache-2.0; MSW MIT. |

---

## 16. Desktop — Tauri v2 + bollard

### 16.1 Tauri v2

| Field | Value |
|---|---|
| **Technology** | Tauri v2 (Rust shell + system WebView, reusing the React renderer). |
| **Pinned version** | `2.11.2` (latest as of 2026-05-16; source https://github.com/tauri-apps/tauri/releases). Ecosystem: `tauri-bundler 2.9.2`, `wry 0.55.1`, `tao 0.35.3`. `verify-before-use` for newer 2.x. |
| **What it is** | A Rust-based desktop application framework that renders the existing React UI in the OS WebView and exposes a typed Rust command layer. |
| **Why chosen** | Verdict **REBUILD shell → Tauri** (`README.md:54`). The deciding factor is **security, not size**: Tauri's updater "requires signed update artifacts (minisign keypair) — unsigned updates are rejected by the framework itself", which "directly kills" the v1 unsigned auto-install RCE (`B5-desktop-strategy.md:101`, risk R5 `README.md:185`). It also removes Node from the privileged process and shrinks the renderer attack surface (`B5-desktop-strategy.md:104`). The mandatory **code-signed minisign auto-updater** is locked. |
| **Alternatives rejected** | **Keep Electron** — REFACTOR → Tauri; Electron requires *adding* signature verification whereas Tauri makes it the default contract (`B5-desktop-strategy.md:101`). **Per-OS native (Swift/WinUI/GTK)** — KILL: 3× UI surface for an orchestrator UI (`B5-desktop-strategy.md:71,186`). **Thin shell over the mobile C++ core in-process** — KILL for v2: no bundle win (libtorch dominates either way), adds a fragile cross-language RNG-parity invariant, and collapses process isolation (`B5-desktop-strategy.md:56-67,189`). |
| **License** | MIT / Apache-2.0. |

**Open risks to re-verify at build time (do not treat as blockers):** (1) WebKitGTK rendering parity for framer-motion/recharts on Linux — smoke-test before committing (`B5-desktop-strategy.md:108,200`); (2) Tauri sidecar code-signing issues #11778 / #9981 — re-check against the pinned 2.11.2 release; if unresolved, sign the FL client as a non-sidecar external binary (`B5-desktop-strategy.md:125,201`).

### 16.2 bollard (Docker client in Rust)

| Field | Value |
|---|---|
| **Technology** | `bollard` (async Rust Docker API client). |
| **Pinned version** | `verify-before-use` — pin the exact crate version in `Cargo.toml` (`cargo search bollard`). |
| **What it is** | The Rust equivalent of `dockerode`, used to talk to the Docker socket for the Jetson container path. |
| **Why chosen** | Verdict **REFACTOR → bollard** (`README.md` desktop row; `B5-desktop-strategy.md:144,192`). It covers `createContainer`/`start`/`logs(follow)`/`stop`/`remove` and demuxes the multiplexed stream that v1 hand-rolled. The Jetson invariant — no `--runtime nvidia`, explicit `/dev/nvhost-*` device mounts — ports verbatim to a bollard `HostConfig` (`B5-desktop-strategy.md:142`). |
| **Alternatives rejected** | **`dockerode` (v1, Node)** — replaced by the Tauri/Rust migration; bollard also removes the hand-rolled stream demux. |
| **License** | Apache-2.0. |

**Desktop training-engine model (do not change):** keep training as a **subprocess/sidecar** (the PyInstaller-bundled Python client), not in-process C++. This keeps DeComFL RNG parity free (same Python/PyTorch as the server) and preserves process isolation so a libtorch crash does not take down the UI (`B5-desktop-strategy.md:62,166`). The OS keychain holds the JWT via Tauri's keychain command; the renderer never sees the token.

**Code-signing budget (any framework, OS-count not framework-count):** Apple Developer Program $99/yr (notarization free); Windows Azure Trusted Signing ~$120/yr (US/Canada + legal entity required — verify eligibility) or OV/EV Authenticode ~$200–580/yr; Linux unsigned is normal (`B5-desktop-strategy.md:118-127`).

---

## 17. Mobile FL core — React Native + NativeWind + react-native-reusables + libtorch (C++)

### 17.1 React Native + bridge

| Field | Value |
|---|---|
| **Technology** | React Native (TurboModule / JavaScript-Interface (JSI) bridge to the native C++ core). |
| **Pinned version** | `0.8x` line — pin exact (`verify-before-use`; the search found 0.86.x current as of mid-2026 — pin the latest stable 0.8x via `npm view react-native version`). |
| **What it is** | The mobile app shell hosting the on-device FL core via a TurboModule (ObjC++ on iOS, Java-Native-Interface / JNI on Android). |
| **Why chosen** | Verdict **SALVAGE (RN), REBUILD styling** — "Sound TurboModule wiring; unthemed Bootstrap hex" (`README.md:132`). The RN bridge wiring is correct; only the styling and the C++ core need work. |
| **Alternatives rejected** | **Flutter** — would discard the working RN bridge and the shared React/TS knowledge. **Native-only (no RN)** — loses cross-platform UI reuse. |
| **License** | MIT. |

### 17.2 NativeWind + react-native-reusables

| Field | Value |
|---|---|
| **Technology** | NativeWind (Tailwind-for-RN) + react-native-reusables (shadcn/ui ported to RN). |
| **Pinned version** | NativeWind `4.x` `verify-before-use`; react-native-reusables `verify-before-use` (pin exact from npm). |
| **What it is** | NativeWind brings Tailwind utility classes to RN; react-native-reusables provides shadcn-equivalent RN components. |
| **Why chosen** | Locked stack: the OKLCH token package feeds "react-native-reusables / NativeWind (mobile)" (`README.md:55`). One brand, one token source across web/desktop/mobile. |
| **Alternatives rejected** | **Bootstrap hex (v1 mobile)** — REBUILD (`README.md:132`). **React Native Paper** — Material-opinionated; fights the shared OKLCH tokens. |
| **License** | NativeWind MIT; react-native-reusables MIT. |

### 17.3 libtorch (C++ ARM64 on-device core)

| Field | Value |
|---|---|
| **Technology** | libtorch (the PyTorch C++ distribution) + gRPC C++, implementing DeComFL on-device. |
| **Pinned version** | Pin the libtorch version to **match the server-side PyTorch** (`2.12.0`) as closely as the ARM64 build allows — RNG parity across versions is the risk (`B5-desktop-strategy.md:59,203`). `verify-before-use`: confirm an ARM64 libtorch build exists for the target PyTorch and pin it explicitly in `mobile_client/CMakeLists.txt` (`LIBTORCH_DIR`). |
| **What it is** | The native C++17 FL client linking libtorch for on-device tensor math, with dual gRPC channels and parameter chunking already implemented in C++ (`B5-desktop-strategy.md:48`). |
| **Why chosen** | Verdict **SALVAGE (mobile-only), REBUILD/harden the ZO core** (`README.md:45,134`). CPython/PyInstaller cannot ship on mobile, so the C++ core is justified *only on mobile* (`B5-desktop-strategy.md:67`). It must use **CPU-canonical RNG** and pass a **golden-vector parity test** gating Python↔C++ determinism (`README.md:45,113`, risk R3). Harden: float32→correct dtype, add a `requires_grad` filter (`README.md:134`). |
| **Alternatives rejected** | **Ship the Python client on mobile** — impossible (no CPython on iOS/Android in this model). **Reuse the C++ core on desktop too (v2)** — KILL for v2 (`B5-desktop-strategy.md:189`); revisit only as a v3 unification with a passing parity gate. |
| **License** | libtorch BSD-3-Clause; gRPC Apache-2.0. |
| **ARM64 note** | libtorch is ~267 MB (CPU) to ~1.9 GB (CUDA) (`B5-desktop-strategy.md:56`); confirm the ARM64 build and size budget before committing. The gRPC C++ runtime is cross-compiled by the existing scripts (buf does not provide it — §3.2). |

---

## 18. FL orchestration substrate — Kubernetes + FlServerLauncher backends

The substrate is the single biggest **REBUILD**: a long-running, multi-run server keyed on `run_id`, with a durable `fl_runs` lease table and a reconciler loop, replacing v1's per-project `ProcessBuilder` model (capped at 11 ports, state lost on JVM restart, round loop hangs on one straggler) (`README.md:23,43`, `B6-scale-cost.md:36-42`).

### 18.1 FlServerLauncher abstraction (three backends)

| Backend | Pinned version | What / when | Why |
|---|---|---|---|
| **Kubernetes Jobs** (primary, production) | Kubernetes `1.36.x` (latest stable `1.36.1`, 2026-05-13; source https://kubernetes.io/releases/). On AWS use Elastic-Kubernetes-Service (EKS) — pin the EKS-supported 1.3x version `verify-before-use`. Java client: `io.kubernetes:client-java` `verify-before-use`. | One FL server = one Kubernetes `Job`; GPU runs land on `g5`/`g6` nodes via nodeSelector; bin-packed via Karpenter at hyperscale. | Primary production backend per the locked stack and `README.md:43`. EKS+Karpenter wins above ~30–50 steady concurrent tasks because it bin-packs and schedules GPUs that Fargate cannot (`B6-scale-cost.md:52`). |
| **AWS ECS RunTask** (secondary) | ECS is a managed service. Java SDK `software.amazon.awssdk:ecs` `2.25.11` (carried from v1, verified in build.gradle; `verify-before-use` to align with a current 2.x). | One FL server = one Fargate task via `EcsClient.runTask(...)`; per-second billing, scale-to-zero between runs. | Verdict **SALVAGE → complete**: the path is "already coded" but must persist the task Amazon-Resource-Name (ARN), poll/stop, and reconcile (`README.md:101`, `B6-scale-cost.md:204`). The right Series-A primitive (no EKS $73/mo control-plane fee). |
| **LocalProcessLauncher** (dev only) | N/A (the old `ProcessBuilder` model behind the abstraction). | Spawns `python fl_server.py` locally for development only. | Verdict **KILL for non-dev, keep as dev-only** (`README.md:43,99`). Raise the dev port range and fix the reader-thread hazard, but never use in a deployed env. |

| Field | Value |
|---|---|
| **What it is** | A Java interface `FlServerLauncher.launch(run_id, config)` with three pluggable backends; the control plane's `/start` "submit a run to the substrate", not "fork a Python process and grab a port" (`B2-tech-stack.md:148`). |
| **Why chosen** | v1's substrate is "an architectural anti-pattern that Flower/FLARE solve with one long-running multi-run server" (`B2-tech-stack.md:190`). The custom substrate is kept (not Flower/FLARE) **only because** the native C++ mobile client and DeComFL's scalar protocol do not fit Flower's Python-SDK `Parameters` model (`B2-tech-stack.md:148`). |
| **Alternatives rejected** | **Adopt Flower SuperLink/SuperNode (Option A)** — rejected: violates the "no `flwr`" invariant at the substrate level and has no native C++ on-device client (`B2-tech-stack.md:136`). **Adopt NVIDIA FLARE (Option B)** — rejected: heaviest framework, same C++-client gap, NVIDIA-ecosystem coupling (`B2-tech-stack.md:140`). |
| **License** | Kubernetes Apache-2.0; ECS proprietary managed; kubernetes-client-java Apache-2.0; AWS SDK Apache-2.0. |

### 18.2 Durable lease + reconciler (the reliability contract)

The `fl_runs` table is a **lease** in Postgres; the JVM is a **stateless supervisor** over that lease (`README.md:83`). Pseudocode for the two mandatory mechanisms (restated from the locked stack so this doc is self-contained):

```
# Reconciler loop (boot-time + periodic, e.g. every 30s)
for run in fl_runs.where(state in [STARTING, RUNNING]):
    actual = launcher.describe(run.backend_handle)   # k8s Job / ECS task / local pid
    if actual is MISSING and run.lease_expired():
        run.state = FAILED; emit_alert(run)           # crashed/orphaned
    elif actual is RUNNING:
        run.renew_lease(now + LEASE_TTL)              # supervisor heartbeat
    elif actual is SUCCEEDED:
        run.state = COMPLETED

# Round loop MUST have a deadline + minimum-quorum (no infinite hang on a straggler)
def run_round(run, clients, round_idx):
    deadline = now() + ROUND_DEADLINE          # e.g. config per run
    min_quorum = ceil(QUORUM_FRACTION * len(clients))   # e.g. 0.6
    received = []
    while now() < deadline and len(received) < len(clients):
        r = wait_for_next_result(timeout=deadline-now())
        if r: received.append(r)
    if len(received) < min_quorum:
        run.state = FAILED; reason = "quorum_not_met"   # do NOT hang forever
        return
    aggregate(received)        # FedAvg: 1/N ; DeComFL: 1/P factor (see §4.1)
    checkpoint_to_s3(run, round_idx)   # per-round, content-addressed
```

| Field | Value |
|---|---|
| **Why chosen** | Verdict **REBUILD** — v1 has "No run entity, no checkpoint/resume; round loop hangs on one straggler; state lost on restart" (risk R9 `README.md:189`, `B6-scale-cost.md:40`). The deadline + quorum is the locked-stack mandate "no infinite hang on a straggler". |
| **Cost controls (P0, locked):** | **Per-org concurrency quotas + admission control** before lifting the port cap, and **scale-to-zero** orchestration — "Without it, the Fargate/EKS bill is unbounded" (`B6-scale-cost.md:187,188`, risk R10 `README.md:190`). |

---

## 19. Aggregation robustness & privacy — Differential Privacy + robust guard

| Field | Value |
|---|---|
| **Technology** | Differential Privacy (DP) — DP-SGD (Differentially-Private Stochastic-Gradient-Descent) on the FedAvg path, calibrated scalar-DP on DeComFL — plus a robust-mean/clipping aggregation guard. |
| **Pinned version** | If using Opacus for DP-SGD: `verify-before-use` (pin exact, matched to torch 2.12). The scalar-DP for DeComFL and the robust-mean/clipping guard are implemented in-framework against `torch`/`numpy` (no extra dependency). |
| **What it is** | A privacy layer (calibrated noise) and a robustness guard (clip + robust mean) over the aggregation step. |
| **Why chosen** | Verdict **REBUILD** — "none today" (`README.md:57,107`, risk R12 `README.md:192`). **Delete the false "Byzantine-robust" README claim** — "the paper makes no such claim; aggregation is an unguarded mean" (`README.md:106`). Note: DeComFL's scalar-only uploads already structurally kill the Deep-Leakage-from-Gradients (DLG) reconstruction attack family — that is the privacy wedge to *market*, not to overstate (`README.md:14,199`). |
| **Alternatives rejected** | **Claim Byzantine robustness without implementing it (v1)** — KILL; it is false and a liability (`README.md:106`). |
| **License** | Opacus (if used) Apache-2.0. |

---

## 20. Platform observability — Micrometer + Prometheus + Grafana + Loki + Tempo + OTel

| Field | Value |
|---|---|
| **Technology** | Micrometer (JVM metrics) → Prometheus (metrics store) on an internal management port; Grafana (dashboards) + Loki (logs) + Tempo (traces) + OpenTelemetry (OTel) Collector; structlog for structured Python logs. |
| **Pinned versions** | Micrometer + `micrometer-tracing-bridge-otel`: Boot 3.5 BOM-managed (`verify-before-use`). Prometheus `3.12.0` (2026-05-28, https://github.com/prometheus/prometheus/releases). Grafana `13.0.1` (2026-05, https://grafana.com/blog/grafana-13-release-all-the-latest-features/). Loki `3.7.2` (2026-05). Tempo `3.0` (note: 3.0 has breaking config changes — pin and read its migration notes). OTel Collector `0.153.0` (2026-05-26, https://github.com/open-telemetry/opentelemetry-collector-releases/releases). All `verify-before-use`. |
| **What it is** | The metrics/logs/traces stack: Micrometer exposes Prometheus metrics on an internal port; OTel propagates a W3C `traceparent`; Grafana visualizes against Prometheus/Loki/Tempo. |
| **Why chosen** | Verdict **REBUILD** — "deps pinned, imported nowhere" in v1 (`README.md:51,114`). The locked requirement is a **W3C `traceparent` propagated JVM → spawned Python → client → mobile** so one trace spans every hop (`README.md:51,82`). structlog carries `project_id`/`round_idx`/`trace_id`. Self-hosting Grafana/Prom/Loki is correct vs Datadog at scale ($80–120/host/mo is "untenable", `B6-scale-cost.md:139`). |
| **Alternatives rejected** | **Datadog** — cost-prohibitive at hyperscale host counts (`B6-scale-cost.md:139`). **No tracing (v1)** — REBUILD; correlation IDs are required (`README.md:114`). |
| **License** | Micrometer Apache-2.0; Prometheus Apache-2.0; Grafana **AGPLv3** (*flag*: self-hosting internally is fine; redistribution of a modified Grafana triggers copyleft — use Grafana Cloud or unmodified self-host); Loki AGPLv3 (same caveat); Tempo AGPLv3 (same caveat); OTel Collector Apache-2.0. |

**Cardinality discipline (cost control):** keep `client_id` off histogram labels — "label cardinality *is* the observability bill" at scale; send per-client detail to MLflow instead (`B6-scale-cost.md:119,193`).

### 20.1 FL-run telemetry pipeline (salvage + extend)

The `RoundResult` → `/api/internal/runs/{runId}/results` → STOMP → recharts pipeline is **SALVAGE + extend** (`README.md:52,115`; the authoritative path/shape is `04-API-CONTRACTS.md §5`). Two mandatory changes: (1) make the per-round POST **incremental** (v1 batches after the run completes, so the chart only populates at the end — `README.md:52,167`); (2) add a **communication-cost panel** — DeComFL's "bandwidth wedge" — plus per-client small-multiples and the `round_results` bytes/scalars-transmitted columns (`uplink_bytes`/`downlink_bytes`/`scalars_transmitted`, created by the `V7` migration in `03-DATA-MODEL.md §5.2`). recharts is carried from v1 (`^2.15.2`, verified in `frontend/package.json`; pin exact, `verify-before-use`). The result callback uses a **per-run scoped token**, not the single global key (`README.md:50,98`).

---

## 21. Compliance posture (architectural, informs the stack)

| Field | Value |
|---|---|
| **Technology** | SOC 2 (System and Organization Controls 2) Type 2 + HIPAA (Health Insurance Portability and Accountability Act)-readiness architecture; defer FedRAMP (Federal Risk and Authorization Management Program). |
| **What it is** | The compliance program the architecture must support from day one. |
| **Why chosen** | Verdict **REBUILD** — "the healthcare demo makes HIPAA the floor" (`README.md:151,191`, risk R11). The pneumonia/healthcare demo means Protected-Health-Information (PHI) handling, so HIPAA-readiness (data residency via MinIO/RDS in-region, audit-event capture, encryption in transit via mTLS, scoped tokens) is a stack constraint, not an afterthought. |
| **Alternatives rejected** | **FedRAMP now** — deferred; out of scope for the initial program (`README.md:151`). |
| **License** | N/A (program, not software). |

---

## 22. Supply chain — Renovate + vulnerability scanners + SBOM

| Field | Value |
|---|---|
| **Technology** | Renovate (dependency upgrade bot) + per-stack scanners (`pip-audit`, `npm audit`, OWASP/Gradle dependency-check, gitleaks) + Software-Bill-Of-Materials (SBOM) via CycloneDX. |
| **Pinned version** | Renovate: run via the hosted GitHub App or pin the self-hosted action `verify-before-use`. `pip-audit`, `gitleaks`, `@cyclonedx/*`, `dependency-check`: pin each exact in CI `verify-before-use`. |
| **What it is** | Automated dependency upgrades plus security scans and a machine-readable inventory of every dependency. |
| **Why chosen** | Verdict **REBUILD** — "Nothing exists; lockfiles will rot; backend is on an EOL Spring Boot line" (`B7-standards-dx.md:32`). Renovate is chosen over Dependabot for its grouping (Spring stack as one PR, ML stack monthly) which "fits a polyglot repo far better" (`B7-standards-dx.md:198`). gitleaks needs a one-time history baseline before the trunk is locked (`B7-standards-dx.md:200`). SBOM is a SOC 2/HIPAA prerequisite (`B7-standards-dx.md:202`). |
| **Alternatives rejected** | **Dependabot** — weaker grouping for a polyglot monorepo (`B7-standards-dx.md:198`). **No scanning (v1)** — a PR could merge vulnerable/leaked code (`B7-standards-dx.md:14`). |
| **License** | Renovate AGPLv3 (the hosted app is free for OSS/most use); pip-audit Apache-2.0; gitleaks MIT; CycloneDX Apache-2.0. |

---

## 23. Monorepo / CI tooling — Makefile/Taskfile + paths-filter + GitHub Actions

| Field | Value |
|---|---|
| **Technology** | A root `Makefile` (or `Taskfile.yml`) as the one "run all checks" entrypoint, delegating to each unit's native tool, plus `dorny/paths-filter` for affected-only CI, on GitHub Actions. |
| **Pinned version** | GNU Make (system) or Taskfile `verify-before-use`. `dorny/paths-filter`: pin the action to its latest tag `verify-before-use`. `bufbuild/buf-setup-action`: pin latest tag. |
| **What it is** | A thin, language-agnostic task orchestration layer (`make lint`, `make test`, `make proto`, `make build`) that CI and developers both call; paths-filter runs only the units a PR touches. |
| **Why chosen** | Verdict **REBUILD** — "no PR CI today… the highest-leverage, lowest-cost fix" (`README.md:56,141`, `B7-standards-dx.md:14`). The Makefile + paths-filter gives Nx-style affected-only builds "at near-zero cost, for a 6-unit repo" (`B7-standards-dx.md:64`). PR-time `ci.yml` + **branch protection** (required status checks) is the literal mechanism that closes the "PR can merge broken" gap (`B7-standards-dx.md:152`). |
| **Alternatives rejected** | **Bazel** — **REJECT**: true polyglot but "demands a dedicated build infrastructure team" — a full-time role this team does not have (`B7-standards-dx.md:53,59`). **Nx / Turborepo** — Nx is **deferred** to the JS/TS triangle only (frontend + desktop renderer + mobile RN) *if* a shared component library materializes; Turborepo covers only 2 of 6 units (`B7-standards-dx.md:57,65`). **Bazel is rejected; Nx is deferred** — do not adopt either now. |
| **License** | GNU Make GPLv3; Taskfile MIT; dorny/paths-filter MIT; GitHub Actions proprietary platform. |

**CI workflow set (from `B7-standards-dx.md:137-146`):** `ci.yml` (orchestrator → paths-filter), `backend.yml`, `framework.yml`, `frontend.yml`, `desktop.yml`, `mobile.yml`, `proto.yml` (buf lint + breaking + freshness), `security.yml` (gitleaks + Trivy + pip-audit), `release.yml`. **Kill the duplicate `desktop-release.yml`; keep `release-desktop.yml`** (`README.md:142`, `B7-standards-dx.md:26`). Keep expensive macOS/Windows multi-arch builds **tag-gated**, not on the PR path (Linux runners are ~10× cheaper, `B7-standards-dx.md:169`). Per-unit independent versioning via conventional commits + release-please/changesets; the wire contract versions via `buf breaking` within `fedlearn.v2` (`B7-standards-dx.md:213-216`).

---

## 24. Consolidated version-pin table (copy into manifests)

Pin these **exact** values. `verify-before-use` (VBU) entries: run the stated resolution command/source first, then pin the resolved value. All versions verified via web search on 2026-05-29 against the cited sources; re-verify before the build.

### 24.1 Language runtimes

| Component | Pin | Status | Source / resolution |
|---|---|---|---|
| Java (Temurin JDK LTS) | `21.0.7+6` | VBU | https://adoptium.net/temurin/releases/?version=21 |
| Python (CPython) | `3.12.9` | VBU | `pyenv install --list \| grep 3.12` |
| Node.js (Active LTS) | `24.4.0` | VBU | https://nodejs.org/en/about/previous-releases |
| Rust (stable) | `1.87.0` | VBU | `rustup show` / https://forge.rust-lang.org/ |

### 24.2 Control plane + wire contract

| Component | Pin | Status | Source |
|---|---|---|---|
| Spring Boot | `3.5.14` | verified | https://spring.io/blog/2026/04/23/spring-boot-3-5-14-available-now/ |
| Spring gRPC | latest 1.x | VBU | https://github.com/spring-projects/spring-grpc/releases |
| Gradle | `9.5.1` | verified | https://gradle.org/releases/ |
| jjwt-api / -impl / -jackson | `0.12.5` | VBU | matches v1 build.gradle |
| AWS SDK ecs | `2.25.11` | VBU | matches v1; align to current 2.x |
| buf CLI | `1.70.0` | verified | https://github.com/bufbuild/buf/releases |
| protobuf runtime | from buf stubs | VBU | `buf generate` then `pip show protobuf` |
| grpcio / grpcio-tools (Python) | matched pair | VBU | pin to protobuf runtime |
| gRPC C++ runtime (mobile) | current release | VBU | replaces v1 `v1.62.0` in build scripts |

### 24.3 FL framework + data

| Component | Pin | Status | Source |
|---|---|---|---|
| PyTorch (`torch`) | `2.12.0` | verified | https://pypi.org/project/torch/ |
| safetensors | latest 0.4.x | VBU | `pip index versions safetensors` |
| Hugging Face `datasets` | latest | VBU | npm/pip index |
| NumPy | 1.26+/2.x consistent w/ torch | VBU | pin with torch build |
| Opacus (DP-SGD, if used) | latest matched to torch | VBU | pin exact |

### 24.4 Datastore + storage + lineage

| Component | Pin | Status | Source |
|---|---|---|---|
| PostgreSQL (RDS) | `17.10` | verified | https://www.postgresql.org/about/news/postgresql-184-1710-1614-1518-and-1423-released-3297/ |
| Flyway (+ postgresql) | Boot 3.5 BOM | VBU | do not override |
| S3 SDK (Java `s3` / Python `boto3`) | current 2.x / latest | VBU | align to AWS SDK |
| MinIO (self-host) | latest stable image tag | VBU | pin image digest |
| MLflow | `3.12.0` | verified | https://mlflow.org/releases/ |

### 24.5 Real-time + broker

| Component | Pin | Status | Source |
|---|---|---|---|
| STOMP/WebSocket | Boot BOM | VBU | `spring-boot-starter-websocket` |
| `@stomp/stompjs` | `7.1.1` | VBU | matches v1 frontend |
| RabbitMQ (preferred relay) | `4.3.1` | verified | https://github.com/rabbitmq/rabbitmq-server/releases |
| Redis (alt; or Valkey) | `7.4.9` | verified | https://github.com/redis/redis/releases (note license caveat §11) |

### 24.6 Frontend

| Component | Pin | Status | Source |
|---|---|---|---|
| React / React-DOM | `19.x` | VBU | `npm view react version` |
| Vite | `6.x` | VBU | `npm view vite version` |
| TypeScript | `5.x` | VBU | `npm view typescript version` |
| TanStack Query | `5.100.14` | verified | https://www.npmjs.com/package/@tanstack/react-query |
| Zod | `4.4.3` | verified | https://www.npmjs.com/package/zod |
| Tailwind CSS | `4.x` (e.g. `4.1.12`) | VBU | matches v1 frontend |
| shadcn/ui | CLI (no runtime pin) | VBU | `npx shadcn@latest` |
| recharts | `2.15.2` | VBU | matches v1 frontend |
| Vitest | `3.x` | VBU | `npm view vitest version` |
| Playwright | `1.x` | VBU | `npm view @playwright/test version` |
| MSW | `2.14.6` | verified | https://www.npmjs.com/package/msw |

### 24.7 Desktop + mobile

| Component | Pin | Status | Source |
|---|---|---|---|
| Tauri v2 | `2.11.2` | verified | https://github.com/tauri-apps/tauri/releases |
| tauri-bundler / wry / tao | `2.9.2` / `0.55.1` / `0.35.3` | verified | same release set |
| bollard (Rust crate) | latest | VBU | `cargo search bollard` |
| React Native | `0.8x` (latest stable) | VBU | `npm view react-native version` |
| NativeWind | `4.x` | VBU | `npm view nativewind version` |
| react-native-reusables | latest | VBU | npm |
| libtorch (C++ ARM64) | match torch `2.12.0` | VBU | confirm ARM64 build exists |

### 24.8 Orchestration + observability + supply chain

| Component | Pin | Status | Source |
|---|---|---|---|
| Kubernetes (EKS) | `1.36.x` | verified | https://kubernetes.io/releases/ |
| kubernetes-client-java | latest | VBU | Maven Central |
| Prometheus | `3.12.0` | verified | https://github.com/prometheus/prometheus/releases |
| Grafana | `13.0.1` | verified | https://grafana.com/blog/grafana-13-release-all-the-latest-features/ |
| Loki | `3.7.2` | verified | grafana/loki releases |
| Tempo | `3.0` (breaking config) | verified | grafana/tempo release notes |
| OpenTelemetry Collector | `0.153.0` | verified | https://github.com/open-telemetry/opentelemetry-collector-releases/releases |
| Micrometer (+ otel bridge) | Boot 3.5 BOM | VBU | do not override |
| Renovate | hosted app / latest | VBU | GitHub App |
| gitleaks / pip-audit / CycloneDX | latest each | VBU | pin in CI |
| dorny/paths-filter | latest tag | VBU | pin action SHA |
| buf-setup-action | latest tag | VBU | pin action SHA |

---

## 25. Hard invariants the local model must never violate

These are restated from the locked stack and project conventions so they are impossible to lose:

1. **No `flwr` / Flower dependency** anywhere. Custom protobuf only (`package fedlearn.v2`). Remove `flwr-datasets`.
2. **No `Authorization: Bearer` header in the frontend.** Cookie-only HttpOnly JWT; `withCredentials: true`; no `localStorage` token.
3. **Schema is owned by Flyway, not JPA.** New fields = new `V{n}__*.sql`. `test` profile keeps Flyway disabled (in-memory H2 `create-drop`) — do not change that.
4. **Gradle wrapper is committed.** Do not switch the backend to Maven.
5. **gRPC defaults to TLS + mTLS** (identity bound to certificate CN). Do not ship `insecure_channel` as the default.
6. **DeComFL: `1/P` averaging factor + CPU-canonical RNG + symmetric serializer.** Golden-vector Python↔C++ parity test gates determinism.
7. **Jetson:** L4T base image, **no `--runtime nvidia`**, explicit `/dev/nvhost-*` device mounts.
8. **FL round loop has a deadline + minimum-quorum.** Never hang on a straggler.
9. **Per-org concurrency quotas + scale-to-zero** before lifting the port cap.
10. **Delete the false "Byzantine-robust" claim.** Market the DeComFL scalar-only DLG-resistance wedge truthfully.
11. **No AI attribution** in any commit, PR, comment, doc, or changelog. Authorship is human-only.

---

*End of 02-TECH-STACK.md. All file:line citations refer to the v2 audit reports under `/home/anurag/codebase/FedLearn-Platform/docs/audit/2026-05-29/`. All external version claims carry a source URL and were verified on 2026-05-29; re-verify `verify-before-use` items before the build.*
