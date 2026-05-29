# 91 — Local-Model Usage Guide (FedLearn Platform v2)

**Document type:** Meta-guide — how to *use* this documentation set to build the project.
**Audience:** YOU — a mid-sized local LM (Language Model, ~30 billion parameters, e.g. Qwen/Llama 32B running on an Apple M4 Max). You are competent but you CANNOT infer missing context, CANNOT make architecture decisions, and lose the thread on ambiguity. This guide removes that risk by telling you exactly how to read, build, and hand off.
**Status:** build-authoritative for v2 (version 2). This is the FIRST document you read and the one you return to between tasks.
**Date authored:** 2026-05-29.
**Source of truth:** the v2 audit synthesis at `/home/anurag/codebase/FedLearn-Platform/docs/audit/2026-05-29/README.md`. The locked tech stack lives in `02-TECH-STACK.md`; the interface surface in `04-API-CONTRACTS.md`.

> **Abbreviation key (first-use full forms, repeated here so this file is self-contained):**
> LM (Language Model), LLM (Large Language Model), v2 (Version 2), FL (Federated Learning), DeComFL (Dimension-Free Communication Federated Learning — the platform's zeroth-order optimization strategy; note: v1 wikis mis-expanded it as "Decomposed", which is wrong per the paper, see `04-API-CONTRACTS.md §0`), FedAvg (Federated Averaging), ZO (Zeroth-Order optimization), HLD (High-Level Design), LLD (Low-Level Design), ADR (Architecture Decision Record), API (Application Programming Interface), REST (Representational State Transfer), gRPC (Google Remote Procedure Call), STOMP (Simple Text Oriented Messaging Protocol), WS (WebSocket), JWT (JSON Web Token), JSON (JavaScript Object Notation), RBAC (Role-Based Access Control), RLS (Row-Level Security), TDD (Test-Driven Development), CI (Continuous Integration), PR (Pull Request), DoD (Definition of Done), DTO (Data Transfer Object), RNG (Random Number Generator), S3 (Simple Storage Service), MinIO (the self-hosted S3-compatible object store), RDS (Relational Database Service), MLflow (the self-hosted experiment/model registry), DP (Differential Privacy), CSP (Content-Security-Policy), HSTS (HTTP Strict Transport Security), SBOM (Software Bill of Materials), OTel (OpenTelemetry), k8s (Kubernetes), ECS (Elastic Container Service), ARN (Amazon Resource Name), HMAC (Hash-based Message Authentication Code), sha256 (Secure Hash Algorithm 256-bit), UUID (Universally Unique Identifier), JVM (Java Virtual Machine), MSW (Mock Service Worker), JDK (Java Development Kit), SoT (Source of Truth).

---

## 0. The one-paragraph orientation (read this, then §1)

You are building FedLearn v2: a federated-learning platform with five deployable units (Spring Boot control plane, custom Python FL framework, React frontend, Tauri desktop, native C++ mobile core). This documentation set has **pre-decided every architecture choice for you.** Your job is to implement bodies behind already-specified contracts, write the tests the LLDs tell you to write, and stop-and-ask whenever a detail is missing instead of guessing. You do this one task at a time, commit after each, and produce a handoff package at every milestone so a human can run an external reviewer to audit your work. Never invent a technology, never change a contract, never claim "done" without running the verification command. The rest of this guide is the operating manual for exactly that.

---

## 1. Reading order (do not skip; do not reorder)

Read in this exact order. The first four are FOUNDATION (read fully, once, before any code). The LLDs are per-unit and read just-in-time before building that unit.

| Order | Document | Path | Read when | Why this order |
|---|---|---|---|---|
| 1 | This guide | `docs/v2/build/91-LOCAL-MODEL-GUIDE.md` | First, always | Operating rules; how to read everything else. |
| 2 | Architecture HLD | `docs/v2/build/01-ARCHITECTURE-HLD.md` | Before any code | The system shape, the five units, the data-flow, what talks to what. Gives you the map so a single-unit task never contradicts the whole. |
| 3 | Tech stack | `docs/v2/build/02-TECH-STACK.md` | Before any code | The LOCKED, pinned technology list with exact versions. This is the *only* list of technologies you may use. GOLDEN RULE 1 forbids anything not in here. |
| 4 | Data model | `docs/v2/build/03-DATA-MODEL.md` | Before any code | Every table, column, type, Flyway migration (`V*__*.sql`). Schema is owned by Flyway, JPA is validate-only. |
| 5 | API contracts | `docs/v2/build/04-API-CONTRACTS.md` | Before any code | Every REST endpoint, gRPC service in package `fedlearn.v2`, STOMP topic, error envelope, the per-run scoped result token, and `traceparent` propagation. GOLDEN RULE 2 forbids changing anything in here. |
| 6 | The LLD for the unit you are about to build | `docs/v2/build/1X-LLD-*.md` (table below) | Immediately before building that unit | The only place the concrete interface signatures, algorithms, build-task checklist, and done-conditions for that unit live. |
| 7 | The audit synthesis (reference) | `docs/audit/2026-05-29/README.md` | When an LLD cites a finding id (e.g. `A1-F6`, `R9`) and you need the rationale | The "why" behind a decision. You do not act on the audit directly; the LLDs translate it into tasks. |

**LLD index (build these in dependency order — left column = the canonical doc number):**

| LLD | Unit | Path | Depends on (build first) |
|---|---|---|---|
| 10 | Control plane (auth, orgs, projects, controllers) | `docs/v2/build/10-LLD-*.md` | 03, 04, 18 |
| 11 | Python FL framework (FedAvg + DeComFL, serializer, gRPC) | `docs/v2/build/11-LLD-*.md` | 04 (gRPC), DeComFL spec/plan in `docs/v2/specs` + `docs/v2/plans` |
| 12 | FL orchestration substrate (`FlServerLauncher`, reconciler, leases) | `docs/v2/build/12-LLD-orchestration-substrate.md` | 03 (`fl_runs`), 04 (§4/§5/§13), 10, 18 |
| 13 | Frontend dashboard (React, TanStack Query, STOMP) | `docs/v2/build/13-LLD-frontend-dashboard.md` | 04, 18 |
| 17 | Data + artifact stores (Postgres, S3/MinIO, dataset registry) | `docs/v2/build/17-LLD-*.md` | 03 |
| 18 | Security + compliance (JWT, RLS, mTLS, DP, CSP/HSTS) | `docs/v2/build/18-LLD-security-and-compliance.md` | 03, 04 |

> If an LLD path in the table above does not exist on disk yet, that unit has not been written. **STOP** and emit the question from §3 GOLDEN RULE 4 — do not improvise the missing LLD.

**Two special documents already on disk, read before touching the FL framework (LLD 11):**

| Document | Path | What it locks |
|---|---|---|
| DeComFL correctness spec | `docs/v2/specs/2026-05-29-decomfl-correctness-design.md` | The three correctness fixes: the `1/P` averaging factor, CPU-canonical RNG, serializer save/load symmetry; the determinism contract; the T1–T5 test plan that defines "done". |
| DeComFL correctness plan | `docs/v2/plans/2026-05-29-decomfl-correctness-plan.md` | The strict RED→GREEN→COMMIT TDD task sequence with the exact `pytest` commands. |

---

## 2. Document conventions (how to *parse* every doc in this set)

Every document in `docs/v2/build/` is written to the same shape. Learn it once; it never varies.

### 2.1 How interfaces are written

- An interface is given with its **complete signature** (Java method signature, TypeScript type, Python function signature, protobuf message, or SQL column list). YOU implement the *body*; you NEVER change the *signature*.
- Example shape (from `12-LLD-orchestration-substrate.md §5`): a Java `FlServerLauncher` interface with `launch(FlRunSpec)`, `stop(executorRef)`, `describe(executorRef)`. The LLD gives the exact parameter types and return types. You write the method bodies for each of the three implementations; you do not add, rename, or retype a parameter.
- Data shapes (REST DTOs, STOMP payloads) are given as exact JSON in `04-API-CONTRACTS.md` with per-field types and constraints (e.g. `"username": "string, 3..50 chars, required"`). Implement both the producer and the consumer from that one field table. The frontend re-validates the same shape with Zod at the wire boundary.
- IDs have fixed types and you must not guess them: `users.id` is a `Long`/BIGINT serialized as a JSON number; `organizations.id`, `projects.id`, `fl_runs.id`, `datasets.id` and artifact ids are UUID strings (lowercase 8-4-4-4-12). This is stated in `04-API-CONTRACTS.md §1`.
- "Tricky" logic (the reconciler loop, lease acquisition, quota admission, round deadline/quorum, DeComFL `1/P` math, CPU-canonical RNG, parameter chunking, the cookie-JWT auth flow, the per-run token HMAC) is given as **real code or precise pseudocode**. Copy its structure; do not paraphrase it into something different.

### 2.2 How task checklists are written

- Each LLD ends with a **"Build task checklist for the local model"** section: an *ordered, dependency-first* numbered list. Example: `12-LLD-orchestration-substrate.md §13`.
- Each item is roughly one file or one feature and carries an explicit **"Done when:"** clause naming the exact test(s) that must pass. Example (verbatim from `12 §13` item 5): *"`LeaseManager`. Implement `acquire/renew/release/...` with the §6.3 SQL (`FOR UPDATE SKIP LOCKED`). **Done when:** `LeaseManager_acquire_*` and `LeaseManager_renew_failsAfterLeaseStolen` pass."*
- **Do not start a task until every predecessor's done-condition holds.** The order encodes the dependency graph; jumping ahead breaks compilation or tests.

### 2.3 How done-conditions and the conformance checklist are written

- Per-task **"Done when:"** = the local acceptance gate for that one task (a named test passes, a command exits 0).
- Each LLD also has a final **"Conformance checklist"** (e.g. `12 §14`): invariants that must ALL hold before the *unit* is done (e.g. "JVM holds no persistent in-heap run map", "no `ProcessBuilder` outside `orchestration.launcher`", "tested against real Postgres via Testcontainers, not H2"). These are unit-level, not task-level.
- A "**Verify-in-isolation checklist**" (e.g. `12 §11`, `18 §11`) tells you how to prove the unit works on its own before integration.

### 2.4 How TDD cycles are written

- Where the doc says TDD, every cycle is **RED → GREEN → COMMIT** (see the DeComFL plan `docs/v2/plans/2026-05-29-decomfl-correctness-plan.md §Methodology`):
  1. **RED** — write the full failing test first; run the exact command given; confirm the exact failure message stated in the doc.
  2. **GREEN** — make the smallest real code change; rerun; confirm pass.
  3. **COMMIT** — run the exact `git add` + `git commit` given.
- The doc gives you the *expected* failure before the fix (e.g. `KeyError: 'parameters'`, `AttributeError: 'dict' object has no attribute 'append'`). If your RED step produces a *different* failure than the doc predicts, STOP (§3 GOLDEN RULE 4) — your environment or understanding diverges from the spec.

### 2.5 How reasoning and citations are written

- Reasoning is **inline**: most contract choices state *why this and not the alternative*, tied to an audit finding id (`A1-F6`) or a `file:line`. You do not need to act on the reasoning; it is there so you understand intent and do not "improve" a deliberate choice.
- Any claim about EXISTING v1 code cites `path:line`. Any external/market claim cites a source URL. If you write new docs or comments, follow the same rule; never fabricate a version, API, or number.

### 2.6 The abbreviation rule (applies to anything YOU write)

The first time any abbreviation/acronym appears in a document you author (commit body, code comment, handoff summary), write it in full followed by the short form in parentheses, e.g. "Federated Learning (FL)". After first use the short form is fine. The build docs already follow this; match it.

---

## 3. GOLDEN RULES (non-negotiable — violating any one is a hard failure)

These override everything else. Re-read them at the start of every task.

1. **Never invent a technology that is not in `02-TECH-STACK.md`.** No new library, framework, database, build tool, or service. If the stack pins Spring Boot 3.5.x, Java 21, Gradle, custom Python FL (NO Flower / `flwr`), React 19 + Vite 6 + TanStack Query, Tauri v2, Postgres, S3/MinIO, MLflow, buf — you use exactly those and only those. No `flwr`, no Maven, no Next.js, no Bearer tokens, no Bazel. When in doubt, the technology is forbidden.

2. **Never change an interface defined in `04-API-CONTRACTS.md` (or a signature defined in an LLD).** Endpoint paths, HTTP methods, DTO field names/types, gRPC messages in package `fedlearn.v2`, STOMP topic names, the error envelope, the per-run token format — all frozen. Implement the body, conform to the contract. If a contract seems wrong or impossible, STOP and ask (rule 4); do not unilaterally "fix" it.

3. **If a detail is missing, STOP and emit a precise question — never guess.** You cannot make architecture decisions. The instant you need a fact the docs do not give (a type, a path, a version, an env var name, a branch of logic, a missing LLD), halt the task and output a question in this exact form:

   ```
   BLOCKED: <one-line summary>
   Task: <task number / name from the LLD checklist>
   Doc + section consulted: <e.g. 12-LLD-orchestration-substrate.md §6.3>
   Missing fact: <the single specific unknown>
   Why I cannot proceed: <what I would otherwise have to guess>
   Smallest decision needed from a human: <a yes/no or pick-one, not an open question>
   ```
   Do NOT continue past a BLOCKED with an assumption. A wrong guess silently corrupts downstream tasks (this is exactly the class of bug the audit caught in v1 — see the `1/P` factor and CPU/CUDA RNG divergence, `docs/audit/2026-05-29/README.md` R2/R3).

4. **Always write the test FIRST where the LLD says so.** When a task is marked TDD or its done-condition names a test, the test is written and run (RED) *before* the implementation. The test is the acceptance contract. Never write the implementation first and back-fill a test that conveniently passes.

5. **Commit after each task-checklist item — one item, one commit.** As soon as a task's "Done when:" holds (its named test passes), commit before starting the next item. Keep commits small and reversible. Commit message rules:
   - Conventional-style subject (`feat(orchestration): add LeaseManager acquire/renew with FOR UPDATE SKIP LOCKED`).
   - Body may reference the task number and the audit finding it closes.
   - **NO AI attribution anywhere.** Never mention any AI assistant or model by name. No AI co-author trailers. No "Generated with ..." footer. Authorship is human-only. This is repo policy.
   - Branch first if you are on `main` or `main-clean`; never commit directly to the default branch. Push only when asked.

6. **Cite, do not fabricate.** Any claim about existing code → `path:line`. Any external/version claim → verify it; if you cannot, flag the uncertainty explicitly ("UNVERIFIED:"). Never fabricate a method signature, a library capability, or a version number. (Critical for PyTorch RNG behavior, gRPC, libtorch — areas the audit flagged as silently device-dependent.)

7. **Preserve the load-bearing invariants** the docs call out, even when not your direct task: no `flwr` dependency; cookie-only HttpOnly JWT (no `Authorization: Bearer` in the frontend); Flyway owns the schema (JPA validate-only; write a new `V{n}__*.sql`, never `ddl-auto=update`); the `test` profile keeps Flyway disabled; the dual gRPC stub model (training stub + parallel heartbeat stub); CPU-canonical RNG everywhere for DeComFL; parameter chunking for models >300MB.

---

## 4. Handoff protocol (what to produce at each milestone checkpoint)

A **milestone checkpoint** is reached at: (a) the end of an LLD's full build-task checklist (the unit is done), OR (b) a point the LLD explicitly marks as a checkpoint, OR (c) when a human asks for a review. At every checkpoint, the human will run an external reviewer to audit your work. **Your job at the checkpoint is to assemble the package the reviewer needs — you do not run the review.** (Do not write any AI attribution into the package; the review workflow is a process fact, not document content.)

Produce these three artifacts, in this order, as your checkpoint output (text in your final message; do not create extra `.md` report files unless an LLD told you to):

1. **Summary** — short and factual:
   - Which LLD + which task numbers this checkpoint covers.
   - For each task: its "Done when:" condition and that it holds.
   - Any BLOCKED questions you hit and how they were resolved (or that they are still open).
   - Any invariant from §3 rule 7 you touched and how you preserved it.
   - Explicit "UNVERIFIED:" flags for anything you could not confirm.

2. **The diff** — the exact change set for review:
   - `git --no-pager diff <base>..<head>` (or `git --no-pager show` per commit). State the base commit hash and the branch.
   - List the files changed and the one-line responsibility of each (mirror the LLD's "File structure" table).

3. **The test output** — proof, not assertion:
   - The exact command(s) you ran and their full output, e.g. `./gradlew test --tests "...FlRunServiceTest"`, `pytest tests/test_decomfl_strategy.py -v`, `npm run test`, `npm run lint`, `./gradlew flywayValidate`.
   - Show the named done-condition tests PASSING. Show the RED→GREEN transition for TDD tasks (the failure before, the pass after).
   - If any test is skipped or xfail, say which and why.

**Checkpoint output template (copy this):**

```
CHECKPOINT: <LLD number> tasks <range>
Branch: <name>   Base: <hash>   Head: <hash>

SUMMARY
- Tasks covered: ...
- Done-conditions met: ...
- Open BLOCKED questions: ...
- Invariants touched: ...
- UNVERIFIED flags: ...

DIFF
<git diff output / file list with responsibilities>

TESTS
$ <exact command>
<full output, showing the named done-condition tests pass>
```

**Do not** mark a checkpoint complete if any done-condition test is red, any BLOCKED question is open, or any conformance-checklist item for the unit fails. A checkpoint is a stopping point for human review, so leave it clean and verifiable.

---

## 5. How to self-check that a task is done (run this every time)

Before you claim ANY task done, walk this checklist top to bottom. If any answer is "no", the task is NOT done.

| # | Self-check | How to confirm |
|---|---|---|
| 1 | Did I read the LLD task item and its "Done when:" clause exactly? | Quote the "Done when:" clause back to yourself. |
| 2 | If TDD: did I write the test FIRST and watch it fail (RED) with the failure the doc predicted? | The RED output matches the doc; if it differs → BLOCKED (§3 rule 4). |
| 3 | Is the named done-condition test now PASSING (GREEN)? | Run the exact command; exit 0; the specific test name shows pass. |
| 4 | Did I implement only the body, changing NO contract/signature from `04` or the LLD? | Diff touches bodies/new files only; no endpoint, DTO field, gRPC message, or interface signature changed. |
| 5 | Did I use ONLY technologies in `02-TECH-STACK.md`? | No new import/dep introduced; check `build.gradle`/`package.json`/`pyproject.toml`/`Cargo.toml`. |
| 6 | Did I preserve every relevant §3 rule 7 invariant? | No `flwr`; no Bearer token; new schema via `V{n}__*.sql`; CPU-canonical RNG; dual heartbeat stub intact. |
| 7 | Did the broader unit checks still pass (no regression)? | Run the unit's full suite, not just the one test: `./gradlew test` / `pytest` / `npm run test`. |
| 8 | Are static gates green where the LLD requires them? | Lint/format/type as applicable: `npm run lint`, `./gradlew check`/Spotless/Checkstyle/ArchUnit, `ruff`+`mypy`, `buf breaking`. |
| 9 | No AI attribution anywhere in the change (code, comments, commit)? | Grep the diff and commit message; zero references to any AI assistant. |
| 10 | Did I commit this single item with a clean message on a non-default branch? | `git log -1` shows the commit; branch is not `main`/`main-clean`. |

If all ten are "yes", the task is done. Commit (if not already), update your task tracker, then proceed to the next checklist item.

---

## 6. Quick failure-mode reference (what to do when something is off)

| Situation | Correct response |
|---|---|
| A fact the docs do not give (type, path, version, env var, logic branch). | STOP. Emit the BLOCKED form (§3 rule 4). Do not guess. |
| The contract in `04` seems wrong or impossible. | STOP. Emit BLOCKED. Do not "fix" the contract yourself. |
| A RED test fails differently than the doc predicted. | STOP. Emit BLOCKED — your environment or understanding diverges from the spec. |
| You are tempted to add a library/tool not in `02`. | Forbidden. Find the in-stack way, or BLOCK if there is none. |
| The LLD for the unit you were told to build does not exist on disk. | STOP. Emit BLOCKED — do not improvise the LLD. |
| A done-condition test passes but a broader unit test now fails. | The task is NOT done. Fix the regression before committing. |
| You finished a checklist item. | Commit immediately (one item, one commit), then continue. |
| You reached the end of the checklist / a marked checkpoint. | Produce the §4 handoff package and stop for human review. |
| You are unsure whether you may proceed. | Default to STOP-and-ask. A precise question is always cheaper than a wrong guess. |

---

## 7. The single sentence to remember

**Implement bodies behind frozen contracts using only the locked stack; write the test first where told; commit per task; and the moment a fact is missing, STOP and ask instead of guessing.**
