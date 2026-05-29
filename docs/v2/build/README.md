# FedLearn Platform v2 (Version 2) — Build Documentation Index

**Date:** 2026-05-29 · **Status:** build-authoritative.
**Start here:** [`00-OVERVIEW.md`](00-OVERVIEW.md) — the single entry point (vision, architecture, document map, the master glossary, the reading order).

> **Acronyms used in this index** (full form on first use, per house style): v2 (Version 2), FL (Federated Learning), DeComFL (Dimension-Free Communication Federated Learning), HLD (High-Level Design), LLD (Low-Level Design), API (Application Programming Interface), CI (Continuous Integration), TDD (Test-Driven Development), DP (Differential Privacy), JWT (JSON (JavaScript Object Notation) Web Token), gRPC (Google Remote Procedure Call), STOMP (Simple Text Oriented Messaging Protocol).

This set was written for a mid-sized local Language Model to BUILD the FedLearn v2 platform — every architecture choice is pre-decided; you implement bodies behind frozen contracts. **Read [`91-LOCAL-MODEL-GUIDE.md`](91-LOCAL-MODEL-GUIDE.md) first, then [`00-OVERVIEW.md`](00-OVERVIEW.md), then the documents below in the order the overview gives.**

## The documents

| Doc | Title | One-line summary |
|---|---|---|
| [00](00-OVERVIEW.md) | **Master Overview** | **Read 2nd.** The vision, the one-paragraph target architecture, the full document map, the master glossary of every acronym, and the reading/build order. |
| [01](01-ARCHITECTURE-HLD.md) | Architecture HLD | The system shape: the five-unit map, the three end-to-end data-flows, the deployment topology, the eight architecture decisions with reasoning. Defines *what* and *why*, not signatures. |
| [02](02-TECH-STACK.md) | Tech Stack | The LOCKED, pinned technology list with exact versions (§24 pin table) and the eleven hard invariants (§25). The only technologies you may use. |
| [03](03-DATA-MODEL.md) | Data Model | Every table, column, type, and Flyway migration: the V1–V5 identity baseline + the new V6/V7/V8 (dataset registry, `fl_runs` lease, determinism manifests). Flyway owns the schema; JPA validate-only. |
| [04](04-API-CONTRACTS.md) | API Contracts | Every wire contract: REST endpoints, the `fedlearn.v2` gRPC `.proto`, STOMP topics, the error envelope, the per-run scoped result token, the W3C `traceparent` propagation. Frozen — never change. |
| [12](12-LLD-orchestration-substrate.md) | LLD — Orchestration Substrate | The `FlServerLauncher` (Kubernetes Jobs / ECS RunTask / dev LocalProcess), the durable lease, the reconciler loop, per-org quotas, the round deadline + minimum-quorum, run-token-authenticated callbacks. |
| [13](13-LLD-frontend-dashboard.md) | LLD — Frontend Dashboard | The React 19 + Vite 6 SPA: TanStack Query server-state, Zod wire-boundary validation, the V5 role types (fixing the dead admin UI), one shared STOMP connection, the recharts communication-cost panel, Vitest+Playwright+MSW tests. |
| [18](18-LLD-security-and-compliance.md) | LLD — Security & Compliance | Cross-cutting: the three-layer role enum, org-scoped multi-tenant authorization, cookie-only HttpOnly JWT, the per-run scoped token, gRPC mutual-TLS, DP + a robust-mean guard, the SOC-2/HIPAA controls. |
| [90](90-BUILD-SEQUENCE.md) | Build Sequence | The conductor's score: the milestone-ordered build plan (M0 monorepo/CI → M13 production deploy), the dependency graph, per-milestone done-conditions, the human-review gate after every milestone. |
| [91](91-LOCAL-MODEL-GUIDE.md) | Local-Model Usage Guide | **Read 1st.** Your operating manual: the reading order, the seven GOLDEN RULES, the TDD cycle (RED→GREEN→COMMIT), the checkpoint/handoff protocol, the ten-point self-check. |

## Companion documents (read before the FL framework, milestone M3)

| Doc | Path | Summary |
|---|---|---|
| DeComFL correctness **spec** | [`../specs/2026-05-29-decomfl-correctness-design.md`](../specs/2026-05-29-decomfl-correctness-design.md) | The design source of truth: the three bugs (the `1/P` factor, CPU-canonical RNG, serializer symmetry), the determinism contract, the T1–T5 test plan that defines "done". |
| DeComFL correctness **plan** | [`../plans/2026-05-29-decomfl-correctness-plan.md`](../plans/2026-05-29-decomfl-correctness-plan.md) | The strict RED→GREEN→COMMIT TDD task sequence with the exact `pytest` commands and expected failures. |

## Numbering note

Three LLDs are authored on disk: **12** (orchestration substrate), **13** (frontend dashboard), **18** (security & compliance). LLDs for the control plane, FL framework, desktop, mobile, observability, and data/artifact stores are **referenced but not yet authored** — where a milestone needs one, [`90-BUILD-SEQUENCE.md`](90-BUILD-SEQUENCE.md) points at the authoritative existing contract (the proto in `04 §10`, the schema in `03`, the security slices in `18`, the DeComFL spec/plan) and flags the dependency. If a milestone needs an LLD that is not on disk, STOP and ask (`91 §3` GOLDEN RULE 3) — do not improvise it.

**Reference:** the reference architecture, the per-unit salvage/refactor/rebuild/kill decision table, and the risk register (R1–R17) live in [`../../audit/2026-05-29/README.md`](../../audit/2026-05-29/README.md).
