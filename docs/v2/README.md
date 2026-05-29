# FedLearn Platform — v2 (Version 2) Documentation Set

**Date:** 2026-05-29
**Branch:** `main-clean`
**Topic:** DeComFL (Decomposed Federated Learning) correctness work
**Queue origin:** First P0 item of the v2 audit's next-brainstorm queue ([`docs/audit/2026-05-29/README.md`](../audit/2026-05-29/README.md) §5)

> **Acronyms used in this index** (full form on first use, per house style): FL (Federated Learning), DeComFL (Decomposed Federated Learning), ZO (Zeroth-Order) optimization, FedAvg (Federated Averaging), LLM (Large Language Model), RNG (Random Number Generator), TDD (Test-Driven Development), ADR (Architecture Decision Record), API (Application Programming Interface), AWS (Amazon Web Services), gRPC (Google Remote Procedure Call), RIT (Rochester Institute of Technology).

---

## Overview

DeComFL is the platform's only paper-backed differentiator: ZO optimization whose per-round communication is independent of model dimension, giving roughly 1,000,000x less bandwidth than FedAvg for LLMs (RIT / Prof. Haibo Yang; see [`docs/wikis/framework/06_decomfl.md`](../wikis/framework/06_decomfl.md)). The v2 audit found it **broken on its live path in three independent ways**: the server update drops the `1/P` averaging factor (the global model steps P-times too far, 10x at the `P=10` default — [`decomfl_strategy.py:197,200`](../../framework/src/fedlearn/server/decomfl_strategy.py)); perturbation generation is device-dependent, so a seeded `torch.randn` reconstructs a different `z` across CPU/CUDA/MPS and silently corrupts aggregation on any GPU server or mixed-device fleet ([`decomfl_strategy.py:210-219`](../../framework/src/fedlearn/server/decomfl_strategy.py), [`zeroth_order.py:45-48`](../../framework/src/fedlearn/estimators/zeroth_order.py)); and the chunked serializer saves a bare state-dict but loads a wrapped one, raising `KeyError` on every chunked/LLM upload ([`serializer.py:97` vs `:155`](../../framework/src/fedlearn/communication/serializer.py)). This document set fixes all three — plus the two blocking test/RNG-hygiene defects and the two opted-in correctness cleanups — **test-first**: the five tests (T1–T5) are written before the code and become the acceptance contract. The work is correctness-first and scoped tightly; the larger v2 rebuild items remain in the audit queue.

---

## The documents

| Document | What it is | Summary |
|---|---|---|
| [Spec](specs/2026-05-29-decomfl-correctness-design.md) | Design — source of truth | The technical source of truth. Names the three bugs with file:line evidence, the blocking fixes (B-1 stale test, B-2 process-global RNG mutation) and cleanups (C-1 O(KP·N)→O(KP) loop hoist, C-2 bounded history), the locked RNG decision (CPU-canonical `torch` RNG + frozen golden vectors), the determinism contract, and the T1–T5 test plan that defines "done". Everything else builds on this. |
| [Implementation plan](plans/2026-05-29-decomfl-correctness-plan.md) | Phased TDD plan | The step-by-step build. Strict RED→GREEN→COMMIT TDD cycles, one fix per cycle, with exact `pytest` commands run from `framework/` and the precise expected failure before each fix. Written for an agentic worker; tracks progress with checkboxes. |
| [Critical decisions](decisions/2026-05-29-decomfl-correctness-decisions.md) | ADR record | The "why", ranked by stakes. A set of ADRs covering the load-bearing calls — chiefly CPU-canonical RNG (Approach A) over a hand-rolled counter-based PRNG, the language-neutral golden-vector contract that gates a later C++ mobile port, and the scope boundary. Each records the problem, alternatives weighed, the winner, and the cost accepted. |
| [Run-cost analysis](cost/2026-05-29-run-cost-analysis.md) | Cloud cost, plain-language | What it costs per month to run the platform on AWS, written for a non-finance reader. Itemizes each always-on piece and shows why the DeComFL fix is a **cost** story, not only a correctness story: clients are user-owned hardware, so the cloud's data-plane charge is bandwidth (egress) — exactly the line item DeComFL collapses. |
| [Beginner explainer](explainers/2026-05-29-decomfl-explained-for-beginners.md) | Plain-language intro | DeComFL for a first-year computer-science student with no FL background. Uses a cooking-school analogy to explain federated learning and the ZO trick, then walks through the three real bugs and how the tests prove the fixes worked. Includes a full glossary. |

---

## Where to start, by reader

| Reader | Start here | Then |
|---|---|---|
| **Brand-new student** (no FL background) | [Beginner explainer](explainers/2026-05-29-decomfl-explained-for-beginners.md) | Skim the [spec](specs/2026-05-29-decomfl-correctness-design.md) once the vocabulary clicks. |
| **Engineer implementing the fix** | [Spec](specs/2026-05-29-decomfl-correctness-design.md) (the contract) | [Implementation plan](plans/2026-05-29-decomfl-correctness-plan.md) and work the TDD cycles top to bottom. |
| **Manager / founder** | [Critical decisions](decisions/2026-05-29-decomfl-correctness-decisions.md) (what was chosen and why) | [Run-cost analysis](cost/2026-05-29-run-cost-analysis.md) for the money and the bandwidth wedge. |

---

## Context

This is the **first** item from the v2 audit's prioritized next-brainstorm queue — the P0 "DeComFL correctness trifecta" in [`docs/audit/2026-05-29/README.md`](../audit/2026-05-29/README.md) §5. Each queue item becomes its own `brainstorming → writing-plans → implementation` cycle; this set is that cycle for the trifecta. Items deferred to their own queue entries (C++ mobile port, mono-vs-poly repo decision, checkpoint/resume, differential-privacy/robust aggregation, the full per-run determinism manifest) are listed under the spec's "Out of scope" section.
