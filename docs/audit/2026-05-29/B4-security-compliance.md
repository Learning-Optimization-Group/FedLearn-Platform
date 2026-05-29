# B4 — Security, Threat Model & Compliance

**Audit date:** 2026-05-29
**Scope:** FL-specific threat model; platform security (multi-tenant isolation, secrets, cookie-auth contract); compliance floor for a production-grade FL startup.
**Builds on:** `docs/audit/2026-05-27/01-backend.md` (C2, C5, C6, C7, H1, H2, H3), `03-framework.md` (C2, C3, C4, H5), and `README.md` Themes 1 & 5. This report **extends** those findings into a threat-model frame and a compliance posture; it does not re-litigate the role-name break (C1) already covered.

---

## Executive summary

The platform's single biggest **latent security asset** is the DeComFL data plane: zeroth-order training uploads only **gradient scalars + integer seeds** (`fedlearn.proto:139-146,172-177`), never gradient vectors or model tensors. This structurally eliminates the entire deep-leakage-from-gradients (DLG) attack family that dominates the FL security literature — confirmed by the ZOO-VFL result that scalar-only zeroth-order channels "prevent the reverse-sum and backdoor attacks totally" because intermediate gradients are never transmitted. **This is a defensible privacy wedge and should be marketed as one.** It does **not**, however, make the system private: there is **zero differential privacy** anywhere in `framework/` (one grep hit, a comment), no Byzantine-robust aggregation, no client identity binding on gRPC, and the FedAvg path still ships full tensors and inherits the full DLG surface.

On the platform side, the cookie-auth contract is sound in shape but has three concrete holes: (1) **`cookie.secure=false` over plain HTTP in `ec2demo`** (`application-ec2demo.properties:16`) puts the session JWT on the wire in cleartext; (2) **multi-tenant isolation is project-scoped, not org-scoped** — `AuthorizationService` (entire file) never checks `org_id`, and `getDiscoverProjects` (`ProjectService.java:410-422`) leaks PUBLIC project metadata across every tenant; (3) **JWT validation lacks issuer/audience/jti** (carried from 2026-05-27 H2), so tokens are non-revocable and cross-environment-replayable.

On compliance: the pneumonia/chest-X-ray healthcare demo (`docs/guides/pneumonia_demo_plan.md`, referenced in CLAUDE.md) makes **HIPAA the de-facto floor** the moment a US clinical partner touches the system — even though FL's premise is that raw PHI never leaves the client. The recommended posture is **SOC 2 Type 2 (Security + Confidentiality) as the table-stakes baseline ($20-40k, 3-6 mo), HIPAA-readiness architecture from day one (AWS BAA, KMS-CMK encryption, audit logging — most of which SOC 2 work already builds), GDPR as a design constraint (the model-weight right-to-erasure problem is unsolved in FL and must be disclosed contractually), and FedRAMP explicitly deferred** ($800k-$2M, 12-24 mo) until a federal contract justifies it.

**Verdict headline:** the DeComFL privacy property is **salvage-and-amplify**; the gRPC trust model and multi-tenant authz are **rebuild**; secrets handling and the cookie contract are **refactor**.

---

## Part 1 — FL-specific threat model

### 1.1 Threat catalogue mapped to this platform

| Threat | What it is | Exposure on **this** platform | Evidence |
|---|---|---|---|
| **Gradient leakage / DLG reconstruction** | Reconstruct a client's training samples from shared gradients | **DeComFL path: structurally near-eliminated** (scalars+seeds only). **FedAvg path: fully exposed** (ships full tensors via `SubmitModelUpdate` / streaming). | `fedlearn.proto:172-177` (scalars), `serializer.py` full-tensor path; FedAvg `strategy.py` |
| **Membership inference** | Decide whether a specific record was in a client's training set | Present on **both** paths. Scalar uploads still correlate with loss landscape; no DP noise to blunt it. | No DP in `framework/` (grep: 0 hits); ZOO-VFL notes scalar channels still need DP for inference defense |
| **Property / attribute inference** | Infer a population property of a client's private data | Present; unmitigated. Aggregation has no noise floor. | same as above |
| **Model poisoning** | Malicious client sends crafted updates to degrade/backdoor the global model | **Unmitigated.** No Byzantine-robust aggregation. FedAvg sample-weights are only clamped (`MAX_SAMPLES`), not validated. | `strategy.py:81` `MAX_SAMPLES=100_000` (clamp, not defense — 2026-05-27 H2); no Krum/median/trimmed-mean |
| **Data poisoning** | Client trains on poisoned local data (label-flip, backdoor trigger) | **Unmitigated and undetectable** — server never sees data; no anomaly scoring on updates. | no per-client update scoring anywhere |
| **Byzantine clients** | Arbitrary/colluding malicious clients (crash, send NaN/Inf, collude) | **Unmitigated.** `coordinator.py:69` WARNs on "suspicious payload" but still aggregates min-clients average. | `coordinator.py:55-69` |
| **Sybil attack** | One adversary registers as many fake clients to dominate aggregation | **Wide open.** `RegisterClient` takes a self-asserted `client_id` string; no auth, no per-client cert, no rate limit. | `fedlearn.proto:41` `RegisterClientRequest`; gRPC has no client auth by default |
| **Free-rider** | Client takes the global model, contributes nothing real, still "participates" | Undetectable — no contribution measurement, no `num_examples` validation beyond the clamp. | `coordinator.py:55` |
| **gRPC plaintext over WAN** | On-path attacker reads/forges all FL traffic | **Default-on insecurity.** TLS exists but is opt-in via env var; default is `add_insecure_port`. | `server.py:126`, `grpc_client.py:55`; audit item #37; 2026-05-27 framework C4 |

### 1.2 Does DeComFL's scalar-only upload reduce the leakage surface? — **Yes, materially, but it is not privacy.**

This is the most important nuance in the report and the one most likely to be over- or under-claimed.

**What the design actually transmits (verified):**
- Uplink: `SubmitGradientScalarsRequest { client_id, trained_on_round, GradientScalars gradients, num_examples }` where `GradientScalars` is `[local_step][perturbation] -> double` (`fedlearn.proto:139-177`). For K local steps and P perturbations, that is **K×P doubles per client per round** — independent of model dimension P_model. This is the paper's O(K·P) communication claim, and it is real in the proto.
- Downlink: `PerturbationSeeds` (integers) + `RebuildHistory` (`fedlearn.proto:129-167`). Seeds are public RNG inputs, not secrets.

**Why this shrinks the attack surface (cite):** The classic DLG/iDLG/GradInversion family reconstructs inputs by matching a *dummy gradient* to the *transmitted gradient vector*. DeComFL never transmits a gradient vector — only its **scalar projection onto a random direction** g·u for a known seed-derived u. The zeroth-order-VFL literature is explicit that "only a scalar value, rather than a full gradient vector is transmitted … thus can prevent the reverse-sum and backdoor attacks totally" ([ZOO-VFL / DPZV, arXiv:2502.20565](https://arxiv.org/html/2502.20565)). So the **highest-severity, best-published FL attack class is structurally out of reach on the DeComFL path.**

**Why it is still not "private" (the part to NOT over-claim):**
1. **Membership/property inference survives.** The same paper injects *calibrated scalar DP noise* precisely because scalar-only channels still leak enough to support targeted inference attacks. DeComFL-as-implemented injects **no noise** — `decomfl_strategy.py` has no DP, no clipping. The scalar g·u over many rounds/perturbations is an informative observation of the client's loss landscape.
2. **The seeds are known.** Because u is reconstructible from the public seed, a curious server with even a few scalars can run a low-dimensional inference attack along known directions — strictly easier than a black-box attack.
3. **Only the DeComFL strategy benefits.** The platform also ships **FedAvg**, which transmits full tensors and is fully DLG-exposed. If a customer runs FedAvg "because it converges faster," the privacy property evaporates silently. There is no guardrail forcing the privacy-preserving strategy.
4. **Compression ≠ privacy.** The prior audit's instinct that compressed gradients leak less is **refuted by current literature** — HCGLA and DLCG reconstruct recognizable images from *highly compressed* gradients ([ScienceDirect S0925231224011202](https://www.sciencedirect.com/science/article/abs/pii/S0925231224011202); [IEEE 10003066](https://ieeexplore.ieee.org/document/10003066/)). The DeComFL property comes from being **scalar-projected**, not from being **small** — do not conflate the two in marketing.

**Net:** treat DeComFL scalar-upload as a **strong communication-channel privacy reduction** (kills the DLG family), and layer **client-side DP-SGD or scalar-DP noise + secure aggregation** on top to address the residual membership/inference surface for a HIPAA/GDPR posture. This is a salvage-and-amplify story, and it is the single most commercially defensible security claim the platform has.

### 1.3 Mitigation roadmap (FL data plane)

| Mitigation | Why / fit to this platform | Verdict on current state |
|---|---|---|
| **Mutual-TLS on gRPC, default-on in deployed profiles** | TLS+`require_client_auth` already implemented (`server.py:107-122`, `grpc_client.py:52-72`) — it is simply **off by default**. Flip the default: refuse `add_insecure_port` unless `dev`. This also gives **per-client cert identity**, killing the Sybil/spoofed-`client_id` problem in one move. | **rebuild trust model** (code exists, posture is wrong) |
| **Client identity binding** | Replace self-asserted `client_id` (`fedlearn.proto:41`) with the mTLS cert CN, and bind it to a backend-issued enrollment token (the platform already mints `X-Internal-Key`; mint per-client tokens the same way). | **rebuild** |
| **Byzantine-robust aggregation** | Add coordinate-wise median / trimmed-mean / Krum as a selectable aggregator. Trimmed-mean is the cheapest fit for DeComFL scalars (per-(k,p) coordinate). Note the literature caveat: these need an assumed Byzantine fraction and fail under adaptive/majority attacks ([USENIX Fang sec20](https://www.usenix.org/system/files/sec20summer_fang_prepub.pdf)). | **rebuild** (absent today) |
| **Differential privacy** | Client-side DP-SGD (FedAvg path) and scalar-DP noise (DeComFL path, per DPZV). Tunable ε per project; surface ε in the run telemetry (ties to B3). Required for any defensible HIPAA/GDPR claim. | **rebuild** (absent) |
| **Secure aggregation (Bonawitz-style)** | Server sees only the **sum** of masked updates, not individuals. High value for FedAvg; **largely redundant for DeComFL** (scalars already low-information) — so scope it to the FedAvg path only to control cost. | **rebuild** (absent), scoped |
| **Update size + content bounds** | Already flagged: bound streaming upload size (2026-05-27 framework H5, `grpc_servicer.py:177-194`) and reject NaN/Inf scalars before aggregation. Cheap anti-DoS + anti-Byzantine floor. | **refactor** |
| **Bake `compressed` into proto, fail-closed** | 2026-05-27 framework C3 — env-var-inferred compression is a correctness *and* a security foot-gun (mismatched peers feed attacker-influenced bytes to decompress). `weights_only=True` blocks RCE today but is a load-bearing invariant that must be tested. | **refactor** |

---

## Part 2 — Platform security

### 2.1 Multi-tenant isolation — **the structural gap is enforcement, not schema**

V5 added `organizations`, `organization_memberships`, and `projects.org_id NOT NULL` (`V5__identity_foundations.sql:7,17`; CLAUDE.md "Identity layers"). The **schema models tenancy correctly**, but the **authorization layer does not enforce it**:

- `AuthorizationService` (entire file, 85 lines) checks **only** project ownership and **project-level** membership (`isOwner`, `hasMembership`, `requireParticipant`). It **never references `org_id`** — grep for org-scoping in services returned nothing relevant. A user in Org A who is added as a CLIENT to a project owned by Org B gets full participant access regardless of org boundary. The org layer is decorative in the hot path.
- `getDiscoverProjects()` (`ProjectService.java:410-422`) returns **every PUBLIC project across all orgs** — name, owner username, model type, description — with **no `org_id` filter**. This is a cross-tenant metadata leak: Org A enumerates Org B's project names and owners. For a B2B FL product where the *project topic itself* can be sensitive ("Mercy Hospital pneumonia model"), this is a confidentiality finding, not a cosmetic one.
- `isAdmin()` (`AuthorizationService.java:42`) checks `ROLE_ADMIN`, which the 2026-05-27 backend C1 proved is **never emitted** (only `ROLE_PLATFORM_ADMIN` exists). So every admin bypass in this class is **dead** — which today *fails closed* (admins are denied), but the moment C1 is "fixed" by renaming, a PLATFORM_ADMIN gets unconditional cross-org access with **no org-membership check** — exactly the bypass CLAUDE.md says PLATFORM_ADMIN is *supposed* to have, but with no audit annotation on most of those paths (2026-05-27 H3).

**Recommendation:** introduce a single `requireOrgScope(project)` (or a JPA `@Filter`/row-level predicate) that every project-scoped read/write funnels through, asserting the caller shares the project's `org_id` (or is PLATFORM_ADMIN, audited). Add `org_id` filtering to `findDiscoverable`. This is the **most important platform-security change** in v2: tenancy isolation is a SOC 2 *and* HIPAA control, and it is currently unenforced. **Verdict: rebuild.**

### 2.2 Secrets — **shape is right, three concrete leaks**

What's good (cite): base profile refuses to boot without `APP_JWT_SECRET` / `APP_INTERNAL_API_KEY` / `CORS_ALLOWED_ORIGINS` (CLAUDE.md; confirmed by `application-ec2demo.properties:9-10` and `application-production.properties:61,72` using `${VAR}` with no fallback). Dev secrets are clearly fenced and labeled PUBLIC (`application-dev.properties:26-29`). The internal API key uses constant-time compare (`InternalApiKeyFilter.java:69-72`) — correct.

What's wrong:
1. **Bootstrap admin password logged at WARN** under loose profile check (2026-05-27 backend C6, `BootstrapRunner.java:139-141`). In a deployed env with centralized logging this spills a credential to the log sink — a HIPAA audit finding. **Move to a 0600 sidecar.**
2. **Secret reaches the spawned Python FL server via process environment** (`FlowerServerManager.java:188-189,386` puts `FEDLEARN_INTERNAL_API_KEY` into `pb.environment()`). Env-passed secrets are readable via `/proc/<pid>/environ` by any process under the same UID and frequently leak into crash dumps and `ps -E`. For v2's ECS/Fargate path use a secrets store (AWS Secrets Manager / SSM Parameter Store with KMS) and inject at task definition, not via `ProcessBuilder` env from a parent JVM that also holds it in memory.
3. **No secret rotation story.** A single static `APP_INTERNAL_API_KEY` and a single HMAC `APP_JWT_SECRET` with no `kid`/rotation. JWTs are non-revocable (see 2.3). For SOC 2 you need a documented rotation procedure; architecturally, move to short-lived asymmetric JWT signing (rotatable JWKS) so secret rotation doesn't invalidate the verifier.

**Verdict: refactor** (good bones, fix the three leaks; the env-passing one becomes rebuild on the Fargate path).

### 2.3 Cookie-auth contract — **sound design, two transport/lifecycle holes**

The contract (CLAUDE.md): HttpOnly + SameSite + Secure `jwtToken` cookie, `withCredentials`, no Bearer, no `localStorage`, same cookie on the STOMP handshake via `JwtHandshakeInterceptor`. This is the **right** design and resists the token-theft-via-XSS class better than localStorage-Bearer. Keep it. But:

1. **`ec2demo` runs `cookie.secure=false` over plain HTTP** (`application-ec2demo.properties:15-16`). The session JWT travels in cleartext on every request — a passive WAN observer captures it and replays it (no jti, no rotation). The comment says "EC2 demo runs plain HTTP." For any demo touching real or realistic PHI this is disqualifying. **Put TLS in front (the nginx `:443` the flow already references) and flip `secure=true`.** This is the platform-layer twin of the gRPC-plaintext finding.
2. **CSRF disabled + `SameSite=Lax`** (`SecurityConfig.java:130`; 2026-05-27 backend H1). `Lax` permits top-level navigation POSTs; with CSRF off, a cross-site form-POST to a state-changing endpoint succeeds with the cookie attached. Mitigation: `SameSite=Strict` for the session cookie in deployed profiles + an `Origin`/`Referer` allowlist check on mutating endpoints (the CORS allowlist already exists — reuse it server-side).
3. **JWT has no issuer/audience/jti/clock-skew** (2026-05-27 backend H2, `JwtTokenProvider.java:54-57`). Consequences for compliance: (a) **no revocation** — a stolen or post-termination token is valid until expiry; HIPAA/SOC 2 expect session revocation on access-change; (b) a `dev` token (public secret) is structurally indistinguishable from a prod token of the same shape — add `requireIssuer`/`requireAudience` so environments can't cross-replay. Add a `jti` + a small deny-list (or short expiry + refresh) to get revocation.
4. **Frontend has no CSP** (2026-05-27 README Theme 5). With `SameSite=Lax` cookies, one injected vendor XSS performs credentialed POSTs. Add a strict `Content-Security-Policy` (backend response header is preferable to a `<meta>` so it covers all responses).

**Verdict: refactor.**

### 2.4 Audit logging — **good foundation, coverage gap**

`audit_events` exists with `org_id`/`actor_user_id`/`action`/`occurred_at` and three composite indexes including `(org_id, occurred_at)` (`V5__identity_foundations.sql:43-57`) — exactly the shape you want for tenant-scoped, time-bounded audit retrieval (a HIPAA §164.312(b) and SOC 2 CC7 control). The `@Auditable` aspect proceeds-then-writes so caller rollback rolls back the audit row (CLAUDE.md) — correct transactional semantics. **The gap is coverage:** essentially zero annotations outside login/logout/register (2026-05-27 backend H3) — no audit on project create/delete, start/stop FL server, membership grants, role changes, or access-request decisions. For any compliance posture, **mutating + authorization-relevant actions must be audited**, and the audit log must be **append-only with a retention policy** (the prior audit notes the table is unbounded; add monthly partitioning + retention — 2026-05-27 README Phase 3). **Verdict: refactor (extend coverage, add retention + tamper-evidence).**

---

## Part 3 — Compliance floor (recommendation)

### 3.1 The recommendation, up front

| Posture | Recommendation | Trigger | Indicative cost / time |
|---|---|---|---|
| **SOC 2 Type 2** (Security + Confidentiality TSC) | **DO — baseline, start now** | First enterprise B2B deal | ~$20-40k first year, **3-6 mo** with automation (Vanta/Drata/Secureframe) ([Workstreet](https://www.workstreet.com/blog/soc-2-for-startups), [Sprinto](https://sprinto.com/blog/soc-2-type-2/)) |
| **HIPAA-readiness** (architecture + BAA chain) | **DESIGN FOR NOW, certify when a covered entity signs** | Any US clinical partner / the pneumonia demo going real | Incremental on top of SOC 2; AWS BAA is free, the controls overlap ~70% with SOC 2 |
| **GDPR** | **DESIGN CONSTRAINT now, DPA + disclosures before any EU data** | First EU client/data subject | Process + contracts; the model-weight erasure problem must be disclosed (see 3.4) |
| **FedRAMP** | **DEFER explicitly** | A federal contract that requires it | $800k-$2M, **12-24 mo** ([Vanta](https://www.vanta.com/collection/fedramp/fedramp-cost), [Secureframe](https://secureframe.com/hub/fedramp/costs)) — do not pursue speculatively |

**Rationale:** SOC 2 Type 2 is table stakes for B2B procurement and pays for itself on a single ~$100k enterprise deal ([Skedda](https://www.skedda.com/insights/soc-2-type-2)). It is also the **cheapest path to ~70% of HIPAA's technical safeguards** (encryption, access control, audit logging, change management), so doing SOC 2 first and HIPAA-readiness alongside is strictly more efficient than the reverse. FedRAMP is a different universe of cost/time and should be gated behind an actual federal opportunity — pursuing it speculatively would consume a seed-stage runway.

### 3.2 HIPAA — what it requires architecturally (and why FL helps but does not exempt you)

The pneumonia/chest-X-ray demo (CLAUDE.md cross-network demo plan) means **PHI is realistically in scope**. FL's "data never leaves the client" premise is a genuine HIPAA advantage — the **platform itself may never store ePHI** if it only ever sees gradient scalars — but this must be **architecturally guaranteed and contractually asserted**, not assumed. The moment a client uploads a model artifact, a log line, or an error trace that contains a patient image path, ePHI lands on the platform.

If the platform (or any artifact it stores) touches ePHI, HIPAA's Security Rule requires:

- **AWS BAA executed** and PHI confined to **HIPAA-eligible services only** (AWS lists 160+ eligible services, updated 2026-02-10; eligibility varies by Region) ([AWS HIPAA](https://aws.amazon.com/compliance/hipaa-compliance/), [Accountable](https://www.accountablehq.com/post/how-to-get-a-baa-with-aws-steps-requirements-and-covered-hipaa-services)). Practically: EC2/ECS, RDS/Aurora, S3, KMS, CloudWatch are eligible; verify each before placing PHI.
- **Encryption at rest and in transit** via **KMS, customer-managed keys (CMK)** ([AWS HIPAA](https://aws.amazon.com/compliance/hipaa-compliance/)). The proposed 2025/2026 HIPAA Security Rule update would make encryption at rest, encryption in transit, **MFA**, network segmentation, and **annual penetration testing** mandatory with no exceptions ([Exabeam](https://www.exabeam.com/explainers/hipaa-compliance/hipaa-on-aws-requirements-and-best-practices/)) — design to the stricter bar now.
  - **Gap today:** gRPC plaintext (§1, audit #37) and `ec2demo` cookie `secure=false` (§2.3) both **violate encryption-in-transit**. The TLS/mTLS machinery already exists in the framework — flip it on.
- **Audit controls** (§164.312(b)) — `audit_events` is the right primitive; extend coverage + retention (§2.4). Retain ≥6 years.
- **Access controls + unique user identification** — the multi-tenant enforcement gap (§2.1) and dead admin role (C1) are direct HIPAA access-control findings.
- **BAA chain to clients/clinics** — each covered-entity client needs a BAA with the startup; the startup needs one with AWS. The FL clients themselves (hospital edge nodes) are the covered entity's responsibility, but the enrollment/identity binding (§1.3) is what lets you scope that boundary.
- **Data residency** — pin PHI-bearing resources to in-scope US Regions.

### 3.3 SOC 2 Type 2 — concrete control mapping to this codebase

Start with **Security (mandatory) + Confidentiality** TSC; add Availability/Privacy later as customers demand ([Workstreet](https://www.workstreet.com/blog/soc-2-for-startups)). The audit attests controls operated over a 3-12 month window ([Sprinto](https://sprinto.com/blog/soc-2-type-2/)). Direct mappings to findings above:

- **CC6 (logical access):** multi-tenant org enforcement (§2.1), JWT revocation/issuer (§2.3), MFA on the platform, secret rotation (§2.2). All currently gapped.
- **CC7 (monitoring):** `@Auditable` coverage + retention (§2.4); ties to B3 observability (alerting on auth failures — `AuditingAuthenticationFailureHandler` exists, wire it to metrics).
- **CC8 (change management):** the **load-bearing CI gap** from 2026-05-27 README Theme 4 — no PR-time gate means no enforced change-control evidence. SOC 2 auditors ask for this directly.
- **Confidentiality:** encryption in transit (gRPC TLS, cookie `secure`), the cross-org metadata leak (§2.1).

### 3.4 GDPR — the FL-specific landmine: right-to-erasure on model weights

GDPR Art. 17 (right to be forgotten) is **not satisfied by deleting a client's local data** — once a client's update has been aggregated over rounds, "its effects have already gradually permeated through a large number of clients" ([iQua FedUnlearning](https://iqua.ece.toronto.edu/research/fedunlearning/), [arXiv:2411.17126](https://arxiv.org/pdf/2411.17126)). The aggregated **model weights may themselves be personal data**, and machine unlearning in FL is an open research problem. Practical posture:

- **DeComFL helps here too**: because the server stores **seed history + scalar gradient history** (`decomfl_strategy.py:66-67`, `RebuildHistory` in proto), a model is *reconstructible from its history*. That same history is what makes **certified/replay-based unlearning** tractable — you can recompute a model excluding a client's scalar contributions for given rounds. This is a genuine, paper-aligned differentiator worth building (ties to C3 reproducibility).
- **Contractually**, disclose that full erasure from a deployed aggregated model requires retraining/unlearning and may not be instantaneous — and implement the DeComFL-history-replay unlearning path as the technical answer.
- **DPA + data-residency** (EU Region pinning) before any EU data subject; minimize what the platform stores about clients (the self-asserted `client_id` and any IP in logs are personal data).

### 3.5 FedRAMP — defer, with rationale

FedRAMP Moderate is $500k-$2M and 12-24 months ([Vanta](https://www.vanta.com/collection/fedramp/fedramp-cost), [Knox](https://knoxsystems.com/resources/fedramp-authorization-timeline)). The new **FedRAMP 20x** track (Phase 3 from April 2026) promises pilot authorizations in months rather than years and changes startup entry dynamics ([Convox](https://www.convox.com/blog/fedramp-authorization-2026-guide-saas-companies)) — worth tracking, but still **gated behind an actual federal customer**. Building SOC 2 + HIPAA first creates ~60-70% of the control evidence FedRAMP reuses, so deferring loses little. **Do not pursue speculatively at seed stage.**

---

## Decision table (verdicts)

| Module / subsystem | Verdict | One-line rationale |
|---|---|---|
| **DeComFL scalar-upload privacy property** | **salvage** (amplify) | Structurally kills the DLG attack family; the platform's best privacy wedge — keep, market, and layer DP on top. |
| **gRPC trust model (plaintext default, self-asserted client_id)** | **rebuild** | TLS/mTLS code exists but is off-by-default; client identity is unauthenticated — flip default-secure + bind identity to cert. |
| **Multi-tenant authorization (org isolation)** | **rebuild** | `AuthorizationService` enforces project scope only, never `org_id`; `getDiscoverProjects` leaks cross-tenant metadata. |
| **FL aggregation robustness (Byzantine/poisoning/DP)** | **rebuild** | No robust aggregation, no DP — `MAX_SAMPLES` clamp is not a defense; required for any HIPAA/GDPR claim. |
| **Cookie-auth contract** | **refactor** | Right design; fix `secure=false`-over-HTTP, add JWT issuer/aud/jti + revocation, add CSP + Origin check. |
| **Secrets handling** | **refactor** (rebuild on Fargate) | Boot-fails-without-secrets is correct; fix password logging, env-passed internal key, and missing rotation. |
| **Audit logging (`audit_events` + `@Auditable`)** | **refactor** | Correct schema + transactional semantics; extend coverage to all mutating/authz actions + retention + tamper-evidence. |
| **Compliance program (none today)** | **rebuild** (greenfield) | Stand up SOC 2 Type 2 + HIPAA-readiness architecture; defer FedRAMP. |

---

## Prioritized recommendations

**P0 — transport & tenancy (security-critical, mostly config flips):**
1. Default-on gRPC mTLS in all deployed profiles; refuse `add_insecure_port` outside `dev` (`server.py:126`, `grpc_client.py:55`). Bind `client_id` to the mTLS cert CN.
2. Front `ec2demo` with TLS and set `cookie.secure=true` (`application-ec2demo.properties:16`).
3. Enforce org isolation in a single `requireOrgScope` chokepoint; add `org_id` filter to `getDiscoverProjects` (`ProjectService.java:410-422`).
4. Move bootstrap password off the WARN log to a 0600 sidecar (`BootstrapRunner.java:139-141`).

**P1 — auth lifecycle & compliance scaffolding:**
5. Add JWT `iss`/`aud`/`jti` + revocation (deny-list or short-lived + refresh) (`JwtTokenProvider.java:54-57`); rotate to asymmetric JWKS signing.
6. Add strict CSP header + `Origin`/`Referer` check on mutating endpoints; `SameSite=Strict` in deployed profiles.
7. Extend `@Auditable` to all mutating/authz endpoints; add `audit_events` retention/partitioning + append-only/tamper-evidence.
8. Begin SOC 2 Type 2 (Security + Confidentiality) with a compliance-automation platform; stand up PR-time CI (the CC8 change-management evidence, per 2026-05-27 Theme 4).

**P2 — FL data-plane hardening (the privacy product):**
9. Add client-side DP (DP-SGD on FedAvg; scalar-DP on DeComFL per DPZV) with per-project ε surfaced in run telemetry.
10. Add selectable Byzantine-robust aggregation (trimmed-mean first); reject NaN/Inf scalars and bound upload size (2026-05-27 framework H5).
11. Add secure aggregation scoped to the FedAvg path only (DeComFL scalars make it largely redundant).
12. Build DeComFL-history-replay unlearning as the GDPR Art. 17 technical answer (ties to C3).
13. Execute AWS BAA + KMS-CMK encryption-at-rest before any real PHI; pin PHI resources to in-scope US Regions.

---

## Sources

- AWS HIPAA Eligible Services & BAA — https://aws.amazon.com/compliance/hipaa-compliance/
- AWS BAA covered services / steps — https://www.accountablehq.com/post/how-to-get-a-baa-with-aws-steps-requirements-and-covered-hipaa-services
- HIPAA on AWS best practices (proposed 2025/26 rule: encryption, MFA, pen-test mandatory) — https://www.exabeam.com/explainers/hipaa-compliance/hipaa-on-aws-requirements-and-best-practices/
- SOC 2 for startups (scope, cost, timeline) — https://www.workstreet.com/blog/soc-2-for-startups
- SOC 2 Type 2 requirements/process/cost — https://sprinto.com/blog/soc-2-type-2/
- SOC 2 Type 2 cost/ROI — https://www.skedda.com/insights/soc-2-type-2
- FedRAMP cost — https://www.vanta.com/collection/fedramp/fedramp-cost ; https://secureframe.com/hub/fedramp/costs
- FedRAMP timeline / 20x — https://knoxsystems.com/resources/fedramp-authorization-timeline ; https://www.convox.com/blog/fedramp-authorization-2026-guide-saas-companies
- Deep Leakage from Gradients survey — https://dl.acm.org/doi/10.1007/s10462-023-10550-z
- Leakage from compressed gradients (HCGLA) — https://www.sciencedirect.com/science/article/abs/pii/S0925231224011202 ; https://ieeexplore.ieee.org/document/10003066/
- Zeroth-order VFL scalar-only channel prevents reverse-sum/backdoor; needs scalar DP — https://arxiv.org/html/2502.20565
- Byzantine-robust aggregation / Krum/trimmed-mean limits — https://www.usenix.org/system/files/sec20summer_fang_prepub.pdf
- FL unlearning / GDPR Art. 17 — https://iqua.ece.toronto.edu/research/fedunlearning/ ; https://arxiv.org/pdf/2411.17126
