# SOC 2 + HIPAA Controls Checklist (Compliance Evidence Index)

> **Owning document:** `../18-LLD-security-and-compliance.md` (sections 11.x and 11.y). This file is the expanded, standalone version of those tables. **Audience:** a mid-sized local language model (~30 billion parameters) building FedLearn v2, plus the humans who will later run a SOC 2 / HIPAA audit. Every control below states exactly what to build and what evidence proves it.

> **Acronyms used in this document** (full form on first use, per the doc-set house style): SOC 2 (System and Organization Controls 2), TSC (Trust Services Criteria), CC (Common Criteria — the SOC 2 control families CC1–CC9), HIPAA (Health Insurance Portability and Accountability Act), PHI (Protected Health Information), ePHI (electronic PHI), BAA (Business Associate Agreement), GDPR (General Data Protection Regulation), FedRAMP (Federal Risk and Authorization Management Program), FL (Federated Learning), DeComFL (the platform's zeroth-order FL strategy), FedAvg (Federated Averaging), DP (Differential Privacy), DP-SGD (Differentially-Private Stochastic Gradient Descent), DLG (Deep Leakage from Gradients), TLS (Transport Layer Security), mTLS (mutual TLS), HSTS (HTTP Strict Transport Security — HTTP is HyperText Transfer Protocol), CSP (Content-Security-Policy), JWT (JSON Web Token — JSON is JavaScript Object Notation), MFA (Multi-Factor Authentication), RLS (Row-Level Security), RBAC (Role-Based Access Control), KMS (Key Management Service), CMK (Customer-Managed Key), SSM (AWS Systems Manager), RDS (Relational Database Service), S3 (Simple Storage Service), EC2 (Elastic Compute Cloud), ECS (Elastic Container Service), EKS (Elastic Kubernetes Service), AWS (Amazon Web Services), VPC (Virtual Private Cloud), CI (Continuous Integration), CD (Continuous Delivery), SBOM (Software Bill of Materials), PR (Pull Request), IR (Incident Response), RPO (Recovery Point Objective), RTO (Recovery Time Objective), MDC (Mapped Diagnostic Context), gRPC (Google Remote Procedure Call), WS (WebSocket), STOMP (Simple Text Oriented Messaging Protocol), CN (Common Name of an X.509 certificate), ε (epsilon — the privacy-loss parameter of differential privacy).

---

## 1. Purpose and how to use this checklist

This is the **single compliance evidence index** for FedLearn v2. It answers three questions per control: (a) **what** the control requires, (b) **where in this codebase** it is implemented (so the local model knows what to build and the auditor knows where to look), and (c) **what artifact proves** the control operated.

**Reasoning — why this doc exists.** SOC 2 Type 2 is table-stakes for any business-to-business deal and is also the cheapest path to roughly 70% of HIPAA's technical safeguards, so the two are pursued together (`../../../audit/2026-05-29/B4-security-compliance.md:120`). The healthcare/pneumonia demo makes HIPAA the de-facto floor the moment a United-States clinical partner touches the system (`B4:15`, risk **R11** in `../../../audit/2026-05-29/README.md`). A control without a written, locatable evidence trail does not exist for an auditor — hence this index.

**Status legend** (matches the doc-set verdict vocabulary):

| Status | Meaning |
|---|---|
| **rebuild** | Does not exist in v1; the local model must build it new. |
| **refactor** | A correct foundation exists; extend or harden it. |
| **salvage** | Exists and is correct; keep and document. |
| **process** | Not code — an organizational policy/procedure the company stands up in a compliance-automation platform (for example Vanta, Drata, or Secureframe). Listed so coverage is complete; the local model does **not** implement these. |

**Evidence-artifact** column = the concrete thing an auditor is handed (a test name, a config file, a screenshot of a setting, a runbook, a log query). For code controls, the **Done-condition** is the automated check that proves it.

**Scope locked for the first audit window:** SOC 2 Type 2 across the **Security** (mandatory) and **Confidentiality** Trust Services Criteria; HIPAA-readiness architecture in parallel; GDPR as a design constraint; FedRAMP explicitly deferred (`B4:115-120, 138`). Availability and Privacy TSC are added later as customers demand.

---

## 2. Compliance program summary (the posture)

| Program | Decision | Cost / timeline (cited) | Trigger |
|---|---|---|---|
| **SOC 2 Type 2** (Security + Confidentiality) | **Do now — baseline** | ~$20,000–$40,000 first year, 3–6 months with a compliance-automation platform | First enterprise business-to-business deal |
| **HIPAA-readiness** (architecture + BAA chain) | **Design for now; certify when a covered entity signs** | Incremental on top of SOC 2; the AWS BAA is free; controls overlap ~70% with SOC 2 | Any United-States clinical partner / the pneumonia demo going real |
| **GDPR** | **Design constraint** | Design-time only | Any European Union data subject |
| **FedRAMP** | **Deferred** | $500,000–$2,000,000, 12–24 months | A real federal contract |

**Reasoning.** SOC 2 first because it pays for itself on a single ~$100,000 enterprise deal and builds most of HIPAA's technical safeguards anyway; FedRAMP is a different universe of cost and time and is gated behind an actual federal customer — pursuing it speculatively would consume a seed-stage runway (`B4:120, 155`). The newer **FedRAMP 20x** track (pilot authorizations from April 2026) is worth tracking but does not change the defer decision.

---

## 3. SOC 2 Type 2 — controls by Trust Services Criteria family

The audit attests that controls **operated over a 3–12 month window** (`B4:138`), so every code control below must be **continuously enforced** (a one-time fix is not enough — the CI gate and the tests are what make it continuous).

### 3.1 CC1–CC5, CC9 — governance & risk (mostly process)

These are organizational controls the company stands up in the compliance-automation platform; the local model does not implement them, but they are listed so the index is complete.

| Control | Requirement | Status | Evidence artifact |
|---|---|---|---|
| CC1.1–CC1.5 | Control environment: org chart, board oversight, code of conduct, background checks, defined security roles | process | HR records, org chart, signed policies in the automation platform |
| CC2.1–CC2.3 | Communication of security objectives internally and to customers | process | Published security page, internal security policy, customer-facing terms |
| CC3.1–CC3.4 | Risk assessment program (annual + on change), fraud risk | process | Risk register, annual risk-assessment report |
| CC4.1–CC4.2 | Monitoring of controls (internal control evaluations) | process + tech | Quarterly access reviews; the audit-log review query (CC7 below) |
| CC5.1–CC5.3 | Control activities mapped to risks | process | Control matrix (this document is part of it) |
| CC9.1–CC9.2 | Risk mitigation incl. **vendor management and BAAs** | process | Signed AWS BAA, sub-processor list, vendor reviews |

### 3.2 CC6 — Logical and physical access (the core technical family)

| ID | Control | Implementation in this codebase | Status | Evidence artifact / Done-condition |
|---|---|---|---|---|
| CC6.1 | **Multi-tenant logical isolation** — a user reaches only their organization's data | `TenantPredicate` + `requireOrgScope(project)` funnels every project-scoped read/write; `org_id` filter added to discovery (`../18-LLD-security-and-compliance.md` §6.1; `B4:79`) | **rebuild** | Test `TenantIsolationIntegrationTest.cross_org_read_is_403` (a cross-org request returns HTTP 403). Optional defense-in-depth: PostgreSQL native RLS policy (`18-LLD §6.1` note). |
| CC6.1 | **Authentication** — cookie-only HttpOnly JWT, no Bearer token, no JavaScript-readable token | `SecurityFilterChain` + the cookie contract (`18-LLD §5, §6.5`); `cookie.secure=true` outside dev | **salvage/refactor** | Test asserting the `jwtToken` cookie is `HttpOnly`+`Secure`+`SameSite=Lax` and no token appears in any response body. |
| CC6.2 | **Unique user identification + provisioning** | `users.id` unique; role enum (platform/org/project) replacing the dead `ADMIN` string (`18-LLD §6.2`; `A1-F1`) | **refactor** | `AdminControllerIntegrationTest.platform_admin_reaches_admin_routes` passes seeding `PLATFORM_ADMIN`; the old test seeding the literal `"ADMIN"` is deleted. |
| CC6.2 | **Session revocation** on password/role change or termination | `tokenVersion` integer compared on each request (`18-LLD §5.2`; `A1-F7`, `B4:98`) | **rebuild** | Test: changing a user's `tokenVersion` invalidates an outstanding token on the next request. |
| CC6.3 | **Authorization / least privilege** — role enum gates admin and mutating actions | `@EnableMethodSecurity` + `@PreAuthorize("hasRole('PLATFORM_ADMIN')")` (never the legacy `'ADMIN'`) (`18-LLD §11, §13 task 7`) | **refactor** | Every `/api/admin/**` route returns 403 to a non-admin and 200 to a `PLATFORM_ADMIN` in the integration tests. |
| CC6.6 | **Encryption in transit** on every boundary | gRPC TLS+mTLS default-secure (`18-LLD §6.3`); HSTS header + `cookie.secure=true` on the web side (`18-LLD §6.5`); WebSocket over TLS (`B4:130`) | **rebuild/refactor** | Plaintext gRPC channel is refused in `ec2demo`/`production`; an HSTS header is present on every HTTPS response. |
| CC6.6 | **FL transport peer authentication** — only enrolled machines speak the protocol | mTLS identity bound to certificate CN + a backend-issued enrollment token (`18-LLD §6.3`; `B4:33-35, 59`) | **rebuild** | A client presenting no/invalid client certificate or an unrecognized enrollment token is rejected at channel open (anti-Sybil test). |
| CC6.7 | **Encryption at rest** | RDS + S3/MinIO encryption with KMS customer-managed keys; secrets in AWS Secrets Manager / SSM Parameter Store with KMS (`18-LLD §11; B4:128-129`) | **rebuild** (infrastructure) | Terraform/infra config shows RDS `storage_encrypted=true` with a CMK; S3 bucket default encryption with the CMK; no plaintext secret in any committed file (gitleaks clean). |
| CC6.7 | **Secret management + rotation** | Secrets Manager rotation runbook; per-run and enrollment tokens are short-lived; move the bootstrap-admin password off WARN logging to a `0600` sidecar (`18-LLD §11; B4:86, 88`) | **rebuild** | A documented rotation procedure exists; no credential is ever written to the log sink (a log-scan test). |
| CC6.x | **MFA on the platform** for platform-admin and org-admin | Enforced at the identity layer (`18-LLD §11.y; B4:132`) | **rebuild/process** | MFA-required setting screenshot + a policy that admin accounts cannot disable it. |

### 3.3 CC7 — System operations (monitoring, audit, incident response)

| ID | Control | Implementation in this codebase | Status | Evidence artifact / Done-condition |
|---|---|---|---|---|
| CC7.1 | **Audit logging coverage** — every mutating and authorization-relevant action is recorded | `@Auditable` aspect (proceed-then-write, so caller rollback rolls back the audit row) applied to project create/delete, run start/stop, membership grants, role changes, access-request decisions (`18-LLD §11; B4:105`, `03-DATA-MODEL.md:438`) | **refactor** (schema exists, coverage is the gap) | A test asserts an `audit_events` row is written for each annotated action; coverage list reviewed against the mutating-endpoint inventory. |
| CC7.1 | **Append-only audit log with retention** | `audit_events` append-only, JSONB metadata, monthly partitioning + ≥6-year retention (HIPAA requirement, reused here) (`B4:131; 03-DATA-MODEL.md:438`) | **refactor** | Retention/partitioning migration present; no `UPDATE`/`DELETE` grant on `audit_events` for the app role. |
| CC7.2 | **Alerting on security events** — authentication failures, authorization denials | Wire an authentication-failure handler to Micrometer metrics; alert in the observability stack (`18-LLD §11.y`; `16-LLD-observability.md`) | **refactor** | A burst of auth failures raises a metric an alert rule fires on (test the metric increments). |
| CC7.3–CC7.5 | **Incident response** — detect, respond, recover, communicate | IR runbook + the on-call/alert path; ties to the observability traces (W3C traceparent across services) | process + tech | An IR runbook document; a tabletop-exercise record. |

### 3.4 CC8 — Change management

| ID | Control | Implementation in this codebase | Status | Evidence artifact / Done-condition |
|---|---|---|---|---|
| CC8.1 | **Pull-Request-time CI gate** — no change merges without passing checks | `ci.yml` (Gradle test, pytest, Vitest, lint, buf breaking-change gate) + branch protection with required checks (`18-LLD §11.y`; `02-TECH-STACK.md` monorepo/CI; `../../../audit/2026-05-29/README.md` Theme 4) | **rebuild** | Branch protection requires the CI checks; a deliberately failing PR cannot merge. |
| CC8.1 | **Vulnerability scanning + SBOM on every build** | `security.yml`: gitleaks, pip-audit, Gradle dependency-check, `npm audit`, a CycloneDX SBOM (`18-LLD §11.y; 02-TECH-STACK.md:624`) | **rebuild** | The scans run on every PR; an SBOM artifact is produced per build. |
| CC8.1 | **Code review on every change** | Branch protection requires ≥1 approving review | process + config | Branch-protection setting screenshot. |

### 3.5 Confidentiality (C1)

| ID | Control | Implementation in this codebase | Status | Evidence artifact / Done-condition |
|---|---|---|---|---|
| C1.1 | **No cross-tenant data leakage** | Org-scoped discovery (`findDiscoverable` filters `org_id`) + WebSocket/STOMP topic authorization so a user cannot subscribe to another org's `/topic/logs/{projectId}` or `/topic/results/{projectId}` (`18-LLD §6.1, §6.x; B4:79`) | **rebuild** | Test: subscribing to another org's topic is rejected; discovery never returns another org's metadata. |
| C1.2 | **Confidential data identified, encrypted, and disposed** | Data classification (the FL data plane treats raw data as client-owned and never stored); encryption at rest (CC6.7); artifact lifecycle in the artifact store (`17-LLD-data-and-artifacts.md`) | rebuild/process | Data-classification policy; the FL-premise guarantee (section 4.4 below). |

---

## 4. HIPAA Security Rule — safeguards mapping

The proposed 2025/2026 HIPAA Security Rule update would make encryption at rest, encryption in transit, MFA, network segmentation, and **annual penetration testing** mandatory with no exceptions — **design to the stricter bar now** (`B4:129`).

### 4.1 Administrative safeguards (§164.308)

| Control | v2 answer | Status | Evidence |
|---|---|---|---|
| Security management / risk analysis (§164.308(a)(1)) | Annual risk analysis + this control matrix | process | Risk-analysis report |
| Assigned security responsibility (§164.308(a)(2)) | A named security owner | process | Org chart / appointment record |
| Workforce security + access management (§164.308(a)(3)-(4)) | Role enum + org-scoped authorization + quarterly access reviews (`18-LLD §6.1-6.2`) | refactor + process | Access-review records |
| Security awareness training (§164.308(a)(5)) | Annual workforce training | process | Training completion records |
| Security incident procedures (§164.308(a)(6)) | The IR runbook (CC7.3) | process + tech | IR runbook |
| Contingency plan (§164.308(a)(7)) | Backup + restore for RDS and the artifact store; defined RPO/RTO; the orchestration reconciler resumes runs from checkpoints (`12-LLD-orchestration-substrate.md`, `17-LLD-data-and-artifacts.md`) | rebuild + process | Restore-test record; documented RPO/RTO |
| **Business Associate Agreements (§164.308(b))** | BAA chain: the startup ↔ AWS (free), and each covered-entity client ↔ the startup (`B4:133`) | process | Executed AWS BAA; customer BAA template |

### 4.2 Physical safeguards (§164.310)

| Control | v2 answer | Status | Evidence |
|---|---|---|---|
| Facility access, workstation & device/media controls | Inherited from AWS under the shared-responsibility model (covered by the AWS BAA); the platform runs no owned data-center hardware | process | AWS BAA + AWS compliance attestations |

### 4.3 Technical safeguards (§164.312)

| Control | v2 answer | Status | Evidence |
|---|---|---|---|
| Access control + unique user id (§164.312(a)) | Role enum + org-scoped authorization + automatic session expiry via short JWT lifetime + `tokenVersion` revocation (`18-LLD §5.2, §6`) | refactor/rebuild | The CC6 tests above |
| Audit controls (§164.312(b)) | `audit_events` append-only, ≥6-year retention, monthly partitioning, `@Auditable` coverage (`B4:131; 03-DATA-MODEL.md:438`) | refactor | The CC7.1 tests above |
| Integrity (§164.312(c)) | Content-addressed model artifacts (sha256 keys) so any tampering changes the hash (`17-LLD-data-and-artifacts.md`); append-only audit log | rebuild | An artifact whose bytes change fails its sha256 check |
| Person/entity authentication (§164.312(d)) | Cookie JWT for humans; mTLS cert + enrollment token for FL machines (`18-LLD §5, §6.3`) | rebuild | The CC6.6 anti-Sybil test |
| Transmission security (§164.312(e)) | gRPC mTLS + HSTS + `cookie.secure` (encryption in transit) (`18-LLD §6.3, §6.5`) | rebuild/refactor | Plaintext channel refused outside dev |

### 4.4 HIPAA-eligible services, data residency, and the FL advantage

- **HIPAA-eligible services only + AWS BAA:** confine ePHI to EC2/ECS/EKS, RDS, S3, KMS, CloudWatch — verify eligibility per Region and execute the AWS BAA **before** any real PHI (`B4:128, 133`).
- **Encryption everywhere:** KMS customer-managed keys for at-rest; mTLS/TLS for in-transit (sections 4.3, CC6.6/6.7).
- **Data residency:** pin PHI-bearing RDS, S3, and MinIO resources to in-scope United-States Regions (`B4:134`).
- **Annual penetration test + MFA:** design to the proposed-rule bar now (`B4:129`).
- **The FL premise as a HIPAA advantage — with a caveat.** Raw PHI never leaves the client; if the platform only ever sees gradient **scalars**, it may never store ePHI at all. This is the single most defensible privacy claim the platform has (`B4:124`, DeComFL scalar-only DLG-resistance, `18-LLD §10`). **But it must be architecturally guaranteed, not assumed:** no patient image path may appear in a log line, an error trace, an artifact name, or a metric label. **Done-condition:** a log/trace scrubbing test asserts no client-supplied path or filename is emitted; the guarantee is also contractually asserted in the customer BAA.

---

## 5. GDPR design constraints (not a certification — a design obligation)

| Constraint | v2 answer | Status |
|---|---|---|
| Right to erasure / Article 17 in FL | The model-weight unlearning problem is **unsolved** in federated learning and must be **disclosed contractually** (the platform cannot perfectly "forget" a client's contribution from already-aggregated weights without retraining) (`B4:15`; FL-unlearning reference below) | process + disclosure |
| Data-processing transparency | Surface the differential-privacy ε per project in run telemetry so a data controller can document the privacy budget (`B4:62`; `16-LLD-observability.md`) | rebuild |
| Lawful basis / data minimization | The FL premise (raw data stays on the client) is data-minimization by construction | salvage |

---

## 6. FL data-plane privacy controls (the commercially load-bearing ones)

| Control | v2 answer | Status | Reference |
|---|---|---|---|
| DLG-resistance (scalar-only uploads) | DeComFL transmits scalars + seeds, not gradient vectors — structurally kills the Deep-Leakage-from-Gradients attack family | salvage/amplify | `18-LLD §10; B4:53` |
| Differential privacy | DP-SGD on the FedAvg path; scalar-DP noise on the DeComFL path; tunable ε per project, surfaced in telemetry | **rebuild** | `B4:62` |
| Secure aggregation | Bonawitz-style masked sum on the **FedAvg path only** (largely redundant for DeComFL's already-low-information scalars — scoped to control cost) | **rebuild**, scoped | `B4:63` |
| Robust aggregation guard | Robust-mean / clipping guard against outlier/poisoning influence; **delete the false "Byzantine-robust" README claim** (it is a liability) | **rebuild** | `B4:30, 62`; `../../../audit/2026-05-29/B4-security-compliance.md` |

**Reasoning.** A broken or absent privacy layer is not just a missing feature — the v1 README made a **false "Byzantine-robust" claim** that is a direct compliance and legal liability; deleting the claim and shipping the real, opt-in guard is mandatory before any HIPAA/GDPR assertion (`18-LLD §11; B4:30`).

---

## 7. Cross-reference map (control → where it is built)

| Area | Primary Low-Level Design / file |
|---|---|
| Authentication, authorization, tenant isolation, mTLS, DP/robust guard | `../18-LLD-security-and-compliance.md` |
| Role enum, audit-log coverage, secret handling | `../10-LLD-backend-control-plane.md` |
| `audit_events`, `fl_runs`, artifact tables, retention/partitioning | `../03-DATA-MODEL.md` |
| Encryption-in-transit on the FL channel, DeComFL scalar privacy | `../11-LLD-fl-framework.md` |
| Backup/restore, checkpoint-resume (contingency plan) | `../12-LLD-orchestration-substrate.md`, `../17-LLD-data-and-artifacts.md` |
| Content-addressed artifact integrity, ε telemetry | `../16-LLD-observability.md`, `../17-LLD-data-and-artifacts.md` |
| CI gate, vuln scans, SBOM (CC8) | `../02-TECH-STACK.md`, `../90-BUILD-SEQUENCE.md` |
| CSP/HSTS, `cookie.secure`, WS topic authorization | `../13-LLD-frontend-dashboard.md`, `../18-LLD-security-and-compliance.md` |

---

## 8. Implementation done-checklist (the code controls the local model must complete)

Tick each only when its automated check passes (these are the SOC 2/HIPAA-relevant subset of `18-LLD-security-and-compliance.md` §13):

- [ ] Role enum replaces every `hasRole('ADMIN')` with `hasRole('PLATFORM_ADMIN')`; admin integration test passes seeding `PLATFORM_ADMIN`; the `"ADMIN"`-seeding test is deleted. (CC6.2/6.3, HIPAA access control)
- [ ] `TenantPredicate` + `requireOrgScope` enforced on every project-scoped path; `cross_org_read_is_403` passes; `findDiscoverable` filters `org_id`. (CC6.1, C1.1)
- [ ] `tokenVersion` revocation: changing it invalidates outstanding tokens. (CC6.2)
- [ ] gRPC default-secure: plaintext channel refused in `ec2demo`/`production`; mTLS cert CN + enrollment token required; anti-Sybil test passes. (CC6.6, HIPAA §164.312(d)(e))
- [ ] HSTS + `cookie.secure=true` + CSP present on web responses. (CC6.6, Confidentiality)
- [ ] `@Auditable` covers all mutating/authz actions; an `audit_events` row is written per action; the table is append-only with monthly partitioning + ≥6-year retention. (CC7.1, HIPAA §164.312(b))
- [ ] Auth-failure metric increments and an alert rule fires. (CC7.2)
- [ ] `ci.yml` + branch protection: a failing PR cannot merge. (CC8.1)
- [ ] `security.yml`: gitleaks + pip-audit + Gradle dependency-check + `npm audit` + CycloneDX SBOM run per PR. (CC8.1)
- [ ] DP layer (DP-SGD on FedAvg, scalar-DP on DeComFL) with tunable ε surfaced in telemetry; robust-mean/clipping guard present; the false "Byzantine-robust" README claim deleted. (Section 6)
- [ ] Encryption at rest: RDS + S3/MinIO with KMS CMK; no plaintext secret committed; bootstrap-admin password never logged. (CC6.7, HIPAA at-rest)
- [ ] Log/trace scrubbing test: no client-supplied patient path/filename is ever emitted to logs, traces, metric labels, or artifact names. (HIPAA FL-premise guarantee)

**Infrastructure / process controls** (not code — for the human, tracked in the compliance-automation platform): execute the AWS BAA; configure KMS CMK + encryption + United-States Region pinning; enable MFA for admins; stand up the SOC 2 program (Vanta/Drata/Secureframe); write the IR runbook, contingency/restore test, access-review cadence, vendor/BAA register; schedule the annual penetration test.

---

## 9. Sources

- SOC 2 for startups (scope, cost, timeline) — https://www.workstreet.com/blog/soc-2-for-startups
- SOC 2 Type 2 requirements / process / cost — https://sprinto.com/blog/soc-2-type-2/
- SOC 2 Type 2 cost / return on investment — https://www.skedda.com/insights/soc-2-type-2
- AWS HIPAA-eligible services & BAA — https://aws.amazon.com/compliance/hipaa-compliance/
- AWS BAA covered services / steps — https://www.accountablehq.com/post/how-to-get-a-baa-with-aws-steps-requirements-and-covered-hipaa-services
- HIPAA on AWS best practices (proposed 2025/2026 rule: encryption, MFA, penetration testing mandatory) — https://www.exabeam.com/explainers/hipaa-compliance/hipaa-on-aws-requirements-and-best-practices/
- FedRAMP cost / defer rationale — https://www.vanta.com/collection/fedramp/fedramp-cost
- Zeroth-order vertical FL scalar-only / scalar-DP — https://arxiv.org/html/2502.20565
- Byzantine-robust aggregation limits — https://www.usenix.org/system/files/sec20summer_fang_prepub.pdf
- Federated unlearning / GDPR Article 17 — https://arxiv.org/pdf/2411.17126
