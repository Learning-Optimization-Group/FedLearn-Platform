# 18 — Low-Level Design (LLD): Security & Compliance (cross-cutting)

**Document type:** Production build specification — Low-Level Design (LLD) for a **cross-cutting** unit.
**Audience:** a mid-sized local Large Language Model (LLM, ~30 billion parameters) implementing the build. Everything here is pre-decided: exact class names, method signatures, environment-variable names, profiles, and commands. The local model implements method **bodies**; this document gives the **contracts**. Where a body is genuinely tricky (the per-run token Hash-based Message Authentication Code (HMAC) verify, the Row-Level-Security (RLS) chokepoint, the mutual Transport-Layer-Security (mTLS) trust evaluation, the Differential-Privacy (DP) clip-and-noise), real code or precise pseudocode is given.
**Status:** authoritative for v2 (version 2). Conforms exactly to the four foundation documents in this directory: `01-ARCHITECTURE-HLD.md`, `02-TECH-STACK.md`, `03-DATA-MODEL.md`, `04-API-CONTRACTS.md`. Where this document and those disagree, those win — file an issue, do not improvise.
**Date authored:** 2026-05-29.
**Sourced from:** the v2 audit synthesis `/home/anurag/codebase/FedLearn-Platform/docs/audit/2026-05-29/README.md` and the depth reports `B4-security-compliance.md` and `A1-backend.md` in the same directory. Existing-code claims cite `path:line`; external/market claims cite a source Uniform Resource Locator (URL).

> **Abbreviation key (first-use full forms, repeated here for self-containment so no section depends on another):**
> LLD (Low-Level Design), LLM (Large Language Model), API (Application Programming Interface), REST (Representational State Transfer), gRPC (Google Remote Procedure Call), RPC (Remote Procedure Call), STOMP (Simple Text Oriented Messaging Protocol), WS (WebSocket), JWT (JSON Web Token), JSON (JavaScript Object Notation), HTTP (HyperText Transfer Protocol), HTTPS (HTTP Secure), TLS (Transport Layer Security), mTLS (mutual TLS), CN (Common Name of an X.509 certificate), SAN (Subject Alternative Name), CA (Certificate Authority), CSR (Certificate Signing Request), PKI (Public Key Infrastructure), HMAC (Hash-based Message Authentication Code), JWKS (JSON Web Key Set), RLS (Row-Level Security), RBAC (Role-Based Access Control), DP (Differential Privacy), DP-SGD (Differentially-Private Stochastic Gradient Descent), DLG (Deep Leakage from Gradients), ZO (Zeroth-Order optimization), FL (Federated Learning), DeComFL (Dimension-Free Communication Federated Learning — the platform's zeroth-order FL strategy; the v1 wiki "Decomposed" expansion is wrong per the paper, `04-API-CONTRACTS.md:9`), FedAvg (Federated Averaging), CSP (Content-Security-Policy), HSTS (HTTP Strict Transport Security), CSRF (Cross-Site Request Forgery), XSS (Cross-Site Scripting), PII (Personally Identifiable Information), PHI (Protected Health Information), ePHI (electronic PHI), SOC 2 (System and Organization Controls 2), TSC (Trust Services Criteria), HIPAA (Health Insurance Portability and Accountability Act), BAA (Business Associate Agreement), GDPR (General Data Protection Regulation), FedRAMP (Federal Risk and Authorization Management Program), KMS (Key Management Service), CMK (Customer-Managed Key), RDS (Relational Database Service), S3 (Simple Storage Service), MinIO (the self-hosted S3-compatible object store), ECS (Elastic Container Service), EKS (Elastic Kubernetes Service), k8s (Kubernetes), ARN (Amazon Resource Name), AWS (Amazon Web Services), SBOM (Software Bill of Materials), MFA (Multi-Factor Authentication), MDC (Mapped Diagnostic Context), DTO (Data Transfer Object), SpEL (Spring Expression Language), JPA (Jakarta Persistence API), UUID (Universally Unique Identifier), VPC (Virtual Private Cloud), W3C (World Wide Web Consortium), OTel (OpenTelemetry), CI (Continuous Integration), sha256 (Secure Hash Algorithm 256-bit), Sybil (an attack where one adversary forges many identities).

---

## 1. Purpose & single responsibility of this unit

This unit is the **cross-cutting security and compliance layer** of FedLearn v2. Its single responsibility is to make every trust decision in the platform: **who is authenticated** (cookie-only HttpOnly JWT), **what they may touch** (org-scoped, RLS-style authorization with the collapsed role enum), **which machines may speak the FL protocol** (gRPC default-secure with TLS + mTLS, identity bound to certificate CN plus a backend enrollment token), **how the FL data plane resists attack** (the DeComFL scalar-only DLG-resistance property, a Differential-Privacy layer, and a robust-mean/clipping aggregation guard), and **how all of the above produces the evidence** a SOC 2 Type 2 audit and HIPAA-readiness architecture require (encryption at rest/in transit, append-only audit logging, secrets management, data residency). It owns no business logic of its own; it is the set of filters, interceptors, query predicates, certificate material, and aggregation guards that every other unit defers to.

---

## 2. Position in the system — dependencies and interfaces

### 2.1 What this unit depends on

| Depends on | Why | Reference |
|---|---|---|
| Control plane (Spring Boot 3.5.14, Java 21) | Hosts the `SecurityFilterChain`, `@PreAuthorize` gates, STOMP interceptors, and the per-run-token verifier. | `01-ARCHITECTURE-HLD.md` unit 1; `02-TECH-STACK.md:96` |
| PostgreSQL (RDS 17.10) + Flyway | Stores `users`/`organizations`/`organization_memberships`/`projects`/`fl_runs`/`audit_events`; the RLS predicates and the `org_id NOT NULL` rule (rule R-C) run here. | `03-DATA-MODEL.md:87`; `02-TECH-STACK.md:241` |
| Secrets store (AWS Secrets Manager / SSM Parameter Store with KMS-CMK) | Source of `APP_JWT_SECRET`, `app.internal.run-token-secret`, mTLS private keys, SMTP creds. | `B4-security-compliance.md:87` |
| AWS KMS-CMK, RDS/S3 encryption | Encryption at rest for the HIPAA-readiness floor. | `B4-security-compliance.md:128-129` |
| OTel + structlog/MDC | Binds `trace_id` to every audit line and error; W3C `traceparent` carries no PII. | `04-API-CONTRACTS.md:1085` |

### 2.2 What depends on this unit

Every other unit. The control plane REST handlers cannot run without the filter chain; the orchestration substrate cannot mint a launch without an enrollment token and a run token; the FL framework cannot open a channel without the mTLS material; the frontend cannot authenticate without the cookie contract; the datastore cannot serve a row without passing the org-scope predicate.

### 2.3 Interfaces this unit CONSUMES (by exact name, from `04-API-CONTRACTS.md`)

- The **role enum** model `PlatformRole {USER, PLATFORM_ADMIN}`, `OrgRole {OWNER, ADMIN, MEMBER}`, `ProjectRole {MEMBER, CLIENT}` (`04-API-CONTRACTS.md:49-53`).
- The **auth-requirement notation** `PUBLIC | AUTH | ORG_MEMBER(p) | ORG_ADMIN(p) | PROJECT_PARTICIPANT(p) | PLATFORM_ADMIN | RUN_TOKEN(r)` (`04-API-CONTRACTS.md:56-62`) — this unit is the code that evaluates each one.
- The **standard error envelope** with stable `code` registry (`04-API-CONTRACTS.md:963-1010`) — every authn/authz denial emits one of `NOT_AUTHENTICATED(401)`, `BAD_CREDENTIALS(401)`, `RUN_TOKEN_INVALID(401)`, `ACCOUNT_NOT_VERIFIED(403)`, `FORBIDDEN(403)`, `RUN_TOKEN_MISMATCH(403)`, `RUN_TERMINAL(409)`, `RATE_LIMITED(429)`.
- The **per-run scoped token** format and validation pseudocode (`04-API-CONTRACTS.md:1014-1062`) and its env-var injection contract (`FEDLEARN_RUN_ID`, `FEDLEARN_RUN_TOKEN`, `FEDLEARN_BACKEND_URL`, `TRACEPARENT`).
- The **`fedlearn.v2` gRPC framing rules** including channel security, identity authz, codec whitelist, and the `enrollment_token` field of `RegisterClientRequest` (`04-API-CONTRACTS.md:668-685, 869-880`).
- The **W3C `traceparent` propagation contract** and its "never put PII in baggage" caveat (`04-API-CONTRACTS.md:1066-1085`).

### 2.4 Interfaces this unit EXPOSES (by exact endpoint / RPC / topic name)

- Guards every REST route in `04-API-CONTRACTS.md` §2–§9 with the auth-requirement column: e.g. `POST /api/auth/login` (`PUBLIC`, sets `Set-Cookie: jwtToken`), `GET /api/auth/me` (the silent-401 probe), `POST /api/projects` (`ORG_MEMBER`), `GET /api/admin/users` (`PLATFORM_ADMIN`).
- Guards the internal-callback routes `POST /api/internal/runs/{runId}/results | /finished | /checkpoint | /status` with `RUN_TOKEN(runId)` (`04-API-CONTRACTS.md:350-355`).
- Guards the STOMP subscriptions `/topic/logs/{projectId}`, `/topic/results/{projectId}`, `/topic/status/{projectId}`, `/topic/runs/{projectId}` with a per-`SUBSCRIBE` `PROJECT_PARTICIPANT` check (`04-API-CONTRACTS.md:884-895`).
- Guards the gRPC service `FederatedLearningService` (`04-API-CONTRACTS.md:629-647`): the mTLS handshake + `RegisterClient` enrollment-token check is the admission gate; the DP/robust guard sits at the aggregation step before `SubmitGradientScalars`/`SubmitModelUpdate` results are folded in.

---

## 3. Tech stack for this unit (pinned, from `02-TECH-STACK.md`)

| Concern | Technology + pin | One-line reasoning |
|---|---|---|
| Control plane | Spring Boot `3.5.14`, Java `21.0.7+6` (`02-TECH-STACK.md:96, 48`) | Salvaged control plane bumped off End-Of-Life 3.4.5; Spring Security carries the filter chain. |
| JWT library | `io.jsonwebtoken:jjwt-api/-impl/-jackson` `0.12.5` (`02-TECH-STACK.md:109-111`) | Carried from v1; signs/verifies the cookie JWT; v2 adds issuer/audience/jti. |
| Password hashing | Spring Security `BCryptPasswordEncoder` (in `spring-boot-starter-security`) | Existing `SecurityConfig.java:56-58`; salvaged, no change. |
| gRPC + TLS/mTLS | Spring gRPC (GA 1.x, `02-TECH-STACK.md:120`); Python `grpcio` (matched pair, `02-TECH-STACK.md:161`); C++ gRPC runtime (mobile) | Default-secure channel; mTLS gives per-client cert identity that closes the Sybil-open `client_id`. |
| Rate limiting | Bucket4j (pin `verify-before-use`) | Throttles `/api/auth/register|login|password/forgot` per `A1-F5`; emits `429 RATE_LIMITED`. |
| Differential Privacy | Opacus for DP-SGD on FedAvg (pin `verify-before-use`, matched to torch `2.12.0`); scalar-DP for DeComFL implemented in-framework against `torch`/`numpy` (`02-TECH-STACK.md:580-581`) | The privacy layer over aggregation; no extra dep for the scalar-DP path. |
| Secrets at rest | AWS Secrets Manager / SSM Parameter Store + KMS-CMK (`B4-security-compliance.md:87, 128`) | HIPAA encryption-at-rest + a documented rotation story (SOC 2 CC6). |
| Audit log | `audit_events` table (`03-DATA-MODEL.md:438-452`) + `@Auditable` aspect (proceed-then-write) | Append-only tenant-scoped trail; HIPAA §164.312(b), SOC 2 CC7. |
| Supply-chain scans | gitleaks, pip-audit, Gradle dependency-check, `npm audit`, CycloneDX SBOM (`02-TECH-STACK.md:624`) | SOC 2 CC8 change-management evidence; SBOM is a SOC 2/HIPAA prerequisite. |
| Compliance program | SOC 2 Type 2 (Security + Confidentiality TSC) + HIPAA-readiness; defer FedRAMP (`02-TECH-STACK.md:612`) | The pneumonia/healthcare demo makes HIPAA the floor, risk R11. |

Do not substitute alternatives (no Bearer-token auth, no OAuth2 redirect flow, no plaintext gRPC default, no Byzantine-robustness claim — see §12 and `02-TECH-STACK.md:756-764`).

---

## 4. Module / file structure

All paths are absolute under the repo root `/home/anurag/codebase/FedLearn-Platform/`. Java package root is `com.federated.fl_platform_api` (existing v1 package, salvaged).

```
backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/
├── config/
│   ├── SecurityConfig.java              # the SecurityFilterChain bean; CSP/HSTS headers; CSRF; CORS (§5.1)
│   ├── WebSocketSecurityConfig.java     # registers JwtChannelInterceptor for STOMP SUBSCRIBE authz (§6.4)
│   └── GrpcTlsConfig.java               # loads server keystore + trust store; configures mTLS require-client-auth (§6.3)
├── security/
│   ├── PlatformRole.java                # enum {USER, PLATFORM_ADMIN}  (kills the ADMIN/PLATFORM_ADMIN drift)
│   ├── OrgRole.java                     # enum {OWNER, ADMIN, MEMBER}
│   ├── ProjectRole.java                 # enum {MEMBER, CLIENT}
│   ├── JwtTokenProvider.java            # generate/validate cookie JWT WITH iss/aud/jti/skew/tokenVersion (§5.2)
│   ├── JwtAuthenticationFilter.java     # reads jwtToken cookie, sets SecurityContext (salvaged, hardened)
│   ├── CustomUserDetailsService.java    # emits ROLE_PLATFORM_ADMIN authority (must match @PreAuthorize)
│   ├── JwtHandshakeInterceptor.java     # WS handshake-time cookie auth (salvaged)
│   ├── JwtChannelInterceptor.java       # WS CONNECT re-validate + SUBSCRIBE topic authz (§6.4)
│   ├── RunTokenService.java             # mint + verify the per-run scoped token (HMAC-SHA256) (§6.2)
│   ├── RunTokenFilter.java              # gate /api/internal/** on RUN_TOKEN(runId) (replaces InternalApiKeyFilter)
│   ├── EnrollmentTokenService.java      # mint + verify the gRPC client enrollment token (anti-Sybil) (§6.3)
│   ├── OrgScope.java                    # request-scoped holder of the caller's org_id set + platformRole
│   ├── OrgScopeFilter.java              # populates OrgScope from the authenticated principal each request
│   └── RateLimitFilter.java            # Bucket4j throttle on the three public auth endpoints
├── authz/
│   ├── AuthorizationService.java        # requireOrgScope / requireParticipant / requireOrgAdmin / isPlatformAdmin (§6.1)
│   └── TenantPredicate.java             # builds the `org_id IN (:scope)` JPA/SQL predicate (RLS-style) (§6.1)
├── audit/
│   ├── Auditable.java                   # @Auditable(action, targetType, targetIdParam) annotation
│   ├── AuditAspect.java                 # proceeds-then-writes audit_events row (salvaged); Jackson serializer
│   └── AuditAction.java                 # enum of audited actions (RUN_START, ROLE_CHANGE, ...)
└── error/
    └── GlobalExceptionHandler.java      # maps authn/authz exceptions -> the standard error envelope (§9)

framework/src/fedlearn/security/
├── tls.py                               # build grpc.ssl_server_credentials / ssl_channel_credentials (§6.3)
├── identity.py                          # extract cert CN from peer; verify enrollment_token; bind to client_id
├── dp.py                                # DP-SGD (FedAvg) + scalar-DP (DeComFL) noise + clipping (§6.6)
└── robust.py                            # robust-mean / coordinate-wise trimmed-mean / clip guard (§6.6)

deploy/security/                         # NOT application code — PKI + secrets material, gitignored keys
├── ca/                                  # root + intermediate CA (offline root; intermediate signs leaf certs)
├── issue-server-cert.sh                 # issues the FL-server leaf cert (SAN = run service DNS)
├── issue-client-cert.sh                 # issues an FL-client leaf cert (CN = stable client identity)
└── README.md                            # rotation + revocation runbook (SOC 2 evidence)

docs/v2/build/controls/
└── soc2-hipaa-controls-checklist.md     # the controls checklist of §11 (the compliance evidence index)
```

One responsibility per file is stated in the inline comments above. The local model creates each file; §13 orders them.

---

## 5. Key interfaces & type signatures — the platform-auth half

### 5.1 `SecurityConfig` outline (the filter chain — full contract)

This is the v2 replacement for `SecurityConfig.java` (verified v1 at `backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/config/SecurityConfig.java`). The v1 chain disables CSRF (`SecurityConfig.java:114`), sets only `frameOptions.sameOrigin` (`:115`), and `permitAll`s `/api/internal/**` behind the global-key `InternalApiKeyFilter` (`:122, 127`). v2 keeps the shape and changes exactly the lines the audit flagged.

```java
@Configuration
@EnableWebSecurity
@EnableMethodSecurity                       // keeps @PreAuthorize on controllers/services
public class SecurityConfig {

  // beans carried from v1 unchanged: passwordEncoder() -> BCryptPasswordEncoder,
  // authenticationProvider(), authenticationManager(), corsConfigurationSource()
  // (CORS allowlist from app.cors.allowed-origins; allowCredentials=true; SecurityConfig.java:74-93)

  @Bean
  public SecurityFilterChain filterChain(
      HttpSecurity http,
      Environment env,
      JwtAuthenticationFilter jwtAuthFilter,
      RunTokenFilter runTokenFilter,           // REPLACES InternalApiKeyFilter
      RateLimitFilter rateLimitFilter,
      OrgScopeFilter orgScopeFilter
  ) throws Exception {

    List<String> publicPaths = new ArrayList<>(List.of(
        "/api/auth/**", "/ws-logs/**", "/error", "/actuator/health"));
    // /h2-console/** ONLY in dev (carried from SecurityConfig.java:107-110); never in deployed profiles.
    if (env.acceptsProfiles(p -> p.test("dev"))) publicPaths.add("/h2-console/**");

    http
      .cors(c -> c.configurationSource(corsConfigurationSource()))
      .csrf(csrf -> csrf.disable())          // safe ONLY because SameSite=Strict + Origin check (§6.5); see reasoning
      .headers(h -> h
          .frameOptions(f -> f.deny())                       // tightened from v1 sameOrigin
          .httpStrictTransportSecurity(hsts -> hsts          // HSTS (Spring default on; pin maxAge + subdomains)
              .includeSubDomains(true).maxAgeInSeconds(31_536_000).preload(true))
          .contentSecurityPolicy(csp -> csp
              .policyDirectives(CSP_VALUE))                  // see CSP_VALUE constant below (§6.5)
          .referrerPolicy(r -> r.policy(STRICT_ORIGIN_WHEN_CROSS_ORIGIN))
          .contentTypeOptions(Customizer.withDefaults())     // X-Content-Type-Options: nosniff
          .permissionsPolicy(pp -> pp.policy("geolocation=(), microphone=(), camera=()")))
      .authorizeHttpRequests(authz -> authz
          .requestMatchers(HttpMethod.OPTIONS, "/**").permitAll()
          .requestMatchers(publicPaths.toArray(new String[0])).permitAll()
          // /api/internal/** is permitAll HERE because RunTokenFilter rejects any request
          // without a valid per-run scoped token before Spring Security authz runs.
          .requestMatchers("/api/internal/**").permitAll()
          .anyRequest().authenticated())
      .sessionManagement(s -> s.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
      .authenticationProvider(authenticationProvider())
      .addFilterBefore(rateLimitFilter,  UsernamePasswordAuthenticationFilter.class)
      .addFilterBefore(runTokenFilter,   UsernamePasswordAuthenticationFilter.class)
      .addFilterBefore(jwtAuthFilter,    UsernamePasswordAuthenticationFilter.class)
      .addFilterAfter (orgScopeFilter,   JwtAuthenticationFilter.class);   // OrgScope after authn

    return http.build();
  }
}
```

Filter order (locked, request-time top to bottom): `RateLimitFilter` → `RunTokenFilter` → `JwtAuthenticationFilter` → `OrgScopeFilter` → Spring authz → `@PreAuthorize`. Reasoning: rate-limit before any work; the run-token path is fully self-contained (no SecurityContext needed); the JWT path establishes the principal; `OrgScope` is derived once per request from the established principal so the RLS predicate and `@PreAuthorize` SpEL both read the same authoritative scope.

### 5.2 `JwtTokenProvider` (hardened — full signatures)

v1 emits only `sub/iat/exp` (`JwtTokenProvider.java:41-46`) and validates only subject + expiry (`:54-57`) — no issuer, audience, jti, clock-skew, or revocation (`A1-F7`, `B4-security-compliance.md:98`). v2 adds them.

```java
@Component
public class JwtTokenProvider {

  // Config-bound (see §8): app.jwt.secret, app.jwt.issuer, app.jwt.audience,
  // app.jwt.expiration-ms, app.jwt.clock-skew-seconds
  public String generateToken(UserPrincipal principal);
  // claims: sub=username, iss=app.jwt.issuer, aud=app.jwt.audience, iat, exp,
  //         jti=UUID, tokenVersion=principal.tokenVersion  (bumped on password change / forced logout)

  public Jws<Claims> parseAndValidate(String token);
  // builds parser with .requireIssuer(issuer).requireAudience(audience)
  //                     .clockSkewSeconds(clockSkew).verifyWith(key)
  // throws JwtException on signature/iss/aud/exp failure -> caller maps to 401 NOT_AUTHENTICATED

  public boolean isRevoked(Claims claims);
  // true if claims.tokenVersion != currentTokenVersion(claims.sub)  -> revocation without a deny-list store

  public String getUsername(Claims claims);   // claims.getSubject()
  public String getJti(Claims claims);         // claims.getId(); logged into audit + MDC for traceability
}
```

Reasoning for `tokenVersion` instead of a Redis deny-list: a monotonic per-user integer column (`users.token_version`, add in a future `V{n}` migration or reuse the existing user row) gives **revocation on password-change / forced-logout / role-change** with a single integer compare and no extra store. `iss`/`aud` stop a `dev` token (public secret, `02-TECH-STACK.md` dev profile) from being replayed against a deployed environment of the same shape (`B4-security-compliance.md:98`). Keep the symmetric HMAC signing of v1 (`Keys.hmacShaKeyFor`, `JwtTokenProvider.java:32`) for the build; the asymmetric/JWKS rotation (`B4-security-compliance.md:88`) is a P2 follow-up, not this LLD's scope.

### 5.3 Cookie attributes (locked, from `04-API-CONTRACTS.md:35`)

`Set-Cookie: jwtToken=<jwt>; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=3600`. `Secure` is bound to `app.auth.cookie.secure` (true outside `dev`). `SameSite=Strict` is tightened from v1's `Lax` (`B4-security-compliance.md:97`). Logout sets `Max-Age=0`. **No `Authorization: Bearer`, no `localStorage`, no JS-readable token** anywhere (`02-TECH-STACK.md:756`).

### 5.4 The role enum (kills the v1 admin lockout)

```java
public enum PlatformRole { USER, PLATFORM_ADMIN }
public enum OrgRole      { OWNER, ADMIN, MEMBER }
public enum ProjectRole  { MEMBER, CLIENT }
```

`CustomUserDetailsService` MUST emit authority `"ROLE_" + platformRole.name()` = `ROLE_PLATFORM_ADMIN`, and every admin route uses `@PreAuthorize("hasRole('PLATFORM_ADMIN')")` (`04-API-CONTRACTS.md:64`). These two strings must agree — the v1 critical bug `A1-F1` was exactly their disagreement (`@PreAuthorize("hasRole('ADMIN')")` vs an emitted `ROLE_PLATFORM_ADMIN`, `A1-backend.md:57-60`). Jackson deserializes `UpdateUserRoleRequest.platformRole` straight to the enum, so `valueOf` rejects a typo at the wire boundary (`04-API-CONTRACTS.md:460`). The database `CHECK (platform_role IN ('USER','PLATFORM_ADMIN'))` constraint (added in `V6`, `03-DATA-MODEL.md:593`) is the third, defense-in-depth guard.

### 5.5 `AuthorizationService` (the org-scoped chokepoint — full signatures)

v1's `AuthorizationService` checks only project ownership/membership and **never references `org_id`** (`B4-security-compliance.md:75`, `A1-F9`). v2 rebuilds it around org scope.

```java
@Service
public class AuthorizationService {

  // Authn helpers
  boolean isPlatformAdmin(UserPrincipal caller);          // caller.platformRole == PLATFORM_ADMIN

  // THE chokepoint every tenant-owned read/write funnels through.
  void requireOrgScope(UserPrincipal caller, UUID orgId);
  // pass iff: isPlatformAdmin(caller) OR caller is a member of orgId. Else throw ForbiddenException -> 403.

  void requireOrgMember(UserPrincipal caller, UUID orgId);             // any OrgRole in orgId, or platform admin
  void requireOrgAdmin (UserPrincipal caller, UUID orgId);             // OrgRole in {OWNER,ADMIN}, or platform admin
  void requireParticipant(UserPrincipal caller, UUID projectId);       // owner OR project_memberships row
                                                                       //   OR requireOrgAdmin(project.org) OR platform admin
  void requireParticipantForRun(UserPrincipal caller, UUID runId);     // resolve run.project_id then requireParticipant

  // The org_id set the caller may see; the source of the RLS predicate.
  Set<UUID> visibleOrgIds(UserPrincipal caller);                       // platform admin -> the sentinel ALL_ORGS
}
```

`TenantPredicate.orgScoped(root, cb, scope)` builds `org_id IN (:visibleOrgIds)`; for a platform admin it returns the always-true predicate (audited — see §6.1). Every repository method that reads a tenant-owned row applies this predicate; there is no row the filter can miss because `org_id` is `NOT NULL` on every tenant table (rule R-C, `03-DATA-MODEL.md:87`).

---

## 6. Core algorithms & flows (the hard parts, in code/pseudocode)

### 6.1 Org-scoped multi-tenant isolation (RLS-style, no cross-org leakage)

**The single rule:** a tenant-owned row is readable/writable only if the caller shares its `org_id`, or the caller is `PLATFORM_ADMIN` (and that bypass is audited). This is enforced in **two layers** so a single missed `WHERE` cannot leak:

1. **Application chokepoint** — every project/run/dataset/artifact handler calls `AuthorizationService.requireOrgScope(caller, resource.orgId)` (or `requireParticipant`) before returning data. The `org_id` is **never** taken from the request body — it is resolved server-side from the resource id (`04-API-CONTRACTS.md:37`). For create, the body `orgId` is validated against the caller's memberships (`04-API-CONTRACTS.md:189`).
2. **Query predicate** — every tenant-scoped repository read appends `TenantPredicate.orgScoped(...)`. This catches the case where a developer forgets the chokepoint: the query physically cannot return another org's rows.

```java
// TenantPredicate — the RLS-style filter applied to every tenant-owned query.
public final class TenantPredicate {
  public static final UUID ALL_ORGS = UUID.fromString("ffffffff-ffff-ffff-ffff-ffffffffffff");

  public static Predicate orgScoped(Root<?> root, CriteriaBuilder cb, Set<UUID> scope) {
    if (scope.contains(ALL_ORGS)) return cb.conjunction();           // platform admin: no filter (AUDITED upstream)
    if (scope.isEmpty())          return cb.disjunction();           // belongs to no org: see nothing
    return root.get("orgId").in(scope);                              // org_id IN (...)
  }
}
```

**The cross-org leak this kills (verified v1 bug):** `getDiscoverProjects()` returned every `PUBLIC` project across all orgs — name, owner, model type, description — with no `org_id` filter (`B4-security-compliance.md:76`, `ProjectService.java:410-422`). For a B2B FL product where the project topic itself is sensitive ("Mercy Hospital pneumonia model"), this is a confidentiality finding. v2 discovery is org-scoped: a PUBLIC project is discoverable only within its own org unless an explicit cross-org share is modeled later.

> **Stronger option (recommended for the HIPAA tier, optional for the build):** also enable PostgreSQL native RLS — `ALTER TABLE projects ENABLE ROW LEVEL SECURITY;` plus a `CREATE POLICY org_isolation USING (org_id = current_setting('app.current_org')::uuid)`, with the app setting `SET app.current_org = ...` per transaction. This is the defense-in-depth backstop if the JPA predicate is ever bypassed by a raw query. It is not required to pass the build but is named so the local model knows the upgrade path; the JPA `TenantPredicate` is the mandatory layer.

### 6.2 The cookie-only JWT auth contract, end to end

ASCII sequence (login → authenticated call → STOMP → logout); conforms to `04-API-CONTRACTS.md` §2 and Flow A of `01-ARCHITECTURE-HLD.md:136-169`:

```
Browser (React, withCredentials:true)        Control plane (Spring Boot)              Postgres
   │ 1. POST /api/auth/login {username,pass}        │                                     │
   │───────────────────────────────────────────────▶│ 2. authenticationManager.authenticate
   │                                                 │    BCrypt verify; gate status==ACTIVE
   │                                                 │    && email_verified (A1-F5)        │
   │                                                 │────────── SELECT user ─────────────▶│
   │ 3. 200 MeResponse                               │◀───────── row ──────────────────────│
   │    Set-Cookie: jwtToken=<jwt>;                  │  JwtTokenProvider.generateToken      │
   │    HttpOnly; Secure; SameSite=Strict; Max-Age=3600                                    │
   │◀───────────────────────────────────────────────│  claims: sub,iss,aud,iat,exp,jti,tokenVersion
   │                                                 │                                     │
   │ 4. GET /api/auth/me  (silent 401 probe)         │ 5. JwtAuthenticationFilter:          │
   │    (cookie sent automatically)                  │    read jwtToken cookie ->           │
   │───────────────────────────────────────────────▶│    parseAndValidate (iss/aud/exp/sig)│
   │                                                 │    isRevoked? (tokenVersion compare) │
   │◀───────────── 200 MeResponse (or 401) ──────────│    set SecurityContext; OrgScopeFilter populates OrgScope
   │                                                 │                                     │
   │ 6. POST /api/projects {orgId,...}               │ 7. @PreAuthorize ORG_MEMBER ->       │
   │───────────────────────────────────────────────▶│    requireOrgScope(caller, orgId)    │
   │                                                 │    TenantPredicate on every read     │
   │◀────────────── 201 ProjectResponseDto ──────────│                                     │
   │                                                 │                                     │
   │ 8. WS CONNECT /ws-logs  (cookie on handshake)   │ 9. JwtHandshakeInterceptor: auth at  │
   │───────────────────────────────────────────────▶│    UPGRADE (no subscribe-then-check) │
   │ 10. SUBSCRIBE /topic/results/{projectId}        │ 11. JwtChannelInterceptor:           │
   │───────────────────────────────────────────────▶│    parse dest -> requireParticipant  │
   │                                                 │    reject frame if not a participant │
   │ 12. POST /api/auth/logout                       │ 13. Set-Cookie: jwtToken=; Max-Age=0 │
   │───────────────────────────────────────────────▶│    bump users.token_version (revoke) │
   │◀──────────────── 204 ───────────────────────────│                                     │
```

The Axios interceptor on the frontend swallows `401` **only** on `/api/auth/me` (the probe), and fires an `authError`/redirect on `401` elsewhere — preserved verbatim from v1 (`04-API-CONTRACTS.md:82`).

### 6.2.1 Per-run scoped result token (the internal-callback auth — full verify code)

Replaces the v1 single global `APP_INTERNAL_API_KEY` that let any FL task write results for **any** project (`A1-F6`, `B4-security-compliance.md`; v1 `SecurityConfig.java:122` permits `/api/internal/**` behind that one global key). Token format and claims are locked in `04-API-CONTRACTS.md:1018-1031`. The verifier (`RunTokenService.verify`, called by `RunTokenFilter`):

```java
// RunTokenService.verify(authHeader, pathRunId) -> RunContext   (throws -> mapped to the envelope codes)
RunContext verify(String authHeader, UUID pathRunId) {
  if (authHeader == null || !authHeader.startsWith("Bearer flrun_"))
      throw new RunTokenInvalid();                                  // 401 RUN_TOKEN_INVALID
  String raw       = authHeader.substring("Bearer ".length());      // "flrun_<payloadB64>.<sigB64>"
  String body      = raw.substring("flrun_".length());
  int dot          = body.lastIndexOf('.');
  byte[] payloadB64= body.substring(0, dot).getBytes(UTF_8);
  byte[] sig       = base64UrlDecode(body.substring(dot + 1));
  byte[] expected  = hmacSha256(secret /* app.internal.run-token-secret */, payloadB64);
  if (!MessageDigest.isEqual(expected, sig)) throw new RunTokenInvalid();   // CONSTANT-TIME compare, 401
  Claims c = json(base64UrlDecode(new String(payloadB64, UTF_8)));          // {runId,projectId,orgId,issuedAt,expiresAt,nonce}
  if (now().getEpochSecond() > c.expiresAt)  throw new RunTokenInvalid();   // 401
  if (!c.runId.equals(pathRunId))            throw new RunTokenMismatch();  // 403 RUN_TOKEN_MISMATCH
  FlRun run = flRunRepository.findById(c.runId).orElseThrow(RunNotFound::new); // 404 RUN_NOT_FOUND
  if (TERMINAL_STATES.contains(run.getStatus())) throw new RunTerminal();   // 409 RUN_TERMINAL
  return new RunContext(c.runId, c.projectId, c.orgId);                     // scope comes from the TOKEN, never the body
}
```

`RunTokenService.mint(FlRun run)` is the launch-time counterpart: it builds the payload, HMAC-signs it with `app.internal.run-token-secret`, and returns `flrun_<payloadB64>.<sigB64>`. The orchestration substrate injects it as `FEDLEARN_RUN_TOKEN` (with `FEDLEARN_RUN_ID`, `FEDLEARN_BACKEND_URL`, `TRACEPARENT`) into the k8s Job / ECS task override / dev process env (`04-API-CONTRACTS.md:1052-1062`). `MessageDigest.isEqual` is constant-time — preserve the v1 strength (v1's `InternalApiKeyFilter:69-72` already did constant-time compare; do not regress to `.equals`).

### 6.3 gRPC default-secure (TLS + mTLS) with cert-CN-bound identity + enrollment token

**The Sybil hole this closes (verified):** v1's `RegisterClient` takes a self-asserted `client_id` string with no auth, no per-client cert, no rate limit, and gRPC defaults to `add_insecure_port` (`B4-security-compliance.md:33, 35`). One adversary registers as many fake clients to dominate aggregation. v2 makes the channel default-secure and binds identity to the cert plus a backend-minted enrollment token (`README.md` risk R6; `01-ARCHITECTURE-HLD.md:258`).

**PKI layout (`deploy/security/`):** an offline root CA signs one intermediate CA; the intermediate signs short-lived leaf certs. The FL **server** leaf has a SAN of the run's service DNS (`fl-run-<id>.fl.svc.cluster.local`); each FL **client** leaf has a stable, meaningful CN that is the client's enrolled identity (e.g. `client:<orgId>:<deviceLabel>`). The intermediate CA's certificate is the trust anchor in both directions (server trusts client leafs, clients trust the server leaf).

**Server side (Python, `framework/src/fedlearn/security/tls.py`):**

```python
def build_server_credentials(profile: str) -> grpc.ServerCredentials:
    # default-secure: refuse plaintext outside dev (locked invariant, 02-TECH-STACK.md:759)
    if profile == "dev":
        return None                      # caller uses add_insecure_port ONLY when this is None AND profile==dev
    server_key  = read_secret("FEDLEARN_GRPC_SERVER_KEY")
    server_crt  = read_secret("FEDLEARN_GRPC_SERVER_CERT")
    ca_crt      = read_secret("FEDLEARN_GRPC_CA_CERT")     # intermediate CA bundle
    return grpc.ssl_server_credentials(
        [(server_key, server_crt)],
        root_certificates=ca_crt,
        require_client_auth=True,        # mTLS: every client MUST present a cert signed by our CA
    )

# server bind (fl_server.py): refuse insecure unless dev
creds = build_server_credentials(profile)
if creds is None and profile == "dev":
    server.add_insecure_port(f"[::]:{port}")          # loopback dev only
elif creds is not None:
    server.add_secure_port(f"[::]:{port}", creds)
else:
    raise RuntimeError("refusing to start FL server with no TLS outside dev")   # fail-closed
```

**Identity binding (Python, `framework/src/fedlearn/security/identity.py`):** in the `RegisterClient` handler, read the peer cert CN from the gRPC `ServicerContext` auth context, verify the `enrollment_token` (HMAC over `{client_cn, run_id, org_id, exp}` minted by the backend), and require CN consistency:

```python
def authorize_register(context, req: RegisterClientRequest, profile: str) -> str:
    if profile == "dev":
        return req.client_id                         # dev: trust the asserted id (loopback, no mTLS)
    peer_cn = extract_peer_cn(context)               # from context.auth_context()['x509_common_name'][0]
    if peer_cn is None:
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "no client cert")
    claims = enrollment.verify(req.enrollment_token) # HMAC verify; raises -> abort UNAUTHENTICATED
    if claims.client_cn != peer_cn or claims.run_id != req.run_id:
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "enrollment/cert mismatch")
    if claims.exp < now():
        context.abort(grpc.StatusCode.UNAUTHENTICATED, "enrollment expired")
    return peer_cn                                    # the TRUSTED identity; req.client_id is display-only
```

The trusted client identity used for aggregation accounting, rate limiting, and audit is **`peer_cn`**, never the self-asserted `req.client_id` (`04-API-CONTRACTS.md:669, 874`). `enrollment_token` is minted by `EnrollmentTokenService` on the backend when a client is enrolled to a run; it is the gRPC-side analogue of the per-run result token (§6.2.1). All five gRPC error mappings from `04-API-CONTRACTS.md:880` apply: `UNAUTHENTICATED` for bad cert/token, `INVALID_ARGUMENT` for bad `protocol_version`/`codec`, etc.

**Why mTLS and not a token-only scheme:** mTLS gives a cryptographic per-client identity at the transport layer that one move both encrypts the channel (HIPAA encryption-in-transit, `B4-security-compliance.md:130`) and authenticates the peer (anti-Sybil). The TLS+mTLS code already existed in v1 but was off by default (`B4-security-compliance.md:59`); v2 flips the default — this is "rebuild the trust model, the code exists, the posture is wrong."

### 6.4 STOMP topic-level authorization (close the WS cross-tenant leak)

v1 authenticates the WS handshake (good, `A1-backend.md:75`) but has **no `SUBSCRIBE`-frame check**, so any authenticated user can subscribe to any project's live logs/metrics (`A1-backend.md:76`). v2 adds the check in `JwtChannelInterceptor`:

```java
// JwtChannelInterceptor.preSend — on SUBSCRIBE, parse the destination and authorize the participant.
if (StompCommand.SUBSCRIBE.equals(accessor.getCommand())) {
    String dest = accessor.getDestination();                 // e.g. /topic/results/{projectId}
    UUID projectId = parseProjectId(dest);                   // matches /topic/(logs|results|status|runs)/{uuid}
    UserPrincipal caller = principalFrom(accessor);          // set at CONNECT re-validation
    authorizationService.requireParticipant(caller, projectId);   // throws -> frame rejected
}
```

Also fix the v1 anonymous `AuthenticationException` subclasses (`JwtChannelInterceptor.java:71,78`, `A1-backend.md:76`) — use a named, mapped exception. Topics are keyed on `projectId` (not `runId`) per `04-API-CONTRACTS.md:897`; the payload carries `runId` for client-side filtering.

### 6.5 CSP, HSTS, CSRF, and the Origin check

The header set is configured in `SecurityConfig.headers(...)` (§5.1). The CSP constant:

```java
// CSP_VALUE — strict; the SPA is bundled (Vite), so no inline scripts are needed in production.
// Adjust connect-src to the deployed API + ws origin; keep 'self' as the base.
static final String CSP_VALUE =
    "default-src 'self'; " +
    "script-src 'self'; " +
    "style-src 'self' 'unsafe-inline'; " +     // Tailwind injects styles; scripts stay locked down
    "img-src 'self' data:; " +
    "connect-src 'self' https: wss:; " +        // REST over https, STOMP over wss
    "frame-ancestors 'none'; " +
    "base-uri 'self'; " +
    "form-action 'self'";
```

**CSRF reasoning (why disabled is acceptable here, and what compensates):** the session is a cookie, so CSRF is a real surface. v1 disabled CSRF with `SameSite=Lax` (`SecurityConfig.java:114`; `B4-security-compliance.md:97`) — a top-level navigation POST could ride the cookie. v2 compensates with **`SameSite=Strict`** (the browser does not attach the cookie to any cross-site request, killing the classic CSRF vector) **plus** a server-side `Origin`/`Referer` allowlist check on every mutating endpoint, reusing the CORS allowlist (`B4-security-compliance.md:97`). This is the documented v2 posture; do not re-enable Spring's CSRF token (it adds a token-echo dance the cookie+SameSite+Origin trio already covers for an SPA). HSTS (`max-age=31536000; includeSubDomains; preload`) forces HTTPS — the platform-layer twin of the gRPC default-secure decision; it also remediates the v1 `ec2demo` `cookie.secure=false`-over-HTTP finding (`B4-security-compliance.md:96`).

### 6.6 The FL threat model + mitigations

The threat catalogue is `B4-security-compliance.md:25-35`. Mapping each threat to the v2 mitigation:

| Threat | v2 exposure & mitigation | Where implemented |
|---|---|---|
| **Gradient leakage / DLG reconstruction** | **DeComFL path: structurally near-eliminated** — only scalars `g·u` + integer seeds cross the wire, never a gradient vector, so the DLG/iDLG/GradInversion family has nothing to invert (`B4-security-compliance.md:37-45`; the "wedge"). **FedAvg path: fully exposed** — full tensors ship; mitigate with DP-SGD + secure aggregation scoped to FedAvg. | proto `fedlearn.v2` SubmitGradientScalars (`04-API-CONTRACTS.md:837-848`); `framework/.../dp.py` |
| **Membership / property inference** | Survives on **both** paths (scalars still correlate with the loss landscape; seeds are public so a curious server can probe known directions, `B4-security-compliance.md:48-49`). Mitigate with calibrated **scalar-DP** on DeComFL and **DP-SGD** on FedAvg, ε per run surfaced in telemetry. | `framework/.../dp.py`; `dpEnabled/dpNoiseMultiplier/dpClipNorm` (`04-API-CONTRACTS.md:228-231`) |
| **Model poisoning** | Unmitigated in v1 (no robust aggregation; `MAX_SAMPLES` is a clamp, not a defense, `B4-security-compliance.md:30`). v2 adds a selectable **robust-mean / coordinate-wise trimmed-mean** guard + per-coordinate clip. | `framework/.../robust.py`; `robustClipTau` (`04-API-CONTRACTS.md:231`) |
| **Data poisoning (label-flip / backdoor)** | Server never sees data; undetectable directly. Partial mitigation: robust aggregation blunts the influence of outlier updates; reject NaN/Inf scalars before aggregation. | `framework/.../robust.py` (NaN/Inf reject) |
| **Byzantine clients (NaN/Inf, collusion)** | Unmitigated in v1; v2 rejects non-finite scalars and applies trimmed-mean. **Caveat (honest):** trimmed-mean assumes a bounded Byzantine fraction and fails under adaptive/majority attacks ([USENIX Fang sec20](https://www.usenix.org/system/files/sec20summer_fang_prepub.pdf)) — do **not** market this as "Byzantine-robust." | `framework/.../robust.py` |
| **Sybil (forge many client_id)** | **Closed** by mTLS cert-CN identity + per-run `enrollment_token` (§6.3). One adversary cannot mint identities without CA-signed certs. | `framework/.../identity.py`; `GrpcTlsConfig` |
| **Free-rider** | Partially addressed by contribution accounting keyed on the trusted `peer_cn`; full detection deferred. | `framework/.../identity.py` |
| **gRPC plaintext over WAN** | **Closed** by default-secure TLS+mTLS (§6.3); plaintext only on dev loopback. | `framework/.../tls.py` |

**The DP layer — exact algorithm (the genuinely tricky math):**

```python
# framework/src/fedlearn/security/dp.py

# FedAvg path: per-sample gradient clipping + Gaussian noise (DP-SGD, Abadi et al.).
# Use Opacus PrivacyEngine where it fits; the contract the framework MUST honor:
def dpsgd_clip_and_noise(per_sample_grads, clip_norm, noise_multiplier, generator):
    # 1. clip each per-sample gradient to L2 norm <= clip_norm
    clipped = [g * min(1.0, clip_norm / (g.norm(2) + 1e-12)) for g in per_sample_grads]
    summed  = sum(clipped)
    # 2. add Gaussian noise calibrated to the clip (CPU-canonical generator, B1-C2)
    noise   = torch.normal(0.0, noise_multiplier * clip_norm, size=summed.shape, generator=generator)
    return (summed + noise) / len(per_sample_grads)

# DeComFL path: calibrated SCALAR DP noise on the gradient scalar g (per DPZV, arXiv:2502.20565).
# DeComFL transmits g = (loss(theta + mu*z) - loss(theta)) / mu  per (local_step, perturbation).
# Clip the magnitude of g, then add scalar Gaussian noise BEFORE the client uploads it.
def scalar_dp(g: float, clip_tau: float, noise_multiplier: float, generator) -> float:
    g_clipped = max(-clip_tau, min(clip_tau, g))                 # bound the scalar magnitude
    noise     = torch.normal(0.0, noise_multiplier * clip_tau, size=(1,), generator=generator).item()
    return g_clipped + noise
```

Both noise draws MUST use the **CPU-canonical `torch.Generator`** (the DeComFL correctness fix bug-2, `02-TECH-STACK.md:205`; `03-DATA-MODEL.md:548`) so a CUDA/MPS server does not silently diverge from CPU clients. The robust guard (`robust.py`) applies a coordinate-wise trimmed-mean over the per-client scalars/tensors after clipping; reject any update containing NaN/Inf with a counted, audited drop.

**Privacy claim discipline (locked):** market DeComFL's scalar-only upload as a **DLG-attack-family eliminator** (true, `B4-security-compliance.md:45`); **delete the false "Byzantine-robust" README claim** (`README.md:106`, `02-TECH-STACK.md:583`); never conflate "compressed" with "private" — the property comes from being scalar-projected, not from being small (`B4-security-compliance.md:51`).

### 6.7 GDPR right-to-erasure note (FL-specific, must be disclosed)

GDPR Art. 17 is **not** satisfied by deleting a client's local data — aggregated model weights may themselves be personal data and unlearning in FL is an open problem (`B4-security-compliance.md:147`). **DeComFL helps:** because the server stores seed history + scalar gradient history (`RebuildHistory` in proto, `04-API-CONTRACTS.md:811-819`), a model is reconstructible from its history, which makes **replay-based unlearning** tractable (recompute excluding a client's contributions for given rounds). Implement that as the technical answer and **disclose contractually** that full erasure may require retraining/unlearning and is not instantaneous. This is a P2 build item; the LLD records the design so the local model does not invent a different erasure story.

---

## 7. Data it owns (exact tables/columns, from `03-DATA-MODEL.md`)

This unit does not own a plane of its own; it reads and writes specific columns across the identity and audit tables.

| Table | Columns this unit reads/writes | Reference |
|---|---|---|
| `users` | `platform_role` (enum source; `CHECK` in V6), `password` (BCrypt), `status`, `email_verified`, plus a `token_version` integer for JWT revocation (add in a future `V{n}` if not present). | `03-DATA-MODEL.md:425-433, 593` |
| `organizations`, `organization_memberships` | `org_role` (`OWNER/ADMIN/MEMBER`) drives `requireOrgMember/requireOrgAdmin`; composite key `(org_id UUID, user_id BIGINT)`. | `03-DATA-MODEL.md:416-423` |
| `projects` | `org_id NOT NULL` (the tenancy anchor for `TenantPredicate`), `user_id` (implicit owner). | `03-DATA-MODEL.md:435, 463` |
| `project_memberships` | `role` (`MEMBER/CLIENT/OWNER`; `CHECK` added in V6) drives `requireParticipant`. | `03-DATA-MODEL.md:374-385, 597` |
| `fl_runs` | `org_id`, `project_id`, `status` (the `RunTokenService` resolves run→project/org and rejects terminal runs). | `03-DATA-MODEL.md:695-733` |
| `audit_events` | writes `occurred_at, actor_user_id, org_id, action, target_type, target_id, metadata (JSONB), request_ip, user_agent`. **`metadata` is JSONB in v2** (the V5 `CLOB` is the §6 defect fixed in V7, `03-DATA-MODEL.md:446, 664`). | `03-DATA-MODEL.md:438-452` |

**In-memory structures (request-scoped, owned by this unit):**

```java
@RequestScope class OrgScope {            // populated by OrgScopeFilter from the authenticated principal
  UUID  userId;                            // Long on the wire, UUID-free here
  PlatformRole platformRole;
  Set<UUID> visibleOrgIds;                 // {ALL_ORGS} for platform admin; else the caller's org memberships
}
record RunContext(UUID runId, UUID projectId, UUID orgId) {}   // built by RunTokenService.verify; never from body
record UserPrincipal(Long userId, String username, PlatformRole platformRole, int tokenVersion) {}
```

The audit row write goes through `AuditAspect` which **proceeds the join point first, then writes the row in the same transaction**, so a caller rollback rolls back the audit row too (salvaged correct semantics, `A1-backend.md:147`; the project conventions). v2 must extend `@Auditable` coverage from v1's 2 annotations (`A1-backend.md:147`) to all mutating + authorization-relevant actions: `RUN_START`, `RUN_STOP`, `PROJECT_CREATE/DELETE`, `MEMBERSHIP_GRANT/REVOKE`, `ROLE_CHANGE`, `ACCESS_REQUEST_DECIDE`, `PLATFORM_ADMIN_ORG_BYPASS` (audit every cross-org platform-admin access — `B4-security-compliance.md:77`).

---

## 8. Configuration & environment variables (exact names, types, defaults, profiles)

| Env var / property | Type | Default | Profile(s) | Purpose |
|---|---|---|---|---|
| `APP_JWT_SECRET` (`app.jwt.secret`) | string (base64) | none — **boot fails if absent** | all | HMAC key for the cookie JWT (`SecurityConfig`/`JwtTokenProvider`). Public dev value only in `dev`. |
| `app.jwt.issuer` | string | `fedlearn` | all | JWT `iss`; environment-binds the token (`B4-security-compliance.md:98`). |
| `app.jwt.audience` | string | `fedlearn-web` | all | JWT `aud`. |
| `app.jwt.expiration-ms` | long | `3600000` | all | Access-token lifetime (1h). |
| `app.jwt.clock-skew-seconds` | int | `30` | all | Allowed skew on `exp`/`iat`. |
| `app.auth.cookie.secure` | bool | `true` (`false` only in `dev`) | all | `Secure` flag on `jwtToken`. **Must be `true` in `ec2demo`/`production`** (fixes `B4-security-compliance.md:96`). |
| `app.auth.cookie.same-site` | enum | `Strict` (`Lax` allowed only in `dev`) | all | Tightened from v1 `Lax` (`B4-security-compliance.md:97`). |
| `APP_INTERNAL_API_KEY` | string | **removed in v2** | — | The v1 global key is deleted; replaced by per-run tokens. Do not reintroduce. |
| `app.internal.run-token-secret` | string | none — **boot fails if absent** outside dev | all | HMAC key for the per-run scoped token (§6.2.1). From the secrets manager. |
| `app.enrollment.token-secret` | string | none — **boot fails if absent** outside dev | all | HMAC key for the gRPC client enrollment token (§6.3). |
| `CORS_ALLOWED_ORIGINS` (`app.cors.allowed-origins`) | csv | none — **boot fails if absent** | all | Origin allowlist; reused server-side for the mutating-endpoint Origin check (§6.5). |
| `app.security.csp` | string | the `CSP_VALUE` of §6.5 | all | Overridable CSP for env-specific `connect-src`. |
| `app.security.hsts.max-age-seconds` | long | `31536000` | non-dev | HSTS max-age. |
| `app.ratelimit.auth.capacity` | int | `10` | all | Bucket4j tokens per window on auth endpoints. |
| `FEDLEARN_GRPC_SERVER_CERT` / `_KEY` / `FEDLEARN_GRPC_CA_CERT` | path/secret | none outside dev | non-dev | FL-server TLS material; mTLS trust anchor (§6.3). |
| `FEDLEARN_RUN_TOKEN` | string | injected per run | runtime | The per-run token the FL server sends on callbacks (`04-API-CONTRACTS.md:1057`). |
| `FEDLEARN_RUN_ID` / `FEDLEARN_BACKEND_URL` / `FEDLEARN_PROJECT_ID` / `TRACEPARENT` | string | injected per run | runtime | Callback context + trace (`04-API-CONTRACTS.md:1056-1060`). |
| `FEDLEARN_GRPC_INSECURE` | bool | `false` | `dev` only honored | If ever true outside `dev`, the FL server **must refuse to start** (fail-closed, §6.3). |

The base Spring profile must **refuse to boot** without `APP_JWT_SECRET`, `app.internal.run-token-secret`, `app.enrollment.token-secret`, and `CORS_ALLOWED_ORIGINS` (extends the v1 fail-fast posture, `A1-backend.md:153`). `dev` carries public, clearly-fenced values; `production`/`ec2demo` carry no fallbacks (`B4-security-compliance.md:83`).

---

## 9. Error handling & edge cases

Every failure maps to the standard error envelope (`04-API-CONTRACTS.md` §12) via `GlobalExceptionHandler`. Enumerate the real modes:

| Failure mode | Detection | Handling / response |
|---|---|---|
| JWT signature/iss/aud/exp invalid | `JwtTokenProvider.parseAndValidate` throws `JwtException` | `401 NOT_AUTHENTICATED`; clear nothing (let the client re-auth). On `/api/auth/me` the frontend swallows it silently. |
| JWT valid but `tokenVersion` stale (revoked) | `isRevoked(claims)` true | `401 NOT_AUTHENTICATED`; forces re-login after a password/role change or forced logout. |
| Login by `PENDING`/unverified account | `status != ACTIVE \|\| !email_verified` | `403 ACCOUNT_NOT_VERIFIED` (fixes `A1-F5` — v1 never enforced this). |
| Cross-org read attempt | `requireOrgScope` fails AND `TenantPredicate` returns no rows | `403 FORBIDDEN`; the predicate also makes the query physically empty (defense in depth). |
| WS `SUBSCRIBE` to a non-participant topic | `JwtChannelInterceptor` `requireParticipant` throws | Reject the `SUBSCRIBE` frame (named exception, not anonymous — fixes `A1` C7). |
| Run token: bad HMAC / expired | `RunTokenService.verify` constant-time compare fails / `exp` passed | `401 RUN_TOKEN_INVALID`. |
| Run token: `runId` ≠ path | claim mismatch | `403 RUN_TOKEN_MISMATCH`. |
| Run token for a terminal run | `run.status ∈ TERMINAL_STATES` | `409 RUN_TERMINAL` (stops late/replayed callbacks from mutating a finished run). |
| gRPC client: no/invalid cert | mTLS handshake fails / `peer_cn == null` | gRPC `UNAUTHENTICATED`; connection refused (fail-closed). |
| gRPC client: enrollment/cert mismatch | `authorize_register` checks | gRPC `UNAUTHENTICATED`. |
| FL server started without TLS outside dev | `build_server_credentials` returns None and profile≠dev | `RuntimeError` at boot — **fail closed**, never silently fall back to plaintext (the v1 anti-pattern). |
| Rate limit exceeded on auth endpoint | Bucket4j bucket empty | `429 RATE_LIMITED` with `Retry-After`. |
| Aggregation receives NaN/Inf scalar | `robust.py` finite check | Drop the offending update, increment a counted+audited metric, continue with quorum (never crash the round). |
| Origin/Referer mismatch on a mutating endpoint | server-side allowlist check | `403 FORBIDDEN` (CSRF compensation, §6.5). |
| Platform-admin cross-org access | `isPlatformAdmin` true bypasses scope | Allow, but **write a `PLATFORM_ADMIN_ORG_BYPASS` audit row** (the bypass is legitimate but must be evidenced — `B4-security-compliance.md:77`). |

Never return a raw stack trace or a bare string; `INTERNAL_ERROR(500)` carries a generic message with details only in logs (`04-API-CONTRACTS.md:1010`).

---

## 10. Testing strategy

Backend tests with JUnit 5 + Spring Boot Test + Testcontainers-Postgres (so RLS predicates run against real Postgres, `A1-F10`). Framework tests with pytest. Name and assert exactly:

| Test (name) | Framework | Asserts |
|---|---|---|
| `SecurityConfigHeadersTest.csp_hsts_nosniff_present()` | Spring MVC test | Every response carries the CSP, `Strict-Transport-Security`, `X-Content-Type-Options: nosniff`, `Referrer-Policy` headers. |
| `JwtTokenProviderTest.rejects_wrong_issuer_audience()` | JUnit | A token with a foreign `iss`/`aud` fails `parseAndValidate` (env cross-replay blocked). |
| `JwtTokenProviderTest.revokes_on_token_version_bump()` | JUnit | After `token_version` increments, a previously valid token is `isRevoked()`. |
| `AdminControllerIntegrationTest.platform_admin_reaches_admin_routes()` | Testcontainers | Seed `PLATFORM_ADMIN` (NOT the literal `"ADMIN"` that masked `A1-F1`); assert `200` on `/api/admin/users`. This test must fail on the v1 code and pass on v2. |
| `TenantIsolationTest.orgA_cannot_read_orgB_project()` | Testcontainers | Caller in org A gets `403`/empty for an org B project; `getDiscoverProjects` returns only same-org PUBLIC projects (closes `B4` cross-org leak). |
| `WsSubscribeAuthzTest.non_participant_subscribe_rejected()` | Spring WS test | A `SUBSCRIBE /topic/results/{otherProjectId}` by a non-participant is rejected at the frame. |
| `RunTokenServiceTest.mismatched_run_id_is_403()` | JUnit | A token for run X used on `/api/internal/runs/Y/results` → `403 RUN_TOKEN_MISMATCH`; bad HMAC → `401`; terminal run → `409`. |
| `RunTokenServiceTest.constant_time_compare()` | JUnit | Uses `MessageDigest.isEqual` (no early-exit `.equals`). |
| `GrpcMtlsTest.rejects_client_without_cert()` | pytest + grpc | A client connecting without a CA-signed cert is `UNAUTHENTICATED`; with a valid cert + enrollment token, `RegisterClient` returns `ACCEPTED`. |
| `GrpcIdentityTest.self_asserted_client_id_not_trusted()` | pytest | Aggregation accounting keys on `peer_cn`, not `req.client_id` (anti-Sybil). |
| `GrpcServerStartTest.refuses_plaintext_outside_dev()` | pytest | `build_server_credentials("production")` returning None → boot raises (fail-closed). |
| `DpTest.scalar_dp_clips_and_noises_deterministically()` | pytest | With a fixed CPU-canonical generator, `scalar_dp` output is reproducible and bounded by `clip_tau + noise`. |
| `RobustTest.nan_inf_update_dropped()` | pytest | A NaN/Inf scalar is dropped and counted; the round still aggregates over the remaining quorum. |
| `AuditCoverageTest.mutating_actions_are_audited()` | Testcontainers | `RUN_START`, `ROLE_CHANGE`, `MEMBERSHIP_GRANT`, etc. each write an `audit_events` row; a rolled-back transaction writes no row. |
| `RateLimitTest.auth_endpoint_429_after_capacity()` | Spring MVC test | The N+1th `/api/auth/login` in a window returns `429 RATE_LIMITED`. |

CI gates (`02-TECH-STACK.md:624, 644`): `gitleaks` (secret scan), `pip-audit` + Gradle dependency-check + `npm audit` (per-stack vuln scans), CycloneDX SBOM — these run in `security.yml` and are required status checks (SOC 2 CC8 evidence).

---

## 11. Build & run (verify this unit in isolation)

```bash
# 1. Backend security tests (Testcontainers spins real Postgres; RLS predicates run for real)
cd /home/anurag/codebase/FedLearn-Platform/backend/fl-platform-api
SPRING_PROFILES_ACTIVE=test ./gradlew test --tests "com.federated.fl_platform_api.security.*" \
                                           --tests "com.federated.fl_platform_api.authz.*" \
                                           --tests "com.federated.fl_platform_api.audit.*"

# 2. Static + supply-chain gates (the SOC 2 CC8 evidence)
./gradlew dependencyCheckAnalyze            # OWASP/Gradle dependency-check
gitleaks detect --source . --redact         # secret scan (baseline history first, B7)
# pip-audit for the framework:
cd /home/anurag/codebase/FedLearn-Platform/framework && pip-audit

# 3. PKI for local mTLS testing (issues a CA + server + one client leaf into deploy/security/)
cd /home/anurag/codebase/FedLearn-Platform/deploy/security
bash issue-server-cert.sh && bash issue-client-cert.sh client:dev:laptop

# 4. Framework security tests (TLS/identity/DP/robust)
cd /home/anurag/codebase/FedLearn-Platform/framework
pytest src/fedlearn/security/ -v

# 5. Manual mTLS smoke (dev): server with mTLS, client with a valid cert
FEDLEARN_GRPC_SERVER_CERT=deploy/security/server.crt \
FEDLEARN_GRPC_SERVER_KEY=deploy/security/server.key \
FEDLEARN_GRPC_CA_CERT=deploy/security/ca.crt \
SPRING_PROFILES_ACTIVE=ec2demo python run_local_test.py   # connecting without a cert must be UNAUTHENTICATED
```

Verify-in-isolation done-conditions: every test in §10 passes; `gitleaks`/`pip-audit`/dependency-check report no high/critical findings (or only baselined ones); a gRPC client with no cert is refused; `build_server_credentials("production")==None` raises.

---

## 12. Reasoning & alternatives (why this design; what was rejected)

| Decision | Why | Rejected alternative (and why) | Audit tie |
|---|---|---|---|
| Cookie-only HttpOnly JWT, no Bearer | XSS cannot read an HttpOnly cookie; the v1 posture was rated "textbook-correct". | **localStorage + `Authorization: Bearer`** — a JS-readable token is an XSS exfiltration target; explicitly forbidden. | `B4-security-compliance.md:94`; `02-TECH-STACK.md:756` |
| `tokenVersion` for revocation | One integer compare gives revocation on password/role change without a Redis deny-list. | **Stateless JWT with no revocation (v1)** — a stolen/post-termination token is valid until expiry; a HIPAA/SOC 2 finding. **Full JWKS/asymmetric rotation** — deferred to P2; out of scope for this LLD. | `A1-F7`; `B4-security-compliance.md:98` |
| Org-scoped `TenantPredicate` (RLS-style) chokepoint + `org_id NOT NULL` | A single missed `WHERE` cannot leak because the predicate is applied at the query layer and `org_id` is on every tenant row. | **Application-code-only project scope (v1)** — `AuthorizationService` never checked `org_id`; cross-org PUBLIC metadata leaked. | `B4-security-compliance.md:75-79`; `A1-F9` |
| Per-run scoped HMAC token | A compromised task can write only its own run's telemetry; `runId` mismatch + HMAC both block impersonation. | **Single global `APP_INTERNAL_API_KEY` (v1)** — any task could write any project's results (broken object-level auth). | `A1-F6`; `04-API-CONTRACTS.md:1014` |
| gRPC default-secure TLS+mTLS, identity = cert CN + enrollment token | One move encrypts the channel (HIPAA in-transit) and authenticates the peer (anti-Sybil); the code already existed unused. | **Plaintext-by-default + self-asserted `client_id` (v1)** — on-path read/forge + unbounded Sybil. | `B4-security-compliance.md:33-35, 59`; risk R6 |
| DP layer (DP-SGD on FedAvg, scalar-DP on DeComFL) + robust-mean/clip guard | Blunts membership/property inference and outlier/poisoning influence; required for any defensible HIPAA/GDPR claim. | **No DP / no robust aggregation (v1)** with a **false "Byzantine-robust" README claim** — delete the claim (it is a liability), implement the real, opt-in guard. | `B4-security-compliance.md:30, 62`; `README.md:106` |
| Market DeComFL scalar-only as DLG-eliminator, not as "private" | Structurally kills the highest-severity FL attack family; honest scoping protects the moat. | **Over-claim "private" or conflate "compressed"=="private"** — refuted by literature (HCGLA reconstructs from compressed gradients); membership inference survives. | `B4-security-compliance.md:45, 51` |
| SOC 2 Type 2 (Security + Confidentiality) + HIPAA-readiness now; defer FedRAMP | SOC 2 is table-stakes for B2B and builds ~70% of HIPAA's technical safeguards; the healthcare demo makes HIPAA the floor. | **FedRAMP now** ($800k–$2M, 12–24 mo) — pursue only behind a real federal contract; speculative pursuit burns seed runway. | `B4-security-compliance.md:113-120, 155` |
| `SameSite=Strict` + Origin check instead of CSRF tokens | The trio (HttpOnly cookie + `SameSite=Strict` + server-side Origin allowlist) closes CSRF for an SPA without the token-echo dance. | **`SameSite=Lax` + CSRF disabled (v1)** — top-level navigation POST rode the cookie. **Spring CSRF token** — redundant given Strict+Origin for a non-form SPA. | `B4-security-compliance.md:97` |

### 11.x HIPAA-readiness architecture (what the stack must provide)

| HIPAA Security Rule control | v2 architecture answer | Reference |
|---|---|---|
| Encryption in transit | gRPC TLS+mTLS default-secure (§6.3); HSTS + `cookie.secure=true` on the web side (§6.5). | `B4-security-compliance.md:130` |
| Encryption at rest | RDS + S3/MinIO encryption with **KMS customer-managed keys (CMK)**; secrets in Secrets Manager/SSM with KMS. | `B4-security-compliance.md:128-129` |
| HIPAA-eligible services only + AWS BAA | Confine ePHI to EC2/ECS/EKS, RDS, S3, KMS, CloudWatch (verify eligibility per Region); execute the AWS BAA before any real PHI. | `B4-security-compliance.md:128, 133` |
| Audit controls (§164.312(b)) | `audit_events` (append-only, JSONB metadata) with ≥6-year retention + monthly partitioning; `@Auditable` on all mutating/authz actions. | `B4-security-compliance.md:131`; `03-DATA-MODEL.md:438` |
| Access control + unique user id | The role enum (kills the dead-admin path), org-scoped authz, MFA on the platform. | `B4-security-compliance.md:132` |
| Data residency | Pin PHI-bearing RDS/S3/MinIO to in-scope US Regions. | `B4-security-compliance.md:134` |
| The FL premise as a HIPAA advantage | Raw PHI never leaves the client (the load-bearing invariant); the platform may never store ePHI if it only sees scalars — but this must be **architecturally guaranteed** (no patient paths in logs/error traces) and **contractually asserted**, not assumed. | `B4-security-compliance.md:124` |

### 11.y SOC 2 Type 2 controls checklist (write to `docs/v2/build/controls/soc2-hipaa-controls-checklist.md`)

| TSC | Control | This-codebase evidence | Status driver |
|---|---|---|---|
| CC6 (logical access) | Multi-tenant org enforcement | `TenantPredicate` + `requireOrgScope` (§6.1) | rebuild |
| CC6 | Session revocation | `tokenVersion` revocation (§5.2) | rebuild |
| CC6 | MFA on the platform | platform-admin + org-admin MFA | rebuild |
| CC6 | Secret rotation | Secrets Manager rotation runbook; per-run/enrollment tokens are short-lived | rebuild |
| CC7 (monitoring) | Audit coverage + retention | `@Auditable` on all mutations; `audit_events` retention/partitioning | refactor |
| CC7 | Alert on auth failures | wire `AuditingAuthenticationFailureHandler` to Micrometer metrics | refactor |
| CC8 (change mgmt) | PR-time CI gate | `ci.yml` + branch protection (required checks) | rebuild |
| CC8 | Vuln scans + SBOM | `security.yml`: gitleaks, pip-audit, dependency-check, CycloneDX | rebuild |
| Confidentiality | Encryption in transit | gRPC mTLS + HSTS + `cookie.secure` | rebuild/refactor |
| Confidentiality | No cross-tenant leak | org-scoped discovery + WS topic authz | rebuild |

Sources for the compliance posture (carry these URLs into the checklist): SOC 2 for startups — https://www.workstreet.com/blog/soc-2-for-startups ; SOC 2 Type 2 process/cost — https://sprinto.com/blog/soc-2-type-2/ ; AWS HIPAA eligible services/BAA — https://aws.amazon.com/compliance/hipaa-compliance/ ; HIPAA-on-AWS proposed-rule (encryption/MFA/pen-test) — https://www.exabeam.com/explainers/hipaa-compliance/hipaa-on-aws-requirements-and-best-practices/ ; FedRAMP cost/defer — https://www.vanta.com/collection/fedramp/fedramp-cost ; ZO-VFL scalar-only / scalar-DP — https://arxiv.org/html/2502.20565 ; Byzantine-robust limits — https://www.usenix.org/system/files/sec20summer_fang_prepub.pdf ; FL unlearning / GDPR Art.17 — https://arxiv.org/pdf/2411.17126 .

---

## 13. Build task checklist for the local model (ordered, dependency-first)

Each task is ~one file/feature with a clear done-condition. Execute in order.

1. **Role enums.** Create `security/PlatformRole.java`, `OrgRole.java`, `ProjectRole.java`. **Done:** the three enums compile with exactly the values in §5.4.
2. **`CustomUserDetailsService` authority.** Ensure it emits `ROLE_PLATFORM_ADMIN` from `PlatformRole`. **Done:** unit test shows authority string `ROLE_PLATFORM_ADMIN` for a `PLATFORM_ADMIN` user.
3. **`JwtTokenProvider` hardening.** Add `iss`/`aud`/`jti`/`tokenVersion`/clock-skew to `generateToken`/`parseAndValidate`; add `isRevoked`. **Done:** `JwtTokenProviderTest.rejects_wrong_issuer_audience` and `revokes_on_token_version_bump` pass.
4. **`SecurityConfig` headers + chain.** Add CSP/HSTS/nosniff/Referrer-Policy/Permissions-Policy; `frameOptions.deny`; wire the new filter order (§5.1). **Done:** `SecurityConfigHeadersTest` passes; chain compiles.
5. **`UserPrincipal` + `OrgScope` + `OrgScopeFilter`.** Populate `visibleOrgIds` from memberships (`ALL_ORGS` for platform admin). **Done:** an integration test shows `OrgScope` populated for both a member and a platform admin.
6. **`AuthorizationService` + `TenantPredicate`.** Implement `requireOrgScope/requireOrgMember/requireOrgAdmin/requireParticipant/visibleOrgIds` and the `org_id IN (:scope)` predicate. **Done:** `TenantIsolationTest.orgA_cannot_read_orgB_project` passes.
7. **`@PreAuthorize` alignment.** Replace every `hasRole('ADMIN')` with `hasRole('PLATFORM_ADMIN')` across admin/user controllers. **Done:** `AdminControllerIntegrationTest.platform_admin_reaches_admin_routes` passes seeding `PLATFORM_ADMIN` (and the old test seeding `"ADMIN"` is deleted).
8. **`RunTokenService` + `RunTokenFilter`.** Implement `mint`/`verify` (HMAC-SHA256, constant-time, §6.2.1); gate `/api/internal/**`. Delete `InternalApiKeyFilter` and `APP_INTERNAL_API_KEY`. **Done:** `RunTokenServiceTest` (mismatch=403, bad HMAC=401, terminal=409, constant-time) passes.
9. **`JwtChannelInterceptor` SUBSCRIBE authz.** Parse the destination, call `requireParticipant`; replace the anonymous `AuthenticationException` subclasses. **Done:** `WsSubscribeAuthzTest.non_participant_subscribe_rejected` passes.
10. **`RateLimitFilter`.** Bucket4j on `/api/auth/register|login|password/forgot`; emit `429 RATE_LIMITED`. **Done:** `RateLimitTest.auth_endpoint_429_after_capacity` passes.
11. **`@Auditable` coverage + `AuditAspect` Jackson serializer.** Annotate all mutating/authz actions incl. `PLATFORM_ADMIN_ORG_BYPASS`. **Done:** `AuditCoverageTest.mutating_actions_are_audited` passes; metadata serializes via Jackson into JSONB.
12. **`GlobalExceptionHandler` mappings.** Map every authn/authz exception to the §9 envelope codes. **Done:** each failure mode in §9 returns the listed code.
13. **PKI scripts.** Author `deploy/security/issue-server-cert.sh` / `issue-client-cert.sh` + the rotation README. **Done:** running them produces a CA, a server leaf (service-DNS SAN), and a client leaf (identity CN).
14. **`tls.py` + `GrpcTlsConfig`.** `build_server_credentials` (fail-closed outside dev) and the Python/Spring channel config. **Done:** `GrpcServerStartTest.refuses_plaintext_outside_dev` passes.
15. **`identity.py` + `EnrollmentTokenService`.** Extract peer CN, verify enrollment token, trust `peer_cn` not `client_id`. **Done:** `GrpcMtlsTest.rejects_client_without_cert` and `GrpcIdentityTest.self_asserted_client_id_not_trusted` pass.
16. **`dp.py`.** Implement `dpsgd_clip_and_noise` (FedAvg) and `scalar_dp` (DeComFL) with the CPU-canonical generator. **Done:** `DpTest.scalar_dp_clips_and_noises_deterministically` passes.
17. **`robust.py`.** Coordinate-wise trimmed-mean + clip + NaN/Inf reject. **Done:** `RobustTest.nan_inf_update_dropped` passes.
18. **Config + boot guards.** Add all §8 properties; make the base profile refuse to boot without the four required secrets; remove the false "Byzantine-robust" README claim. **Done:** boot fails fast without secrets; README claim is gone.
19. **CI security gates.** Add `security.yml` (gitleaks, pip-audit, dependency-check, CycloneDX SBOM) as required checks. **Done:** the gates run on PR and block merge on high/critical findings.
20. **Controls checklist doc.** Write `docs/v2/build/controls/soc2-hipaa-controls-checklist.md` from §11.x/§11.y. **Done:** the file exists with every CC6/CC7/CC8/Confidentiality row and its codebase evidence.

---

*End of 18-LLD-security-and-compliance.md. Existing-code citations refer to files under `/home/anurag/codebase/FedLearn-Platform/backend/fl-platform-api/` (verified during authoring: `SecurityConfig.java:114-128`, `JwtTokenProvider.java:41-46,54-57`, `InternalApiKeyFilter.java:69-72`). Audit citations refer to `/home/anurag/codebase/FedLearn-Platform/docs/audit/2026-05-29/`. External/compliance claims carry source URLs in §11.y; re-verify before relying on cost/timeline figures.*
