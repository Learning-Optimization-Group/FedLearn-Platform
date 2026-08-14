# 02 - Security and Authentication

The FedLearn backend implements a robust, multi-layered security architecture designed to handle both standard REST API clients (React) and internal Machine Learning servers (Python).

> ✅ **Branch reality.** The **role and org-scope material** — `PlatformRole` / `PLATFORM_ADMIN`, the three role layers, `OrgScope` / `OrgScopeFilter`, `organization_memberships`, and the `@Auditable` audit trail — is present on this branch (the `V4`–`V7` identity migrations; the coarse `users.role IN (USER, ADMIN)` column from `V2` has been superseded). See [06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md).
>
> Two mechanisms on this page have changed materially since the page was first written, and §1 and §3
> now describe their current form: **Bearer tokens are no longer a co-equal source** — the browser
> path is cookie-only and a Bearer header authenticates only for a marked native client (SE-9); and
> the internal callback gate takes **two** credentials, `X-Internal-Key` *and* a per-run
> `X-Internal-Run-Token` scoped to one project (SE-7). It also does not grant a `ROLE_INTERNAL`
> authority — it never establishes an `Authentication` at all.

## 1. REST API Security (JWT)

The platform uses **JSON Web Tokens (JWT)** for stateless user authentication. When a user logs in via `/api/auth/login`, the `AuthController` authenticates the credentials, mints a token with `JwtTokenProvider`, and sets it as an `HttpOnly` `jwtToken` cookie on `path=/`.

Its two policy attributes are per-profile, and `production` deliberately differs (SE-21):

| Profile | `app.auth.cookie.secure` | `app.auth.cookie.same-site` |
|---|---|---|
| base / `dev` | `false` | `Lax` |
| `ec2demo` | `false` (the demo runs plain HTTP) | `Lax` |
| `production` | `true` | `Strict` |

`maxAge` is **not** configured separately — it is derived from `app.jwt.expiration-ms` so the cookie
cannot outlive the JWT (a valid-looking cookie past the token's `exp` produces silent 401s, SE-8).
Note that `application.properties` still declares an `app.auth.cookie.max-age-seconds` key that
nothing reads; it is vestigial, not a knob.

The login **response body carries `accessToken` only for a native client** — one that sent the `X-FedLearn-Client` marker. A browser login response carries `username` / `email` / `role` and nothing else, so the SPA never holds a JS-readable token (SE-8). This is the same marker that gates Bearer acceptance below.

### Token Extraction Strategy — cookie-first, Bearer only for native clients (SE-9)

`JwtAuthenticationFilter` intercepts every request. The two sources are **not** interchangeable and the header is **not** checked first:

```java
// JwtAuthenticationFilter.doFilterInternal
String jwt = readJwtCookie(request);                 // 1. always honour the browser cookie
if (jwt == null && isNativeClient(request)) {        // 2. Bearer ONLY for a marked native client
    jwt = readBearerToken(request);
}
if (jwt == null) { filterChain.doFilter(request, response); return; }   // anonymous
```

`isNativeClient` is true only when the request carries a non-blank **`X-FedLearn-Client`** header
(`JwtAuthenticationFilter.NATIVE_CLIENT_HEADER`). The reasoning is that the two client worlds want
different storage: the browser SPA keeps the token in an `HttpOnly` cookie JS cannot read (defeating
XSS exfiltration) and never sends an `Authorization` header, while the mobile/desktop clients cannot
use that cookie and instead read the token from the login response body, stash it in platform secure
storage (Keychain / EncryptedSharedPreferences / Electron `safeStorage`) and replay it as
`Authorization: Bearer <jwt>`. Keeping Bearer behind an explicit marker is fail-closed: a
browser-origin request that presents a Bearer header *without* the marker is treated as anonymous —
the header is ignored, never authenticated from. The marker is an intent signal, not a secret; a
marked request still runs every check below.

Once a token is extracted, three further gates apply before an `Authentication` is set:

- **Signature, expiry and audience.** `JwtTokenProvider.validateToken` requires the audience claim
  `fedlearn-web` (`WEB_AUDIENCE`), deliberately distinct from the FL connection token's
  `fedlearn-fl-server` (`ConnectionTokenService.AUDIENCE`). This is **SE-20**: an FL token — or a
  legacy audience-less one — signed with the same HMAC key cannot be replayed against the web
  boundary. Preserve the audience check when touching either token type.
- **Revocation (SE-8).** `TokenRevocationService.isRevoked(jti)` is consulted, so logout actually
  ends a session rather than waiting out the token's expiry.
- **Account lifecycle.** `CustomUserDetailsService` throws `DisabledException` for a user who is not
  `ACTIVE` or is soft-deleted, before any principal is built — so a suspension takes effect on the
  very next request instead of surviving until token expiry. The filter catches it and treats the
  request as anonymous.

`GET /api/auth/me` is the SPA's silent 401 probe: the Axios interceptor swallows 401s on that one
endpoint, while a 401 anywhere else fires an `authError` event and redirects to `/login`.

### Login throttling (SE-4)

`LoginRateLimiter` is an in-memory sliding-window throttle on **failed** logins: 5 failures inside a
15-minute window lock a key, old failures age out so the lock is temporary, and a successful login
resets it. `AuthController` feeds it **two** keys per attempt — one per username and one per source
IP — so neither a single-account brute force nor a username-spraying source can hammer
`/api/auth/login` indefinitely. It is per-instance by construction; a shared store is the documented
hardening for a multi-replica deployment. The injected `Clock` is a test seam, so window expiry is
testable without sleeping.

---

## 2. WebSocket Security (STOMP)

WebSockets present a unique security challenge because the initial connection is an HTTP Upgrade request, but subsequent messages flow over a persistent, non-HTTP TCP channel.

We secure WebSockets at **three distinct layers** — the first two establish *who* the caller is, the
third decides *what they may subscribe to*:

### Layer 1: Handshake Interception (`JwtHandshakeInterceptor.java`)

This is the most critical defense. Before the WebSocket channel is even opened, the `JwtHandshakeInterceptor` validates the token on the initial HTTP upgrade request.

If the token is invalid or missing, the handshake is rejected with a `401 Unauthorized` status, completely preventing an unauthenticated client from opening a socket.

```java
// Inside JwtHandshakeInterceptor.java
@Override
public boolean beforeHandshake(ServerHttpRequest request, ServerHttpResponse response,
                               WebSocketHandler wsHandler, Map<String, Object> attributes) {
    String token = extractToken(request);
    if (token == null || !jwtTokenProvider.validateToken(token, userDetails)) {
        reject(response);
        return false; // Connection dropped before socket opens
    }
    
    // Store authenticated principal in attributes to pass to STOMP layer
    attributes.put(PRINCIPAL_ATTR, auth);
    return true; 
}
```

### Layer 2: STOMP Channel Interception (`JwtChannelInterceptor.java`)

Once the socket is open, STOMP clients send `CONNECT` frames. The `JwtChannelInterceptor` takes the `PRINCIPAL_ATTR` saved during the handshake and officially promotes it into the STOMP message header context, rejecting an unauthenticated `CONNECT`. This ensures that any `@SubscribeMapping` or message routing logic has access to the user's identity.

### Layer 3: Per-destination SUBSCRIBE authorization (`StompSubscriptionInterceptor`, BA-5)

Identity alone is not authorization. Without this third gate, **any** authenticated user could
`SUBSCRIBE` to any project's stream — across tenants — and receive its broadcasts, because layers 1
and 2 only prove who is asking. The interceptor inspects each `SUBSCRIBE` frame's destination against
the four project-scoped prefixes `WebSocketService` publishes to —
`/topic/logs/`, `/topic/status/`, `/topic/results/`, `/topic/inference/` — parses the trailing project
id, and runs **the same org-scope + participant check the REST read path enforces** before the frame
reaches the broker. Within `/topic` the rule is deny-by-default: an unrecognised `/topic/...`
destination is rejected rather than passed through, and any destination containing a `*`/`?` pattern
is refused outright (the simple broker matches SUBSCRIBE destinations as Ant patterns, so an ungated
`/topic/**` would otherwise receive every project's broadcasts). Only the other namespaces
(`/user/**` user-destinations, `/app/**` app destinations) pass through untouched.

Registration order in `WebSocketConfig.configureClientInboundChannel` is load-bearing:
`registration.interceptors(jwtChannelInterceptor, stompSubscriptionInterceptor)` — the subscription
gate must run second, once the session principal exists.

---

## 3. Machine-to-Machine Security (Internal API Key)

The Python Federated Learning server — spawned as a **local process** on the backend host via the `FlServerProcessRunner` seam, which shells out to the `fl-runtime/` scripts — needs to send training results and status updates back to the Spring Boot API. This local-process path is the only supported deployment architecture: managed-task (ECS/Fargate) orchestration was implemented once, removed along with the AWS SDK, and is deferred to `OP-12`; setting `ecs.cluster-name` today is rejected at boot by `FlOrchestrationModeValidator` in every profile.

**Problem:** The Python script is not a "User". It doesn't have a username/password, and generating temporary JWTs for the script introduces unnecessary complexity.

**Solution:** The `InternalApiKeyFilter` — **two** credentials, not one.

All internal reporting endpoints sit under `/api/internal/**`. `SecurityConfig` marks that prefix
`permitAll()` at the chain level *only because* `InternalApiKeyFilter` runs before
`UsernamePasswordAuthenticationFilter` and rejects anything unauthorised before Spring Security sees
it. The filter never establishes an `Authentication` and never grants a `ROLE_INTERNAL` authority —
it is a pure gate that either forwards the request or ends it.

Its checks, in order:

1. **Path scope.** Anything outside `/api/internal/` is forwarded untouched.
2. **Server configured.** An empty `app.internal.api-key` is a **401** with an ERROR log — fail-closed,
   so a misconfigured deployment cannot silently accept unauthenticated callbacks.
3. **Shared key.** The request must present **`X-Internal-Key`** (not `X-Internal-API-Key`), compared
   with `MessageDigest.isEqual` — constant time, so the key cannot be recovered by timing. Missing or
   mismatched → **401**.
4. **Per-run token (SE-7).** The shared key is only the outer gate. The request must also present
   **`X-Internal-Run-Token`**, which `RunTokenRegistry` resolves to a scope carrying the run's
   `projectId`. Every `/api/internal/**` route puts its target project id in the **5th path segment**
   (`/api/internal/{results|benchmarks}/{projectId}/…`, `/api/internal/projects/{projectId}/artifacts`,
   `/api/internal/runs/{projectId}/{runId}/verify-connection-token`), and the filter extracts it and
   compares. Unknown/absent token or a non-project path → **401**; a token scoped to a *different*
   project → **403**. So a leaked run token can only mutate its own project.

`FlServerManager` mints that run token fresh at each spawn and hands both credentials to the child
process through the environment (`FEDLEARN_INTERNAL_API_KEY`, `FEDLEARN_INTERNAL_RUN_TOKEN`) — see
[04 - Federated Orchestration](04_federated_orchestration.md). Stopping a run evicts its token from
the registry, and `StartupReconciler` rehydrates the registry from `runs.internal_token_hash` for
exactly the runs it re-adopts after a backend restart (`V16`), so a survivor's callbacks keep working
while a reaped run's token stays dead.

---

## 4. Authorization: Roles & Org-Scoped Isolation

Once a request is *authenticated* (sections 1–2), a second layer decides what it
is *allowed* to do. Authorization spans three independent role layers and an
organization-scoped tenant boundary. This section is a summary; the full
treatment lives in [06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md).

### The three role layers

| Layer | Column | Enum | Values |
|---|---|---|---|
| **Platform** | `users.platform_role` | `PlatformRole` | `USER`, `PROJECT_OWNER`, `PLATFORM_ADMIN` |
| **Organization** | `organization_memberships.org_role` | `OrgRole` | `OWNER`, `ADMIN`, `MEMBER` |
| **Project** | `project_memberships.role` | `MembershipRole` | `OWNER`, `MEMBER`, `CLIENT` |

`CustomUserDetailsService` maps the user's `PlatformRole` to a single Spring
authority via `PlatformRole.authority()` (`ROLE_USER` / `ROLE_PROJECT_OWNER` /
`ROLE_PLATFORM_ADMIN`), so admin-only routes gate with
`@PreAuthorize("hasRole('PLATFORM_ADMIN')")` (`@EnableMethodSecurity` is on in
`SecurityConfig`), and project creation gates on
`AuthorizationService.requireCanCreateProject()` —
`ROLE_PROJECT_OWNER` or platform admin. Authorities are reloaded from the database
on every request, so a role change takes effect immediately without re-login.
`PLATFORM_ADMIN` is the privileged tier and **bypasses org-scope checks**. The same
service also enforces a lifecycle gate: it throws `DisabledException` (→ 401) for
any user whose `status != ACTIVE` or who is soft-deleted, *before* the password
check.

`SecurityConfig` also carries two non-role rules worth knowing: `/actuator/**` is
`hasRole('PLATFORM_ADMIN')` (SE-5 — a plain user could otherwise `POST
/actuator/loggers` to flip log levels), with only `/actuator/health` left public for
load-balancer probes; and `/api/users/me/profile` is `permitAll()` at the chain
level so `ProfileController` can 401 anonymous callers itself (the chain's default
entry point would 403, which the SPA's interceptor does not treat as "log in
again").

> **Note (fixed):** `AuthorizationService.isAdmin` formerly checked the literal
> `"ROLE_ADMIN"` — a string that no longer existed after the role rename, so the
> check silently failed. It is now `isPlatformAdmin()` and checks
> `ROLE_PLATFORM_ADMIN`.

### Org-scoped multi-tenant isolation

The P0 data-isolation mechanism guarantees a caller only sees/mutates projects in
organizations they belong to:

- **`OrgScope`** — a `@RequestScope` bean holding `Set<UUID> visibleOrgIds` plus an
  `unrestricted` flag; `allows(orgId)` is the single decision point.
- **`OrgScopeFilter`** — an `OncePerRequestFilter` registered to run *after*
  `JwtAuthenticationFilter`. It loads the caller's org ids from
  `organization_memberships` into `OrgScope`; a platform admin is marked
  `unrestricted`. A user with **no** memberships falls back to the single
  bootstrap `DEFAULT_ORG_ID` — a transitional rule so the current single-org demo
  keeps returning projects.
- **`AuthorizationService.requireOrgScope(UUID)`** — used on **mutation** paths;
  throws **403** when the project's org is out of scope.
- **`AuthorizationService.isInOrgScope(UUID)`** — the boolean form used on **read**
  paths to instead return **404**, so cross-tenant project *existence* is not
  leaked.

These gates are applied uniformly across `ProjectService`, `MembershipService`,
`AccessRequestService`, and `ClientApiService`; list queries are pushed down
org-scoped (`ProjectRepository.findOwnedOrMemberOfInOrgs` /
`findDiscoverableInOrgs`).

---

## 5. The `@Auditable` Audit Trail

Security-relevant mutations emit an `audit_events` row through a declarative
annotation + AOP advice (no audit logic smeared across services):

- `@Auditable(action = ..., targetIdParam = ..., targetType = ...)` marks a method.
- `AuditAspect` is an `@Around` advice that runs the method **first** and persists
  the audit row only on success (a thrown exception writes no row). It writes in
  the **same transaction** as the audited mutation, so a rollback drops the audit
  row too — and clears the `AuditContext` thread-local on the exception path so
  staged metadata can't leak across pooled threads. Metadata is serialised to the
  JSONB `metadata` column via Jackson.

Login is audited *outside* the aspect (credential failures throw before any
`@Auditable` body runs): `AuditingAuthenticationSuccessHandler` (also stamps
`last_login_at`) and `AuditingAuthenticationFailureHandler` write
`USER_LOGIN_SUCCEEDED` / `USER_LOGIN_FAILED`. Full detail — including the
`AuditAction` vocabulary and the JSON-serialisation fix — is in
[06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md).
