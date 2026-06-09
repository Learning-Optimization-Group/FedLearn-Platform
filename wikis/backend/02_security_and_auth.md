# 02 - Security and Authentication

The FedLearn backend implements a robust, multi-layered security architecture designed to handle both standard REST API clients (React) and internal Machine Learning servers (Python).

## 1. REST API Security (JWT)

The platform uses **JSON Web Tokens (JWT)** for stateless user authentication. When a user logs in via `/api/auth/login`, the `AuthController` generates a token using `JwtTokenProvider` and returns it to the client.

### Token Extraction Strategy

The `JwtAuthenticationFilter` intercepts every incoming HTTP request and attempts to extract the token from two potential sources in priority order:

```java
String authHeader = request.getHeader("Authorization");
String jwt = null;

// 1. Check Authorization Header (used by programmatic clients)
if (authHeader != null && authHeader.startsWith("Bearer ")) {
    jwt = authHeader.substring(7);
} 
// 2. Check HttpOnly Cookies (used securely by the browser)
else if (request.getCookies() != null) {
    for (jakarta.servlet.http.Cookie cookie : request.getCookies()) {
        if ("jwtToken".equals(cookie.getName())) {
            jwt = cookie.getValue();
            break;
        }
    }
}
```

By supporting both mechanisms, the backend is highly flexible while remaining secure against Cross-Site Scripting (XSS) when using `HttpOnly` cookies on the web.

---

## 2. WebSocket Security (STOMP)

WebSockets present a unique security challenge because the initial connection is an HTTP Upgrade request, but subsequent messages flow over a persistent, non-HTTP TCP channel.

We secure WebSockets at **two distinct layers**:

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

Once the socket is open, STOMP clients send `CONNECT` frames. The `JwtChannelInterceptor` takes the `PRINCIPAL_ATTR` saved during the handshake and officially promotes it into the STOMP message header context. This ensures that any `@SubscribeMapping` or message routing logic has access to the user's identity.

---

## 3. Machine-to-Machine Security (Internal API Key)

The Python Federated Learning server (spawned by AWS ECS or `ProcessBuilder`) needs to send training results and status updates back to the Spring Boot API.

**Problem:** The Python script is not a "User". It doesn't have a username/password, and generating temporary JWTs for the script introduces unnecessary complexity.

**Solution:** The `InternalApiKeyFilter`.

All internal reporting endpoints are placed under `/api/internal/**`. The `SecurityConfig` allows these endpoints to bypass the standard JWT checks. Instead, they are intercepted by the `InternalApiKeyFilter`.

```java
// Inside InternalApiKeyFilter.java
@Override
protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response,
                                FilterChain filterChain) throws ServletException, IOException {
    
    String requestApiKey = request.getHeader("X-Internal-API-Key");

    if (internalApiKey.equals(requestApiKey)) {
        // Authenticate the request as an internal service
        UsernamePasswordAuthenticationToken authentication =
                new UsernamePasswordAuthenticationToken("internal-service", null,
                        Collections.singletonList(new SimpleGrantedAuthority("ROLE_INTERNAL")));
        SecurityContextHolder.getContext().setAuthentication(authentication);
    } else {
        response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        return;
    }
    
    filterChain.doFilter(request, response);
}
```

The Spring Boot backend securely passes this `FEDLEARN_INTERNAL_API_KEY` to the Python server via environment variables during the orchestration phase (see `FlowerServerManager`), ensuring that only backend-spawned ML processes can hit the internal endpoints.

---

## 4. Authorization: Roles & Org-Scoped Isolation

Once a request is *authenticated* (sections 1–2), a second layer decides what it
is *allowed* to do. Authorization spans three independent role layers and an
organization-scoped tenant boundary. This section is a summary; the full
treatment lives in [06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md).

### The three role layers

| Layer | Column | Enum | Values |
|---|---|---|---|
| **Platform** | `users.platform_role` | `PlatformRole` | `USER`, `PLATFORM_ADMIN` |
| **Organization** | `organization_memberships.org_role` | `OrgRole` | `OWNER`, `ADMIN`, `MEMBER` |
| **Project** | `project_memberships.role` | `MembershipRole` | `OWNER`, `MEMBER`, `CLIENT` |

`CustomUserDetailsService` maps the user's `PlatformRole` to a single Spring
authority via `PlatformRole.authority()` (`ROLE_USER` / `ROLE_PLATFORM_ADMIN`), so
admin-only routes gate with `@PreAuthorize("hasRole('PLATFORM_ADMIN')")`
(`@EnableMethodSecurity` is on in `SecurityConfig`). `PLATFORM_ADMIN` is the
privileged tier and **bypasses org-scope checks**. The same service also enforces
a lifecycle gate: it throws `DisabledException` (→ 401) for any user whose
`status != ACTIVE` or who is soft-deleted, *before* the password check.

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
