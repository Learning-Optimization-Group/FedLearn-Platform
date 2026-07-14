# 06 - Identity, Multi-Tenancy & Audit

This document covers the identity subsystem layered on top of the original
single-user model: a **three-layer role model**, **organization-scoped
multi-tenant isolation**, an **audit trail**, and the **email + bootstrap**
plumbing that seeds the first administrator. It is the deepest part of the
backend's authorization story; the JWT/cookie/WebSocket mechanics that establish
*who* the caller is live in [02 - Security and Authentication](02_security_and_auth.md).

> ✅ **Branch reality (read this first).** This entire subsystem **IS present and
> committed on this branch.** Authorization layers the original single-user model:
> `users.id` is `BIGINT`, `projects.id` is `UUID`, and the **highest committed Flyway
> migration is `V19`**. The identity foundations landed in **`V4`–`V7`**
> (`V5__identity_foundations.sql`, `V6__identity_hardening.sql`,
> `V7__owner_role_and_approval_workflows.sql`): the three-layer platform/org/project
> role model, `organization_memberships` / `project_memberships`,
> `users.platform_role` / `PLATFORM_ADMIN`, `projects.org_id`, the `audit_events`
> table + `@Auditable` / `AuditAction` aspect, the `EmailService` stack, and the
> `APP_BOOTSTRAP_ADMIN_*` bootstrap **all exist here** (`PlatformRole`, `OrgRole`,
> `OrgScopeFilter`, `AuthorizationService`, and the membership/audit repositories are
> all under `com.federated.fl_platform_api`). The single coarse `users.role IN (USER,
> ADMIN)` column from `V2` was the original model and has since been superseded.
>
> The subsystem is enforced on the **backend** (the membership / admin /
> access-request / user-search / client endpoints) AND surfaced in the **frontend**:
> role-gated routes (`RoleRoute allow={['PLATFORM_ADMIN']}` / `['PROJECT_OWNER', …]`),
> the `AdminDashboard` / `OwnerDashboard` / `ClientDashboard`, and the
> owner-promotion / deletion-request approval flows. The clients ship the **Ember**
> design system.

---

## 1. The Three Role Layers

Authorization is split across three independent layers. Each is a per-user enum
persisted in its own table; they do **not** form a single inherited hierarchy.

| Layer | Column / Table | Enum | Values |
|---|---|---|---|
| **Platform** | `users.platform_role` | `PlatformRole` | `USER`, `PLATFORM_ADMIN` |
| **Organization** | `organization_memberships.org_role` | `OrgRole` | `OWNER`, `ADMIN`, `MEMBER` |
| **Project** | `project_memberships.role` | `MembershipRole` | `OWNER`, `MEMBER`, `CLIENT` |

### 1.1 Platform role

`PlatformRole` is the coarse, platform-wide role used for endpoint
authorization. It replaces the old single `users.role` column (renamed to
`platform_role` in the V5 migration). `PLATFORM_ADMIN` is the privileged tier:
it **bypasses organization-scope checks entirely**.

The enum owns its own Spring Security authority mapping:

```java
// PlatformRole.java
public enum PlatformRole {
    USER,
    PLATFORM_ADMIN;

    /** The Spring Security authority string (ROLE_USER / ROLE_PLATFORM_ADMIN). */
    public String authority() {
        return "ROLE_" + name();
    }
}
```

`CustomUserDetailsService` resolves the user's `PlatformRole` to a single
`SimpleGrantedAuthority(role.authority())`, so admin-only routes gate on it with
`@PreAuthorize("hasRole('PLATFORM_ADMIN')")` (see `AdminController`,
`TestEmailController`). `@EnableMethodSecurity` in `SecurityConfig` activates the
`@PreAuthorize` machinery.

> **Bug fixed this session.** `AuthorizationService` previously had an `isAdmin`
> helper that checked the literal authority `"ROLE_ADMIN"` — a string that no
> longer existed once the role was renamed, so every admin check silently
> failed. It was renamed to **`isPlatformAdmin()`** and now checks
> `ROLE_PLATFORM_ADMIN`. The V6 migration also normalizes any legacy
> `platform_role = 'ADMIN'` rows to `'PLATFORM_ADMIN'`.

### 1.2 Organization role

`OrgRole` (`OWNER`/`ADMIN`/`MEMBER`) is held per user per organization in
`organization_memberships`, keyed by the composite primary key
`(org_id UUID, user_id BIGINT)` modelled with `@IdClass(OrganizationMembershipId.class)`.
The enum exposes an ordinal-based `atLeast(...)` comparator for future
role-threshold checks. Today these rows primarily drive **org-scope resolution**
(§3) and ownership of the bootstrap org.

### 1.3 Project role

`MembershipRole` (`OWNER`/`MEMBER`/`CLIENT`) lives in `project_memberships`,
keyed by the embeddable composite key `ProjectMembershipId(projectId UUID, userId BIGINT)`.
Two additional concepts hang off project membership:

- **`JoinedVia`** — provenance of the membership row:
  `OWNER_ADD`, `PUBLIC_JOIN`, `REQUEST_APPROVED`, `OWNER_SELF`.
- **`ProjectVisibility`** — `PRIVATE` (default) or `PUBLIC`, on `projects.visibility`.
- **`ProjectAccessRequest`** — the join-request workflow for PRIVATE projects
  (`AccessRequestStatus` = `PENDING`/`APPROVED`/`DENIED`), see §5.

A subtlety: a project's `OWNER` is decided by `projects.user_id`, **not** by a
membership row. `role = OWNER` rows are created lazily (`JoinedVia.OWNER_SELF`)
only to carry the owner's `partition_id` when they self-connect as a client, and
permission logic ignores them. The API hides `OWNER` rows from membership
listings.

---

## 2. New Entities

All of the following live in `model/`:

| Entity | PK | Notable columns |
|---|---|---|
| `Organization` | `id UUID` | `name`, unique `slug`, soft-delete `deleted_at`, `created_at`/`updated_at` |
| `OrganizationMembership` (+ `OrganizationMembershipId`) | `(org_id, user_id)` | `org_role`, `created_at`; `@Check` on `org_role` |
| `ProjectMembership` (+ `ProjectMembershipId`) | `(project_id, user_id)` | `role`, `partition_id`, `joined_via`, `added_by`, `added_at` |
| `ProjectAccessRequest` | `id BIGSERIAL` | `requested_role`, `status`, `message`, `requested_at`, `decided_at`, `decided_by`; `UNIQUE(project_id, user_id)` |
| `AuditEvent` | `id UUID` | `action`, `actor_user_id`, `org_id`, `target_type`/`target_id`, JSONB `metadata`, `request_ip`, `user_agent` |

The two pre-existing core entities gained columns:

### `User` (extended)

The single `role` column became `platformRole` (a `PlatformRole` enum), plus a
lifecycle/profile column set:

```java
@Enumerated(EnumType.STRING)
@Column(name = "platform_role", nullable = false, length = 32)
private PlatformRole platformRole = PlatformRole.USER;

@Enumerated(EnumType.STRING)
@Column(nullable = false, length = 16)
private UserStatus status = UserStatus.ACTIVE;   // PENDING | ACTIVE | SUSPENDED

private Instant deletedAt;        // soft-delete tombstone
private Boolean emailVerified = false;
private String  displayName;
private String  avatarUrl;
private Instant lastLoginAt;     // stamped on successful login
```

`UserStatus` gates authentication: `CustomUserDetailsService` throws
`DisabledException` (→ 401) for any user whose `status != ACTIVE` or whose
`deletedAt` is set, *before* the password check, so a suspended/deleted account
cannot log in and account existence is not leaked.

### `Project` (extended)

```java
@Column(name = "org_id", nullable = false)
private UUID orgId;                              // V5 pinned every project to an org

@Enumerated(EnumType.STRING)
@Column(nullable = false, length = 32)
private ProjectVisibility visibility = ProjectVisibility.PRIVATE;

// Model-Hub columns (publish toggle)
private boolean modelPublished = false;
private String  modelDescription;
private String  modelTags;
private Instant modelPublishedAt;
```

`org_id` is `NOT NULL` (see §6). New projects are pinned to the owner's first org
membership, falling back to the single Default org for membership-less users.

---

## 3. Organization-Scoped Multi-Tenant Isolation

This is the **P0 data-isolation mechanism**: it guarantees that a caller can only
see and mutate projects belonging to organizations they are a member of. It is a
gate layered *on top of* the existing ownership/membership checks — both must
pass.

### 3.1 `OrgScope` — the request-scoped scope holder

A `@RequestScope` bean holding the set of org ids the caller may see, plus an
`unrestricted` flag for platform admins:

```java
@Component
@RequestScope
public class OrgScope {
    private Set<UUID> visibleOrgIds = Collections.emptySet();
    private boolean unrestricted = false;

    public void set(Set<UUID> visibleOrgIds, boolean unrestricted) { ... }

    /** True if the caller may access the given org (or is unrestricted). */
    public boolean allows(UUID orgId) {
        return unrestricted || visibleOrgIds.contains(orgId);
    }
}
```

### 3.2 `OrgScopeFilter` — populating the scope

`OrgScopeFilter` is an `OncePerRequestFilter` wired in `SecurityConfig` to run
**after** `JwtAuthenticationFilter` (`.addFilterAfter(orgScopeFilter, JwtAuthenticationFilter.class)`),
so the `SecurityContext` is already populated when it resolves memberships. For
each authenticated request it sets the scope:

- **Platform admin** → `orgScope.set(Set.of(), true)` (unrestricted; sees every org).
- **Regular user** → the user's org ids loaded from `organization_memberships`.

```java
private Set<UUID> resolveVisibleOrgIds(User user) {
    Set<UUID> orgIds = orgMembershipRepository.findByUserId(user.getId()).stream()
            .map(OrganizationMembership::getOrgId)
            .collect(Collectors.toCollection(HashSet::new));
    if (orgIds.isEmpty()) {
        orgIds.add(ProjectService.DEFAULT_ORG_ID);   // transitional single-org fallback
    }
    return orgIds;
}
```

**The transitional fallback is load-bearing.** Regular registrations are *not*
auto-added to `organization_memberships` — only the bootstrap admin is seeded. A
naive `org_id IN (visibleOrgIds)` filter would therefore make `visibleOrgIds`
empty for normal users and return **zero** projects on their dashboard. So a user
with no memberships falls back to the single bootstrap `DEFAULT_ORG_ID`
(`00000000-0000-0000-0000-000000000001`, seeded by V5). This preserves today's
single-org demo behaviour while making multi-org isolation real the moment
explicit memberships exist. `OrgScopeFilter` and `ProjectService` share the exact
same constant so there is no duplicate UUID literal.

### 3.3 The enforcement gates — `requireOrgScope` / `isInOrgScope`

`AuthorizationService` exposes two forms, chosen by the calling path:

```java
/** Mutation paths: throw 403 if the project's org is outside the caller's scope. */
public void requireOrgScope(UUID orgId) {
    if (orgScope != null && orgScope.allows(orgId)) return;
    throw new AccessDeniedException("Project is outside your organization scope");
}

/** Read/list paths: boolean form, used to translate out-of-scope into a 404. */
public boolean isInOrgScope(UUID orgId) {
    return orgScope != null && orgScope.allows(orgId);
}
```

The distinction is deliberate: **mutations** (`start`, `stop`, `delete`, add/remove
member, submit/decide request, client connect) call `requireOrgScope(...)` and
fail with a hard **403**. **Pure reads** (`getProject`, list memberships, list
access-requests) use `isInOrgScope(...)` / `orgScope.allows(...)` and instead throw
**404**, so cross-tenant project *existence* is never leaked.

These gates are applied consistently across `ProjectService` **and** the sibling
services `MembershipService`, `AccessRequestService`, and `ClientApiService`. The
list endpoints don't just filter post-hoc — they push the scope into the query.
`ProjectRepository` carries org-scoped variants:

```java
List<Project> findOwnedOrMemberOfInOrgs(Long userId, Collection<UUID> orgIds);
List<Project> findDiscoverableInOrgs(Long userId, ProjectVisibility visibility,
                                     Collection<UUID> orgIds);
```

Unrestricted callers (platform admins) take the unscoped `findOwnedOrMemberOf` /
`findDiscoverable` queries; everyone else takes the `...InOrgs` variants bound to
`orgScope.visibleOrgIds()`.

---

## 4. The Audit Trail

Every security-relevant mutation can emit an `audit_events` row. The mechanism is
a declarative annotation plus an AOP advice, so audit logic is not smeared across
the services.

### 4.1 `@Auditable` + `AuditAspect`

`@Auditable` marks a method as audit-emitting:

```java
@Auditable(action = AuditAction.PROJECT_CREATED, targetType = "PROJECT")
public ProjectResponseDto createProject(CreateProjectRequest request) { ... }

@Auditable(action = AuditAction.RUN_STARTED, targetIdParam = "projectId", targetType = "PROJECT")
public ProjectResponseDto startServerForProject(UUID projectId, StartProject request) { ... }
```

`AuditAspect` is an `@Around` advice on `@annotation(...Auditable)`. Its key
semantics:

- **After-success only.** It runs `pjp.proceed()` *first* and writes the audit row
  only if the method returns normally. A thrown exception writes **no** row.
- **Same transaction as the mutation.** When the audited method is
  `@Transactional`, the audit insert joins that transaction — a rolled-back
  mutation rolls back its audit row too (no orphan audit entries).
- **Metadata via JSON.** Methods can stash key/value metadata on
  `AuditContext` (a `ThreadLocal` sidecar); the aspect *drains* it after
  `proceed()` and serialises it for the JSONB `metadata` column.
- **ThreadLocal hygiene.** On the exception path the aspect calls
  `AuditContext.drain()` to clear any staged metadata so it can't leak onto the
  next request that reuses the pooled servlet thread.

```java
// AuditAspect.record(...)
try {
    result = pjp.proceed();         // run first; audit only after success
} catch (Throwable t) {
    AuditContext.drain();           // clear staged metadata, write no row
    throw t;
}
// ... resolve targetId, @CurrentOrg orgId, actor, ip, user-agent ...
String meta = serialise(AuditContext.drain());   // Jackson → valid JSON
repo.save(AuditEvent.builder()...build());
```

> **Bug fixed this session.** Metadata was previously serialised with hand-rolled
> string concatenation, which could emit **invalid JSON** for values containing
> quotes, backslashes, newlines, or control characters. Because `metadata` is a
> JSONB column (V6), an invalid string would fail the insert and — since the
> aspect writes in the *same transaction* as the audited mutation — roll the
> mutation back. It now serialises through **Jackson** (`ObjectMapper.writeValueAsString`),
> with a `null`-metadata fallback (and a warning log) on the unlikely
> serialisation failure, so the audit row still persists and the business mutation
> is never sacrificed for an unserialisable blob.

The actor id is resolved from the `SecurityContext` principal
(`UserDetails.getUsername()` → `User.id`); `request_ip` and `user_agent` come from
the current `HttpServletRequest`. A method parameter annotated `@CurrentOrg`
supplies `org_id`.

### 4.2 `AuditEvent` and `AuditAction`

`AuditEvent` columns: `id UUID`, `occurred_at`, `actor_user_id BIGINT` (nullable —
system/unauthenticated events), `org_id UUID`, `action` (enum), `target_type`/
`target_id`, JSONB `metadata`, `request_ip`, `user_agent`.

`AuditAction` is the enum vocabulary. The values actually instrumented today
include: `USER_REGISTERED`, `USER_LOGIN_SUCCEEDED`, `USER_LOGIN_FAILED`,
`USER_LOGGED_OUT`, `BOOTSTRAP_ADMIN_CREATED`, `BOOTSTRAP_ORG_CREATED`,
`PROJECT_CREATED`, `PROJECT_DELETED`, `RUN_STARTED`, `RUN_STOPPED`,
`PROJECT_MEMBER_ADDED`, `PROJECT_MEMBER_REMOVED`, `PLATFORM_ADMIN_ORG_BYPASS`. The
`ORG_*` and remaining `USER_*` values are reserved in the enum for instrumentation
in later sub-specs.

### 4.3 Login auditing (outside the aspect)

Login can't ride the aspect because credential failures throw before any
`@Auditable` body runs. Instead two plain collaborators do it:

- `AuditingAuthenticationSuccessHandler.onSuccess(...)` — called by `AuthController`
  after a successful authentication: stamps `users.last_login_at` and writes a
  `USER_LOGIN_SUCCEEDED` row.
- `AuditingAuthenticationFailureHandler.onFailure(...)` — called from the
  `catch (AuthenticationException)` block: writes a `USER_LOGIN_FAILED` row keyed
  by the submitted *username only* (never a user id — avoids a
  existence/timing oracle).

`USER_REGISTERED` and `USER_LOGGED_OUT` *do* ride the aspect, via `@Auditable` on
the `register` and `logout` controller methods. The `ACTIVE`-status gate that
rejects non-active users at login lives in `CustomUserDetailsService` (§2).

---

## 5. Membership & Access-Request Workflow

Three sibling services implement the project-participation lifecycle, all
org-scope-gated:

- **`MembershipService`** (`add` / `remove` / `list`) — owners and platform
  admins may add/remove any role; a project `MEMBER` may add/remove only `CLIENT`.
  `OWNER` memberships are not user-creatable. `add`/`remove` are `@Auditable`
  (`PROJECT_MEMBER_ADDED` / `PROJECT_MEMBER_REMOVED`).
- **`AccessRequestService`** (`submit` / `listForProject` / `decide` / `listMine`) —
  joining a **PUBLIC** project creates a `CLIENT` membership immediately
  (`JoinedVia.PUBLIC_JOIN`); requesting a **PRIVATE** project upserts a `PENDING`
  `ProjectAccessRequest` and notifies the owner/members. `decide` approves
  (→ `CLIENT` membership, `JoinedVia.REQUEST_APPROVED`) or denies.
- **`ClientApiService`** (`listForCurrentUser` / `getConnection`) — the FL-client
  view. `getConnection` takes a pessimistic write lock on the project row
  (`ProjectRepository.lockById`) and assigns a **sticky `partition_id`** to the
  membership, serialising concurrent client connections so partitions never
  collide.

---

## 6. Flyway Migrations (V4–V6)

The schema for this subsystem lands across three migrations
(`src/main/resources/db/migration/`). The base profile runs JPA in `validate`
mode, so these migrations are the source of truth — except the `test` profile,
which disables Flyway and builds from JPA `create-drop` on H2 (so V6's
Postgres-only DDL never runs there).

| Migration | Adds |
|---|---|
| **V4** `__project_membership_and_model_hub.sql` | `projects.visibility` (default `PRIVATE`), Model-Hub columns; `project_memberships` and `project_access_requests` tables with their indexes. One `ALTER` per column (H2 rejects multi-clause `ALTER TABLE`). |
| **V5** `__identity_foundations.sql` | `organizations`, `organization_memberships`; renames `users.role` → `platform_role` and adds the lifecycle/profile columns; `audit_events` (metadata initially `CLOB`); `projects.org_id`. Backfills a single **Default** org (`...0001`), enrolls every existing user, marks project owners as org `OWNER`, then sets `projects.org_id NOT NULL`. |
| **V6** `__identity_hardening.sql` (Postgres-only) | Normalises legacy `platform_role = 'ADMIN'` → `'PLATFORM_ADMIN'`; adds a `CHECK (platform_role IN ('USER','PLATFORM_ADMIN'))`; promotes `audit_events.metadata` from `CLOB` to native **JSONB**. |

```sql
-- V5: rename the single role column and pin projects to an org
ALTER TABLE users ALTER COLUMN role RENAME TO platform_role;
ALTER TABLE projects ADD COLUMN org_id UUID REFERENCES organizations(id);
-- ... backfill Default org, enroll users ...
ALTER TABLE projects ALTER COLUMN org_id SET NOT NULL;

-- V6: harden the role column and upgrade metadata to JSONB
UPDATE users SET platform_role = 'PLATFORM_ADMIN' WHERE platform_role = 'ADMIN';
ALTER TABLE users ADD CONSTRAINT chk_users_platform_role
    CHECK (platform_role IN ('USER','PLATFORM_ADMIN'));
ALTER TABLE audit_events ALTER COLUMN metadata TYPE JSONB USING (NULLIF(metadata,'')::jsonb);
```

---

## 7. Email & First-Run Bootstrap

### 7.1 Email

A small pluggable email layer (`email/`):

- `EmailService` — the interface (`send(EmailMessage)`).
- `LoggingEmailService` — **dev** adapter: logs the message and writes a
  `.eml` file under `target/sent-emails/`; never throws.
- `SmtpEmailService` — **prod** adapter: sends a multipart MIME message via
  Spring's `JavaMailSender`; wraps transport failures in `EmailDeliveryException`.

`EmailConfig` selects the bean by `app.email.provider`: `smtp` activates
`SmtpEmailService`, otherwise the `LoggingEmailService` is registered as the
fallback (`@ConditionalOnMissingBean`).

`TestEmailController` exposes `POST /api/admin/test-email?to=...` for a delivery
smoke test. It is doubly guarded: flag-gated by
`app.email.test-endpoint.enabled=true` (the bean — and thus the route — does not
exist otherwise) and `@PreAuthorize("hasRole('PLATFORM_ADMIN')")`.

### 7.2 Bootstrap

`BootstrapRunner` is an `ApplicationRunner` (`@Profile("!test")`) that idempotently
seeds the **first** `PLATFORM_ADMIN` user and a default `Organization` at startup,
driven by `app.bootstrap.*` env vars (`APP_BOOTSTRAP_ADMIN_EMAIL`,
`APP_BOOTSTRAP_ADMIN_USERNAME`, `APP_BOOTSTRAP_ADMIN_PASSWORD`,
`APP_BOOTSTRAP_PLATFORM_ORG_NAME`). Behaviour:

- No `admin-email` configured → no-op.
- A `PLATFORM_ADMIN` already exists → no-op.
- Otherwise → find-or-create the Platform org by slug, create the admin
  (`status=ACTIVE`, `emailVerified=true`), add an `OWNER`
  `OrganizationMembership`, and emit `BOOTSTRAP_ORG_CREATED` +
  `BOOTSTRAP_ADMIN_CREATED` audit rows (actor `null` — system).

Password resolution fails fast: an explicit `admin-password` is used verbatim;
otherwise a random one is generated and WARN-logged **only** under the `dev`
profile; in any non-dev profile a missing password throws at startup.

---

## 8. New REST Endpoints (summary)

All of these are live on the backend but **not yet surfaced in any client UI**
(see the scope note at the top).

| Method & path | Controller | Purpose |
|---|---|---|
| `GET/POST /api/projects/{id}/memberships`, `DELETE .../{userId}` | `MembershipController` | List / add / remove project members |
| `POST/GET /api/projects/{id}/access-requests`, `PUT .../{requestId}` | `AccessRequestController` | Submit / list / decide join requests |
| `GET /api/my/access-requests` | `MyRequestsController` | The caller's own outstanding requests |
| `GET /api/projects/discover`, `GET /api/projects/{id}`, `PATCH /api/projects/{id}` | `ProjectController` | Discover feed, single-project read (404-on-out-of-scope), update name/description/visibility |
| `GET /api/admin/users`, `GET /api/admin/users/{id}`, `PUT /api/admin/users/{id}/role`, `GET /api/admin/projects` | `AdminController` (`@PreAuthorize PLATFORM_ADMIN`) | Platform admin console (via `AdminService`); role update refuses to demote the last admin |
| `GET /api/users/search?q=` | `UserSearchController` | Username prefix search; per-caller rate-limited (`UserSearchService`) |
| `GET /api/client/projects`, `GET /api/client/projects/{id}/connection` | `ClientApiController` (`ClientApiService`) | FL-client project list + sticky partition assignment |
| `POST /api/admin/test-email?to=` | `TestEmailController` (flag-gated, `PLATFORM_ADMIN`) | Email delivery smoke test |
