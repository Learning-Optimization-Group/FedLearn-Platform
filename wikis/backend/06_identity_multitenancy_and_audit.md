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
> migration is `V23`**. The identity foundations landed in **`V4`–`V7`**
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
> owner-promotion / deletion-request approval flows. The clients ship the **Ledger**
> design system — navy structural ink on quiet paper surfaces, light-first (canvas
> `#F6F3EE`, surface `#FFFFFF`, ink `#191A1C`, accent `#1C314D`), generated from
> `design/tokens.json` by `design/build-tokens.mjs`. *Ledger superseded Ember, which
> superseded Instrument*: if you find the burnt-orange-on-warm-paper Ember palette in
> a doc or a token set, it is two cycles stale.

---

## 1. The Three Role Layers

Authorization is split across three independent layers. Each is a per-user enum
persisted in its own table; they do **not** form a single inherited hierarchy.

| Layer | Column / Table | Enum | Values |
|---|---|---|---|
| **Platform** | `users.platform_role` | `PlatformRole` | `USER`, `PROJECT_OWNER`, `PLATFORM_ADMIN` |
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
    /** Default tier. May join/train projects (as a CLIENT) but may not create them. */
    USER,
    /**
     * May create and own projects (admin-granted via the owner-promotion workflow).
     * Per-project ownership of a specific project is still tracked by
     * projects.user_id; this role only gates the capability to create one.
     */
    PROJECT_OWNER,
    /** Platform administrator. Unrestricted across orgs; approves owner/deletion requests. */
    PLATFORM_ADMIN;

    /** The Spring Security authority string (ROLE_USER / ROLE_PLATFORM_ADMIN). */
    public String authority() {
        return "ROLE_" + name();
    }
}
```

`PROJECT_OWNER` was added by the **V7** migration, which drops and re-adds
`chk_users_platform_role` to widen the column's CHECK domain to all three values.
`AuthorizationService.canCreateProjects()` gates project creation on
`hasAuthority("ROLE_PROJECT_OWNER") || isPlatformAdmin()`.

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
- **`ProjectVisibility`** — three tiers on `projects.visibility`: `PUBLIC`
  (discoverable, auto-join), `RESTRICTED` (discoverable, owner-approved request),
  `PRIVATE` (hidden, invite-only; the column default). `RESTRICTED` needed no
  schema change — `projects.visibility` is a plain `VARCHAR(32)` with no CHECK
  constraint, so the value set is owned entirely by the enum (see V7).
- **`ProjectAccessRequest`** — the join-request workflow (`AccessRequestStatus` =
  `PENDING`/`APPROVED`/`DENIED`), see §5.

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
| `OwnerPromotionRequest` (V7) | `id BIGSERIAL` | `user`, `status` (`AccessRequestStatus`), `message`, `requested_at`, `decided_at`, `decided_by` — a `USER` asking to become a `PROJECT_OWNER` |
| `ProjectDeletionRequest` (V7) | `id BIGSERIAL` | `project`, `requested_by`, `status` (`AccessRequestStatus`), `reason`, `requested_at`, `decided_at`, `decided_by` — an owner asking an admin to delete a project |
| `AuditEvent` | `id UUID` | `action`, `actor_user_id`, `org_id`, `target_type`/`target_id`, JSONB `metadata`, `request_ip`, `user_agent` |

All three request types share the one `AccessRequestStatus` vocabulary
(`PENDING`/`APPROVED`/`DENIED`), which is why an admin decision endpoint for any of
them takes the same `DecideAccessRequestRequest` body.

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

@Enumerated(EnumType.STRING)                     // V22/V23: which parameters a run federates
@Column(name = "training_arm", nullable = false, length = 32)
private TrainingArm trainingArm = TrainingArm.FULL;

// Model-Hub columns (publish toggle)
private boolean modelPublished = false;
private String  modelDescription;
private String  modelTags;
private Instant modelPublishedAt;
```

`org_id` is `NOT NULL` (see §6). New projects are pinned to the owner's first org
membership, falling back to the single Default org for membership-less users. The
`trainingArm` default is the mechanism by which every pre-`V22` project keeps its
exact behaviour — see [03 - Project Management Lifecycle](03_project_management.md).

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

`AuditAction` is the enum vocabulary, and it is deliberately wider than what is
wired. The values **actually emitted today** — verified by walking every `@Auditable`
annotation and every direct `AuditEvent.builder()` call in `src/main/java` — are:

| Emitted via | Actions |
|---|---|
| `@Auditable` + `AuditAspect` | `USER_REGISTERED`, `USER_LOGGED_OUT` (`AuthController`); `PROJECT_CREATED`, `PROJECT_DELETED`, `RUN_STARTED`, `RUN_STOPPED` (`ProjectService`); `PROJECT_MEMBER_ADDED`, `PROJECT_MEMBER_REMOVED` (`MembershipService`); `USER_SUSPENDED`, `USER_REACTIVATED` (`AdminService`); `USER_PROFILE_UPDATED` (`ProfileService`) |
| Written directly (outside the aspect) | `USER_LOGIN_SUCCEEDED`, `USER_LOGIN_FAILED` (the auditing auth handlers); `BOOTSTRAP_ORG_CREATED`, `BOOTSTRAP_ADMIN_CREATED` (`BootstrapRunner`); `USER_PASSWORD_CHANGED` (`ProfileService`) |

Everything else in the enum is **reserved, not wired** — the whole `ORG_*` block,
`USER_EMAIL_VERIFIED`, `USER_DELETED`, `USER_PLATFORM_ROLE_CHANGED`, and
`PLATFORM_ADMIN_ORG_BYPASS`. That last one is worth calling out because earlier
revisions of this page listed it as instrumented: it is declared and never emitted
anywhere in the source, so **an admin's org-scope bypass currently leaves no audit
row**. Treat it as a known gap rather than a working control.

Note also that `AdminService.searchAuditEvents` parses an `action` query parameter
with `AuditAction.valueOf(...)`, so the admin explorer can filter on any name in the
enum — including the reserved ones, which will simply match nothing.

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
  `submit` rejects the owner (**400**) and an existing `MEMBER`/`CLIENT` (**409**)
  first, then branches on the visibility tier:
  - **PUBLIC** → a `CLIENT` membership immediately (`JoinedVia.PUBLIC_JOIN`).
  - **RESTRICTED** → upsert a `PENDING` `ProjectAccessRequest` (explicitly resetting
    `status`/`requestedAt`/`decidedAt`/`decidedBy`, because `@PrePersist` does not
    fire on an UPDATE) and notify the owner and members.
  - **PRIVATE** → the **same 404 a nonexistent project id would produce**. Not a 403:
    a distinct 403 here would be an existence oracle (404 for a missing id vs 403 for
    a real private one), and PRIVATE is hidden as well as invite-only. The owner adds
    participants directly through `MembershipController`. (The owner and
    existing-participant branches have already returned by this point, so only
    outsiders reach it.)

  `decide` approves (→ `CLIENT` membership, `JoinedVia.REQUEST_APPROVED`) or denies.
- **`ClientApiService`** (`listForCurrentUser` / `getOne` / `join` / `getConnection`)
  — the FL-client view. `join` is the desktop/mobile counterpart of `submit`: PUBLIC
  joins outright, RESTRICTED is refused with "request access from the web app", and
  PRIVATE 404s. `getConnection` requires an **active run** (otherwise
  `ProjectStateException`) and delegates to `RunService.enroll(activeRunId)`, which is
  where partition assignment now lives:
  - it takes a pessimistic write lock on the **run** row (`RunRepository.lockById`),
    so concurrent enrollments serialise;
  - it re-checks org scope and owner-or-`CLIENT` membership, and that the run is
    `RUNNING` with a port;
  - an existing enrollment is reused; otherwise the next `partitionId` is
    `max + 1`, refused with "Run is full (K=…)" once a `SHARDED` run reaches
    `clientsPerRound`. `UNIQUE(run_id, partition_id)` (`V8`) makes the invariant the
    database's, not the service's;
  - it mints a **connection token sized to the whole run** (`ttlForRun(numRounds)`,
    SE-14) rather than a fixed TTL — a long run would otherwise outlive its token and
    have its clients rejected mid-training once `require-client-auth` is on;
  - if client-cert issuance is enabled (SE-12, off by default) it also mints a
    short-lived mTLS client cert bound to this user and run.

  The returned `ClientConnectionDto` carries the gRPC endpoint, partition id,
  connection token, the run's **strategy**, and the project's **training arm** — the
  last two so the client picks the matching code path instead of defaulting to FedAvg
  and uploading a full state dict against a server expecting a head. Both are always
  stated, never inferred from silence.

---

## 6. Flyway Migrations

The identity subsystem's own schema lands across four migrations (`V4`–`V7`), but
this page is also the wiki's index of the **whole** migration history, because the
later ones interlock with it. `src/main/resources/db/migration/` currently holds
`V1`–**`V23`**. `ls` sorts lexicographically, so `V5`–`V9` list *after* `V21` — the
last line of an `ls` is not the highest version.

Every profile runs PostgreSQL and JPA in `validate` mode, so these migrations are
the source of truth — except the `test` profile, which disables Flyway and builds
from JPA `create-drop` against Testcontainers Postgres
(`jdbc:tc:postgresql:16.6-alpine`) for the bulk suite, so these files don't run
there. The dedicated `V*MigrationTest` classes flip Flyway back on (§6.1).

### 6.1 The identity migrations (V4–V7)

| Migration | Adds |
|---|---|
| **V4** `__project_membership_and_model_hub.sql` | `projects.visibility` (default `PRIVATE`), Model-Hub columns; `project_memberships` and `project_access_requests` tables with their indexes. Written as one `ALTER` per column — a legacy constraint from when `dev` ran on H2, which rejects multi-clause `ALTER TABLE`. Later migrations (e.g. `V17`, `V20`) use the multi-clause form freely, because Postgres is now the only target. |
| **V5** `__identity_foundations.sql` | `organizations`, `organization_memberships`; renames `users.role` → `platform_role` and adds the lifecycle/profile columns; `audit_events` (metadata initially a plain `TEXT` column); `projects.org_id`. Backfills a single **Default** org (`...0001`), enrolls every existing user, marks project owners as org `OWNER`, then sets `projects.org_id NOT NULL`. |
| **V6** `__identity_hardening.sql` | Normalises legacy `platform_role = 'ADMIN'` → `'PLATFORM_ADMIN'`; adds `chk_users_platform_role` — `CHECK (platform_role IN ('USER','PLATFORM_ADMIN'))`; promotes `audit_events.metadata` from `TEXT` to native **JSONB**. |
| **V7** `__owner_role_and_approval_workflows.sql` | Widens `chk_users_platform_role` to include **`PROJECT_OWNER`** (drops and re-adds the constraint); creates `owner_promotion_requests` (`USER` → `PROJECT_OWNER`, admin-approved) and `project_deletion_requests` (owner-requested, admin-approved) with their status indexes. The `RESTRICTED` visibility tier needs no DDL — `projects.visibility` carries no CHECK constraint. |

### 6.2 The rest of the history (V1–V3, V8–V23)

| Migration | Adds |
|---|---|
| **V1** `__init.sql` | The original schema. Establishes the `TIMESTAMP WITH TIME ZONE` convention for every point-in-time column. |
| **V2** `__add_user_role.sql` | The original coarse `users.role IN ('USER','ADMIN')` — superseded by `V5`/`V6`. |
| **V3** `__server_logs_fk_projects.sql` | `server_logs → projects` FK. |
| **V8** `__run_lifecycle_and_enrollment.sql` | The **`Run` aggregate** — one training execution of a project, and the source of truth for live FL-server state — plus `run_enrollments` with run-scoped partition assignment under `UNIQUE(run_id, partition_id)`. Additive: `projects.status`/`server_port` stay as a mirror so existing readers keep working. |
| **V9** `__project_requirements_override.sql` | `projects.requirements_override` (JSON text) — the owner may tighten the recipe's device requirements; merged most-restrictive-wins at read time. |
| **V10** `__project_task_type.sql` | `projects.task_type` for `LLM_LORA` (generative vs classification). NULL is read as `SEQ_CLASSIFICATION`. |
| **V11** `__benchmarks.sql` | `benchmark_rounds` (the rich per-round metric vector) + `benchmark_runs` (a denormalized one-row-per-project rollup for the admin dashboard). Deliberately decoupled from `round_result`, which stays the lightweight live-telemetry path. |
| **V12** `__model_artifact_registry.sql` | The content-addressed registry keystone: `artifact_blobs`, `model_artifacts`, `artifact_lineage`. See [07](07_artifact_registry.md). |
| **V13** `__timestamptz_convention.sql` | Retypes the raw `TIMESTAMP` columns `V8` introduced to `timestamptz` (`AT TIME ZONE 'UTC'` — lossless and order-preserving, since JPA already mapped them to `Instant`). Restates the convention: **every future timestamp column must be `timestamptz`**. Pre-`V8` raw-`TIMESTAMP` columns from `V5` are explicitly out of scope. |
| **V14** `__project_init_status.sql` | `projects.init_status` (NOT NULL, default `DONE`) — the one-time model-init phase (BA-1), which needs its own column because status is otherwise derived from the active run and a project mid-init has none. |
| **V15** `__run_process_tracking.sql` | `runs.server_pid` + `runs.process_started_at` — the OS identity a `StartupReconciler` needs to tell a survivor from a dead run, and to defend against PID reuse (BA-3). |
| **V16** `__run_internal_token_hash.sql` | `runs.internal_token_hash` — the SHA-256 (never the plaintext) of the per-run internal token, so the reconciler can rehydrate `RunTokenRegistry` for exactly the runs it re-adopts. |
| **V17** `__project_dp_policy.sql` | `projects.regulated` / `dp_enabled` (NOT NULL, default `FALSE`) + the nullable `dp_target_epsilon` / `dp_delta` / `dp_clip_norm` (SE-11). Completeness is validated in Java at creation and again at the run-start gate — deliberately no CHECK constraints, matching the `V14` convention. |
| **V18** `__artifact_marketplace_publish.sql` | `model_artifacts.published` / `published_at` + a `(org_id, kind, published)` index. Discovery stays strictly inside `OrgScope`; a cross-org marketplace is a separate, threat-model-sensitive effort and deliberately not enabled. |
| **V19** `__cascade_delete_run_subtree.sql` | `ON DELETE CASCADE` on `runs.project_id` and `run_enrollments.run_id`. Before this, **any project that had ever been started could not be deleted** — `V8` created those two FKs with no `ON DELETE` action, so Postgres raised `23503`, surfaced as an opaque 409. Registry rows are deliberately left alone (`SET NULL` / untouched / `RESTRICT`). |
| **V20** `__project_derivation.sql` | `projects.init_from_pretrained` (NOT NULL, default `FALSE`), `base_ref_sha256`, `derivation_spec` — the opt-in record of a project deriving from a pretrained/frozen base instead of training from scratch. `chk_projects_base_ref_sha256_hex` pins a present ref to lowercase-hex sha256, matching `V12`'s content-address convention. A NULL derivation behaves exactly as before. |
| **V21** `__base_ref_unique_index.sql` | The partial unique index `uq_base_ref_org_model ON model_artifacts (org_id, base_model_ref) WHERE kind = 'BASE_REF'`. `findOrCreateBaseRef` was a non-atomic read-then-insert with no backing constraint, so two concurrent adapter registrations in one org over the same base could each insert a `BASE_REF` and fork their `ADAPTER_OF` edges. The index makes one-per-`(org, base)` a DB invariant *and* backs the `ON CONFLICT DO NOTHING` the service now uses. It fails loudly if a deployed DB already holds duplicates. |
| **V22** `__project_training_arm.sql` | `projects.training_arm VARCHAR(32) NOT NULL DEFAULT 'FULL'` + `chk_projects_training_arm CHECK (training_arm IN ('FULL','FROZEN_HEAD'))`. The `DEFAULT` is load-bearing for backward compatibility: every pre-existing project trained every parameter, so backfilling them to `FULL` preserves their behaviour exactly. |
| **V23** `__training_arm_ova_lp.sql` | Drops and re-adds that CHECK widened to `('FULL','FROZEN_HEAD','OVA_LP')` for the OvA-LP arm (arXiv:2511.05028). |

The arm's full contract — DTO pattern, entity default, immutability after creation,
and why the CHECK is the last line of defence — is in
[03 - Project Management Lifecycle](03_project_management.md).

### 6.3 The `V*MigrationTest` classes bypass the `test` profile — deliberately

There are **twelve** of them, and they all share one unusual shape that is easy to
break by copying an ordinary test instead:

```java
@SpringBootTest
@ActiveProfiles("dev")                       // NOT "test" — that profile disables Flyway
@TestPropertySource(properties = {
        "spring.datasource.url=jdbc:tc:postgresql:16.6-alpine:///fedlearn_v23_arm",  // per-test DB
        "spring.datasource.driver-class-name=org.testcontainers.jdbc.ContainerDatabaseDriver",
        "spring.jpa.hibernate.ddl-auto=validate",   // or "none"
        "spring.flyway.enabled=true",               // the point of the whole exercise
        "app.jwt.secret=…", "app.internal.api-key=…", "app.cors.allowed-origins=…"
})
```

The reason is that the bulk suite's `test` profile builds its schema from the JPA
entities (`create-drop`, Flyway off). A column generated from an entity proves
nothing about the migration that is supposed to create it, and a **backfill of
existing rows cannot be exercised at all** without the real migration running. So
each of these classes runs every migration in order against a real Postgres in its
own Testcontainers database, then asserts against `information_schema` and real
inserts.

Current classes and the database name each claims:
`V5MigrationTest`, `V6MigrationTest` (identity) · `V8MigrationTest`,
`V9MigrationTest`, `V13TimestamptzMigrationTest` (runs) · `V11BenchmarkMigrationTest`
· `V12ModelRegistryMigrationTest`, `V21BaseRefUniqueMigrationTest` (registry) ·
`V17MigrationTest` (DP) · `V19ProjectDeletionCascadeMigrationTest` ·
`V20DerivationMigrationTest` · `V22TrainingArmMigrationTest`.

Copy that shape for a new migration test; do **not** reach for
`@ActiveProfiles("test")`. And keep Flyway disabled for the `test` profile itself —
migrations must validate against `dev`/`ec2demo`/`production` only.

```sql
-- V5: rename the single role column and pin projects to an org
ALTER TABLE users RENAME COLUMN role TO platform_role;
ALTER TABLE projects ADD COLUMN org_id UUID REFERENCES organizations(id);
-- ... backfill Default org, enroll users ...
ALTER TABLE projects ALTER COLUMN org_id SET NOT NULL;

-- V6: harden the role column and upgrade metadata to JSONB
UPDATE users SET platform_role = 'PLATFORM_ADMIN' WHERE platform_role = 'ADMIN';
ALTER TABLE users ADD CONSTRAINT chk_users_platform_role
    CHECK (platform_role IN ('USER','PLATFORM_ADMIN'));
ALTER TABLE audit_events ALTER COLUMN metadata TYPE JSONB USING (NULLIF(metadata,'')::jsonb);

-- V7: widen the role domain to the third tier (IF EXISTS guards dev DBs baselined past V6)
ALTER TABLE users DROP CONSTRAINT IF EXISTS chk_users_platform_role;
ALTER TABLE users ADD CONSTRAINT chk_users_platform_role
    CHECK (platform_role IN ('USER','PROJECT_OWNER','PLATFORM_ADMIN'));
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

These are live on the backend **and** surfaced in the frontend — see the branch-reality
note at the top: `App.tsx` wraps the owner routes in
`RoleRoute allow={['PROJECT_OWNER', 'PLATFORM_ADMIN']}` and the admin routes in
`RoleRoute allow={['PLATFORM_ADMIN']}`.

| Method & path | Controller | Purpose |
|---|---|---|
| `GET/POST /api/projects/{id}/memberships`, `DELETE .../{userId}` | `MembershipController` | List / add / remove project members |
| `POST/GET /api/projects/{id}/access-requests`, `PUT .../{requestId}` | `AccessRequestController` | Submit / list / decide join requests |
| `GET /api/my/access-requests` | `MyRequestsController` | The caller's own outstanding requests |
| `POST /api/owner-requests`, `GET /api/owner-requests/mine` | `OwnerRequestController` | A `USER` asks to be promoted to `PROJECT_OWNER`, and tracks that request |
| `GET /api/projects/discover`, `GET /api/projects/{id}`, `PATCH /api/projects/{id}` | `ProjectController` | Discover feed, single-project read (404-on-out-of-scope), update name/description/visibility/requirements. Full route table in [03](03_project_management.md) |
| `POST /api/projects/{id}/deletion-request`, `GET .../deletion-request` | `ProjectController` (`ProjectDeletionService`) | Owner files / inspects a deletion request (**204** when there is none) |
| `GET /api/users/search?q=` | `UserSearchController` | Username prefix search; per-caller rate-limited (`UserSearchService`) |
| `GET/PATCH /api/users/me/profile` | `ProfileController` | Self-service profile. `permitAll` at the chain level so the controller can 401 anonymously; `USER_PROFILE_UPDATED` / `USER_PASSWORD_CHANGED` are audited |
| `GET /api/users`, `POST /api/users`, `DELETE /api/users/{id}` | `UserController` | Legacy user-management surface; the `GET`/`DELETE` are `@PreAuthorize PLATFORM_ADMIN` |
| `GET /api/client/projects`, `GET /api/client/projects/{id}`, `POST .../join`, `GET .../connection` | `ClientApiController` (`ClientApiService`) | FL-client project list, single read, join, and per-run enrollment + connection token |
| `POST /api/admin/test-email?to=` | `TestEmailController` (flag-gated, `PLATFORM_ADMIN`) | Email delivery smoke test |

`AdminController` is class-level `@PreAuthorize("hasRole('PLATFORM_ADMIN')")` and is
big enough to warrant its own table:

| Method & path | Purpose |
|---|---|
| `GET /api/admin/overview` | The admin console landing summary |
| `GET /api/admin/users`, `GET /api/admin/users/{id}` | User directory and detail |
| `GET /api/admin/users/search?q=&role=&status=&page=&size=` | Search-first user directory, server-side paged (`PagedResponseDto`) |
| `PUT /api/admin/users/{id}/role` | Change a platform role; refuses to demote the last admin |
| `PUT /api/admin/users/{id}/status` | Suspend / reactivate. Dispatched to a **separate service method per transition** so each carries its own `@Auditable` action and the aspect fires through the Spring proxy — a self-invocation would bypass it |
| `GET /api/admin/projects`, `GET /api/admin/projects/search?q=&status=&visibility=&page=&size=` | Project directory, flat and search-first |
| `GET /api/admin/audit-events?actor=&action=&targetType=&from=&to=&page=&size=` | The audit-event explorer |
| `GET /api/admin/owner-requests?status=`, `PUT /api/admin/owner-requests/{id}` | The owner-promotion queue (`OwnerPromotionService`) |
| `GET /api/admin/deletion-requests?status=`, `PUT /api/admin/deletion-requests/{id}` | The project-deletion queue; an `APPROVED` decision calls `ProjectService.deleteProject` in the admin's own security context (`ProjectDeletionService`) |
