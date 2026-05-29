# 03 — DATA MODEL (FedLearn Platform v2)

**Audience:** a mid-sized local LLM (Large Language Model, ~30 billion parameters) implementing the
schema. Every column, type, constraint, and migration filename below is **pre-decided**. Do not
choose alternatives. Implement exactly what is written.

**Status:** Build specification (greenfield v2). The V1–V5 baseline described in §2 already exists in
the repository; the v2 additions (V6–V8 in §5) are **new** migrations you will author.

**Source of truth:** the v2 audit synthesis at
`/home/anurag/codebase/FedLearn-Platform/docs/audit/2026-05-29/README.md` and the three depth reports
`A1-backend.md`, `C2-data-engineering.md`, `C3-reproducibility.md`. Every design claim below cites the
audit finding it derives from (e.g. "A1-F8", "C2 §3.2", "C3 §5.3").

---

## 0. Definitions of every abbreviation used in this document

The first occurrence of each acronym is expanded here once; thereafter the short form is used.

| Short form | Full form |
|---|---|
| LLM | Large Language Model |
| ER | Entity-Relationship |
| DDL | Data Definition Language |
| DML | Data Manipulation Language |
| SQL | Structured Query Language |
| OLTP | Online Transaction Processing |
| RDS | (AWS) Relational Database Service |
| JPA | Jakarta Persistence API |
| ORM | Object-Relational Mapping |
| PK | Primary Key |
| FK | Foreign Key |
| RBAC | Role-Based Access Control |
| RLS | Row-Level Security |
| JWT | JSON Web Token (JSON = JavaScript Object Notation) |
| FL | Federated Learning |
| DeComFL | Dimension-Free Communication Federated Learning (zeroth-order optimization; the v1 wiki's "Decomposed" expansion is wrong per the paper, `B1-paper-alignment.md:33`) |
| FedAvg | Federated Averaging |
| RNG | Random Number Generator |
| ZO | Zeroth-Order (optimization) |
| gRPC | Google Remote Procedure Call |
| S3 | (AWS) Simple Storage Service |
| MinIO | (an S3-compatible self-hosted object store; not an acronym) |
| MLflow | (Machine-Learning lifecycle tool; not an acronym) |
| sha256 | Secure Hash Algorithm 256-bit |
| JSONB | JSON Binary (PostgreSQL binary JSON column type) |
| CLOB | Character Large Object (SQL large-text type) |
| TEXT | (PostgreSQL unbounded variable-length string type) |
| UUID | Universally Unique Identifier |
| UUIDv7 | UUID version 7 (time-ordered) |
| BIGINT | 64-bit signed integer SQL type |
| BIGSERIAL | PostgreSQL auto-incrementing 64-bit integer |
| TIMESTAMPTZ | TIMESTAMP WITH TIME ZONE (PostgreSQL) |
| IID | Independent and Identically Distributed |
| non-IID | not Independent and Identically Distributed |
| ECS | (AWS) Elastic Container Service |
| ARN | (AWS) Amazon Resource Name |
| k8s | Kubernetes |
| API | Application Programming Interface |
| CSV | Comma-Separated Values |
| DP | Differential Privacy |
| HIPAA | Health Insurance Portability and Accountability Act |
| SOC 2 | System and Organization Controls 2 |
| CI | Continuous Integration |
| TTL | Time-To-Live |
| HA | High Availability |
| JVM | Java Virtual Machine |
| DLG | Deep Leakage from Gradients |

---

## 1. Scope and the four hard rules of this data model

This document defines the **control-plane OLTP schema** owned by PostgreSQL (managed AWS RDS in
production; see audit README §1.1 "OLTP datastore"). It does **not** define raw training data storage —
per the FL invariant, raw training features and labels live only on the client and never enter any table
here (C2 §3.1, the "two-plane model"). What the control plane stores is **metadata, lineage, leases, and
non-reversible fingerprints**.

Four hard rules govern every table below. They are pre-decided; do not deviate.

| # | Rule | Why (audit finding) |
|---|---|---|
| R-A | **Flyway owns the schema. JPA runs `validate`-only** in `dev`/`ec2demo`/`production`. Every schema change is a new `V{n}__*.sql` file. Never use `ddl-auto=update`. | A1 "salvage the Flyway-owns-schema discipline (it's correct and worth keeping)"; A1-F10. |
| R-B | **PostgreSQL is the only target dialect.** No `CLOB`. Use `TEXT` for unbounded strings and `JSONB` for structured blobs. The V5 `audit_events.metadata CLOB` is a bug to fix (§6). | A1-F10; README §3 conflict 7 ("Must fix `audit_events.metadata CLOB`→`TEXT/JSONB`"). |
| R-C | **`org_id` is on every tenant-owned row and is `NOT NULL`.** Tenant isolation is enforced by org-scoped query filters / RLS-style predicates, not by application code alone. | A1-F9; README §1.1 "AuthZ … `org_id`-scoped checks + RLS-style query filters". |
| R-D | **Mixed key types are frozen, not extended.** `users.id` is `BIGINT`. `organizations.id`, `projects.id`, and **all new v2 top-level entities** are `UUID`. Composite keys that join a user to a UUID entity mix `(UUID, BIGINT)`. v2 does **not** migrate `users.id` (the migration cost is not justified for this build; see §3.4). | A1-F8 (flagged the tax) reconciled against build scope: README locks the existing identity model. |

> **Note on the V1–V5 baseline being incomplete on disk.** Only `V1__init.sql`, `V2__add_user_role.sql`,
> and `V3__server_logs_fk_projects.sql` are present under
> `backend/fl-platform-api/src/main/resources/db/migration/`. `V4__project_membership_and_model_hub.sql`
> and `V5__identity_foundations.sql` exist only under `backend/fl-platform-api/build/resources/main/db/migration/`
> (the compiled-resources copy). **Before authoring V6, copy V4 and V5 into the `src/main/resources`
> migration directory** so Flyway has the full ordered chain. The DDL of V4/V5 reproduced in §2 is taken
> verbatim from the `build/resources` copies.

---

## 2. ER overview diagram (ASCII) — all entities

The diagram shows existing (V1–V5) and new (V6–V8) entities together. `[BIGINT]` / `[UUID]` annotate the
PK type. Crow's-foot notation: `||` = exactly one, `o{` / `}o` = zero-or-many, `}|` = one-or-many.

```
                              IDENTITY / TENANCY PLANE
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │                                                                                    │
 │   users [BIGINT]                          organizations [UUID]                     │
 │   ─────────────                           ──────────────────                       │
 │   id (PK, BIGSERIAL)  ◀──┐                id (PK, UUID)  ◀──────┐                   │
 │   username UNIQUE        │                name                 │                   │
 │   email UNIQUE           │                slug UNIQUE           │                   │
 │   password (hashed)      │                created/updated/del   │                   │
 │   platform_role          │                                      │                   │
 │   status                 │                                      │                   │
 │   email_verified         │  organization_memberships            │                   │
 │   display_name ...       │  ────────────────────────            │                   │
 │                          ├─o{ org_id (FK,UUID) }o────────────────┤                   │
 │                          │    user_id (FK,BIGINT)                │                   │
 │                          │    org_role {OWNER,ADMIN,MEMBER}      │                   │
 │                          │    PK (org_id, user_id)  ← MIXED KEY  │                   │
 │                          │                                       │                   │
 └──────────────────────────┼───────────────────────────────────────┼──────────────────┘
                            │                                       │
              ┌─────────────┘                                       │ org_id (FK, NOT NULL)
              │                                                     ▼
              │                                  projects [UUID]
              │                                  ──────────────
              │                                  id (PK, UUID)  ◀────────────┐
              │                          ┌──────▶ user_id (FK,BIGINT) owner   │
              │                          │        org_id (FK,UUID) NOT NULL   │
              │                          │        name UNIQUE                 │
              │                          │        model_type / model_name     │
              │                          │        status / visibility         │
              │                          │        model_published ...         │
              │                          │   v2 ADD: dataset_version_id (FK)   │
              │                          │   v2 ADD: partition_recipe_id (FK)  │
              │                          │                                     │
              │  project_memberships     │                                     │
              │  ───────────────────     │                                     │
              ├─o{ project_id (FK,UUID) }o┤                                     │
              │    user_id (FK,BIGINT)    │                                     │
              │    role {MEMBER,CLIENT,OWNER}                                   │
              │    PK (project_id, user_id) ← MIXED KEY                         │
              │                                                                 │
              │  project_access_requests  │                                     │
              ├─o{ project_id (FK,UUID) }o┘                                     │
              │    user_id (FK,BIGINT)                                          │
              │    status {PENDING,APPROVED,DENIED}                             │
              │                                                                 │
              │  audit_events [UUID]                                            │
              └─o{ actor_user_id (FK,BIGINT, nullable)                          │
                   org_id (FK,UUID, nullable)                                   │
                   action / target_type / target_id                            │
                   metadata JSONB  ← v2 FIX (was CLOB)                          │
                                                                                │
 ══════════════════════════════════════════════════════════════════════════════│════
                              FL-RUN / LINEAGE PLANE (v2 NEW)                    │
                                                                                │
   datasets [UUID]                  dataset_versions [UUID]                     │
   ──────────────                   ──────────────────────                     │
   id (PK,UUID) ◀──────────┐        id (PK,UUID) ◀───────────┐                  │
   org_id (FK,UUID)        │        dataset_id (FK,UUID) }o───┤                  │
   name                    └──o{    version (INT, monotonic)  │                  │
   modality                         content_hash CHAR(64)     │                  │
   created_by (FK,BIGINT)           schema_json JSONB         │                  │
                                    sample_count BIGINT       │                  │
                                                              │                  │
   partition_recipes [UUID]                                   │                  │
   ────────────────────────                                   │                  │
   id (PK,UUID) ◀────────────┐                                │                  │
   dataset_version_id (FK) }o─────────────────────────────────┘                  │
   partitioner {DIRICHLET_LABEL,DIRICHLET_QTY,SHARD,NATURAL}                      │
   num_partitions / alpha / data_seed                                            │
   recipe_hash CHAR(64)                                                          │
        ▲                  ▲                                                     │
        │                  │ (projects pin one version + one recipe — FKs above) │
        │                  └──────────────────────────────────────── projects ──┘
        │
        │  fl_runs [UUID]  (the run lease + state + lineage aggregate)
        │  ──────────────
        │  id == run_id (PK,UUID) ◀──────────────────────┐
        │  project_id (FK,UUID) NOT NULL                 │
        │  org_id (FK,UUID) NOT NULL                     │
        │  status {PENDING,STARTING,RUNNING,             │
        │          SUCCEEDED,FAILED,STOPPED}             │
        │  lease_owner / lease_expires_at  ← reconciler  │
        │  launcher {K8S_JOB,ECS_RUN_TASK,LOCAL_PROCESS} │
        │  executor_ref / grpc_endpoint                  │
        │  round_idx INT                                 │
        │  strategy {DeComFL,FedAvg}                     │
        │  config JSONB  (K,P,eta,mu,seed,num_rounds...) │
        │  dataset_version_id (FK,UUID, nullable)        │
        ├──partition_recipe_id (FK,UUID, nullable)       │
        │  initial_model_artifact_id (FK,UUID, nullable) │
        │  mlflow_run_id                                 │
        │  requested_by (FK,BIGINT)                      │
        │  created_at / updated_at / started_at / ended  │
        │                                                │
        │  round_results [UUID]   (incremental per-round)│
        └─o{ fl_run_id (FK,UUID) NOT NULL }o─────────────┤
             round_idx INT                               │
             loss / accuracy / val_loss / val_accuracy   │
             num_clients_reported INT                    │
             uplink_bytes / downlink_bytes BIGINT        │
             scalars_transmitted BIGINT  ← DeComFL wedge │
             round_started_at / round_ended_at           │
                                                         │
   model_artifacts [UUID]  (content-addressed S3/MinIO)  │
   ──────────────────────                                │
   id (PK,UUID)  ◀──────────────────────────────────────┘ (initial/final FKs)
   org_id (FK,UUID) NOT NULL
   sha256 CHAR(64) UNIQUE-per-org
   storage_uri (s3://bucket/<sha256>)
   size_bytes / kind {INITIAL,CHECKPOINT,FINAL}
   fl_run_id (FK,UUID, nullable)
   round_idx INT (nullable; set for CHECKPOINT)

   determinism_manifests [UUID]  (one per fl_run)
   ────────────────────────────
   id (PK,UUID)
   fl_run_id (FK,UUID) UNIQUE NOT NULL  ← 1:1 with fl_runs
   torch_version / framework_git_sha / proto_version
   rng_device='cpu' / rng_engine / use_deterministic_algorithms
   seed BIGINT
   initial_model_sha256 / dataset_split_sha256
   manifest_json JSONB  (full §5.2-C3 manifest)
```

**Cardinality summary (read this if the diagram is dense):**

| Relationship | Cardinality | Enforced by |
|---|---|---|
| organization 1—* organization_memberships *—1 user | many-to-many via join table | composite PK `(org_id, user_id)` |
| organization 1—* projects | one org owns many projects | `projects.org_id NOT NULL` FK |
| project 1—* project_memberships *—1 user | many-to-many via join table | composite PK `(project_id, user_id)` |
| organization 1—* datasets 1—* dataset_versions 1—* partition_recipes | strict hierarchy | FK chain + `ON DELETE CASCADE` |
| project *—1 dataset_version, project *—1 partition_recipe | a project pins one of each | nullable FKs on `projects` |
| project 1—* fl_runs | a project is a template; each launch is a run | `fl_runs.project_id NOT NULL` FK |
| fl_run 1—* round_results | one row per round, incremental | `round_results.fl_run_id NOT NULL` FK |
| fl_run 1—1 determinism_manifest | exactly one manifest per run | `determinism_manifests.fl_run_id UNIQUE NOT NULL` |
| fl_run 1—* model_artifacts | initial + per-round checkpoints + final | `model_artifacts.fl_run_id` nullable FK |

---

## 3. Multi-tenant identity model (V1–V5, the salvaged baseline)

This section is the **canonical contract** for the identity layer. It reproduces the existing V1–V5 DDL
exactly (verified against the on-disk files) and states the key-type rules the local model must obey when
writing any FK to these tables.

### 3.1 Three orthogonal role layers — collapse the strings to enums in code

Three role layers exist, all per-user (audit README "Identity layers (V5)"). v2 keeps the **column shapes**
but the application code MUST collapse the role strings to Java enums to kill the `ADMIN` vs
`PLATFORM_ADMIN` drift that 403'd the bootstrap admin (A1-F1, the headline backend bug). The DDL keeps the
`VARCHAR` + `CHECK` constraint as the database-level guard; the enum is the application-level guard.

| Layer | Column | Allowed values | DB guard | Meaning |
|---|---|---|---|---|
| Platform | `users.platform_role` | `USER`, `PLATFORM_ADMIN` | `CHECK` (added in V6, §5.1) | `PLATFORM_ADMIN` bypasses org-membership checks. |
| Organisation | `organization_memberships.org_role` | `OWNER`, `ADMIN`, `MEMBER` | `CHECK` (exists in V5) | Tenant-scoped admin. |
| Project | `project_memberships.role` | `MEMBER`, `CLIENT`, `OWNER` | none in V1–V5 (add `CHECK` in V6, §5.1) | `OWNER` is implicit via `projects.user_id`; the `OWNER` row only holds `partition_id`. |

> **Critical fix carried from A1-F1:** the V2 migration's bootstrap comment says
> `UPDATE users SET role = 'ADMIN'`, which is wrong — the working value is `PLATFORM_ADMIN` and the column
> is `platform_role` after V5. Do **not** reproduce that comment. The V6 migration (§5.1) adds the
> `CHECK (platform_role IN ('USER','PLATFORM_ADMIN'))` constraint so a typo'd role is rejected at the
> database boundary, making the A1-F1 class structurally impossible.

### 3.2 Key-type rules (R-D, stated precisely so the local model never guesses)

| Entity | PK column | PK type | JPA generation | Source |
|---|---|---|---|---|
| `users` | `id` | `BIGINT` (`BIGSERIAL`) | `GenerationType.IDENTITY` → Java `Long` | `V1__init.sql:5-12`; `User.java:13-15` |
| `organizations` | `id` | `UUID` | application-assigned `UUID` | `V5__identity_foundations.sql:7-14` |
| `projects` | `id` | `UUID` | `GenerationType.AUTO` → `UUID` | `V1__init.sql:14-24`; `Project.java:13-15` |
| `round_result` (V1, renamed/replaced in v2 — see §5.2) | `id` | `UUID` | `GenerationType.AUTO` | `V1__init.sql:28-35`; `RoundResult.java:10-12` |
| `audit_events` | `id` | `UUID` | application-assigned | `V5__identity_foundations.sql:43-54` |
| **all new v2 top-level entities** | `id` | `UUID` | application-assigned `UUID` | this document, R-D |

**Composite (mixed) keys — the local model MUST type these exactly as shown:**

```java
// OrganizationMembershipId — (UUID orgId, Long userId)
// ProjectMembershipId      — (UUID projectId, Long userId)
// Both are @Embeddable / @IdClass composite keys. orgId/projectId are UUID; userId is Long.
```

A new v2 table that references a user uses `BIGINT` for the user FK and `UUID` for the org/project/run FK.
Example: `model_artifacts.org_id UUID` + `created_by BIGINT REFERENCES users(id)`.

### 3.3 V1–V5 DDL reproduced verbatim (the baseline you build on)

`V1__init.sql` (users, projects, round_result, server_logs):

```sql
CREATE TABLE users (
    id            BIGSERIAL PRIMARY KEY,
    username      VARCHAR(50)  NOT NULL UNIQUE,
    email         VARCHAR(100) NOT NULL UNIQUE,
    password      VARCHAR(255) NOT NULL,
    created_at    TIMESTAMP WITH TIME ZONE  NOT NULL,
    updated_at    TIMESTAMP WITH TIME ZONE  NOT NULL
);

CREATE TABLE projects (
    id           UUID         PRIMARY KEY,
    name         VARCHAR(255) NOT NULL UNIQUE,
    model_type   VARCHAR(255) NOT NULL,
    model_name   VARCHAR(255) NOT NULL,
    server_port  INTEGER,
    model_path   VARCHAR(1024),
    optimizer    VARCHAR(64),
    status       VARCHAR(32)  NOT NULL,
    user_id      BIGINT REFERENCES users(id) ON DELETE SET NULL
);
CREATE INDEX idx_projects_user_id ON projects(user_id);

CREATE TABLE round_result (
    id              UUID    PRIMARY KEY,
    project_id      UUID    NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    server_round    INTEGER NOT NULL,
    loss            DOUBLE PRECISION,
    accuracy        DOUBLE PRECISION,
    gpu_utilization DOUBLE PRECISION
);
CREATE INDEX idx_round_result_project_id ON round_result(project_id);

CREATE TABLE server_logs (
    id          BIGSERIAL   PRIMARY KEY,
    project_id  UUID        NOT NULL,
    level       VARCHAR(16) NOT NULL,
    message     TEXT        NOT NULL,
    stack_trace TEXT,
    timestamp   TIMESTAMP WITH TIME ZONE NOT NULL
);
CREATE INDEX idx_server_logs_project_id ON server_logs(project_id);
CREATE INDEX idx_server_logs_timestamp  ON server_logs(timestamp);
```

`V2__add_user_role.sql` (adds `role`, later renamed to `platform_role` in V5):

```sql
ALTER TABLE users
    ADD COLUMN role VARCHAR(32) NOT NULL DEFAULT 'USER';
CREATE INDEX idx_users_role ON users(role);
```

`V3__server_logs_fk_projects.sql` (FK + cascade for server_logs):

```sql
DELETE FROM server_logs
 WHERE project_id NOT IN (SELECT id FROM projects);
ALTER TABLE server_logs
    ADD CONSTRAINT fk_server_logs_project
        FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE;
```

`V4__project_membership_and_model_hub.sql` (visibility, Model Hub columns, memberships, access requests):

```sql
ALTER TABLE projects
    ADD COLUMN visibility VARCHAR(32) NOT NULL DEFAULT 'PRIVATE';
CREATE INDEX idx_projects_visibility ON projects(visibility);

ALTER TABLE projects ADD COLUMN model_published    BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE projects ADD COLUMN model_description  TEXT;
ALTER TABLE projects ADD COLUMN model_tags         VARCHAR(512);
ALTER TABLE projects ADD COLUMN model_published_at TIMESTAMP WITH TIME ZONE;
CREATE INDEX idx_projects_model_published ON projects(model_published);

CREATE TABLE project_memberships (
    project_id     UUID         NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id        BIGINT       NOT NULL REFERENCES users(id)    ON DELETE CASCADE,
    role           VARCHAR(32)  NOT NULL,    -- 'MEMBER' | 'CLIENT' | 'OWNER'
    partition_id   INTEGER,
    joined_via     VARCHAR(32)  NOT NULL,    -- 'OWNER_ADD'|'PUBLIC_JOIN'|'REQUEST_APPROVED'|'OWNER_SELF'
    added_by       BIGINT       REFERENCES users(id) ON DELETE SET NULL,
    added_at       TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (project_id, user_id)
);
CREATE INDEX idx_project_memberships_user_id ON project_memberships(user_id);
CREATE INDEX idx_project_memberships_role    ON project_memberships(project_id, role);

CREATE TABLE project_access_requests (
    id              BIGSERIAL PRIMARY KEY,
    project_id      UUID         NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id         BIGINT       NOT NULL REFERENCES users(id)    ON DELETE CASCADE,
    requested_role  VARCHAR(32)  NOT NULL,    -- always 'CLIENT' in v1
    status          VARCHAR(32)  NOT NULL,    -- 'PENDING' | 'APPROVED' | 'DENIED'
    message         TEXT,
    requested_at    TIMESTAMP WITH TIME ZONE NOT NULL,
    decided_at      TIMESTAMP WITH TIME ZONE,
    decided_by      BIGINT       REFERENCES users(id) ON DELETE SET NULL,
    UNIQUE (project_id, user_id)
);
CREATE INDEX idx_par_project_status ON project_access_requests(project_id, status);
CREATE INDEX idx_par_user_id        ON project_access_requests(user_id);
```

`V5__identity_foundations.sql` (orgs, org memberships, user lifecycle, audit log). **Note the two
PostgreSQL-portability defects flagged in §6: `TIMESTAMP` without time zone, and `metadata CLOB`.**

```sql
CREATE TABLE organizations (
    id            UUID PRIMARY KEY,
    name          VARCHAR(120) NOT NULL,
    slug          VARCHAR(64)  NOT NULL UNIQUE,
    created_at    TIMESTAMP    NOT NULL,    -- ← §6 defect: should be TIMESTAMPTZ
    updated_at    TIMESTAMP    NOT NULL,
    deleted_at    TIMESTAMP
);

CREATE TABLE organization_memberships (
    org_id     UUID         NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    user_id    BIGINT       NOT NULL REFERENCES users(id)         ON DELETE CASCADE,
    org_role   VARCHAR(16)  NOT NULL CHECK (org_role IN ('OWNER','ADMIN','MEMBER')),
    created_at TIMESTAMP    NOT NULL,
    PRIMARY KEY (org_id, user_id)        -- ← MIXED KEY (UUID, BIGINT)
);
CREATE INDEX idx_org_mem_user ON organization_memberships(user_id);

ALTER TABLE users ALTER COLUMN role RENAME TO platform_role;
ALTER TABLE users ADD COLUMN status         VARCHAR(16) NOT NULL DEFAULT 'ACTIVE';
ALTER TABLE users ADD COLUMN deleted_at     TIMESTAMP;
ALTER TABLE users ADD COLUMN email_verified BOOLEAN     NOT NULL DEFAULT FALSE;
ALTER TABLE users ADD COLUMN display_name   VARCHAR(120);
ALTER TABLE users ADD COLUMN avatar_url     VARCHAR(512);
ALTER TABLE users ADD COLUMN last_login_at  TIMESTAMP;
ALTER TABLE users ADD CONSTRAINT chk_users_status
    CHECK (status IN ('PENDING','ACTIVE','SUSPENDED'));

ALTER TABLE projects ADD COLUMN org_id UUID REFERENCES organizations(id);
CREATE INDEX idx_projects_org ON projects(org_id);

CREATE TABLE audit_events (
    id              UUID PRIMARY KEY,
    occurred_at     TIMESTAMP    NOT NULL,
    actor_user_id   BIGINT       REFERENCES users(id),
    org_id          UUID         REFERENCES organizations(id),
    action          VARCHAR(64)  NOT NULL,
    target_type     VARCHAR(32),
    target_id       VARCHAR(64),
    metadata        CLOB,                 -- ← §6 defect: CLOB is not PostgreSQL; must be JSONB
    request_ip      VARCHAR(45),
    user_agent      VARCHAR(256)
);
CREATE INDEX idx_audit_org_time    ON audit_events(org_id, occurred_at);
CREATE INDEX idx_audit_actor_time  ON audit_events(actor_user_id, occurred_at);
CREATE INDEX idx_audit_action_time ON audit_events(action, occurred_at);

-- Backfill (Default org, owner promotion, pin projects to org, then NOT NULL):
INSERT INTO organizations (id, name, slug, created_at, updated_at)
VALUES ('00000000-0000-0000-0000-000000000001', 'Default', 'default',
        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
INSERT INTO organization_memberships (org_id, user_id, org_role, created_at)
SELECT '00000000-0000-0000-0000-000000000001', id, 'MEMBER', CURRENT_TIMESTAMP FROM users;
UPDATE organization_memberships SET org_role = 'OWNER'
WHERE user_id IN (SELECT DISTINCT user_id FROM projects);
UPDATE projects SET org_id = '00000000-0000-0000-0000-000000000001' WHERE org_id IS NULL;
ALTER TABLE projects ALTER COLUMN org_id SET NOT NULL;
```

### 3.4 Reasoning — why this identity model, why not change it now

- **Why keep `users.id BIGINT` while everything else is `UUID` (R-D)?** A1-F8 correctly flags the mixed
  strategy as a long-term tax (enumerable user ids, no client-side id pre-generation, harder export). But
  migrating `users.id` touches every FK in V1–V5 (`projects.user_id`, both membership join tables,
  `audit_events.actor_user_id`, access requests) and is a one-time, high-blast-radius change. The v2 build
  scope (README §1.1) **locks the existing identity model** and spends the rebuild budget on the FL
  substrate, registry, and lineage. Decision: **freeze the split, do not extend the BIGINT footprint** —
  all *new* top-level entities are `UUID`. This bounds the tax without paying the migration cost mid-build.
  If a later epic migrates `users.id`, it does so when the table is still small (A1-F8's own advice).
- **Why `org_id NOT NULL` on every tenant row (R-C)?** A1-F9: tenant isolation today is application-code
  only, so one missing `WHERE org_id = ?` is a cross-tenant leak. Putting `org_id NOT NULL` on every
  tenant-owned table makes the org-scoped RLS predicate (or `@Filter`) total — there is no row the filter
  can miss. This is the database-level half of the README's "RLS-style query filters" decision.
- **Why composite (not surrogate) PKs on the membership tables?** They are pure join tables; the natural
  key `(org_id, user_id)` / `(project_id, user_id)` is also the uniqueness constraint ("one membership per
  user per org/project"). A surrogate would need a separate unique index anyway. This matches V4/V5 as
  built; v2 does not change it.

---

## 4. v2 new entities — design rationale (before the DDL in §5)

Each new table maps to a specific audit finding. Read this table; then the DDL in §5 is the literal
encoding of it.

| New table | Audit driver | One-line purpose |
|---|---|---|
| `datasets` | C2 §3.2 ("dataset registry … none exists") | Per-org named dataset; the lineage root. |
| `dataset_versions` | C2 §3.2; C2-R14 | Immutable, content-hash-addressed version of a dataset's schema+stats manifest. |
| `partition_recipes` | C2 §3.2; C2 §2.3 ("seed is doubly load-bearing") | Reproducible non-IID split: partitioner + N + alpha + **data_seed** (distinct from optimizer seed) + content hash. |
| `fl_runs` | A1-F2/F4 (run entity, lease, race); C1; C3 §5.3 | The durable run lease + state + lineage aggregate, keyed on `run_id`. Survives JVM restarts; the reconciler operates on it. |
| `round_results` (replaces V1 `round_result`) | C3 §5.3; README §1.1 "FL-run telemetry … incremental"; C2 comm-cost | Per-round metrics **including bytes/scalars transmitted** (the DeComFL bandwidth wedge), FK'd to the run. |
| `model_artifacts` | A1 (S3 TODOs); C1; C3 §5.5 | Content-addressed (sha256) initial/checkpoint/final models in S3/MinIO, wired to runs. |
| `determinism_manifests` | C3 §5.2 (the manifest); C3-R14 | One per run: torch/git/proto versions, CPU-canonical RNG declaration, seed, model/dataset hashes. |

### 4.1 Why `fl_runs` is a *lease* table, not just a status table (A1-F2/F4, C1)

The v1 control plane tracked running FL servers in an in-memory `ConcurrentHashMap<UUID, Process>`. A JVM
restart **orphaned every running Python child** and the backend had no record they existed (A1-F2.3). v2
makes the JVM a **stateless supervisor over a database lease** (README §1.2 "the JVM is a stateless
supervisor over the `fl_runs` DB lease"). The mechanism:

1. Every non-terminal run row carries `lease_owner` (the supervisor instance id, e.g. the pod name) and
   `lease_expires_at` (a future timestamp the owning instance renews on a heartbeat).
2. A **reconciler loop** (`@Scheduled`) on each supervisor instance periodically:
   - claims runs whose `lease_expires_at < now()` (the previous owner died) via an atomic
     `UPDATE ... WHERE lease_expires_at < now()` with optimistic concurrency,
   - `poll()`s the launcher (`KubernetesJobLauncher.poll(executor_ref)` etc.) and reconciles real executor
     state → `status` → STOMP,
   - reaps orphans on boot.
3. The **round loop deadline + minimum-quorum** (README §1.1 "Round loop MUST have a deadline +
   minimum-quorum") is enforced inside the FL server, but its parameters (`num_rounds`, `min_clients`,
   `round_deadline_seconds`) live in `fl_runs.config` so the supervisor can also kill a run whose
   `updated_at` has not advanced past the deadline.

The **partial unique index** `(project_id) WHERE status IN ('PENDING','STARTING','RUNNING')` closes the
A1-F4 check-then-act race declaratively: the second concurrent `/start` for one project gets a constraint
violation → HTTP 409, with no application-level lock needed.

### 4.2 Why content-addressing (sha256) for datasets and models (C2 §3.2, C3 §5.5)

v1's "version key" was a pickle filename `ecg_clients{N}_alpha{alpha}_frac{frac}_seed{seed}.pkl`
(C2 §2.3) — change the data contents without changing N/alpha/frac/seed and you silently reuse a stale
split (a reproducibility trap). v2 keys versions on a **sha256 content hash**:

- `dataset_versions.content_hash = sha256(canonical schema + aggregate-stats manifest)` — raw data is
  client-private and the platform never sees it, so the hash covers metadata, not raw rows (C2 §6, "content
  hash of client-private data").
- `partition_recipes.recipe_hash = sha256(partitioner | num_partitions | alpha | data_seed | content_hash)`
  — the reproducibility key that *replaces* the pickle filename.
- `model_artifacts.sha256 = sha256(model bytes)` — the S3 object key is the hash, so the same bytes are
  stored once (deduplicated) and an artifact is immutable by construction.

### 4.3 Why two seed namespaces (C2 §2.3, C3 §5.1)

The v1 seed was **doubly load-bearing**: one global `seed=42` controlled both the data partition layout and
the DeComFL perturbation RNG, coupling two things that must be independently reproducible. v2 splits them:

- `partition_recipes.data_seed BIGINT` — the **data** seed, fed to `np.random.Generator(np.random.PCG64(...))`
  in the `DirichletLabelPartitioner`. Never the global RNG.
- `fl_runs.config -> 'seed'` and `determinism_manifests.seed` — the **optimizer** seed for DeComFL
  perturbations, used with the CPU-canonical `torch.Generator` (C3 §5.1, the bug-2 fix).

These are stored in different rows precisely so re-partitioning data does not perturb the optimizer's
randomness and vice versa.

---

## 5. The new Flyway migrations — exact filenames, order, and full DDL

Author these three files in `backend/fl-platform-api/src/main/resources/db/migration/`, in this exact
order. They depend on V1–V5 being present in that same directory (copy V4/V5 from `build/resources` first;
see §1 note).

| Order | Filename | Adds |
|---|---|---|
| V6 | `V6__dataset_registry.sql` | `datasets`, `dataset_versions`, `partition_recipes`; `projects.dataset_version_id` + `projects.partition_recipe_id`; the platform/project role `CHECK` constraints (A1-F1 guard). |
| V7 | `V7__fl_runs_and_artifacts.sql` | `fl_runs` (lease/state), `round_results` (incremental per-round, replaces V1 `round_result`), `model_artifacts`; backfill `round_result` → `round_results`; fix `audit_events.metadata` CLOB→JSONB. |
| V8 | `V8__determinism_manifest.sql` | `determinism_manifests` (1:1 with `fl_runs`). |

> **Why this split into three files and this order?** V6 is pure additive registry + guards (no data
> migration, lowest risk, lands first). V7 introduces the run aggregate and **must** run after V6 because
> `fl_runs` and `projects` FK the registry tables; it also performs the only data migration
> (`round_result` → `round_results`). V8 depends on `fl_runs` existing, so it is last. Keeping the
> determinism manifest in its own migration isolates a table whose columns (torch/proto versions) may
> evolve as the framework pins change — a future `V{n}` can alter it without touching the run lease.

### 5.0 PostgreSQL extension prerequisite

`fl_runs.config`, `audit_events.metadata`, and the registry use `JSONB`, which is built into PostgreSQL.
No extension is required for JSONB. `gen_random_uuid()` is used in defaults; it requires the `pgcrypto`
extension on older PostgreSQL but is **built-in since PostgreSQL 13**. The v2 target is RDS PostgreSQL `17.10`
(pinned in `02-TECH-STACK.md §5.1`), so `gen_random_uuid()` is available without an extension. Where the application assigns the
UUID (the JPA path), the default is harmless; where SQL inserts a row directly, it is used.

### 5.1 `V6__dataset_registry.sql`

```sql
-- =====================================================================
-- V6: Dataset / partition registry (content-hash keyed) + role CHECKs.
-- Drivers: C2 §3.2 (registry), C2 §2.3 (data_seed split), A1-F1 (role guard).
-- PostgreSQL only. No CLOB. UUID PKs for all new top-level entities (R-D).
-- =====================================================================

-- --- Role guards (A1-F1): make the bootstrap-admin lockout impossible at the DB. ---
-- platform_role was renamed from `role` in V5; constrain it now.
ALTER TABLE users ADD CONSTRAINT chk_users_platform_role
    CHECK (platform_role IN ('USER','PLATFORM_ADMIN'));

-- project_memberships.role had no DB guard in V4; add it.
ALTER TABLE project_memberships ADD CONSTRAINT chk_project_membership_role
    CHECK (role IN ('MEMBER','CLIENT','OWNER'));

-- --- datasets: per-org named dataset; the lineage root. ---
CREATE TABLE datasets (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id       UUID         NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    name         VARCHAR(255) NOT NULL,
    modality     VARCHAR(32)  NOT NULL
                 CHECK (modality IN ('TABULAR','IMAGE','TEXT','TIMESERIES')),
    created_by   BIGINT       REFERENCES users(id) ON DELETE SET NULL,
    created_at   TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    UNIQUE (org_id, name)
);
CREATE INDEX idx_datasets_org ON datasets(org_id);

-- --- dataset_versions: immutable, content-addressed. content_hash is the anchor. ---
CREATE TABLE dataset_versions (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id     UUID    NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    version        INTEGER NOT NULL,            -- monotonic per dataset, app-assigned
    content_hash   CHAR(64) NOT NULL,           -- sha256 (hex) of canonical schema+stats manifest
    schema_json    JSONB   NOT NULL,            -- declared schema contract (modality, feature_shape,
                                                --   feature_dtype, label_set, num_classes)
    sample_count   BIGINT,                      -- aggregate metadata only; raw rows never stored
    created_at     TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    UNIQUE (dataset_id, version),
    UNIQUE (dataset_id, content_hash)
);
CREATE INDEX idx_dataset_versions_dataset ON dataset_versions(dataset_id);

-- --- partition_recipes: reproducible non-IID split. data_seed is DISTINCT from optimizer seed. ---
CREATE TABLE partition_recipes (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_version_id UUID    NOT NULL REFERENCES dataset_versions(id) ON DELETE CASCADE,
    partitioner        VARCHAR(32) NOT NULL
                       CHECK (partitioner IN ('DIRICHLET_LABEL','DIRICHLET_QTY','SHARD','NATURAL')),
    num_partitions     INTEGER NOT NULL CHECK (num_partitions >= 1),
    alpha              DOUBLE PRECISION,        -- NULL for non-Dirichlet partitioners
    data_seed          BIGINT  NOT NULL,        -- DATA seed; fed to np.random.PCG64, NEVER global RNG
    recipe_hash        CHAR(64) NOT NULL,       -- sha256(partitioner|N|alpha|data_seed|content_hash)
    created_at         TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    UNIQUE (dataset_version_id, recipe_hash)
);
CREATE INDEX idx_partition_recipes_version ON partition_recipes(dataset_version_id);

-- --- Lineage: a project pins exactly one (dataset_version, partition_recipe). Nullable until set. ---
ALTER TABLE projects ADD COLUMN dataset_version_id  UUID REFERENCES dataset_versions(id);
ALTER TABLE projects ADD COLUMN partition_recipe_id UUID REFERENCES partition_recipes(id);
CREATE INDEX idx_projects_dataset_version  ON projects(dataset_version_id);
CREATE INDEX idx_projects_partition_recipe ON projects(partition_recipe_id);
```

### 5.2 `V7__fl_runs_and_artifacts.sql`

```sql
-- =====================================================================
-- V7: FL run lease/state aggregate + incremental round results + artifacts.
-- Drivers: A1-F2 (run entity/lease), A1-F4 (one-active-run race),
--          C1 (checkpoint/resume), C3 §5.3 (run lineage),
--          README §1.1 (incremental per-round POST, comm-cost wedge),
--          A1 audit-log note (metadata CLOB -> JSONB, §6).
-- PostgreSQL only. No CLOB.
-- =====================================================================

-- --- §6 FIX: audit_events.metadata was CLOB (not a PostgreSQL type). Convert to JSONB. ---
-- USING ... ::jsonb casts any existing text rows; CLOB rows were stored as text under H2.
ALTER TABLE audit_events
    ALTER COLUMN metadata TYPE JSONB USING (metadata::jsonb);

-- §6 hardening: align V5 TIMESTAMP columns to TIMESTAMPTZ for tz-correctness on real Postgres.
-- (See §6 reasoning. Safe: existing values are interpreted in the session tz.)
ALTER TABLE organizations ALTER COLUMN created_at TYPE TIMESTAMP WITH TIME ZONE;
ALTER TABLE organizations ALTER COLUMN updated_at TYPE TIMESTAMP WITH TIME ZONE;
ALTER TABLE organizations ALTER COLUMN deleted_at TYPE TIMESTAMP WITH TIME ZONE;
ALTER TABLE organization_memberships ALTER COLUMN created_at TYPE TIMESTAMP WITH TIME ZONE;
ALTER TABLE users ALTER COLUMN deleted_at    TYPE TIMESTAMP WITH TIME ZONE;
ALTER TABLE users ALTER COLUMN last_login_at TYPE TIMESTAMP WITH TIME ZONE;
ALTER TABLE audit_events ALTER COLUMN occurred_at TYPE TIMESTAMP WITH TIME ZONE;

-- --- model_artifacts: content-addressed (sha256) S3/MinIO objects, wired to runs. ---
CREATE TABLE model_artifacts (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id       UUID     NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    sha256       CHAR(64) NOT NULL,            -- sha256 (hex) of the model bytes; the S3 object key
    storage_uri  VARCHAR(512) NOT NULL,        -- e.g. s3://fedlearn-artifacts/<sha256>
    size_bytes   BIGINT   NOT NULL,
    kind         VARCHAR(16) NOT NULL
                 CHECK (kind IN ('INITIAL','CHECKPOINT','FINAL')),
    fl_run_id    UUID,                          -- nullable; FK added after fl_runs exists (below)
    round_idx    INTEGER,                       -- set for CHECKPOINT; NULL otherwise
    created_at   TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    UNIQUE (org_id, sha256)                     -- dedupe identical bytes within a tenant
);
CREATE INDEX idx_model_artifacts_org   ON model_artifacts(org_id);
CREATE INDEX idx_model_artifacts_run   ON model_artifacts(fl_run_id);

-- --- fl_runs: the durable run lease + state + lineage aggregate. id == run_id. ---
CREATE TABLE fl_runs (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),   -- the run_id
    project_id               UUID    NOT NULL REFERENCES projects(id)      ON DELETE CASCADE,
    org_id                   UUID    NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    status                   VARCHAR(16) NOT NULL DEFAULT 'PENDING'
                             CHECK (status IN ('PENDING','STARTING','RUNNING',
                                               'SUCCEEDED','FAILED','STOPPED')),

    -- Lease (reconciler ownership): the JVM is a stateless supervisor over this lease.
    lease_owner              VARCHAR(128),          -- supervisor instance id (pod name / host id)
    lease_expires_at         TIMESTAMP WITH TIME ZONE,

    -- Executor binding (FlServerLauncher abstraction).
    launcher                 VARCHAR(16) NOT NULL
                             CHECK (launcher IN ('K8S_JOB','ECS_RUN_TASK','LOCAL_PROCESS')),
    executor_ref             VARCHAR(512),          -- k8s Job name / ECS task ARN / local PID
    grpc_endpoint            VARCHAR(255),          -- host:port clients dial

    -- Progress + algorithm.
    round_idx                INTEGER NOT NULL DEFAULT 0,
    strategy                 VARCHAR(16) NOT NULL
                             CHECK (strategy IN ('DeComFL','FedAvg')),
    config                   JSONB   NOT NULL,      -- {seed,K,P,eta,mu,num_rounds,min_clients,
                                                    --  round_deadline_seconds,dp,...}; see §5.4

    -- Lineage pins (nullable: a dev run may not pin a registry version).
    dataset_version_id       UUID REFERENCES dataset_versions(id),
    partition_recipe_id      UUID REFERENCES partition_recipes(id),
    initial_model_artifact_id UUID REFERENCES model_artifacts(id),
    final_model_artifact_id   UUID REFERENCES model_artifacts(id),
    mlflow_run_id            VARCHAR(64),           -- link-out to MLflow tracking run

    -- Provenance / timestamps.
    requested_by             BIGINT REFERENCES users(id) ON DELETE SET NULL,
    created_at               TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    updated_at               TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    started_at               TIMESTAMP WITH TIME ZONE,
    ended_at                 TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_fl_runs_project       ON fl_runs(project_id);
CREATE INDEX idx_fl_runs_org           ON fl_runs(org_id);
CREATE INDEX idx_fl_runs_status        ON fl_runs(status);
-- Reconciler scan: find non-terminal runs whose lease has expired.
CREATE INDEX idx_fl_runs_lease_active  ON fl_runs(lease_expires_at)
    WHERE status IN ('PENDING','STARTING','RUNNING');
-- A1-F4 fix: at most ONE active run per project, enforced declaratively.
CREATE UNIQUE INDEX uq_fl_runs_one_active_per_project ON fl_runs(project_id)
    WHERE status IN ('PENDING','STARTING','RUNNING');

-- Now wire model_artifacts.fl_run_id as a real FK (fl_runs exists).
ALTER TABLE model_artifacts
    ADD CONSTRAINT fk_model_artifacts_run
        FOREIGN KEY (fl_run_id) REFERENCES fl_runs(id) ON DELETE SET NULL;

-- --- round_results: incremental per-round metrics, FK'd to the RUN (not the project). ---
-- Replaces the V1 `round_result` table. Adds comm-cost columns (the DeComFL wedge).
CREATE TABLE round_results (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fl_run_id             UUID    NOT NULL REFERENCES fl_runs(id) ON DELETE CASCADE,
    round_idx             INTEGER NOT NULL,
    loss                  DOUBLE PRECISION,
    accuracy              DOUBLE PRECISION,
    val_loss              DOUBLE PRECISION,
    val_accuracy          DOUBLE PRECISION,
    num_clients_reported  INTEGER,
    -- Communication-cost panel (README §1.1 "communication-cost panel … DeComFL's bandwidth wedge").
    uplink_bytes          BIGINT,            -- total client->server bytes this round
    downlink_bytes        BIGINT,            -- total server->client bytes this round
    scalars_transmitted   BIGINT,            -- DeComFL: count of ZO scalars (the O(1)-per-round proof)
    gpu_utilization       DOUBLE PRECISION,  -- carried from V1 round_result
    round_started_at      TIMESTAMP WITH TIME ZONE,
    round_ended_at        TIMESTAMP WITH TIME ZONE,
    created_at            TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
    UNIQUE (fl_run_id, round_idx)            -- one row per round; idempotent incremental POST
);
CREATE INDEX idx_round_results_run ON round_results(fl_run_id);

-- --- Data migration: synthesize one fl_run per project that has old round_result rows, ---
-- --- then copy the V1 rows into round_results under that synthetic run. ---
-- (C3 §9 risk 4: existing rows have no run; backfill a synthetic run per project.)
INSERT INTO fl_runs (id, project_id, org_id, status, launcher, strategy, config,
                     round_idx, requested_by, created_at, updated_at)
SELECT gen_random_uuid(), p.id, p.org_id, 'SUCCEEDED', 'LOCAL_PROCESS',
       COALESCE(NULLIF(p.optimizer,''),'FedAvg'),
       '{}'::jsonb,
       COALESCE((SELECT MAX(rr.server_round) FROM round_result rr WHERE rr.project_id = p.id), 0),
       p.user_id, now(), now()
FROM projects p
WHERE EXISTS (SELECT 1 FROM round_result rr WHERE rr.project_id = p.id);

INSERT INTO round_results (id, fl_run_id, round_idx, loss, accuracy, gpu_utilization, created_at)
SELECT gen_random_uuid(), fr.id, rr.server_round, rr.loss, rr.accuracy, rr.gpu_utilization, now()
FROM round_result rr
JOIN projects p ON p.id = rr.project_id
JOIN fl_runs  fr ON fr.project_id = p.id AND fr.launcher = 'LOCAL_PROCESS' AND fr.status = 'SUCCEEDED';

-- Drop the legacy V1 table now that its data is migrated.
DROP TABLE round_result;
```

> **Note on the strategy `CHECK` and the backfill:** the synthetic-run backfill coerces
> `projects.optimizer` to `'FedAvg'` when it is null/empty, because `fl_runs.strategy` is `NOT NULL` with a
> `CHECK` allowing only `'DeComFL'`/`'FedAvg'`. If your existing `projects.optimizer` rows hold values
> outside that set (e.g. `'adam'`), change the `COALESCE` expression to map them — verify the distinct
> values with `SELECT DISTINCT optimizer FROM projects;` against the target database **before** running V7.
> Flagged as uncertain: the exact legacy `optimizer` value domain is not knowable from the schema alone.

### 5.3 `V8__determinism_manifest.sql`

```sql
-- =====================================================================
-- V8: Determinism manifest (one per fl_run). Driver: C3 §5.2 / §5.3.
-- Captures everything needed to assert "this run started from THAT model
-- and THAT data split, with a CPU-canonical RNG and a pinned torch."
-- PostgreSQL only.
-- =====================================================================
CREATE TABLE determinism_manifests (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fl_run_id                   UUID NOT NULL UNIQUE REFERENCES fl_runs(id) ON DELETE CASCADE,
    framework_git_sha           VARCHAR(40)  NOT NULL,
    proto_version               VARCHAR(32)  NOT NULL,    -- 'fedlearn.v2'
    torch_version               VARCHAR(32)  NOT NULL,    -- e.g. '2.12.0' (the pinned torch, 02-TECH-STACK §4.1)
    torch_cuda_version          VARCHAR(16),              -- NULL when CPU-only
    rng_device                  VARCHAR(8)   NOT NULL DEFAULT 'cpu'
                                CHECK (rng_device = 'cpu'),  -- CPU-canonical invariant (C3 §5.1)
    rng_engine                  VARCHAR(64)  NOT NULL,    -- 'torch.Generator(cpu)'
    use_deterministic_algorithms BOOLEAN     NOT NULL DEFAULT FALSE,
    seed                        BIGINT       NOT NULL,    -- optimizer/perturbation seed
    initial_model_sha256        CHAR(64),                 -- lineage anchor (model start)
    dataset_split_sha256        CHAR(64),                 -- lineage anchor (data split)
    platform_os                 VARCHAR(16),              -- 'linux'
    platform_arch               VARCHAR(16),              -- 'x86_64' | 'arm64'
    manifest_json               JSONB        NOT NULL,    -- the full C3 §5.2 manifest, verbatim
    created_at                  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);
CREATE INDEX idx_determinism_manifests_run ON determinism_manifests(fl_run_id);
```

### 5.4 `fl_runs.config` JSONB contract (the local model must serialize exactly this shape)

`config` carries the user-facing run inputs that C3 §4.2 says were hidden constants in v1. The JPA entity
maps this to a typed record; the on-disk JSONB shape is:

```json
{
  "seed": 42,
  "num_rounds": 50,
  "min_clients": 2,
  "round_deadline_seconds": 600,
  "decomfl": { "K": 1, "P": 10, "eta": 0.001, "mu": 0.001 },
  "dp": { "enabled": false, "mechanism": "scalar_dp", "epsilon": null, "clip_norm": null },
  "robust_guard": { "enabled": false, "method": "trimmed_mean", "trim_fraction": null }
}
```

| Field | Type | Meaning | Source |
|---|---|---|---|
| `seed` | int | Optimizer/perturbation seed (CPU-canonical RNG). | C3 §5.2; spec design row "RNG determinism". |
| `num_rounds` | int | Total FL rounds. | A1 v2 target ("num_rounds"). |
| `min_clients` | int | Minimum-quorum: a round proceeds only with ≥ this many reporting clients. | README §1.1 "minimum-quorum". |
| `round_deadline_seconds` | int | Round deadline: the round closes after this even if not all clients reported (no infinite hang on a straggler). | README §1.1 "Round loop MUST have a deadline". |
| `decomfl.K` | int | Local steps per round. | spec `2026-05-29-decomfl-correctness-design.md` (K). |
| `decomfl.P` | int | Number of perturbations. The `1/P` averaging factor is the bug-1 fix; P=10 default. | spec bug-1. |
| `decomfl.eta` | float | Learning rate (η). | spec hyperparameters. |
| `decomfl.mu` | float | Smoothing radius (μ) for the ZO gradient estimate `g=(f(x+μz)−f(x))/μ`. | spec hyperparameters. |
| `dp` | object | Differential Privacy config (scalar-DP on DeComFL, DP-SGD on FedAvg). Disabled by default. | README §1 "Aggregation robustness/privacy". |
| `robust_guard` | object | Robust-mean/clipping guard. Disabled by default. | README §1 "robust-mean/clipping guard". |

> **Why `config` is JSONB, not columns:** the DeComFL/FedAvg/DP/robust-guard parameter set will evolve and
> differs per strategy (FedAvg has no K/P/η/μ). A single typed `JSONB` keeps the run aggregate stable while
> the strategy parameters change, and PostgreSQL JSONB is queryable/indexable if needed. The *stable*
> lineage fields that you query and join on (seed, strategy, versions) are promoted to real columns on
> `fl_runs`/`determinism_manifests`; only the volatile strategy hyperparameters live in `config`. This is
> the same split C3 §5.3 draws between "determinism manifest core" columns and the rest.

---

## 6. PostgreSQL portability fixes (CLOB → TEXT/JSONB, and TIMESTAMP → TIMESTAMPTZ)

The README (§3 conflict 7) and A1-F10 require fixing v1 schema that only *emulated* on H2-with-`MODE=PostgreSQL`
and breaks or misbehaves on real RDS PostgreSQL. There are two defects in V5:

| Defect | Where | Why it is wrong on PostgreSQL | Fix (in V7, §5.2) |
|---|---|---|---|
| `metadata CLOB` | `V5__identity_foundations.sql:51` | `CLOB` is **not a PostgreSQL type**. H2 accepts it under `MODE=PostgreSQL`; RDS does not. Audit log metadata is structured JSON, so it should be queryable. | `ALTER TABLE audit_events ALTER COLUMN metadata TYPE JSONB USING (metadata::jsonb);` |
| `TIMESTAMP` (no time zone) | `V5` `organizations`, `organization_memberships`, several `users` columns, `audit_events.occurred_at` | V1 used `TIMESTAMP WITH TIME ZONE` (the correct, tz-aware type the JPA `Instant` maps to); V5 regressed to bare `TIMESTAMP`. On real PostgreSQL bare `TIMESTAMP` drops the offset, so cross-region timestamps drift. | The `ALTER COLUMN ... TYPE TIMESTAMP WITH TIME ZONE` block in V7. |

> **Why fix these in V7 and not a separate migration:** they are part of the same "make the schema
> PostgreSQL-real before the managed-Postgres cutover" workstream, and V7 is already the data-migration
> migration (it touches `audit_events` is the only cross-cutting table). Keeping the fixes adjacent to the
> run-aggregate work makes the cutover a single reviewable diff. **Hard requirement (A1-F10):** add a
> Testcontainers-PostgreSQL CI profile that runs V1→V8 against real PostgreSQL `17.10`, because the `test`
> profile disables Flyway and uses H2 `create-drop` — so these migrations are otherwise *never* validated
> against the real dialect. Do **not** change the `test` profile's Flyway-disabled behavior (it must stay
> in-memory `create-drop` per the repo invariant); add a *new* Testcontainers profile alongside it.

---

## 7. JPA mapping notes (so the entity layer matches the DDL, validate-only)

JPA runs in `validate` mode against this schema; entities must match the columns exactly or boot fails.
Key mappings the local model must implement:

| Table | JPA entity | PK strategy | Notable column mappings |
|---|---|---|---|
| `fl_runs` | `FlRun` | `@Id UUID` (app-assigned or `gen_random_uuid()` default) | `config` and lineage-pin FKs; `status`/`strategy`/`launcher` as `@Enumerated(EnumType.STRING)` enums whose names match the `CHECK` sets exactly. |
| `round_results` | `RoundResult` (repointed FK) | `@Id UUID` | `@ManyToOne FlRun flRun` on `fl_run_id` (was `project` on `project_id` in V1; C3 §5.3 "re-point RoundResult at fl_run_id"). |
| `datasets` | `Dataset` | `@Id UUID` | `modality` enum. |
| `dataset_versions` | `DatasetVersion` | `@Id UUID` | `schema_json` as a `JdbcTypeCode(SqlTypes.JSON)` field. |
| `partition_recipes` | `PartitionRecipe` | `@Id UUID` | `data_seed` as `Long`. |
| `model_artifacts` | `ModelArtifact` | `@Id UUID` | `kind` enum; `sha256` `CHAR(64)`. |
| `determinism_manifests` | `DeterminismManifest` | `@Id UUID` | `@OneToOne FlRun` on `fl_run_id` (UNIQUE). |
| `audit_events` | `AuditEvent` | `@Id UUID` | `metadata` JSONB via `@JdbcTypeCode(SqlTypes.JSON)` (was `String`/CLOB). |

For JSONB columns use Hibernate 6's built-in JSON mapping:
`@JdbcTypeCode(org.hibernate.type.SqlTypes.JSON) private JsonNode config;` (or a typed record). Do not add
a third-party JSONB type library — Hibernate 6 (Spring Boot 3.5) maps JSONB natively.

---

## 8. Per-run scoped result-token note (data-model touchpoint for A1-F6)

The internal result callback `/api/internal/runs/{runId}/results` (the authoritative path is `04-API-CONTRACTS.md §5`) is, in v2, authenticated by a **per-run scoped
token**, not the single global `APP_INTERNAL_API_KEY` (A1-F6; README §1.1 "per-run scoped result tokens").
The token is signed by the backend and bound to `(fl_run_id, project_id, org_id)`; it is injected into the
launcher environment at launch. **It is not stored as a table** (it is a short-lived signed JWT minted at
launch and validated by signature, so there is no `result_tokens` table). The data-model consequence is
only that `fl_runs` carries the `(id, project_id, org_id)` triple the token binds to — which it does. The
local model does **not** add a token table; it mints/validates tokens in the security layer using these
three `fl_runs` columns as the binding claims.

---

## 9. Final checklist for the implementing model

Before declaring the data model complete, verify each item:

1. [ ] V4 and V5 are copied into `backend/fl-platform-api/src/main/resources/db/migration/` so the chain V1→V8 is complete in one directory (§1 note).
2. [ ] `V6__dataset_registry.sql`, `V7__fl_runs_and_artifacts.sql`, `V8__determinism_manifest.sql` exist with the DDL in §5, in that order.
3. [ ] No `CLOB` anywhere in V6–V8; `audit_events.metadata` is `JSONB` after V7 (§6).
4. [ ] Every tenant-owned new table (`datasets`, `model_artifacts`, `fl_runs`) has `org_id UUID NOT NULL` (R-C).
5. [ ] All new top-level PKs are `UUID`; user FKs are `BIGINT` (R-D); composite membership keys stay `(UUID, BIGINT)`.
6. [ ] `fl_runs` has the partial unique index `uq_fl_runs_one_active_per_project` (A1-F4) and the lease index `idx_fl_runs_lease_active`.
7. [ ] `round_results` is FK'd to `fl_run_id` (not `project_id`) and has `uplink_bytes`/`downlink_bytes`/`scalars_transmitted` (the comm-cost wedge).
8. [ ] `determinism_manifests` is 1:1 with `fl_runs` (`fl_run_id UNIQUE NOT NULL`) and enforces `rng_device = 'cpu'` (C3 §5.1).
9. [ ] The V7 backfill migrates V1 `round_result` rows into `round_results` under synthetic runs, then drops `round_result`; the `optimizer`→`strategy` coercion was verified against the target DB (§5.2 note).
10. [ ] `platform_role` and `project_memberships.role` `CHECK` constraints are added in V6 (A1-F1 DB guard).
11. [ ] A Testcontainers-PostgreSQL CI profile runs V1→V8 against real PostgreSQL `17.10`; the `test` profile's Flyway-disabled/H2 `create-drop` behavior is unchanged (§6).
12. [ ] JPA entities map JSONB via Hibernate 6 `@JdbcTypeCode(SqlTypes.JSON)`; enums via `@Enumerated(EnumType.STRING)` matching the `CHECK` sets exactly (§7).

---

*End of 03-DATA-MODEL.md. All claims about existing code cite file:line against `main-clean`; all design
decisions cite the v2 audit synthesis or its depth reports (A1-backend.md, C2-data-engineering.md,
C3-reproducibility.md) under `docs/audit/2026-05-29/`.*
