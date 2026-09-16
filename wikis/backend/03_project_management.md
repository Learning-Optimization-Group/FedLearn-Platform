# 03 - Project Management Lifecycle

This document explains how a Federated Learning Project is created, initialized, and managed within the Spring Boot backend.

> ✅ **Branch reality.** The **org-scoping and audit layers** described here — `projects.org_id` (the `V5` migration), `AuthorizationService`, `OrgScopeFilter`, `isPlatformAdmin()`, and the `@Auditable` annotations — are present on this branch (the `V4`–`V7` identity migrations). A project is owned by a `User` and pinned to an org via `projects.org_id`. See [06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md).
>
> Three things on this page changed after the page was first written and are described in their
> current form below, not their historical one: model **initialization is asynchronous** (BA-1) and
> `createProject` returns 201 while the project is still `INITIALIZING`; a project's **status is
> derived from its active run**, not read from the column (BA-4); and **direct deletion is
> platform-admin only** — owners file a deletion request an admin approves. The **training arm**
> (`projects.training_arm`, `V22`/`V23`) is new material with no earlier counterpart at all.

## 1. The `ProjectController` route surface

Everything below hangs off one controller. These are the routes that actually exist
(`controller/ProjectController.java`); membership, access-request and admin routes for a project live
on their own controllers and are catalogued in
[06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md).

| Method & path | Purpose |
|---|---|
| `POST /api/projects` | Create. Body `CreateProjectRequest`; returns **201** with the project already visible as `INITIALIZING`. |
| `GET /api/projects` | The caller's own projects (owned or joined), org-scoped. |
| `GET /api/projects/discover` | The discover feed — `DiscoverProjectDto`, org-scoped. |
| `GET /api/projects/{id}` | One project. 404 (not 403) when out of org scope. |
| `PATCH /api/projects/{id}` | Update `name` / `description` / `visibility` / `requirementsOverride`. Owner-or-admin. |
| `POST /api/projects/{id}/start` | Start a run. Body `StartProject`. |
| `POST /api/projects/{id}/stop` | Stop the active run. |
| `GET /api/projects/{id}/results` | Per-round results (`RoundResultDto`). |
| `GET /api/projects/{id}/logs` | One page of logs; `?page=&size=`, default size 200, clamped server-side. |
| `GET /api/projects/{id}/logs/export` | The whole log as a downloadable `.txt` attachment. |
| `POST /api/projects/{id}/deletion-request` | Owner files a deletion request for admin approval → **201**. |
| `GET /api/projects/{id}/deletion-request` | This project's deletion request, or **204** if none (drives the owner's badge). |
| `DELETE /api/projects/{id}` | Delete. **Platform-admin only** — see "Completion and Cleanup". |
| `POST /api/projects/{id}/delete` | `@Deprecated` legacy alias for the `DELETE` above, kept until the desktop and web clients stop calling it. |

## 2. Project Creation Flow

The lifecycle begins when a user submits a configuration via the React dashboard.

### The `CreateProjectRequest` payload
`POST /api/projects` accepts:

| Field | Notes |
|---|---|
| `name` | `@NotEmpty` (e.g. "Pneumonia Detection") |
| `modelType` | `@NotEmpty` — the recipe key (`PNEUMONIA_CNN`, `CNN`, `CIFAR_RESNET18`, `MLP`, `TRANSFORMER`, `LLM_LORA`, `TINYNET_GOLDEN`) |
| `modelName` | e.g. `"net"` |
| `optimizer` | e.g. `"Adam"` |
| `pretrainEpochs` | `@NotNull`, `@Min(0)` |
| `taskType` | `@Pattern(SEQ_CLASSIFICATION\|CAUSAL_LM)` — `LLM_LORA` only |
| `trainingArm` | `@Pattern(FULL\|FROZEN_HEAD\|OVA_LP)`; omitted means `FULL`. See §3 |
| `regulated`, `dpEnabled`, `dpTargetEpsilon`, `dpDelta`, `dpClipNorm` | SE-11 DP policy; completeness is a cross-field rule enforced in the service, not by bean validation |
| `initFromPretrained`, `baseRef`, `derivationSpec` | DA-14 derivation record; all optional, absent == a from-scratch recipe project |
| `requirementsOverride` | Owner may tighten the recipe's device requirements |

### `ProjectService.createProject()`

The method is `@Transactional` and `@Auditable(action = PROJECT_CREATED)`, so a successful creation
writes an `audit_events` row in the same transaction (see
[06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md)).

1. **Gate on the platform role.** `authz.requireCanCreateProject()` — only a `PROJECT_OWNER`
   (admin-granted through the owner-promotion workflow) or a `PLATFORM_ADMIN` may create projects at
   all. A plain `USER` gets a 403 telling them to request owner access.
2. **Validate the DP config (SE-11).** A `regulated` or `dpEnabled` request must carry a complete,
   sane config (`epsilon > 0`, `delta` in (0,1) exclusive, `clipNorm > 0`) — rejected before anything
   is persisted.
3. **Build the entity**, including the training arm (§3), the DP knobs and the derivation record.
4. **Pin to an Organization:** since `V5` made `projects.org_id` **NOT NULL**, the project is pinned
   to the owner's first org membership, falling back to the single `DEFAULT_ORG_ID` for
   membership-less users. `visibility` defaults to `PRIVATE`.
5. **Persist the shell as `INITIALIZING`** (`init_status`, `V14`) and generate the target model path
   `models/<uuid>.npz`.
6. **Dispatch model init asynchronously (BA-1)** and return **201 immediately**. This is the part
   most older prose gets wrong: model initialization spawns an unbounded Python process, and running
   it inside the `@Transactional` request pinned both a DB connection and a Tomcat thread for its
   whole duration. It now runs on a bounded async worker (`ModelInitializationWorker`), dispatched
   from an `afterCommit` transaction synchronization so the worker's own unit of work can see the
   committed row. The worker transitions `init_status` to `DONE` (success) or `FAILED`
   (timeout/error) and broadcasts the change for a polling client. A failed init therefore leaves a
   visible `FAILED` project the owner can inspect and delete — the request no longer rolls the row
   back.

Because status is derived from the active run (BA-4) and no run exists yet, a project mid-init would
otherwise read as the idle `CREATED`; `ProjectStatusService` consults `init_status` **first** and
reports `INITIALIZING`, which the SPA renders as a "Preparing" pill.

> Step 5–6 describe the project's *initial* model file only, written once at creation. They predate,
> and are unrelated to, the content-addressed model artifact registry that now also records every run's
> *trained* output as a versioned, provenance-tracked row (`model_artifacts`) rather than treating this
> `.npz` as the sole, overwritable record. See [07 - Content-Addressed Model Artifact Registry](07_artifact_registry.md)
> for the write path (registration on run completion) and the read path (inference/warm-start now prefer
> the registry over this file when an artifact exists).

## 3. The training arm (`projects.training_arm`)

An **arm** says which parameters a run trains and federates — and, since OvA-LP, what objective it
trains them under. It is a property of the *project*, fixed at creation, and it is the newest
first-class concept in this subsystem.

### Why it is stored at all

Before `V22` the arm was not stored anywhere. `fl-runtime/client.py` inferred it from the recipe key
(`USE_DERIVED = (mt == "FROZEN_DEMO")`), which had three consequences: every recipe had exactly one
hard-coded arm; one recipe could not be run under both arms as the two halves of a comparison; and a
result could not say which arm produced it. That last one is a real, observed bug class — commit
`21699bc` ("frozen arm silently mislabelled its backbone, risking cell overwrites"): when the arm is
implicit, two different experiments write the same result cell.

### The vocabulary

`model/TrainingArm.java` is the Java vocabulary — three constants:

| Arm | What it trains / federates |
|---|---|
| `FULL` | Every parameter is trainable; the whole model rides the wire. |
| `FROZEN_HEAD` | The backbone is frozen and only the head trains, so the wire carries the head alone. |
| `OVA_LP` | One-vs-all linear probing (arXiv:2511.05028) — the *same* frozen encoder and the *same* federated parameter subset as `FROZEN_HEAD`, but trained under C independent one-vs-all binary classifiers instead of one softmax. |

`OVA_LP` is the first arm that differs from another arm in its **objective** rather than its
parameter subset, which is why an arm carries an objective and not merely a trainable set. The Python
catalog is explicit about it: `fl-runtime/recipes.py` declares
`TRAINING_ARMS = ("FULL", "FROZEN_HEAD", "OVA_LP")`,
`ARM_OBJECTIVES = {"FULL": "cross_entropy", "FROZEN_HEAD": "cross_entropy", "OVA_LP": "one_vs_all"}`
and `DEFAULT_ARM = "FULL"`. (The paper's two-stage schedule is *not* implemented; see that file's
`arm_notes`.)

**The enum is the vocabulary; the recipe catalog is the authority.** Which arms a given recipe can
actually run is declared per recipe in `recipes.py` as `supported_arms` / `trainable_spec` —
`PNEUMONIA_CNN` and `CNN` offer `[FULL, FROZEN_HEAD]`, `CIFAR_RESNET18` offers all three, and
`MLP` / `TRANSFORMER` / `LLM_LORA` / `TINYNET_GOLDEN` offer `[FULL]` only. The backend deliberately
does **not** duplicate that table.

### How an arm travels

```
CreateProjectModal (picker, with the measured trade-off attached)
   └─ POST /api/projects  { "trainingArm": "FROZEN_HEAD", ... }
         └─ CreateProjectRequest  @Pattern(FULL|FROZEN_HEAD|OVA_LP)      [400 on anything else]
              └─ ProjectService.createProject → TrainingArm.valueOf(...)
                   └─ projects.training_arm   NOT NULL DEFAULT 'FULL'    [V22/V23 CHECK]
                        ├─ FlServerManager argv: --training-arm <ARM>    (emitted only when != FULL)
                        ├─ ClientConnectionDto.trainingArm               (always stated, never null)
                        └─ ProjectResponseDto.trainingArm                (always stated, never null)
```

Four properties of that chain are load-bearing:

- **Omission means `FULL`, through one code path.** `createProject` sets the arm only when the
  request names one, leaving the entity default (`TrainingArm.FULL`) to supply it otherwise — so a
  pre-arm client keeps its exact previous behaviour without a second branch. `FlServerManager`
  mirrors this on the argv side: `--training-arm` is emitted **only** when the arm is not `FULL`, so
  every pre-existing spawn's command line is byte-identical to before, and both `fl_server.py` and
  `client.py` resolve an omitted arm to `FULL`.
- **The response and the connection payload always state the arm explicitly.** A UI that had to read
  `FULL` out of silence would be re-deriving a default the server already knows, and a frozen project
  would look identical to a full one in every list view. `ClientConnectionDto.trainingArm` matters
  more than cosmetically: the server filters its parameters to the arm's trainable subset, so a
  client that did not know the arm would upload a full state dict against a server expecting the head
  alone.
- **The arm is immutable after creation.** `StartProject` also carries a `trainingArm` (same
  `@Pattern`), but `/start` may only *restate* the project's arm — a mismatch throws
  `ProjectStateException` with a message telling the caller to create a separate project for the
  other arm. Honouring a different arm at start would make two runs of one project incomparable
  while they still share a project identity; silently ignoring it would be a contract that looks like
  it works. Neither is acceptable, so it is refused.
- **The recipe catalog validates the *combination*, on the Python side.** The DTO pattern only bounds
  the vocabulary; whether `PNEUMONIA_CNN` supports `OVA_LP` is answered by
  `recipes.validate_arm()`, not by Java.

### What the picker is shown

`GET /api/model-recipes` (`ModelRecipeController` → `ModelRecipeService`, which shells out to
`recipes.py --describe` once and caches the result for the JVM's lifetime) carries two arm fields on
each `ModelRecipeDto`:

- **`supportedArms`** — that recipe's own list, `null` for a recipe that declares none.
- **`armTradeoff`** — attached **only to recipes that offer a choice of arms**, and generated from
  the measurement campaign's verdict record by `scripts/build_arm_tradeoff.py` into
  `fl-runtime/arm_tradeoff.json`. It is never hand-written, so what the picker claims cannot drift
  from what was measured. The DTO deliberately carries the `caveats` list alongside the numbers,
  because a headline figure shown without them is a claim the measurement does not support (the
  communication ratio is round-budget dependent, and accuracy and on-device latency were measured on
  different hardware).

Two typing decisions in `ArmTradeoff` exist because of specific near-misses: `commRatio` and
`ondeviceRatio` are `Double`, not `Integer` — every measured ratio *happened* to be a large whole
number (3,321x) until `PNEUMONIA_CNN` was measured on the product path at **1.004x**, since that
recipe's classifier is 99.6% of its parameters and freezing saves almost nothing; Jackson coerces a
JSON float into an `Integer` by truncating, which would have quietly rounded that honest figure to
`1`. And `null` means **not measured** — it must never become `0`, which would read as "no saving
measured" rather than "no measurement taken".

There is **no hardcoded Java fallback** for the catalog (DA-10). An earlier duplicate had already
drifted from `recipes.py`, and since the app spawns Python for all training and inference anyway, a
broken catalog should surface loudly rather than be masked by stale data: a load failure throws
`IllegalStateException` and is not cached, so a transient problem recovers on retry.

### The CHECK-constraint contract

`V22` adds the column *and* `chk_projects_training_arm`; `V23` drops and re-adds it widened to
`('FULL','FROZEN_HEAD','OVA_LP')`. The constraint is deliberately narrow and is described in the
migration as the last line of defence: DTO validation can be bypassed by any direct writer (a
migration, an ops script, a future service), and an unrecognised arm would otherwise reach the Python
runtime and fail at FL-server spawn rather than at write time.

That creates a split-brain hazard — a Java enum constant the application believes is valid and the
database rejects — and **`V22TrainingArmMigrationTest` is the guard**. It asserts:

- the column exists, is `NOT NULL`, and its default contains `FULL`;
- an insert that omits the arm lands on `FULL` (the path every pre-existing row took at migration);
- `FROZEN_HEAD` and `OVA_LP` round-trip;
- an unknown arm (`SEMI_FROZEN`) is rejected by the database, with `training_arm` in the message;
- **every `TrainingArm.values()` constant is accepted by the constraint** — so adding a constant
  without shipping a widening migration fails this test instead of surfacing as a write error inside
  a user's federation.

`ArmCatalogAndCreationTest` is its edge-side counterpart: it pins that every enum constant is also
accepted by `CreateProjectRequest`'s `@Pattern`, that an unknown arm is rejected at the edge rather
than at spawn, that `supported_arms` and the measured `arm_tradeoff` survive parsing into
`ModelRecipeDto` (both were silently dropped by `@JsonIgnoreProperties(ignoreUnknown = true)` before
P1-4), and that a fractional communication ratio is not truncated on the wire — Jackson coerces a
JSON float into an `Integer` by truncating, which would have rounded PNEUMONIA_CNN's honest measured
`1.004x` saving to `1`.

## 4. Project Ownership, Membership, and Org Isolation

A project is no longer just "owned by one user." It belongs to an
**Organization** (`org_id` NOT NULL), has a **visibility** (three tiers —
`PUBLIC`/`RESTRICTED`/`PRIVATE`), and can have **project memberships**
(`OWNER`/`MEMBER`/`CLIENT`) and **access requests**. Authorization is centralised in `AuthorizationService` and applies two
layers in sequence.

### Layer 1 — Org-scope (multi-tenant isolation)

Before any ownership check, mutating paths call `authz.requireOrgScope(project.getOrgId())`,
which throws **403** if the project's org is outside the caller's request-scoped
`OrgScope`. Pure-read paths use the boolean `authz.isInOrgScope(...)` (or
`orgScope.allows(...)`) and instead return **404**, so a caller can't even learn
that a cross-tenant project exists.

### Layer 2 — Ownership / membership

```java
public void requireOwnerOrAdmin(Project project) {
    if (isPlatformAdmin() || isOwner(project)) return;
    throw new AccessDeniedException("You do not have access to this project");
}
```

`isPlatformAdmin()` checks the `ROLE_PLATFORM_ADMIN` authority (platform admins
bypass both layers). The service also offers `requireOwnerOrMemberOrAdmin` and
`requireParticipant` for read endpoints that members/clients may see. An
unauthorized `start`/`stop`/`delete` throws `AccessDeniedException` → 403.

### Visibility, memberships & access requests

- **Visibility** (`PATCH /api/projects/{id}`) moves a project between the three
  `ProjectVisibility` tiers — `PUBLIC` / `RESTRICTED` / `PRIVATE`; participants
  are notified on change, skipping the membership rows whose role is `OWNER`
  (the owner's own lazily-created `OWNER_SELF` row).
- **Memberships** (`/api/projects/{id}/memberships`) — owners/admins add/remove
  any role; a `MEMBER` may add/remove only `CLIENT`. Member add/remove are
  `@Auditable`.
- **Access requests** (`/api/projects/{id}/access-requests`) — behaviour branches
  on the tier: **PUBLIC** creates a `CLIENT` membership immediately;
  **RESTRICTED** upserts a `PENDING` request the owner decides; **PRIVATE**
  returns the same **404** a nonexistent project would, so a non-participant
  cannot use the error code as an existence oracle.

See [06 - Identity, Multi-Tenancy & Audit](06_identity_multitenancy_and_audit.md)
for the full membership/access-request workflow and the org-scope mechanism.

### List views are org-scoped

`getProjectsForCurrentUser()` no longer returns a flat owned-or-joined list:
unless the caller is an unrestricted platform admin, it runs the org-scoped
`ProjectRepository.findOwnedOrMemberOfInOrgs(userId, orgScope.visibleOrgIds())`,
so the "My Projects" dashboard only shows projects from the caller's visible
orgs. The discover feed (`GET /api/projects/discover`) is likewise constrained via
`findDiscoverableInOrgs`.

## 5. Starting the Training Server

`POST /api/projects/{projectId}/start` kicks off the machine-learning phase. A `StartProject` body is
required by the controller, but every field *in* it is optional — the service supplies defaults — and
every field it does carry is bounded, so a pathological input (`numRounds = Integer.MAX_VALUE`) is
rejected by bean validation before it can reach the spawn path, where it could exhaust resources or
be interpolated into a command line.

| Field | Constraint | Default if omitted |
|---|---|---|
| `strategy` | `@Pattern(FedAvg\|DeComFL\|FedProx\|FedOpt\|Robust\|FoT)` | derived by `resolveStrategy(modelType, …)` |
| `trainingArm` | `@Pattern(FULL\|FROZEN_HEAD\|OVA_LP)` | the project's own arm (and it may only restate it — §3) |
| `numRounds` | `@Min(1)` `@Max(100)` | 5 |
| `minClients` | `@Min(1)` `@Max(100)` | 1 |
| `clientsPerRound` | bounded | `minClients` |

`FedLoRA` is deliberately **not** in that regex: it is derived server-side for `LLM_LORA` runs by
`resolveStrategy` and is never user-submitted. `FoT` (Federation over Text) is not a gradient
strategy at all — it spawns the standalone `fl_fot_server.py` through its own wrapper property.

The sequence:

1. Load the project, then enforce **org-scope → ownership**
   (`requireOrgScope` → `requireOwnerOrAdmin`). `startServerForProject` is
   `@Auditable(action = RUN_STARTED)`; `stopServerForProject` is `@Auditable(action = RUN_STOPPED)`.
2. Reconcile the requested arm against the project's arm (§3) — restating is fine, disagreeing is a
   `ProjectStateException`.
3. **Take the per-project start lock (BA-2).** The running-check, `Run` creation and spawn are one
   atomic critical section: without it two concurrent `/start` calls both saw
   `isServerRunning == false` and both spawned, leaving one server orphaned and untracked. The loser
   now gets a deterministic 409 (`ProjectStateException`) instead.
4. Create the `Run` row (`RunService.createForStart`), set it as the project's `activeRunId`, then
   call `FlServerManager.startServerForProject(...)` — see
   [04 - Federated Orchestration](04_federated_orchestration.md). If anything after run creation
   throws, the run is marked `FAILED` rather than left dangling.
5. Persist the reserved `serverPort`, mark the run `RUNNING`, and (flag-gated, phone-only, MO-15)
   auto-stage the run's on-device model bundle on a background worker so a mobile client that joins
   finds it at `GET /api/runs/{runId}/model-bundle`. That stage call is wrapped defensively: a
   scheduling failure must never fail a start.
6. Broadcast a `ProjectStatusUpdateDto` over STOMP so the React dashboard updates instantly. The
   status it carries is the **derived** one from `ProjectStatusService`, not the raw column.

## 6. Completion and Cleanup

### Marking as Completed
When the FL Server finishes all its federated rounds, it calls back to the internal API.
`ProjectService.markProjectAsCompleted()` clears the active `serverPort`, marks the active run
`COMPLETED`, and broadcasts the derived status.

### Deletion — admin-only, with an owner-facing request workflow

Direct deletion is **platform-admin only**: `deleteProject` calls `authz.requireOrgScope(...)` then
`authz.requirePlatformAdmin()`, not `requireOwnerOrAdmin`. An owner instead files a request:

```
owner  → POST /api/projects/{id}/deletion-request        (201, PENDING; ProjectDeletionService)
admin  → GET  /api/admin/deletion-requests?status=PENDING
admin  → PUT  /api/admin/deletion-requests/{id}          (APPROVED → calls deleteProject in the admin's context)
```

`GET /api/projects/{id}/deletion-request` returns the project's outstanding request, or **204** when
there is none — that is what drives the owner's "deletion pending" badge.

When `deleteProject` does run:

1. `@Auditable(action = PROJECT_DELETED)` records it.
2. Under the **same per-project start lock** used by `/start` (BA-13), it makes a best-effort attempt
   to stop any running FL server via `FlServerManager.stopServerForProject()` — local Python
   processes tracked in a `ConcurrentHashMap<UUID, ProcessHandle>` and torn down with
   `destroyForcibly()`. Holding the start lock is what stops a concurrent start from spawning and
   tracking a child *after* the stop and *before* the row disappears. (A residual race is documented
   in the code: `startServerForProject` loads the project before acquiring the lock, so a start
   already past its load could still race a delete; closing that needs an in-lock existence
   re-check.)
3. The row is deleted, and FK cascades take the children with it: `ServerLog`, `RoundResult`,
   `ProjectMembership`, `ProjectAccessRequest`, `ProjectDeletionRequest`, the benchmark tables, and —
   since `V19` — the `runs` / `run_enrollments` sub-tree. Before `V19` those two FKs had no
   `ON DELETE` action, so **any project that had ever been started could not be deleted at all**:
   Postgres raised `23503` on `runs_project_id_fkey`, surfaced as an opaque 409.
   Registry rows are deliberately *not* cascaded — `model_artifacts.project_id`/`run_id` are
   `ON DELETE SET NULL` (a provenance row outlives its producer) and `artifact_blobs` are shared, so
   they are never garbage-collected here. See
   [07 - Content-Addressed Model Artifact Registry](07_artifact_registry.md).
