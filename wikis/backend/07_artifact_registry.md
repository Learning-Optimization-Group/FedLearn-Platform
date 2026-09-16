# 07 - Content-Addressed Model Artifact Registry

This page documents the model artifact registry — the subsystem that replaced the platform's original
"one overwritable `.npz` at `projects.model_path`" design with a versioned, content-addressed,
provenance-tracked store. It shipped in slices tagged `DA-1` through `DA-9`, `BA-11`, `SE-11`, and
`FE-12` in the code; this page describes the shipped result, not the roadmap.

> **What changed, in one line.** Before this subsystem, `Project.modelPath` (a single mutable file
> path, `backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/model/Project.java:34`)
> was the only record of a project's model — each completed run overwrote it, with no history, no
> dedup, and no way to say "this LoRA adapter was trained over that base." The registry adds a second,
> parallel source of truth: an immutable, sha256-addressed blob store plus an append-only provenance
> table with a lineage DAG. `projects.model_path` is **not removed** — it is still written every round
> as the training-loop's working file — but reads (inference, warm-start) now prefer the registry when
> an artifact exists, and a run's *final* model is additionally registered as a durable, listable row.
> See "Retiring the `.npz`-overwrite gap" below for exactly what is superseded and what still legitimately
> uses `.npz`.

## 1. The data model

Four tables/entities, split so identical bytes dedup independently of who produced them:

| Concept | Type | Table | Source |
|---|---|---|---|
| Immutable blob | `ArtifactBlob` | `artifact_blobs` (sha256 PK) | `backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/model/ArtifactBlob.java:13` |
| Provenance record | `ModelArtifact` | `model_artifacts` (UUID PK) | `backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/model/ModelArtifact.java:18` |
| Lineage edge | `ArtifactLineage` | `artifact_lineage` (UUID PK) | `backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/model/ArtifactLineage.java:15` |
| What an artifact *is* | `ArtifactKind` enum | — | `backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/model/ArtifactKind.java:7` |

**`ArtifactBlob`** (`artifact_blobs`) is keyed by the lowercase-hex sha256 of its bytes
(`ArtifactBlob.java:16-18`) and carries no tenant or provenance semantics — `sizeBytes` and a `backend`
discriminator (`'LOCAL_FS' | 'S3'`, `ArtifactBlob.java:23-25`) only. Identical bytes from any org or run
collapse to one row (`backend/fl-platform-api/src/main/resources/db/migration/V12__model_artifact_registry.sql:25-31`).

**`ModelArtifact`** (`model_artifacts`) is the per-org, per-run provenance row that points at a blob by
`blobSha256` — deliberately **not unique** (`ModelArtifact.java:29-30`), so two orgs or two runs can
record the same bytes as distinct provenance rows over one deduplicated blob
(`model/ModelArtifact.java:10-11`; schema comment at `V12__model_artifact_registry.sql:10-14`). Key
columns: `orgId` (NOT NULL, tenant pin), `kind` (`ArtifactKind`), nullable `projectId`/`runId` (FK
`ON DELETE SET NULL` — an artifact outlives its producer, `ModelArtifact.java:36-42`,
`V12__model_artifact_registry.sql:38-39`), `recipeKey`, `baseModelRef`, `licenseTag`, `evalCardJson`
(freeform eval-card JSON as TEXT), and `published`/`publishedAt` (added by `V18`, marketplace-only —
see §4). Rows are never updated in place: "a new model is a new row" (`ModelArtifact.java:14`).

**`ArtifactLineage`** (`artifact_lineage`) is a directed `child → parent` edge under a
`relationship` (`LineageRelationship` — `ADAPTER_OF`, `DERIVED_FROM`, `CONTINUED_FROM`;
`backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/model/LineageRelationship.java`).
`UNIQUE(child_id, parent_id, relationship)` plus a `CHECK (child_id <> parent_id)` forbid duplicate
edges and self-loops (`V12__model_artifact_registry.sql:57-65`); FKs to `model_artifacts` are
`ON DELETE RESTRICT` so an edge never dangles (lineage rows are as append-only as the artifacts they
connect).

**`ArtifactKind`** (`ArtifactKind.java:7-14`) has exactly three values:

| Kind | Meaning | Lineage wired on register |
|---|---|---|
| `FULL_CHECKPOINT` | A complete model checkpoint (imaging CNN, etc.) — the air-gap/export unit | `CONTINUED_FROM` the project's prior `FULL_CHECKPOINT` head, if one exists |
| `LORA_ADAPTER` | A federated LoRA/PEFT adapter over a frozen base — the tradable marketplace unit | `ADAPTER_OF` a deduped `BASE_REF`, plus `CONTINUED_FROM` the prior `LORA_ADAPTER` head, if any |
| `BASE_REF` | A reference to a frozen base model an org hosts/uses — many orgs may share one blob | none (it *is* a lineage root) |

A `BASE_REF`'s "content" is a small JSON reference manifest (`{"base_model_ref": ..., "license": ...}`),
**not** the base model's weights, which live upstream — it exists purely so an adapter has something
content-addressed to point `ADAPTER_OF` at, and so the same base dedups across orgs at the blob layer
(`backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/service/ArtifactRegistryService.java:162-183`).
`findOrCreateBaseRef` is a **private** helper of `ArtifactRegistryService`, not a standalone public API —
it is invoked only from inside `register()` when a `LORA_ADAPTER` is registered
(`ArtifactRegistryService.java:109-111`).

**It is race-safe now, and was not originally (V21).** The first implementation was a non-atomic
read-then-insert with *no* backing constraint, so two concurrent adapter registrations in one org over
the same base could each see "absent", each `save()` a `BASE_REF`, and leave their adapters'
`ADAPTER_OF` edges forked across two rows that were supposed to be one. The current shape is:

1. a read-first fast path (`findFirstByOrgIdAndBaseModelRefAndKind`) that avoids the manifest blob
   write on the common hit;
2. otherwise `ModelArtifactRepository.insertBaseRefIfAbsent(...)` — an atomic
   `INSERT … ON CONFLICT DO NOTHING` — where the loser of the race is a silent no-op;
3. a re-read of the single surviving row, which throws `IllegalStateException` if it is somehow still
   missing.

What makes (2) work is the **partial unique index** `uq_base_ref_org_model ON model_artifacts (org_id,
base_model_ref) WHERE kind = 'BASE_REF'` from `V21` — partial, so `LORA_ADAPTER` / `FULL_CHECKPOINT`
rows (which legitimately share those columns) stay unconstrained. `V21` assumes no pre-existing
duplicates and fails loudly if a deployed database already holds some.

## 2. Storage: `ArtifactBlobStore`

The blob store is a small interface (`put`/`get`/`exists`/`backendId`,
`backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/service/ArtifactBlobStore.java:11-24`)
with one implementation, `LocalFsArtifactBlobStore`
(`backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/service/LocalFsArtifactBlobStore.java:22`):

- **Content IS the key.** `put(byte[])` computes the sha256 of the bytes itself — a caller can never
  choose or spoof the key (`LocalFsArtifactBlobStore.java:37-38`).
- **Write-once, idempotent.** If the target path already exists, `put` is a no-op success — writing
  identical bytes twice is not an error (`LocalFsArtifactBlobStore.java:40-42`).
- **Two-level fan-out.** Blobs live at `root/<first-2-hex>/<next-2-hex>/<64-hex-sha256>`
  (`LocalFsArtifactBlobStore.java:86-89`), avoiding one giant flat directory.
- **Atomic write.** Content goes to a temp file in the same directory, then `Files.move(..., ATOMIC_MOVE)`
  — a crash mid-write can never leave a partial blob at the content-addressed path
  (`LocalFsArtifactBlobStore.java:43-54`). A losing writer in a write race just discards its temp file,
  since the winner already wrote identical bytes (`LocalFsArtifactBlobStore.java:49-50`).
- **Integrity-checked on every read.** `get` recomputes the sha256 of the bytes it read and throws
  `IllegalStateException` if it doesn't match the requested key — bit-rot or a swapped file fails loud
  rather than silently serving the wrong weights under the right id
  (`LocalFsArtifactBlobStore.java:70-78`).

Configured root: `app.artifact-store.root` (default `artifact-store`,
`backend/fl-platform-api/src/main/resources/application.properties:207`).

## 3. Write path — registering a run's final model

```
fl_server.py (run completes)
    │  _register_model_artifact()  /  _emit_and_register_lora_bundle()
    │  POST /api/internal/projects/{projectId}/artifacts
    │  multipart: model bytes + kind + recipeKey + baseModelRef? + licenseTag? + evalCard?
    ▼
InternalArtifactController      (X-Internal-Key + X-Internal-Run-Token gated — see 02; controller/InternalArtifactController.java:34)
    │  registry.registerForProject(projectId, bytes, kind, ...)
    ▼
ArtifactRegistryService.registerForProject  (resolves project -> orgId, activeRunId)
    │
    ├─ ArtifactRegistryService.register()
    │     ├─ LORA_ADAPTER without a baseModelRef                             [400]
    │     ├─ SE-11 gate: requireAccountantTraceForDpClaim(evalCardJson)      [may throw 400]
    │     ├─ idempotency: findByRunIdAndKind(runId, kind) -> return the existing row, done
    │     ├─ look up the project's PRIOR head of this kind (pre-insert)
    │     ├─ blobStore.put(bytes) -> sha256           ──▶  artifact_blobs (ArtifactBlobStore)
    │     ├─ INSERT model_artifacts row                ──▶  model_artifacts
    │     ├─ if LORA_ADAPTER: findOrCreateBaseRef(...) ──▶  artifact_lineage (ADAPTER_OF -> BASE_REF)
    │     └─ if a prior head existed                   ──▶  artifact_lineage (CONTINUED_FROM -> prior head)
    ▼
201 { "id": <artifact uuid>, "sha256": <content address> }
```

(`ArtifactRegistryService.java:67-118`; the internal endpoint itself is
`backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/controller/InternalArtifactController.java:34-50`.)

**The idempotency short-circuit is not incidental.** `V12` declares
`uq_model_artifact_run_kind UNIQUE (run_id, kind)` — a run produces at most one artifact of each kind.
Without the pre-check, a retried completion callback (the first POST committed, then the network timed
out) would 500 on that constraint rather than returning the artifact it already created. `run_id` is
NULL for `BASE_REF` and imported artifacts, and Postgres treats NULLs as distinct, so the constraint
does not bind them.

Two Python-side call sites feed this endpoint (`fl-runtime/fl_server.py`):

- **Non-LoRA recipes** (`FULL_CHECKPOINT`): `_register_model_artifact` posts the run's `--model-path`
  `.npz` bytes directly (`fl_server.py:82-116`, called from the final-save block at
  `fl_server.py:1147-1152`) — a full checkpoint's wire format *is* the imaging air-gap `.npz` by
  design (see `framework/src/fedlearn/bundle/BUNDLE_FORMAT.md`, "Serialization"), so registering those
  exact bytes is correct, not a gap.
- **`LLM_LORA` recipes**: `_emit_and_register_lora_bundle` serializes the adapter as safetensors
  (`adapter_to_safetensors`), builds a versioned bundle manifest whose `artifact_sha256` is the hash of
  those exact bytes, and registers **the safetensors bytes** — not the `.npz`
  (`fl_server.py:119-144`; DA-9 bullet 3, see `framework/src/fedlearn/bundle/BUNDLE_FORMAT.md`). On any
  failure building the bundle it falls back to registering the `.npz` so the run is still recorded
  (`fl_server.py:140-143`).

Registration is **additive and non-fatal**: the legacy `projects.model_path` `.npz` write happens
first and is unconditional; the registry POST is wrapped in a broad `try/except` that only logs on
failure (`fl_server.py:102-116`) — a registry outage can never abort a real federated run. Building
the eval card is separately guarded too: a card that cannot be built degrades to `None` rather than
failing the registration.

An eval card is attached at registration time, built from the run's own history/strategy
(`build_eval_card`, `fl_server.py:147-194`). It carries `recipe_key`, `strategy`, `rounds`,
`final_loss`, `final_accuracy`, `torch_version`, `seed`, `framework` — and, since the training arm
landed, two provenance fields worth calling out:

- **`training_arm`**, resolved through `recipes.validate_arm(...)` and recorded **explicitly even when
  it is `FULL`**. The card travels independently of the project row (it is attached to a registered
  artifact, which outlives its producer), so a reader must be able to answer "which arm produced
  this?" from the card alone. Recording `FULL` by *absence* would make a full fine-tune
  indistinguishable from a card written before arms existed.
- **`trainable_prefixes`** — the actual module-name prefixes, not just the arm's name. Two runs can
  share an arm name while freezing different modules, so the name alone is not a checkable provenance
  claim.

If the strategy ran differential privacy, `SE-11` requires the card's
`dp.accounted_epsilon`/`dp.delta` to be present and numeric before the registry will accept a
`dp.enabled: true` claim — an unaccounted DP claim is rejected with `IllegalArgumentException` → 400
(`ArtifactRegistryService.java:188-224`). Three deliberate non-behaviours in that gate: a card with no
`dp` section, a card with `dp.enabled != true`, and a card that is not parseable JSON all pass through
untouched — the last because an unparseable card cannot carry a machine-readable DP claim to police,
and the card is stored opaque by pre-existing contract. When DP is off, `build_eval_card` emits no
`dp` key at all. On the raw-z path (`accounted_epsilon` null) the backend rejects the upload **by
design**: the platform refuses unaccounted DP claims.

## 4. Read path — inference and FL-server warm-start (BA-11)

`RegistryModelResolver` (`backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/service/RegistryModelResolver.java:32`)
is the shared bean two otherwise-unrelated callers both depend on — it exists as a separate bean
specifically because `FlServerManager` cannot depend on `ProjectService` (Spring rejects the
resulting circular reference; `RegistryModelResolver.java:20-24`):

```
ProjectService.resolveInferenceTarget(projectId)          FlServerManager.startLocalServer(project, ...)
        │                                                          │
        └───────────────────────┬──────────────────────────────────┘
                                 ▼
                  RegistryModelResolver.resolveModelPath(project)
                                 │
                  1. headArtifact(project) — skip entirely for "LLM_LORA"
                     (safetensors head; .npz reader can't parse it)      -> Optional.empty()
                  2. else: ModelArtifactRepository
                     .findFirstByProjectIdAndKindOrderByCreatedAtDesc(
                         projectId, FULL_CHECKPOINT)                     -> the project's current head, or empty
                  3. if a head exists: materializeBlob(head.blobSha256)
                       - cache hit?  <cacheDir>/<sha256>.npz already present -> reuse, done
                       - else: blobStore.get(sha256)  [integrity-checked; THROWS on mismatch/IO error]
                              write temp file -> atomic rename to <sha256>.npz
                                 │
                                 ▼
                     Optional<String> localFilesystemPath
```

(`RegistryModelResolver.java:47-105`; call sites at
`backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/service/ProjectService.java:792`
— plus the cheaper `hasModel(...)` probe at `:753`, which drives "is this project inferable?" —
and `backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/orchestration/FlServerManager.java:177`.)

Both callers fall back to the legacy `.npz` path (`project.getModelPath()`) when the resolver returns
`Optional.empty()` — a project with no registry artifact yet (pre-registry data, or a project that
never finished a run) or a LoRA project. Critically, **a fallback only fires on "no artifact"**, never
on "artifact unreadable": if a registry blob exists but fails to read or fails its integrity check,
`ArtifactBlobStore.get` throws unchecked and that exception propagates out of `resolveModelPath`
through **both** call sites — neither `ProjectService` (inference) nor `FlServerManager` (warm-start)
catches it. The only failure `resolveModelPath` itself absorbs is the narrower `IOException` from the
local cache write, which it logs and degrades to `Optional.empty()` — and only then does a caller fall
back (`RegistryModelResolver.java:65-71`). The intent, stated directly in the code, is
fail-loud: a corrupt or unreadable registry head must never be silently masked by the `.npz` fallback,
because that fallback is supposed to mean "no artifact", not "artifact unreadable"
(`RegistryModelResolver.java:56-58`).

Cache config: `app.model-blob-cache.dir` (default `models/blob-cache`,
`application.properties:211`). The materialized file is always named `<sha256>.npz` regardless of the
artifact's original bytes' internal format — accurate for `FULL_CHECKPOINT` (whose registered bytes
already are a `.npz`), and moot for `LLM_LORA` (which this resolver never returns a path for at all,
per step 1 above).

## 5. HTTP surface

| Method & path | Controller | Auth | Purpose |
|---|---|---|---|
| `GET /api/artifacts?projectId=` | `ArtifactController.list` | Session, org-scoped (filtered, never leaks) | The project's artifacts the caller may see, newest first |
| `GET /api/artifacts/{id}` | `ArtifactController.get` | Session, org-scoped (404 on cross-org) | Artifact metadata (`ArtifactDto`, incl. `blobSha256`) |
| `GET /api/artifacts/{id}/blob` | `ArtifactController.blob` | Session, org-scoped (404 on cross-org) | The immutable bytes; integrity-checked on read; sha256 echoed as a strong ETag |
| `GET /api/artifacts/latest?projectId=&kind=` | `ArtifactController.latest` | Session, org-scoped | The project's current head artifact of `kind` (default `FULL_CHECKPOINT`) |
| `GET /api/artifacts/{id}/lineage` | `ArtifactLineageController.lineage` | Session, org-scoped (404 on cross-org) | The provenance chain, base → … → the artifact |
| `POST /api/internal/projects/{id}/artifacts` | `InternalArtifactController.registerArtifact` | `X-Internal-Key` **and** the per-run `X-Internal-Run-Token`, whose scope must match the `{id}` in the path (SE-7) | `fl_server.py`'s registration callback |
| `GET /api/marketplace/adapters` | `MarketplaceController.browse` | Session, org-scoped | Published `LORA_ADAPTER`s the caller's orgs can see, newest-published first |
| `POST /api/marketplace/adapters/{id}/publish` | `MarketplaceController.publish` | Session, owner-or-admin | Publish a `LORA_ADAPTER` to the org marketplace (`FE-12`) |
| `DELETE /api/marketplace/adapters/{id}/publish` | `MarketplaceController.unpublish` | Session, owner-or-admin | Withdraw |

Sources: `backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/controller/ArtifactController.java:37-106`,
`.../controller/ArtifactLineageController.java:24-61`,
`.../controller/InternalArtifactController.java:24-51`,
`.../controller/MarketplaceController.java:28-52`.

The read-side controllers share one rule: a cross-org id is always a **404**, never a 403, so existence
never leaks (`ArtifactController.java:112-129`, `ArtifactLineageController.java:52`, `:73`). The marketplace
splits along the same line: `browse` is org-scoped at the query (unrestricted admins see everything, a
caller with no visible orgs sees nothing), while `publish`/`unpublish` resolve the artifact, check
`orgScope.allows(...)` — 404 if not — and then `requireOwnerOrAdmin(project)`, so an in-org
non-owner gets a 403.

The internal endpoint sits behind `InternalApiKeyFilter`, which gates all of `/api/internal/**` on the
shared `X-Internal-Key` header (constant-time compared, 401 if absent, mismatched, or unconfigured)
**and** on the per-run `X-Internal-Run-Token`, which must be scoped to the same project id the path
names — otherwise 401/403. Full detail in
[02 - Security and Authentication](02_security_and_auth.md)
(`backend/fl-platform-api/src/main/java/com/federated/fl_platform_api/security/InternalApiKeyFilter.java:46-92`).

## 6. Lineage traversal

`ArtifactRegistryService.getLineageChain(artifactId)` walks `ArtifactLineage.findByChildId` recursively
toward parents, then returns the visited set in post-order (parents before children) using a
`LinkedHashMap` for stable ordering and a `seen` set for cycle-safety
(`ArtifactRegistryService.java:138-152`). `ArtifactLineageController` exposes this at
`GET /api/artifacts/{id}/lineage` as a flat list of `{id, kind, sha256, baseModelRef, licenseTag,
createdAt}` (`ArtifactLineageController.java:39-60`) — e.g. for a continued LoRA run, the chain would
read `BASE_REF → LORA_ADAPTER(round 1) → LORA_ADAPTER(round 2)`.

## 7. Flyway migrations

| Migration | Adds |
|---|---|
| `V12__model_artifact_registry.sql` | `artifact_blobs`, `model_artifacts`, `artifact_lineage` — the keystone (`DA-1`). Also `uq_model_artifact_run_kind UNIQUE (run_id, kind)`, `uq_artifact_lineage UNIQUE (child_id, parent_id, relationship)` and `CHECK (child_id <> parent_id)` |
| `V18__artifact_marketplace_publish.sql` | `model_artifacts.published` / `published_at` + a `(org_id, kind, published)` index for the marketplace feed (`FE-12`) |
| `V19__cascade_delete_run_subtree.sql` | Not a registry migration, but it states the registry's deletion policy explicitly: `model_artifacts.project_id`/`run_id` stay `ON DELETE SET NULL`, `artifact_blobs` are untouched (globally deduplicated — a blob another project still references must survive), and `artifact_lineage` stays `ON DELETE RESTRICT`. Refcount-safe blob GC is deferred to BA-11 Chunk C |
| `V21__base_ref_unique_index.sql` | The partial unique index `uq_base_ref_org_model ON model_artifacts (org_id, base_model_ref) WHERE kind = 'BASE_REF'` — makes one `BASE_REF` per `(org, base model)` a DB invariant and backs the race-safe insert-if-absent in `findOrCreateBaseRef` (`DA-3` hardening; see §1) |

`V12`'s own header comment states the intent this page documents: it "replaces the 'one overwritable
`.npz` at `projects.model_path`' model with a versioned, content-addressed, lineage-tracked registry"
while leaving `projects.model_path` "intentionally... untouched (legacy writers still use it)"
(`backend/fl-platform-api/src/main/resources/db/migration/V12__model_artifact_registry.sql:1-20`).

## 8. Related: the adapter bundle format (DA-9)

A `LORA_ADAPTER`'s registered bytes are also packaged as a versioned "bundle" for mobile/marketplace
delivery — safetensors payload + a JSON manifest whose `artifact_sha256` is, by construction, the exact
same content hash the registry stores (`framework/src/fedlearn/bundle/BUNDLE_FORMAT.md`). See that file
for the manifest schema; this page only tracks where its "Fixture-MVP boundary" section needed a
correction (below).

## 9. Retiring the `.npz`-overwrite gap

Older material (design comments, wiki prose, and one bundle-format doc) described or assumed a single
mutable `.npz`, overwritten every round, as the platform's *only* model store — no history, no dedup,
no way to express "this adapter came from that base." That gap is closed by the registry described
above. Concretely, in this pass:

- **`framework/src/fedlearn/bundle/BUNDLE_FORMAT.md`** ("Fixture-MVP boundary" section) stated, as a
  present-tense fact, that "`fl_server.py` currently registers the legacy `.npz` bytes" and listed
  "register the safetensors artifact bytes" as an open follow-on. That is now **stale for the LoRA
  path**: `_emit_and_register_lora_bundle` (landed after that doc was written; see `fl_server.py:119-144`)
  already serializes the adapter to safetensors and registers *those* bytes, not the `.npz`. It remains
  **accurate** that the mobile bundle-*provisioning* path (`scripts/stage_model_bundle.py`) still stages
  a hardcoded `TINYNET_GOLDEN` fixture rather than a project's real recipe — that half of the boundary is
  still open. The doc has been corrected in place (see the diff on this branch) rather than rewritten,
  per the "don't delete history-relevant context" rule — the two follow-ons are now marked done/open
  individually instead of both open.
- **`wikis/backend/01_architecture_overview.md`** ("`ProjectService` ... asks the `ModelInitializer` to
  build a local `.npz` weights file") and **`wikis/backend/03_project_management.md`** ("Determine File
  Path" / "Model Initialization" / "Finalize DB Entry" steps, all `.npz`-based) describe **project
  creation**, not the registry: `ModelInitializer` still writes the project's *initial*, pre-training
  architecture to a `.npz` at creation time, and that description is accurate and unrelated to what the
  registry replaces. Both pages now carry a pointer to this page so a reader isn't left assuming the
  `.npz` is the only place a trained model ever lives — see the one-line addition on each.
- **Everywhere else `.npz`/"overwrite" appears in code** (`RegistryModelResolver`, `ArtifactController`,
  `ArtifactBlobStore`, `ModelArtifact` javadoc, `fl_server.py` comments) is **already correct as
  written** — those comments explicitly call the `.npz` "legacy" or "overwritable" in describing the
  registry that supersedes it. Nothing needed retiring there.

What genuinely still relies on `.npz`, by design, not by gap:

1. **`FULL_CHECKPOINT` byte format.** A full checkpoint's registered bytes are the `.npz` bytes
   themselves — that is the imaging air-gap export format, not a placeholder (`BUNDLE_FORMAT.md`,
   "Serialization").
2. **`RegistryModelResolver`'s local cache filename.** Materialized registry blobs are cached as
   `<sha256>.npz` (`RegistryModelResolver.java:89`) because, for the only kind this resolver ever
   materializes (`FULL_CHECKPOINT`), the bytes really are an `.npz` archive — this is a real, current
   mechanic, not the retired vision.
3. **`projects.model_path` itself.** Still written by `fl_server.py` as the training loop's working
   file — the final `np.savez(save_path, …)` at `fl_server.py:1132-1136` runs *before* and
   independently of registration — and still the fallback read path when no registry artifact exists.
   The registry is additive, not a replacement of the column.
