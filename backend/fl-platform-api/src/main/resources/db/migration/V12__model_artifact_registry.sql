-- V12 — content-addressed model-artifact registry (the keystone, DA-1).
--
-- Replaces the "one overwritable .npz at projects.model_path" model with a versioned,
-- content-addressed, lineage-tracked registry. It deliberately splits TWO identities that the
-- naive single-table design conflates:
--
--   artifact_blobs   — immutable, GLOBALLY content-addressed storage keyed by sha256. Write-once
--                      and deduplicated: identical bytes from any run/org collapse to ONE row.
--                      Carries NO tenant or provenance semantics.
--   model_artifacts  — per-ORG, per-run PROVENANCE rows that POINT at a blob. blob_sha256 is NOT
--                      unique, so two orgs/runs may record identical bytes as separate provenance
--                      rows sharing one blob — storage dedups, provenance never does. This is what
--                      makes a shared frozen base (BASE_REF, used by many orgs) and OrgScopeFilter
--                      tenant isolation both correct, and closes the content-hash existence oracle.
--   artifact_lineage — base->adapter / continued-training DAG edges between model_artifacts.
--
-- Append-only / write-new-not-overwrite: run and project deletion SET NULL the provenance FKs so an
-- artifact outlives its producer; lineage FKs RESTRICT so edges never dangle. projects.model_path is
-- intentionally left untouched (legacy writers still use it) — backfill + a head-artifact pointer are
-- a later slice (DA-2/DA-3).
--
-- Conventions (match V11): UUID PKs assigned by JPA (no DB-side default); TIMESTAMP WITH TIME ZONE;
-- enums as VARCHAR + @Enumerated(STRING) (no native pg enum); JSON as TEXT.

CREATE TABLE artifact_blobs (
    sha256      VARCHAR(64)              PRIMARY KEY,
    size_bytes  BIGINT                   NOT NULL,
    backend     VARCHAR(16)              NOT NULL,   -- 'LOCAL_FS' | 'S3'
    created_at  TIMESTAMP WITH TIME ZONE NOT NULL,
    CONSTRAINT chk_artifact_blobs_sha256_hex CHECK (sha256 ~ '^[0-9a-f]{64}$')
);

CREATE TABLE model_artifacts (
    id             UUID                     PRIMARY KEY,
    org_id         UUID                     NOT NULL REFERENCES organizations(id),
    blob_sha256    VARCHAR(64)              NOT NULL REFERENCES artifact_blobs(sha256),
    kind           VARCHAR(32)              NOT NULL,   -- FULL_CHECKPOINT | LORA_ADAPTER | BASE_REF
    project_id     UUID                     REFERENCES projects(id) ON DELETE SET NULL,
    run_id         UUID                     REFERENCES runs(id)     ON DELETE SET NULL,
    recipe_key     VARCHAR(64),
    base_model_ref VARCHAR(255),
    license_tag    VARCHAR(64),
    eval_card_json TEXT,
    created_by     BIGINT                   REFERENCES users(id)    ON DELETE SET NULL,
    created_at     TIMESTAMP WITH TIME ZONE NOT NULL,
    -- One artifact per (run, kind): a run produces at most one LORA_ADAPTER / FULL_CHECKPOINT.
    -- run_id is NULL for BASE_REF / imported artifacts; Postgres treats NULLs as distinct here,
    -- so many run-less rows are allowed.
    CONSTRAINT uq_model_artifact_run_kind UNIQUE (run_id, kind)
);

CREATE INDEX idx_model_artifacts_org     ON model_artifacts(org_id);
CREATE INDEX idx_model_artifacts_project ON model_artifacts(project_id);
CREATE INDEX idx_model_artifacts_run     ON model_artifacts(run_id);
CREATE INDEX idx_model_artifacts_blob    ON model_artifacts(blob_sha256);

CREATE TABLE artifact_lineage (
    id           UUID                     PRIMARY KEY,
    child_id     UUID                     NOT NULL REFERENCES model_artifacts(id) ON DELETE RESTRICT,
    parent_id    UUID                     NOT NULL REFERENCES model_artifacts(id) ON DELETE RESTRICT,
    relationship VARCHAR(16)              NOT NULL,   -- ADAPTER_OF | DERIVED_FROM | CONTINUED_FROM
    created_at   TIMESTAMP WITH TIME ZONE NOT NULL,
    CONSTRAINT uq_artifact_lineage       UNIQUE (child_id, parent_id, relationship),
    CONSTRAINT chk_artifact_lineage_self CHECK (child_id <> parent_id)
);

CREATE INDEX idx_artifact_lineage_parent ON artifact_lineage(parent_id);
CREATE INDEX idx_artifact_lineage_child  ON artifact_lineage(child_id);
